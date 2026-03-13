"""
MTL Uncertainty Weighting (UW)
Same HPS architecture, but adds a learnable log_vars parameter per task
to automatically weight the per-task losses.
Seed is reset before AR training and again before VA training.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from config import SEED, WINDOW_SIZE, STRIDE, MAX_NORM, EPOCHS, HARDCODED_SPLITS
from data import create_sliding_windows, make_combined_mtl_loader
from dataset_configs.vreed import load_vreed_df, participant_ids
from models import MTLModelUW
from utils import (set_all_seeds, compute_metrics_from_cm)
from training import aggregate_results, save_all_results
from paths import RESULTS_DIR
BASE_OUTPUT_DIR = RESULTS_DIR

BATCH_SIZE  = 26
NUM_TASKS   = 26
SHARED_LR   = 3e-4
TASK_LR     = 1e-4
LOG_VAR_LR  = {'ar': 4e-3, 'va': 1e-3}
L2_TASK     = 1e-5

#BASE_OUTPUT_DIR = '/content/drive/MyDrive/Phase A/results/VREED'
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'VREED_hps_uw_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\nOutput: {OUTPUT_DIR}")

# =============================
# DATA
# =============================
df = load_vreed_df()


# =============================
# HELPERS
# =============================
def _train_uw(label_type, lr_shared, lr_task, lr_logvar, l2_task, train_data_dict):
    loader, _, _ = make_combined_mtl_loader(
        train_data_dict, WINDOW_SIZE, STRIDE,
        label_type=label_type, batch_size=BATCH_SIZE,
        num_tasks=NUM_TASKS, seed=SEED)

    model = MTLModelUW(NUM_TASKS).to(device)
    opt   = optim.Adam([
        {'params': model.shared_parameters(),        'lr': lr_shared},
        {'params': model.task_specific_parameters(), 'lr': lr_task},
        {'params': [model.log_vars],                 'lr': lr_logvar},
    ])
    sched     = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn   = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    ckpt_path = os.path.join(OUTPUT_DIR, f'best_model_{label_type}_hps_uw.pt')

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in loader:
            X_b, y_b, task_ids, _ = [b.to(device) for b in batch]

            if len(torch.unique(task_ids)) != NUM_TASKS:
                raise ValueError(
                    f"Batch has {len(torch.unique(task_ids))} tasks, expected {NUM_TASKS}")

            opt.zero_grad()
            per_sample_loss = loss_fn(model(X_b, task_ids), y_b).squeeze(-1)
            log_vars        = model.log_vars[task_ids]
            precision       = torch.exp(-log_vars)
            weighted_loss   = (precision * per_sample_loss + log_vars).mean()
            l2_reg          = l2_task * sum(p.norm(2)**2
                                            for p in model.task_specific_parameters()
                                            if p.requires_grad)
            total = weighted_loss + l2_reg

            if torch.isnan(total):
                raise ValueError(f"NaN at epoch {epoch+1} [{label_type.upper()}]")

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
            running += total.item()

        avg = running / len(loader)
        sched.step(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label_type.upper()}] Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}  "
                  f"log_vars_mean={model.log_vars.data.mean().item():.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path))
    return model


def _evaluate_all(model_ar, model_va, test_data_dict):
    from sklearn.metrics import confusion_matrix
    results = []
    for task_idx, pid in enumerate(participant_ids):
        test_df = test_data_dict.get(task_idx)
        if test_df is None or len(test_df) == 0:
            continue
        X, y_ar, y_va, _, _ = create_sliding_windows(
            test_df, WINDOW_SIZE, STRIDE, task_id=task_idx)
        if len(X) == 0:
            continue
        X_t    = torch.tensor(X, dtype=torch.float32).to(device)
        tids_t = torch.full((len(X),), task_idx, dtype=torch.long).to(device)

        model_ar.eval(); model_va.eval()
        with torch.no_grad():
            prob_ar = torch.sigmoid(model_ar(X_t, tids_t)).cpu().numpy().flatten()
            prob_va = torch.sigmoid(model_va(X_t, tids_t)).cpu().numpy().flatten()

        pred_ar = (prob_ar > 0.5).astype(int); y_ar_i = y_ar.astype(int)
        pred_va = (prob_va > 0.5).astype(int); y_va_i = y_va.astype(int)

        cm_ar = confusion_matrix(y_ar_i, pred_ar, labels=[0, 1])
        cm_va = confusion_matrix(y_va_i, pred_va, labels=[0, 1])
        ar_acc, ar_prec, ar_rec, ar_f1 = compute_metrics_from_cm(cm_ar)
        va_acc, va_prec, va_rec, va_f1 = compute_metrics_from_cm(cm_va)

        print(f"  Participant {pid}: AR acc={ar_acc:.4f} f1={ar_f1:.4f} | "
              f"VA acc={va_acc:.4f} f1={va_f1:.4f}")

        results.append({
            'task_idx': task_idx, 'participant_id': pid,
            'cm_ar': cm_ar, 'cm_va': cm_va,
            'ar_acc': ar_acc, 'ar_precision': ar_prec, 'ar_recall': ar_rec, 'ar_f1': ar_f1,
            'va_acc': va_acc, 'va_precision': va_prec, 'va_recall': va_rec, 'va_f1': va_f1,
            'y_true_ar': y_ar_i, 'y_pred_ar': pred_ar, 'y_pred_probs_ar': prob_ar,
            'y_true_va': y_va_i, 'y_pred_va': pred_va, 'y_pred_probs_va': prob_va,
        })
    return results

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    train_data, test_data = {}, {}
    for task_idx, pid in enumerate(participant_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        train_data[task_idx] = p_df[p_df['Trial'].isin(HARDCODED_SPLITS[pid]['train'])].reset_index(drop=True)
        test_data[task_idx]  = p_df[p_df['Trial'].isin(HARDCODED_SPLITS[pid]['test'])].reset_index(drop=True)

    print("\n" + "="*60 + "\nTRAINING AR\n" + "="*60)
    set_all_seeds(SEED)
    model_ar = _train_uw('ar', SHARED_LR, TASK_LR, LOG_VAR_LR['ar'], L2_TASK, train_data)

    print("\n" + "="*60 + "\nTRAINING VA\n" + "="*60)
    set_all_seeds(SEED)
    model_va = _train_uw('va', SHARED_LR, TASK_LR, LOG_VAR_LR['va'], L2_TASK, train_data)

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    results = _evaluate_all(model_ar, model_va, test_data)
    agg     = aggregate_results(results)

    results_df, ar_stds, va_stds = save_all_results(
        results, agg, OUTPUT_DIR,
        method_name='MTL-UW',
        misclassification_csv='VREED_hps_uw_misclassification_rates.csv')

    with open(os.path.join(OUTPUT_DIR, 'hps_uw_results.pkl'), 'wb') as f:
        pickle.dump({**agg, 'per_participant': results,
                     'per_participant_table': results_df,
                     **ar_stds, **va_stds}, f)
    print(f"\nAll results saved to: {OUTPUT_DIR}")