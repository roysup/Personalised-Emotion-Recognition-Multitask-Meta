"""
Subject-Independent (SI) Baseline
Train on 20 participants, evaluate on 6 held-out test participants.
"""
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
from config import (SEED, WINDOW_SIZE, STRIDE, EPOCHS, MAX_NORM, N_FOLDS,
                    PSTL_BATCH_SIZE, MTL_SHARED_LR, L2_TASK,
                    HARDCODED_SPLITS, TEST_PARTICIPANTS, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (set_all_seeds, compute_metrics_from_cm,
                   aggregate_mtml_results, make_kfolds,
                   compute_per_participant_stds, print_determinism_summary)
from data import create_sliding_windows, arrays_to_loader
from models import SingleTaskModel
from dataset_configs.vreed import load_vreed_df
hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir       = os.path.join(BASE_OUTPUT_DIR, 'VREED_SI')
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE     = PSTL_BATCH_SIZE
learning_rates = [MTL_SHARED_LR]
l2_lambdas     = [L2_TASK]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_all_seeds(SEED)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df(preserve_trial_order=True)

existing_ids    = set(df['ID'].unique())
participant_ids = sorted([int(pid) for pid in hardcoded_splits.keys() if pid in existing_ids])
print(f"Total participants: {len(participant_ids)}")

test_participants  = list(TEST_PARTICIPANTS)
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {train_participants}\nTest:  {test_participants}")


def train_model(frames, labels, lr, l2_lambda, epochs=EPOCHS):
    model   = SingleTaskModel().to(device)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader  = arrays_to_loader(frames, labels, BATCH_SIZE, shuffle=True, seed=SEED)
    for epoch in range(epochs):
        model.train()
        run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_b), y_b)
            if torch.isnan(loss):
                raise ValueError(f"NaN at epoch {epoch+1} [SI train_model]")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
        sched.step(run / len(loader))
    return model

# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type='AR'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type}] SI\n{'='*60}")
    results     = []
    train_folds = make_kfolds(train_participants, seed=SEED)
    for lr in learning_rates:
        for l2 in l2_lambdas:
            fold_f1s = []
            for fold_i in range(N_FOLDS):
                val_ps = train_folds[fold_i]
                tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
                tr_pte = [f"{p}_{v}" for p in tr_ps if p in hardcoded_splits
                          for v in hardcoded_splits[p]['train']]
                va_pte = [f"{p}_{v}" for p in val_ps if p in hardcoded_splits
                          for v in hardcoded_splits[p]['train']]
                tr_df = df[df['trial_global'].isin(tr_pte)].reset_index(drop=True)
                va_df = df[df['trial_global'].isin(va_pte)].reset_index(drop=True)
                Xtr, _ar, _va, _, _ = create_sliding_windows(
                    tr_df, WINDOW_SIZE, STRIDE, trial_col='trial_global')
                ytr = _ar if label_type.upper() == 'AR' else _va
                Xva, _ar, _va, _, _ = create_sliding_windows(
                    va_df, WINDOW_SIZE, STRIDE, trial_col='trial_global')
                yva = _ar if label_type.upper() == 'AR' else _va
                if len(Xtr) == 0 or len(Xva) == 0:
                    fold_f1s.append(0.0); continue
                model = train_model(Xtr, ytr, lr, l2)
                if model is None:
                    fold_f1s.append(0.0); continue
                model.eval()
                tp = fp = fn = 0
                loader = arrays_to_loader(Xva, yva, BATCH_SIZE, shuffle=False)
                with torch.no_grad():
                    for X_v, y_v in loader:
                        pred = (torch.sigmoid(model(X_v.to(device, non_blocking=True))) > 0.5).float().cpu()
                        tp += (y_v * pred).sum().item()
                        fp += ((1 - y_v) * pred).sum().item()
                        fn += (y_v * (1 - pred)).sum().item()
                p = tp / (tp + fp + 1e-7)
                r = tp / (tp + fn + 1e-7)
                fold_f1s.append(2 * p * r / (p + r + 1e-7))
                print(f"  fold {fold_i+1}: f1={fold_f1s[-1]:.4f}")
            avg = np.mean(fold_f1s)
            results.append({'lr': lr, 'l2': l2, 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
            print(f"  avg f1={avg:.4f}")
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type.lower()}_tuning_results_si.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr'], best['l2']

# =============================
# EVALUATION HELPER
# =============================
def evaluate_per_participant(model, test_participants, test_data, label_type):
    """
    Evaluate a single pooled model per-participant on test data.
    Returns result dicts using the prefixed key convention:
      {p}_acc, {p}_f1, y_true_{p}, y_pred_probs_{p}, ...  where p = label_type.lower()
    """
    p = label_type.lower()   # 'ar' or 'va'
    model.eval()
    results = []
    for pid in sorted(test_participants):
        test_trials = [f"{pid}_{v}" for v in hardcoded_splits[pid]['test']]
        p_df = test_data[test_data['trial_global'].isin(test_trials)].reset_index(drop=True)
        if len(p_df) == 0:
            continue
        X, _ar, _va, _, _ = create_sliding_windows(
            p_df, WINDOW_SIZE, STRIDE, trial_col='trial_global')
        y = _ar if label_type.upper() == 'AR' else _va
        if len(X) == 0:
            continue
        loader = arrays_to_loader(X, y, BATCH_SIZE, shuffle=False)
        probs, trues = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                probs.extend(torch.sigmoid(model(X_b.to(device, non_blocking=True))).cpu().numpy().flatten())
                trues.extend(y_b.numpy().flatten())
        y_true = np.array(trues).astype(int)
        y_prob = np.array(probs)
        y_pred = (y_prob > 0.5).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc, prec, rec, f1 = compute_metrics_from_cm(cm)
        results.append({
            'participant_id':       pid,
            'cm':                   cm,
            f'{p}_acc':             acc,
            f'{p}_precision':       prec,
            f'{p}_recall':          rec,
            f'{p}_f1':              f1,
            f'y_true_{p}':          y_true,
            f'y_pred_{p}':          y_pred,
            f'y_pred_probs_{p}':    y_prob,
        })
        print(f"  Participant {pid}: Acc={acc:.4f} F1={f1:.4f}")
    return results

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    experiment_t0 = time.time()

    best_lr_ar, best_l2_ar = hyperparameter_tuning('AR')
    best_lr_va, best_l2_va = hyperparameter_tuning('VA')

    train_pte  = [f"{p}_{v}" for p in train_participants if p in hardcoded_splits
                  for v in hardcoded_splits[p]['train']]
    test_pte   = [f"{p}_{v}" for p in test_participants if p in hardcoded_splits
                  for v in hardcoded_splits[p]['test']]
    train_data = df[df['trial_global'].isin(train_pte)].reset_index(drop=True)
    test_data  = df[df['trial_global'].isin(test_pte)].reset_index(drop=True)

    print('\n' + '='*60 + '\nTRAINING AR\n' + '='*60)
    Xtr_ar, ytr_ar, _, _, _ = create_sliding_windows(
        train_data, WINDOW_SIZE, STRIDE, trial_col='trial_global')
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_ar = train_model(Xtr_ar, ytr_ar, best_lr_ar, best_l2_ar)
    print(f"  AR training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_ar.state_dict(), os.path.join(output_dir, 'model_ar_si.pth'))

    print('\n' + '='*60 + '\nTRAINING VA\n' + '='*60)
    Xtr_va, _, ytr_va, _, _ = create_sliding_windows(
        train_data, WINDOW_SIZE, STRIDE, trial_col='trial_global')
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_va = train_model(Xtr_va, ytr_va, best_lr_va, best_l2_va)
    print(f"  VA training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_va.state_dict(), os.path.join(output_dir, 'model_va_si.pth'))

    print('\n' + '='*60 + '\nEVALUATION AR\n' + '='*60)
    results_ar = evaluate_per_participant(model_ar, test_participants, test_data, 'AR')
    print('\n' + '='*60 + '\nEVALUATION VA\n' + '='*60)
    results_va = evaluate_per_participant(model_va, test_participants, test_data, 'VA')

    agg = aggregate_mtml_results(results_ar, results_va)

    roc_data = {
        'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
        'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']},
    }
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    for cm, label, fname in [(agg['cm_ar'], 'AR', 'ar_cm_si.png'),
                              (agg['cm_va'], 'VA', 'va_cm_si.png')]:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'{label}=0', f'{label}=1'],
                    yticklabels=[f'{label}=0', f'{label}=1'])
        plt.title(f'Confusion Matrix {label} (SI)')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, fname)); plt.close()

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')

    final_results = {
        'train_participants': train_participants,
        'test_participants':  test_participants,
        'best_hyperparameters': {'AR': {'lr': best_lr_ar, 'l2': best_l2_ar},
                                 'VA': {'lr': best_lr_va, 'l2': best_l2_va}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc', 'precision', 'recall', 'f1', 'auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc', 'precision', 'recall', 'f1', 'auc']},
        **ar_stds,   # keys already have the right names: ar_acc_std, ar_f1_std, etc.
        **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'si_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    print_determinism_summary(
        {f'ar_{k}': final_results[f'ar_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        {f'va_{k}': final_results[f'va_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
