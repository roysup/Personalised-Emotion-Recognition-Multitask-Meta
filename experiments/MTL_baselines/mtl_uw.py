"""
MTL Uncertainty Weighting (UW)
Same HPS architecture, but adds a learnable log_vars parameter per task
to automatically weight the per-task losses.
Seed is reset before AR training and again before VA training.

Usage
-----
    python mtl_uw.py                  # runs on VREED (default)
    python mtl_uw.py --dataset dssn_eq
    python mtl_uw.py --dataset dssn_em
"""
import argparse
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, EPOCHS, MAX_NORM,
                    MTL_SHARED_LR, MTL_TASK_LR,
                    L2_SHARED, L2_TASK, RESULTS_DIR)
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from data import make_mtl_loader
from dataset_configs.loader import load_dataset
from models import MTLModelUW
from utils import set_all_seeds, aggregate_results
from training import save_all_results, evaluate_mtl_all


def parse_args():
    p = argparse.ArgumentParser(description='MTL-UW experiment')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'],
                   help='Dataset to run on (default: vreed)')
    return p.parse_args()


# =============================
# TRAINING
# =============================
def _train_uw(label_type, lr_shared, lr_task, lr_logvar,
              train_data_dict, cfg, device, output_dir):
    num_tasks = cfg['num_tasks']
    loader, _, _ = make_mtl_loader(
        train_data_dict, cfg['window_size'], cfg['stride'],
        label_type=label_type, batch_size=cfg['mtl_batch'], seed=SEED,
        feature_cols=cfg['feature_cols'])

    model = MTLModelUW(num_tasks, input_dim=cfg['input_dim']).to(device)
    opt   = optim.Adam([
        {'params': model.shared_parameters(),        'lr': lr_shared},
        {'params': model.task_specific_parameters(), 'lr': lr_task},
        {'params': [model.log_vars],                 'lr': lr_logvar},
    ])
    sched     = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn   = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    ckpt_path = os.path.join(output_dir, f'best_model_{label_type}_hps_uw.pt')

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in loader:
            X_b, y_b, task_ids, _ = [b.to(device, non_blocking=True) for b in batch]

            if len(torch.unique(task_ids)) != num_tasks:
                raise ValueError(
                    f"Batch has {len(torch.unique(task_ids))} tasks, expected {num_tasks}")

            opt.zero_grad(set_to_none=True)
            per_sample_loss = loss_fn(model(X_b, task_ids), y_b).squeeze(-1)
            log_vars        = model.log_vars[task_ids]
            precision       = torch.exp(-log_vars)
            weighted_loss   = (precision * per_sample_loss + log_vars).mean()
            total           = weighted_loss + model.compute_l2(L2_SHARED, L2_TASK)

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

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    args = parse_args()
    experiment_t0 = time.time()

    df, cfg = load_dataset(args.dataset)
    splits  = cfg['splits']
    p_ids   = cfg['participant_ids']
    prefix  = cfg['results_prefix']

    OUTPUT_DIR = os.path.join(RESULTS_DIR, f'{prefix}_MTL', f'{prefix}_hps_uw_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {OUTPUT_DIR}")

    # Prepare train/test data
    train_data, test_data = {}, {}
    for task_idx, pid in enumerate(p_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        train_data[task_idx] = p_df[p_df['Trial'].isin(splits[pid]['train'])].reset_index(drop=True)
        test_data[task_idx]  = p_df[p_df['Trial'].isin(splits[pid]['test'])].reset_index(drop=True)

    # Get UW log-var LRs from dataset config
    uw_lr_ar = cfg['uw_logvar_lr_ar']
    uw_lr_va = cfg['uw_logvar_lr_va']

    print("\n" + "="*60 + "\nTRAINING AR\n" + "="*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_ar = _train_uw('ar', MTL_SHARED_LR, MTL_TASK_LR, uw_lr_ar,
                          train_data, cfg, device, OUTPUT_DIR)
    print(f"  AR training complete in {time.time() - train_t0:.1f}s")

    print("\n" + "="*60 + "\nTRAINING VA\n" + "="*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_va = _train_uw('va', MTL_SHARED_LR, MTL_TASK_LR, uw_lr_va,
                          train_data, cfg, device, OUTPUT_DIR)
    print(f"  VA training complete in {time.time() - train_t0:.1f}s")

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    results = evaluate_mtl_all(model_ar, model_va, test_data,
                               p_ids, device,
                               cfg['window_size'], cfg['stride'],
                               feature_cols=cfg['feature_cols'])
    agg = aggregate_results(results)

    results_df, ar_stds, va_stds = save_all_results(
        results, agg, OUTPUT_DIR,
        method_name='MTL-UW',
        misclassification_csv=f'{prefix}_hps_uw_misclassification_rates.csv')

    with open(os.path.join(OUTPUT_DIR, 'hps_uw_results.pkl'), 'wb') as f:
        pickle.dump({**agg, 'per_participant': results,
                     'per_participant_table': results_df,
                     **ar_stds, **va_stds}, f)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
