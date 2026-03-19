"""
Population Single-Task Learning (P-STL)
One model trained on all participants pooled together, separately for AR and VA.

Usage
-----
    python pstl.py                  # runs on VREED (default)
    python pstl.py --dataset dssn_eq
    python pstl.py --dataset dssn_em
"""
import argparse
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, EPOCHS, MAX_NORM, N_FOLDS,
                    MTL_SHARED_LR, L2_TASK, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data import create_sliding_windows, arrays_to_loader
from dataset_configs.loader import load_dataset
from models import SingleTaskModel
from utils import set_all_seeds, create_kfold_splits, aggregate_results
from training import evaluate_per_participant, save_all_results


def parse_args():
    p = argparse.ArgumentParser(description='P-STL experiment')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'],
                   help='Dataset to run on (default: vreed)')
    return p.parse_args()


# =============================
# HELPERS
# =============================
def _make_pool_loader(data_df, label_type, cfg, shuffle):
    """Windowed loader over the full pooled DataFrame."""
    X, y_ar, y_va, _, _ = create_sliding_windows(
        data_df, cfg['window_size'], cfg['stride'],
        trial_col='trial_global', feature_cols=cfg['feature_cols'])
    y = y_ar if label_type == 'ar' else y_va
    return arrays_to_loader(X, y, cfg['pstl_batch'], shuffle=shuffle, seed=SEED)


def _train_single(label_type, lr, l2_lambda, train_df, cfg, device):
    model   = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = _make_pool_loader(train_df, label_type, cfg, shuffle=True)
    t0 = time.time()

    for epoch in range(EPOCHS):
        ep_start = time.time()
        model.train()
        running = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_b), y_b)
            if torch.isnan(loss):
                raise ValueError(f"NaN at epoch {epoch+1} [{label_type.upper()}]")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
            running += loss.item()
        sched.step(running / len(loader))
        elapsed = time.time() - ep_start
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label_type.upper()}] Epoch {epoch+1}/{EPOCHS}  "
                  f"loss={running/len(loader):.4f}  ({elapsed:.1f}s)")

    total = time.time() - t0
    print(f"  [{label_type.upper()}] Training complete in {total:.1f}s")
    return model


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, learning_rates, l2_lambdas,
                          train_df, cfg, device, output_dir):
    splits = cfg['splits']
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING  [{label_type.upper()}]  P-STL"
          f"  ({cfg['results_prefix']})\n{'='*60}")
    tuning_t0 = time.time()
    results = []
    for lr in learning_rates:
        for l2 in l2_lambdas:
            fold_f1s = []
            for fold_i in range(N_FOLDS):
                train_ids, val_ids = [], []
                for pid in sorted(splits):
                    folds = create_kfold_splits(splits[pid]['train'], N_FOLDS)
                    tr_v, va_v = folds[fold_i]
                    train_ids += [f"{pid}_{v}" for v in tr_v]
                    val_ids   += [f"{pid}_{v}" for v in va_v]

                tr_fold = train_df[train_df['trial_global'].isin(train_ids)].reset_index(drop=True)
                va_fold = train_df[train_df['trial_global'].isin(val_ids)].reset_index(drop=True)

                if len(tr_fold) == 0 or len(va_fold) == 0:
                    fold_f1s.append(0.0); continue

                model   = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
                opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
                sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
                loss_fn = nn.BCEWithLogitsLoss()

                tr_loader = _make_pool_loader(tr_fold, label_type, cfg, shuffle=True)
                va_loader = _make_pool_loader(va_fold, label_type, cfg, shuffle=False)

                best_f1 = 0.0
                fold_t0 = time.time()
                for ep in range(EPOCHS):
                    model.train()
                    run = 0.0
                    for X_b, y_b in tr_loader:
                        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        loss = loss_fn(model(X_b), y_b)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                        opt.step(); run += loss.item()
                    sched.step(run / len(tr_loader))

                    model.eval()
                    tp = fp = fn = 0
                    with torch.no_grad():
                        for X_v, y_v in va_loader:
                            X_v, y_v = X_v.to(device, non_blocking=True), y_v.to(device, non_blocking=True)
                            pred = (torch.sigmoid(model(X_v)) > 0.5).float()
                            tp  += (y_v * pred).sum().item()
                            fp  += ((1-y_v)*pred).sum().item()
                            fn  += (y_v*(1-pred)).sum().item()
                    p = tp/(tp+fp+1e-7); r = tp/(tp+fn+1e-7)
                    cur_f1 = 2*p*r/(p+r+1e-7)
                    best_f1 = max(best_f1, cur_f1)

                fold_elapsed = time.time() - fold_t0
                fold_f1s.append(best_f1)
                print(f"  fold {fold_i+1}: f1={best_f1:.4f}  (lr={lr}, l2={l2})  "
                      f"[{fold_elapsed:.1f}s]")

            avg = np.mean(fold_f1s)
            results.append({'lr': lr, 'l2': l2, 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
            print(f"  avg f1={avg:.4f}")

    best = max(results, key=lambda x: x['avg_f1'])
    tuning_elapsed = time.time() - tuning_t0
    print(f"\nBest: lr={best['lr']}, l2={best['l2']}, f1={best['avg_f1']:.4f}")
    print(f"HP tuning [{label_type.upper()}] completed in {tuning_elapsed:.1f}s")
    with open(os.path.join(output_dir, f'{label_type}_tuning.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr'], best['l2']


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    args = parse_args()
    exp_t0 = time.time()

    # Load dataset
    df, cfg = load_dataset(args.dataset, preserve_trial_order=True)
    splits  = cfg['splits']
    p_ids   = cfg['participant_ids']
    prefix  = cfg['results_prefix']

    BATCH_SIZE = cfg['pstl_batch']
    OUTPUT_DIR = os.path.join(RESULTS_DIR, f'{prefix}_MTL', f'{prefix}_pstl_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {OUTPUT_DIR}")

    # Ensure trial_global and participant_trial_encoded columns exist
    if 'participant_trial_encoded' not in df.columns:
        id_trial_col = cfg['id_trial_col']
        df['participant_trial_encoded'] = df[id_trial_col].astype(str)
    if 'trial_global' not in df.columns:
        df['trial_global'] = df['ID'].astype(str) + '_' + df['Trial'].astype(str)

    _train_set = {f"{pid}_{vid}" for pid in splits for vid in splits[pid]['train']}
    _test_set  = {f"{pid}_{vid}" for pid in splits for vid in splits[pid]['test']}

    train_df = df[df['participant_trial_encoded'].isin(_train_set)].reset_index(drop=True)
    test_df  = df[df['participant_trial_encoded'].isin(_test_set)].reset_index(drop=True)

    # Hyperparameter tuning
    best_lr_ar, best_l2_ar = hyperparameter_tuning(
        'ar', [MTL_SHARED_LR], [L2_TASK], train_df, cfg, device, OUTPUT_DIR)
    best_lr_va, best_l2_va = hyperparameter_tuning(
        'va', [MTL_SHARED_LR], [L2_TASK], train_df, cfg, device, OUTPUT_DIR)

    # Train final models
    print("\n" + "="*60 + "\nTRAINING AR\n" + "="*60)
    set_all_seeds(SEED)
    model_ar = _train_single('ar', best_lr_ar, best_l2_ar, train_df, cfg, device)
    torch.save(model_ar.state_dict(), os.path.join(OUTPUT_DIR, 'model_ar.pth'))

    print("\n" + "="*60 + "\nTRAINING VA\n" + "="*60)
    set_all_seeds(SEED)
    model_va = _train_single('va', best_lr_va, best_l2_va, train_df, cfg, device)
    torch.save(model_va.state_dict(), os.path.join(OUTPUT_DIR, 'model_va.pth'))

    # Build per-participant test loaders
    test_loaders_ar, test_loaders_va = {}, {}
    for task_idx, pid in enumerate(p_ids):
        test_trials = [f"{pid}_{v}" for v in splits[pid]['test']]
        p_test = test_df[test_df['trial_global'].isin(test_trials)].reset_index(drop=True)
        if len(p_test) == 0:
            continue
        X_test, y_ar_test, y_va_test, _, _ = create_sliding_windows(
            p_test, cfg['window_size'], cfg['stride'],
            feature_cols=cfg['feature_cols'])
        X_t = torch.tensor(X_test, dtype=torch.float32)
        pin = device.type == 'cuda'
        test_loaders_ar[task_idx] = DataLoader(
            TensorDataset(X_t, torch.tensor(y_ar_test, dtype=torch.float32).reshape(-1, 1)),
            batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
        test_loaders_va[task_idx] = DataLoader(
            TensorDataset(X_t, torch.tensor(y_va_test, dtype=torch.float32).reshape(-1, 1)),
            batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    results = evaluate_per_participant(
        (model_ar, model_va), test_loaders_ar, test_loaders_va,
        p_ids, device, is_mtl=False)

    agg = aggregate_results(results)

    results_df, ar_stds, va_stds = save_all_results(
        results, agg, OUTPUT_DIR,
        method_name='P-STL',
        misclassification_csv=f'{prefix}_pstl_misclassification_rates.csv')

    with open(os.path.join(OUTPUT_DIR, 'pstl_results.pkl'), 'wb') as f:
        pickle.dump({**agg, 'per_participant': results,
                     'per_participant_table': results_df,
                     **ar_stds, **va_stds}, f)

    total_time = time.time() - exp_t0
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"Total experiment time: {total_time:.1f}s ({total_time/60:.1f}min)")
