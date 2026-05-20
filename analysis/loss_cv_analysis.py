"""
Analysis 2 — Per-user training-loss coefficient of variation (CV) under MTL-HPS.

Motivation
----------
Uncertainty Weighting (UW) rebalances per-task loss contributions. For UW to
help via that mechanism, there must be meaningful variation in per-user loss
magnitudes to begin with. This script quantifies that variation by computing,
for the trained MTL-HPS model, the mean training-set BCE loss for each user and
reporting the coefficient of variation (CV = std / mean) across users.

It also reports a *training-free* proxy: at the UW optimum, the learned
log-variance satisfies sigma_u^2 = exp(log_vars_u) ~ per-task loss, so the CV of
exp(log_vars) from the UW checkpoint approximates the same quantity. The two
should broadly agree.

Reads
-----
    results/{prefix}_MTL/{prefix}_hps_results/best_model_{ar,va}_hps_tuned.pt
    results/{prefix}_MTL/{prefix}_hps_uw_results/best_model_{ar,va}_hps_uw.pt

Writes
------
    results/{prefix}_MTL/{prefix}_hps_results/analysis/loss_cv_{ar,va}.csv
    results/{prefix}_MTL/{prefix}_hps_results/analysis/loss_cv_summary.csv

Usage
-----
    python loss_cv_analysis.py --dataset vreed
    python loss_cv_analysis.py --dataset dssn_eq
    python loss_cv_analysis.py --dataset dssn_em
"""
import argparse
import os, sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import SEED, RESULTS_DIR
from data import create_sliding_windows
from dataset_configs.loader import load_dataset
from models import MTLModel
from utils import set_all_seeds


def parse_args():
    p = argparse.ArgumentParser(description='Per-user training-loss CV (Analysis 2)')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def per_user_training_loss(model, df, cfg, splits, p_ids, label_type, device):
    """Mean (unweighted) BCE loss on each user's TRAIN trials under the MTL model."""
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    rows = []
    model.eval()
    with torch.no_grad():
        for task_idx, pid in enumerate(p_ids):
            p_df = df[df['ID'] == pid].reset_index(drop=True)
            tr_df = p_df[p_df['Trial'].isin(splits[pid]['train'])].reset_index(drop=True)
            if len(tr_df) == 0:
                rows.append({'participant_id': pid, 'mean_loss': np.nan,
                             'n_windows': 0, 'pos_frac': np.nan})
                continue
            X, y_ar, y_va, _, _ = create_sliding_windows(
                tr_df, cfg['window_size'], cfg['stride'],
                task_id=task_idx, feature_cols=cfg['feature_cols'])
            if len(X) == 0:
                rows.append({'participant_id': pid, 'mean_loss': np.nan,
                             'n_windows': 0, 'pos_frac': np.nan})
                continue
            y = torch.tensor(y_ar if label_type == 'ar' else y_va,
                             dtype=torch.float32).unsqueeze(1).to(device)
            Xt = torch.tensor(X, dtype=torch.float32).to(device)
            tids = torch.full((len(X),), task_idx, dtype=torch.long).to(device)
            logits = model(Xt, tids)
            rows.append({'participant_id': pid,
                         'mean_loss': loss_fn(logits, y).item(),
                         'n_windows': len(X),
                         'pos_frac': float(y.mean().item())})
    return pd.DataFrame(rows)


def logvars_proxy(uw_ckpt_path, device):
    """sigma_u^2 = exp(log_vars) ~ per-user loss at the UW optimum."""
    sd = torch.load(uw_ckpt_path, map_location=device, weights_only=True)
    return np.exp(sd['log_vars'].cpu().numpy())


def cv(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.std(x) / np.mean(x)) if len(x) > 1 and np.mean(x) != 0 else float('nan')


if __name__ == '__main__':
    args = parse_args()
    df, cfg = load_dataset(args.dataset)
    splits = cfg['splits']
    p_ids  = cfg['participant_ids']
    prefix = cfg['results_prefix']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)

    hps_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL', f'{prefix}_hps_results')
    uw_dir  = os.path.join(RESULTS_DIR, f'{prefix}_MTL', f'{prefix}_hps_uw_results')
    out_dir = os.path.join(hps_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for label_type in ['ar', 'va']:
        ckpt = os.path.join(hps_dir, f'best_model_{label_type}_hps_tuned.pt')
        if not os.path.exists(ckpt):
            print(f"Missing HPS checkpoint: {ckpt}"); continue
        model = MTLModel(cfg['num_tasks'], input_dim=cfg['input_dim']).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        df_loss = per_user_training_loss(model, df, cfg, splits, p_ids,
                                         label_type, device)
        deg = df_loss['pos_frac'].isin([0.0, 1.0]) | df_loss['mean_loss'].isna()
        df_loss['degenerate'] = deg.values

        cv_all = cv(df_loss['mean_loss'])
        cv_nd  = cv(df_loss.loc[~df_loss['degenerate'], 'mean_loss'])

        uw_ckpt = os.path.join(uw_dir, f'best_model_{label_type}_hps_uw.pt')
        cv_proxy_all = cv_proxy_nd = float('nan')
        if os.path.exists(uw_ckpt):
            s2 = logvars_proxy(uw_ckpt, device)
            if len(s2) == len(df_loss):
                df_loss['sigma2_proxy'] = s2
                cv_proxy_all = cv(df_loss['sigma2_proxy'])
                cv_proxy_nd  = cv(df_loss.loc[~df_loss['degenerate'], 'sigma2_proxy'])

        df_loss.to_csv(os.path.join(out_dir, f'loss_cv_{label_type}.csv'), index=False)

        print(f"\n[{label_type.upper()}] per-user training-loss CV")
        print(f"  measured  : all={cv_all:.3f}  non-degenerate={cv_nd:.3f}  "
              f"(n_all={len(df_loss)}, n_nd={(~df_loss['degenerate']).sum()})")
        print(f"  UW proxy  : all={cv_proxy_all:.3f}  non-degenerate={cv_proxy_nd:.3f}")
        summary.append({'task': label_type, 'cv_measured_all': cv_all,
                        'cv_measured_nondeg': cv_nd,
                        'cv_proxy_all': cv_proxy_all, 'cv_proxy_nondeg': cv_proxy_nd,
                        'n_all': len(df_loss),
                        'n_nondeg': int((~df_loss['degenerate']).sum())})

    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, 'loss_cv_summary.csv'), index=False)
    print(f"\nDone. Wrote {os.path.join(out_dir, 'loss_cv_summary.csv')}")
