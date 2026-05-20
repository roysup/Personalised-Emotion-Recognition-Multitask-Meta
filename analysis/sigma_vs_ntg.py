"""
Analysis 3 — Learned UW uncertainty (sigma_u) vs. per-user Negative Transfer Gap.

Question
--------
Does UW's learned per-user uncertainty track which users are harmed by parameter
sharing? If sigma_u correlates with NTG, UW is effectively re-weighting users
according to their transfer outcome. If not, UW's benefit comes from a different
channel (e.g. global loss-scale rebalancing) rather than from targeting transfer
harm directly.

Definitions
-----------
    sigma_u = exp(log_vars_u / 2)   from the trained MTL-UW checkpoint
    NTG_u   = MTL_perf_u - STL_perf_u   (positive = MTL benefits this user;
                                         matches the paper's NTG definition)

NTG is computed from per-participant AUC where available (AUC is the headline
metric and is robust to class imbalance); users whose test set is single-class
have undefined AUC and are excluded from the AUC-based correlation. An
accuracy-based NTG is also reported for completeness, but note that accuracy is
degenerate (==1.0) for single-class users and should be read with caution.

Reads
-----
    results/{prefix}_MTL/{prefix}_hps_uw_results/best_model_{ar,va}_hps_uw.pt
    results/{prefix}_MTL/{prefix}_hps_results/per_participant_results.csv   (MTL)
    results/{prefix}_MTL/{prefix}_stl_results/per_participant_results.csv   (STL)

Writes
------
    results/{prefix}_MTL/{prefix}_hps_uw_results/analysis/sigma_vs_ntg_{ar,va}.csv
    results/{prefix}_MTL/{prefix}_hps_uw_results/analysis/sigma_vs_ntg_summary.csv
    results/{prefix}_MTL/{prefix}_hps_uw_results/analysis/sigma_vs_ntg.png

Usage
-----
    python sigma_vs_ntg.py --dataset vreed
"""
import argparse
import os, sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from config import get_dataset_config, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description='sigma_u vs NTG (Analysis 3)')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def load_sigma(uw_ckpt, device='cpu'):
    sd = torch.load(uw_ckpt, map_location=device, weights_only=True)
    log_vars = sd['log_vars'].cpu().numpy()
    return np.exp(log_vars / 2.0)            # sigma_u


def corr_block(sigma, ntg, mask):
    s, n = sigma[mask], ntg[mask]
    if mask.sum() < 4:
        return dict(n=int(mask.sum()), r=np.nan, p=np.nan, rho=np.nan, p_spear=np.nan)
    r, p   = pearsonr(s, n)
    rho, ps = spearmanr(s, n)
    return dict(n=int(mask.sum()), r=r, p=p, rho=rho, p_spear=ps)


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']

    mtl_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL', f'{prefix}_hps_results')
    stl_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL', f'{prefix}_stl_results')
    uw_dir  = os.path.join(RESULTS_DIR, f'{prefix}_MTL', f'{prefix}_hps_uw_results')
    out_dir = os.path.join(uw_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    mtl_pp = pd.read_csv(os.path.join(mtl_dir, 'per_participant_results.csv'))
    stl_pp = pd.read_csv(os.path.join(stl_dir, 'per_participant_results.csv'))
    # Participant order = row order of the MTL per-participant CSV, which is
    # written in task_idx order and therefore aligns by position with log_vars.
    p_ids = mtl_pp['Participant ID'].tolist()
    mtl_pp = mtl_pp.set_index('Participant ID')
    stl_pp = stl_pp.set_index('Participant ID')

    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (label_type, auc_col, acc_col) in zip(
            axes, [('ar', 'AR AUC', 'AR Acc'), ('va', 'VA AUC', 'VA Acc')]):
        uw_ckpt = os.path.join(uw_dir, f'best_model_{label_type}_hps_uw.pt')
        if not os.path.exists(uw_ckpt):
            print(f"Missing UW checkpoint: {uw_ckpt}"); continue
        sigma = load_sigma(uw_ckpt)
        if len(sigma) != len(p_ids):
            print(f"sigma len {len(sigma)} != n participants {len(p_ids)}; abort {label_type}")
            continue

        mtl_auc = mtl_pp[auc_col].reindex(p_ids).values
        stl_auc = stl_pp[auc_col].reindex(p_ids).values
        mtl_acc = mtl_pp[acc_col].reindex(p_ids).values
        stl_acc = stl_pp[acc_col].reindex(p_ids).values

        ntg_auc = mtl_auc - stl_auc        # positive = MTL benefits user (paper convention)
        ntg_acc = mtl_acc - stl_acc

        valid_auc = ~np.isnan(ntg_auc) & ~np.isnan(sigma)
        valid_acc = ~np.isnan(ntg_acc) & ~np.isnan(sigma)

        res_auc = corr_block(sigma, ntg_auc, valid_auc)
        res_acc = corr_block(sigma, ntg_acc, valid_acc)

        df_out = pd.DataFrame({
            'participant_id': p_ids, 'sigma': sigma,
            'mtl_auc': mtl_auc, 'stl_auc': stl_auc, 'ntg_auc': ntg_auc,
            'mtl_acc': mtl_acc, 'stl_acc': stl_acc, 'ntg_acc': ntg_acc,
        })
        df_out.to_csv(os.path.join(out_dir, f'sigma_vs_ntg_{label_type}.csv'),
                      index=False)

        print(f"\n[{label_type.upper()}] sigma_u vs NTG  (NTG = MTL - STL; positive = benefit)")
        print(f"  AUC-based : n={res_auc['n']:2d}  r={res_auc['r']:+.3f} "
              f"p={res_auc['p']:.3f} | rho={res_auc['rho']:+.3f} p={res_auc['p_spear']:.3f}")
        print(f"  Acc-based : n={res_acc['n']:2d}  r={res_acc['r']:+.3f} "
              f"p={res_acc['p']:.3f} | rho={res_acc['rho']:+.3f} p={res_acc['p_spear']:.3f}"
              f"   (acc NTG degenerate for single-class users; interpret with caution)")

        for metric, res in [('auc', res_auc), ('acc', res_acc)]:
            summary.append({'task': label_type, 'ntg_metric': metric, **res})

        ax.scatter(sigma[valid_auc], ntg_auc[valid_auc], s=80, alpha=0.7,
                   color='steelblue')
        if valid_auc.sum() > 1:
            z = np.polyfit(sigma[valid_auc], ntg_auc[valid_auc], 1)
            xs = np.linspace(sigma[valid_auc].min(), sigma[valid_auc].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), 'r--', alpha=0.6)
        ax.axhline(0, color='black', lw=1, alpha=0.5)
        ax.set_xlabel(r'$\sigma_u$ (UW uncertainty)')
        ax.set_ylabel('NTG (MTL AUC - STL AUC)')
        ax.set_title(f'{label_type.upper()}: '
                     f"r={res_auc['r']:+.3f}, p={res_auc['p']:.3f}, n={res_auc['n']}",
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'sigma_vs_ntg.png')
    plt.savefig(fig_path, dpi=300); plt.close()
    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, 'sigma_vs_ntg_summary.csv'), index=False)
    print(f"\nDone. Wrote {fig_path} and sigma_vs_ntg_summary.csv")