"""
Per-user accuracy (STL / MTL-HPS / MTL-UW) with per-user loss overlay.

For each participant, draws grouped accuracy bars for STL, MTL-HPS and MTL-UW on
the left axis, and overlays that user's per-user loss on the right axis. The
loss is sigma_u^2 = exp(log_vars_u), read from the UW checkpoint; at the UW
optimum this equals the per-task training loss the model converged to (it is the
quantity UW actually re-weights on, computed under the real training protocol).

The overlay lets you read off whether UW (green) diverges from HPS (blue) for the
users whose loss is high — i.e. whether UW's reweighting concentrates where the
imbalance is. The right axis is log-scaled because per-user loss spans several
orders of magnitude (single-class users have near-zero loss).

Reads
-----
    results/{prefix}_MTL/{prefix}_stl_results/per_participant_results.csv
    results/{prefix}_MTL/{prefix}_hps_results/per_participant_results.csv
    results/{prefix}_MTL/{prefix}_hps_uw_results/per_participant_results.csv
    results/{prefix}_MTL/{prefix}_hps_uw_results/best_model_{ar,va}_hps_uw.pt

Writes
------
    results/{prefix}_MTL/analysis/acc_loss_overlay_{prefix}.png
    results/{prefix}_MTL/analysis/acc_loss_overlay_{prefix}.csv

Usage
-----
    python per_user_acc_loss_overlay.py --dataset vreed
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

from config import get_dataset_config, RESULTS_DIR

ACC_MODELS = [('stl', 'STL', '#888888'),
              ('hps', 'MTL-HPS', '#1f77b4'),
              ('hps_uw', 'MTL-UW', '#2ca02c')]


def parse_args():
    p = argparse.ArgumentParser(description='Per-user accuracy + loss overlay')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def load_loss(uw_ckpt, device='cpu'):
    """sigma_u^2 = exp(log_vars) ~ per-user training loss at the UW optimum."""
    sd = torch.load(uw_ckpt, map_location=device, weights_only=True)
    return np.exp(sd['log_vars'].cpu().numpy())


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']
    mtl_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL')
    uw_dir = os.path.join(mtl_dir, f'{prefix}_hps_uw_results')
    out_dir = os.path.join(mtl_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    accs = {}
    for tag, _, _ in ACC_MODELS:
        path = os.path.join(mtl_dir, f'{prefix}_{tag}_results',
                            'per_participant_results.csv')
        if not os.path.exists(path):
            print(f"Missing: {path}"); sys.exit(1)
        accs[tag] = pd.read_csv(path).set_index('Participant ID')
    p_ids = list(accs['stl'].index)

    fig, axes = plt.subplots(2, 1, figsize=(max(11, 0.5 * len(p_ids)), 9))
    out_rows = []
    for ax, (dim, acc_col) in zip(axes, [('AR', 'AR Acc'), ('VA', 'VA Acc')]):
        label_type = dim.lower()
        uw_ckpt = os.path.join(uw_dir, f'best_model_{label_type}_hps_uw.pt')
        loss = load_loss(uw_ckpt) if os.path.exists(uw_ckpt) else None
        if loss is not None and len(loss) != len(p_ids):
            print(f"loss len {len(loss)} != n {len(p_ids)}; skipping overlay for {dim}")
            loss = None

        n_models = len(ACC_MODELS)
        width = 0.8 / n_models
        x = np.arange(len(p_ids))
        for i, (tag, name, color) in enumerate(ACC_MODELS):
            vals = accs[tag][acc_col].reindex(p_ids).values.astype(float)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=name, color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(p_ids, rotation=90, fontsize=8)
        ax.set_ylabel(f'{dim} accuracy'); ax.set_xlabel('Participant ID')
        ax.set_ylim(0, 1.02)
        ax.axhline(0.5, color='black', ls='--', lw=1, alpha=0.4)
        ax.set_title(f'{prefix} — {dim}: per-user accuracy with UW loss overlay',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        if loss is not None:
            ax2 = ax.twinx()
            ax2.plot(x, loss, color='black', marker='o', ms=4, lw=1.3,
                     label=r'per-user loss $\sigma_u^2$')
            ax2.set_yscale('log')
            ax2.set_ylabel(r'per-user loss $\sigma_u^2$ (log scale)')
            for pid, a_stl, a_hps, a_uw, lo in zip(
                    p_ids, accs['stl'][acc_col].reindex(p_ids),
                    accs['hps'][acc_col].reindex(p_ids),
                    accs['hps_uw'][acc_col].reindex(p_ids), loss):
                out_rows.append({'participant_id': pid, 'dim': dim,
                                 'stl_acc': a_stl, 'hps_acc': a_hps,
                                 'uw_acc': a_uw, 'uw_loss_sigma2': lo})
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      ncol=n_models + 1, fontsize=9, loc='lower right')
        else:
            ax.legend(ncol=n_models, fontsize=9, loc='lower right')

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'acc_loss_overlay_{prefix}.png')
    plt.savefig(fig_path, dpi=300); plt.close()
    if out_rows:
        pd.DataFrame(out_rows).to_csv(
            os.path.join(out_dir, f'acc_loss_overlay_{prefix}.csv'), index=False)
    print(f"Done. Wrote {fig_path}")
