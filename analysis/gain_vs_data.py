"""
Analysis 5 — Per-user UW / PCGrad gain vs. data availability.

Question
--------
Does the benefit of UW (or PCGrad) over per-user STL depend on how much data a
user has? If UW gains more for low-data users, that supports a "favorable
interaction with data scarcity" explanation for its advantage. A consistent
relationship across datasets/tasks would be needed to claim this.

Definitions
-----------
    gain_u(model)  = acc_u(model) - acc_u(STL)        (per user)
    data_u         = total windows available for user u  (proxy: total_rows in
                     the misclassification-rate CSV, proportional to data volume)

Correlates gain_u against data_u with Spearman rho (rank-based, robust to the
skewed per-user window counts).

Reads
-----
    results/{prefix}_MTL/{prefix}_hps_results/{prefix}_hps_misclassification_rates.csv
    results/{prefix}_MTL/{prefix}_stl_results/{prefix}_stl_misclassification_rates.csv
    results/{prefix}_MTL/{prefix}_hps_uw_results/{prefix}_hps_uw_misclassification_rates.csv
    results/{prefix}_MTL/{prefix}_hps_pcgrad_results/{prefix}_hps_pcgrad_misclassification_rates.csv

Writes
------
    results/{prefix}_MTL/analysis/gain_vs_data_{prefix}.csv
    results/{prefix}_MTL/analysis/gain_vs_data_summary_{prefix}.csv
    results/{prefix}_MTL/analysis/gain_vs_data_{prefix}.png

Usage
-----
    python gain_vs_data.py --dataset vreed
"""
import argparse
import os, sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import get_dataset_config, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description='gain vs data availability (Analysis 5)')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def _misclass(mtl_dir, prefix, sub, tag):
    path = os.path.join(mtl_dir, f'{prefix}_{sub}_results',
                        f'{prefix}_{tag}_misclassification_rates.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).set_index('participant_id')


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']
    mtl_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL')
    out_dir = os.path.join(mtl_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    hps = _misclass(mtl_dir, prefix, 'hps', 'hps')
    stl = _misclass(mtl_dir, prefix, 'stl', 'stl')
    uw  = _misclass(mtl_dir, prefix, 'hps_uw', 'hps_uw')
    pc  = _misclass(mtl_dir, prefix, 'hps_pcgrad', 'hps_pcgrad')
    if stl is None or hps is None:
        print("Missing STL/HPS misclassification CSVs."); sys.exit(1)

    counts = hps['total_rows']               # data-volume proxy per user
    base = pd.DataFrame({'data': counts})
    summary, plot_specs = [], []

    for name, tbl in [('UW', uw), ('PCGrad', pc)]:
        if tbl is None:
            print(f"Missing {name} CSV; skipping."); continue
        for task, col in [('AR', 'ar_accuracy'), ('VA', 'va_accuracy')]:
            gain = (tbl[col] - stl[col]).reindex(counts.index)
            d = counts.reindex(gain.index)
            m = gain.notna() & d.notna()
            base[f'{name}_{task}_gain'] = gain
            if m.sum() >= 4:
                rho, p = spearmanr(d[m], gain[m])
            else:
                rho, p = np.nan, np.nan
            summary.append({'model': name, 'task': task, 'n': int(m.sum()),
                            'spearman_rho': rho, 'p': p,
                            'mean_gain_pct': float(gain[m].mean() * 100)})
            plot_specs.append((f'{name} {task}', d[m], gain[m] * 100, rho, p))
            print(f"  {name} {task}: rho={rho:+.3f} p={p:.3f} "
                  f"(n={m.sum()})  mean_gain={gain[m].mean()*100:+.1f}%")

    base.to_csv(os.path.join(out_dir, f'gain_vs_data_{prefix}.csv'))
    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, f'gain_vs_data_summary_{prefix}.csv'), index=False)

    n = len(plot_specs)
    if n:
        cols = min(4, n); rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows),
                                 squeeze=False)
        for ax, (title, x, y, rho, p) in zip(axes.flat, plot_specs):
            ax.scatter(x, y, s=60, alpha=0.7, color='seagreen')
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, np.poly1d(z)(xs), 'r--', alpha=0.6)
            ax.axhline(0, color='black', lw=1, alpha=0.5)
            ax.set_xlabel('data volume (windows)')
            ax.set_ylabel('gain over STL (%)')
            ax.set_title(f'{title}: rho={rho:+.3f}, p={p:.3f}',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[n:]:
            ax.axis('off')
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f'gain_vs_data_{prefix}.png')
        plt.savefig(fig_path, dpi=300); plt.close()
        print(f"\nDone. Wrote {fig_path}")
