"""
Improved per-participant MTL−STL gain visualisation.

Reads results/{prefix}_MTL_vs_STL_Gains.csv for each dataset and produces:

  1. Sorted diverging bar charts of MTL−STL gains per dataset
     ({prefix}_ntg_sorted_bars.png)
  2. Rescue-effect scatter (STL baseline vs gain) per dataset
     ({prefix}_rescue_scatter.png)
  3. Cross-dataset comparison panel of rescue-effect scatters
     (combined_rescue_effect.png)

Usage
-----
    python analysis/improved_ntg_plots.py                 # all three datasets
    python analysis/improved_ntg_plots.py --dataset vreed
    python analysis/improved_ntg_plots.py --sort_by baseline   # sort bars by STL acc
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from config import RESULTS_DIR

DATASETS = ['VREED', 'DSSN_EQ', 'DSSN_EM']

PAPER_STYLE = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'savefig.dpi': 300,
}

POS_COLOR = '#4c72b0'   # muted blue   — positive gain
NEG_COLOR = '#c44e52'   # muted red    — negative gain
AR_COLOR  = '#4c72b0'
VA_COLOR  = '#55a868'   # muted green


def _load_gains(dataset):
    path = os.path.join(RESULTS_DIR, f'{dataset}_MTL_vs_STL_Gains.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['participant_id'] = df['participant_id'].astype(int)
    return df


# ---------------------------------------------------------------------------
# Sorted diverging bar chart
# ---------------------------------------------------------------------------
def _diverging_bars(ax, values, labels, title, xlabel,
                    sort_by=None, baseline=None):
    """
    Horizontal diverging bars, sorted ascending by `sort_by` (defaults to
    values themselves). If `baseline` is provided, the per-bar STL accuracy
    is shown as a small annotation on the right.
    """
    if sort_by is None:
        sort_by = values
    order = np.argsort(sort_by)
    vals = np.asarray(values)[order]
    labs = np.asarray(labels)[order]
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]

    pos = np.arange(len(vals))
    ax.barh(pos, vals, color=colors, edgecolor='none', alpha=0.85, height=0.7)
    ax.axvline(0, color='black', linewidth=0.8)

    mean_val = np.mean(vals)
    ax.axvline(mean_val, color='black', linewidth=1, linestyle='--', alpha=0.55)
    ax.text(mean_val, len(vals) - 0.5,
            f' μ = {mean_val:+.1f}%',
            va='bottom',
            ha='left' if mean_val >= 0 else 'right',
            fontsize=8, alpha=0.8, style='italic')

    ax.set_yticks(pos)
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight='bold')

    n_pos = int(np.sum(vals > 0))
    n_neg = int(np.sum(vals < 0))
    n = len(vals)
    ax.text(0.98, 0.02,
            f'{n_pos}/{n} benefit · {n_neg}/{n} hurt',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, alpha=0.75,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.9))

    pad = max(abs(vals.min()), abs(vals.max())) * 1.15
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-0.7, len(vals) - 0.3)


def plot_sorted_bars(df, dataset, output_dir, sort_mode='gain'):
    """AR + VA diverging bars side-by-side, sorted by gain or STL baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.5, 0.28 * len(df))))

    if sort_mode == 'baseline':
        ar_sort = df['AR_acc_STL'].values
        va_sort = df['VA_acc_STL'].values
        suffix = '(sorted by STL baseline ↑)'
    else:
        ar_sort = df['AR_gain_%'].values
        va_sort = df['VA_gain_%'].values
        suffix = '(sorted by gain)'

    _diverging_bars(
        axes[0], df['AR_gain_%'].values,
        df['participant_id'].astype(str).values,
        f'Arousal — {dataset}', 'MTL − STL gain (%)',
        sort_by=ar_sort)
    _diverging_bars(
        axes[1], df['VA_gain_%'].values,
        df['participant_id'].astype(str).values,
        f'Valence — {dataset}', 'MTL − STL gain (%)',
        sort_by=va_sort)

    fig.suptitle(f'Per-participant MTL−STL gains  {suffix}',
                 fontsize=11, y=1.005, alpha=0.7)
    plt.tight_layout()
    out = os.path.join(output_dir, f'{dataset}_ntg_sorted_bars.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ---------------------------------------------------------------------------
# Rescue-effect scatter
# ---------------------------------------------------------------------------
def _scatter_panel(ax, x, y, color, xlabel, ylabel, title):
    ax.scatter(x, y, s=60, color=color, alpha=0.75,
               edgecolor='white', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.7)
    ax.axvline(0.5, color='gray', linewidth=0.6, linestyle=':', alpha=0.55)

    if len(x) > 2:
        r, p = pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, slope * xfit + intercept,
                color=color, linewidth=1.6, alpha=0.55)
        stars = ('***' if p < 0.001 else
                 '**'  if p < 0.01  else
                 '*'   if p < 0.05  else '')
        ax.text(0.03, 0.97,
                f'r = {r:+.2f}{stars}\np = {p:.3f}\nn = {len(x)}',
                transform=ax.transAxes, va='top', ha='left',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='lightgray', alpha=0.92))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')


def plot_rescue_scatter(df, dataset, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _scatter_panel(
        axes[0], df['AR_acc_STL'].values, df['AR_gain_%'].values,
        AR_COLOR, 'AR STL baseline accuracy', 'AR MTL − STL gain (%)',
        f'Arousal — {dataset}')
    _scatter_panel(
        axes[1], df['VA_acc_STL'].values, df['VA_gain_%'].values,
        VA_COLOR, 'VA STL baseline accuracy', 'VA MTL − STL gain (%)',
        f'Valence — {dataset}')
    plt.tight_layout()
    out = os.path.join(output_dir, f'{dataset}_rescue_scatter.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ---------------------------------------------------------------------------
# Cross-dataset comparison
# ---------------------------------------------------------------------------
def plot_cross_dataset(all_dfs, output_dir):
    available = [(d, df) for d, df in all_dfs.items() if df is not None]
    if len(available) < 2:
        return
    n = len(available)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.3 * n), sharey='col')
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (dataset, df) in enumerate(available):
        _scatter_panel(
            axes[row, 0], df['AR_acc_STL'].values, df['AR_gain_%'].values,
            AR_COLOR,
            'STL baseline accuracy' if row == n - 1 else '',
            f'{dataset}\nAR gain (%)',
            'Arousal' if row == 0 else '')
        _scatter_panel(
            axes[row, 1], df['VA_acc_STL'].values, df['VA_gain_%'].values,
            VA_COLOR,
            'STL baseline accuracy' if row == n - 1 else '',
            f'VA gain (%)',
            'Valence' if row == 0 else '')

    fig.suptitle('Rescue effect across datasets: low-baseline participants benefit most',
                 fontsize=12, fontweight='bold', y=1.005)
    plt.tight_layout()
    out = os.path.join(output_dir, 'combined_rescue_effect.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  ✓ {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['vreed', 'dssn_eq', 'dssn_em'],
                        help='Single dataset; omit for all three.')
    parser.add_argument('--sort_by', type=str, default='gain',
                        choices=['gain', 'baseline'],
                        help='Sort bars by gain magnitude (default) or STL baseline.')
    args = parser.parse_args()

    plt.rcParams.update(PAPER_STYLE)
    output_dir = os.path.join(RESULTS_DIR, 'improved_ntg_plots')
    os.makedirs(output_dir, exist_ok=True)

    targets = [args.dataset.upper()] if args.dataset else DATASETS
    all_dfs = {}
    for dataset in targets:
        df = _load_gains(dataset)
        all_dfs[dataset] = df
        if df is None:
            print(f'  ✗ {dataset}: gains CSV not found, skipping')
            continue
        print(f'\n{dataset}  (n={len(df)})')
        plot_sorted_bars(df, dataset, output_dir, sort_mode=args.sort_by)
        plot_rescue_scatter(df, dataset, output_dir)

    if not args.dataset:
        print('\nCross-dataset comparison:')
        plot_cross_dataset(all_dfs, output_dir)

    print(f'\nAll figures saved to: {output_dir}')
