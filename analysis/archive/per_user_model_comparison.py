"""
Per-user model comparison plots.

For a given dataset, draws grouped bar charts of per-participant performance for
the four learning paradigms — STL, MTL-HPS, MTL-UW, MTL-PCGrad — with one panel
for arousal (AR) and one for valence (VA).

Metric is selectable. Macro-F1 is the default because it is defined for every
participant; AUC is undefined for single-class participants (majority-class
collapse) and those bars are simply omitted, while accuracy is degenerate
(==1.0) for the same participants and should be read with caution.

Reads
-----
    results/{prefix}_MTL/{prefix}_stl_results/per_participant_results.csv
    results/{prefix}_MTL/{prefix}_hps_results/per_participant_results.csv
    results/{prefix}_MTL/{prefix}_hps_uw_results/per_participant_results.csv
    results/{prefix}_MTL/{prefix}_hps_pcgrad_results/per_participant_results.csv

Writes
------
    results/{prefix}_MTL/analysis/per_user_{metric}_{prefix}.png
    results/{prefix}_MTL/analysis/per_user_{metric}_{prefix}.csv

Usage
-----
    python per_user_model_comparison.py --dataset vreed --metric f1
    python per_user_model_comparison.py --dataset dssn_em --metric auc
    python per_user_model_comparison.py --dataset dssn_eq --metric acc
"""
import argparse
import os, sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import get_dataset_config, RESULTS_DIR

MODELS = [('stl', 'STL'), ('hps', 'MTL-HPS'),
          ('hps_uw', 'MTL-UW'), ('hps_pcgrad', 'MTL-PCGrad')]
COLORS = ['#888888', '#1f77b4', '#2ca02c', '#d62728']
METRIC_COL = {'f1': 'Macro F1', 'auc': 'AUC', 'acc': 'Acc'}


def parse_args():
    p = argparse.ArgumentParser(description='Per-user model comparison plots')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    p.add_argument('--metric', type=str, default='f1',
                   choices=['f1', 'auc', 'acc'])
    return p.parse_args()


def load_models(mtl_dir, prefix):
    frames = {}
    for tag, _ in MODELS:
        path = os.path.join(mtl_dir, f'{prefix}_{tag}_results',
                            'per_participant_results.csv')
        if not os.path.exists(path):
            print(f"Missing: {path}"); continue
        frames[tag] = pd.read_csv(path).set_index('Participant ID')
    return frames


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']
    metric_word = METRIC_COL[args.metric]

    mtl_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL')
    out_dir = os.path.join(mtl_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    frames = load_models(mtl_dir, prefix)
    present = [(tag, name, c) for (tag, name), c in zip(MODELS, COLORS)
               if tag in frames]
    if not present:
        print("No model result files found."); sys.exit(1)

    # participant ordering = STL row order (falls back to first available)
    ref_tag = present[0][0]
    p_ids = list(frames[ref_tag].index)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.45 * len(p_ids)), 9))
    out_rows = []
    for ax, dim in zip(axes, ['AR', 'VA']):
        col = f'{dim} {metric_word}'
        n_models = len(present)
        width = 0.8 / n_models
        x = np.arange(len(p_ids))
        for i, (tag, name, color) in enumerate(present):
            vals = frames[tag][col].reindex(p_ids).values.astype(float)
            ax.bar(x + i * width - 0.4 + width / 2, np.nan_to_num(vals, nan=0.0),
                   width, label=name, color=color, alpha=0.85)
            for pid, v in zip(p_ids, vals):
                out_rows.append({'participant_id': pid, 'dim': dim,
                                 'model': name, args.metric: v})
        ax.set_xticks(x)
        ax.set_xticklabels(p_ids, rotation=90, fontsize=8)
        ax.set_ylabel(f'{dim} {metric_word}')
        ax.set_xlabel('Participant ID')
        ax.set_title(f'{prefix} — {dim} {metric_word} per participant',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(ncol=n_models, fontsize=9)
        if args.metric in ('auc', 'acc'):
            ax.axhline(0.5, color='black', ls='--', lw=1, alpha=0.5)
            ax.set_ylim(0, 1.02)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'per_user_{args.metric}_{prefix}.png')
    plt.savefig(fig_path, dpi=300); plt.close()
    pd.DataFrame(out_rows).to_csv(
        os.path.join(out_dir, f'per_user_{args.metric}_{prefix}.csv'), index=False)
    if args.metric == 'auc':
        for tag, name, _ in present:
            valid = frames[tag]['AR AUC'].notna().sum()
            print(f"  {name}: AR AUC defined for {valid}/{len(p_ids)} participants")
    print(f"Done. Wrote {fig_path}")
