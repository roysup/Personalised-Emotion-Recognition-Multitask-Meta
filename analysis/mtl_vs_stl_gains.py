"""
MTL vs STL Gains Analysis
Computes per-participant accuracy gains of MTL over STL,
saves a CSV summary and a bar chart.

Usage
-----
    python mtl_vs_stl_gains.py                  # runs on VREED (default)
    python mtl_vs_stl_gains.py --dataset dssn_eq
    python mtl_vs_stl_gains.py --dataset dssn_em
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import get_dataset_config, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description='MTL vs STL gains analysis')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'],
                   help='Dataset to analyse (default: vreed)')
    return p.parse_args()


def aggregate_accuracy(df):
    agg = (df.groupby('participant_id')[
               ['ar_misclassified_rows', 'va_misclassified_rows', 'total_rows']]
           .sum()
           .reset_index())
    agg['ar_accuracy'] = 1 - agg['ar_misclassified_rows'] / agg['total_rows']
    agg['va_accuracy'] = 1 - agg['va_misclassified_rows'] / agg['total_rows']
    return agg[['participant_id', 'ar_accuracy', 'va_accuracy']]


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']

    MTL_OUTPUT_DIR = os.path.join(RESULTS_DIR, f'{prefix}_MTL')

    mtl_file = os.path.join(MTL_OUTPUT_DIR, f'{prefix}_hps_results',
                            f'{prefix}_hps_misclassification_rates.csv')
    stl_file = os.path.join(MTL_OUTPUT_DIR, f'{prefix}_stl_results',
                            f'{prefix}_stl_misclassification_rates.csv')

    print('=' * 60)
    print(f'MTL VS STL GAINS ANALYSIS  ({prefix})')
    print('=' * 60)

    for label, path in [('MTL', mtl_file), ('STL', stl_file)]:
        status = '✓' if os.path.exists(path) else '✗'
        print(f'{status} {label}: {path}')

    if not (os.path.exists(mtl_file) and os.path.exists(stl_file)):
        print('\nCannot proceed — run STL and MTL-HPS experiments first.')
        raise SystemExit(1)

    mtl_df = pd.read_csv(mtl_file)
    stl_df = pd.read_csv(stl_file)

    mtl_acc = aggregate_accuracy(mtl_df).rename(
        columns={'ar_accuracy': 'AR_acc_MTL', 'va_accuracy': 'VA_acc_MTL'})
    stl_acc = aggregate_accuracy(stl_df).rename(
        columns={'ar_accuracy': 'AR_acc_STL', 'va_accuracy': 'VA_acc_STL'})

    merged = pd.merge(mtl_acc, stl_acc, on='participant_id')
    merged['AR_gain_%'] = (merged['AR_acc_MTL'] - merged['AR_acc_STL']) * 100
    merged['VA_gain_%'] = (merged['VA_acc_MTL'] - merged['VA_acc_STL']) * 100

    merged['participant_id'] = merged['participant_id'].astype(int)
    merged = merged.sort_values('participant_id').reset_index(drop=True)

    # Save CSV
    output_csv = os.path.join(RESULTS_DIR, f'{prefix}_MTL_vs_STL_Gains.csv')
    merged.to_csv(output_csv, index=False)
    print(f'\n✓ Saved gains to: {output_csv}')

    # Print table
    print('\n' + '=' * 60)
    print('PARTICIPANT-LEVEL GAINS')
    print('=' * 60)
    print(merged.to_string(index=False))

    # Summary statistics
    print('\n' + '=' * 60)
    print('SUMMARY STATISTICS')
    print('=' * 60)
    for label, col in [('Arousal (AR)', 'AR_gain_%'), ('Valence (VA)', 'VA_gain_%')]:
        vals = merged[col]
        print(f'\n{label}:')
        print(f'  Mean:     {vals.mean():+.2f}%')
        print(f'  Std Dev:  {vals.std():.2f}%')
        print(f'  Positive: {(vals > 0).sum()}/{len(vals)} participants')
        print(f'  Negative: {(vals < 0).sum()}/{len(vals)} participants')
        print(f'  Zero:     {(vals == 0).sum()}/{len(vals)} participants')

    # Interpretation
    print('\n' + '=' * 60)
    print('INTERPRETATION')
    print('=' * 60)
    for label, gain_col, acc_col in [
        ('AR', 'AR_gain_%', 'AR_acc_MTL'),
        ('VA', 'VA_gain_%', 'VA_acc_MTL'),
    ]:
        mean = merged[gain_col].mean()
        n_pos = (merged[gain_col] > 0).sum()
        n_neg = (merged[gain_col] < 0).sum()
        n = len(merged)
        if mean > 0:
            print(f'✓ MTL improves {label} by {mean:.2f}% on average '
                  f'({n_pos}/{n} participants benefit)')
        else:
            print(f'✗ MTL hurts {label} by {abs(mean):.2f}% on average '
                  f'({n_neg}/{n} participants worse)')

    # Plot
    ar_color = sns.color_palette('deep')[0]
    va_color = sns.color_palette('deep')[1]
    index = range(len(merged))
    bar_width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar([i - bar_width / 2 for i in index], merged['AR_gain_%'],
            bar_width, label='AR Gain (%)', color=ar_color, alpha=0.8)
    plt.bar([i + bar_width / 2 for i in index], merged['VA_gain_%'],
            bar_width, label='VA Gain (%)', color=va_color, alpha=0.8)
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
    plt.xlabel('Participant ID', fontsize=13)
    plt.ylabel('Accuracy Gain (%)', fontsize=13)
    plt.title(f'MTL vs STL Accuracy Gain per Participant ({prefix})',
              fontsize=15, fontweight='bold')
    plt.xticks(index, merged['participant_id'], rotation=45)
    plt.legend(frameon=True, fontsize=11, loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    plot_file = os.path.join(RESULTS_DIR, f'{prefix}_mtl_vs_stl_gains.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'\n✓ Saved plot to: {plot_file}')
