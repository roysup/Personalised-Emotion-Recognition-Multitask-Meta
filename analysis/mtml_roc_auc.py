"""
MTML All-Methods ROC-AUC Comparison
Loads global_roc_data.pkl from each of the 7 meta/transfer methods and
produces combined ROC curves, a grouped bar chart, and an AUC summary CSV.

Usage
-----
    python mtml_roc_auc.py                  # runs on VREED (default)
    python mtml_roc_auc.py --dataset dssn_eq
    python mtml_roc_auc.py --dataset dssn_em
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from config import get_dataset_config, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description='MTML all-methods ROC-AUC comparison')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'],
                   help='Dataset to analyse (default: vreed)')
    return p.parse_args()


# Method display names → subdirectory suffixes (appended to {prefix}_MTML/{prefix}_)
METHOD_SUBDIRS = {
    'PSTL (SI)':                  'SI',
    'TL-FT':                      'TF',
    'MTL':                        'mtl_retrain',
    'Transfer-MTL':               'transfer_mtl',
    'Pure Meta':                  'pure_meta',
    'Meta-MTL (single task eps)': 'reptile_st',
    'Meta-MTL (multi task eps)':  'reptile_mt',
}

# Special case: Reptile MI uses a different naming pattern in some setups.
# Add it here if you have that experiment.  For now the 7 above match your scripts.

METHOD_ORDER = list(METHOD_SUBDIRS.keys())

COLORS = {
    'PSTL (SI)':                  '#1f77b4',
    'TL-FT':                      '#ff7f0e',
    'MTL':                        '#2ca02c',
    'Transfer-MTL':               '#00CED1',
    'Pure Meta':                  '#9467bd',
    'Meta-MTL (single task eps)': '#FF69B4',
    'Meta-MTL (multi task eps)':  '#DC143C',
}

LINE_STYLES = {
    'PSTL (SI)':                  '-',
    'TL-FT':                      '--',
    'MTL':                        '-.',
    'Transfer-MTL':               ':',
    'Pure Meta':                  '-',
    'Meta-MTL (single task eps)': '-',
    'Meta-MTL (multi task eps)':  '-',
}

MARKERS = {
    'PSTL (SI)':                  'o',
    'TL-FT':                      's',
    'MTL':                        '^',
    'Transfer-MTL':               'D',
    'Pure Meta':                  'v',
    'Meta-MTL (single task eps)': 'p',
    'Meta-MTL (multi task eps)':  '*',
}


def build_roc_paths(prefix):
    """Return {method_name: path_to_global_roc_data.pkl}."""
    mtml_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML')
    return {
        name: os.path.join(mtml_dir, f'{prefix}_{subdir}', 'global_roc_data.pkl')
        for name, subdir in METHOD_SUBDIRS.items()
    }


def load_roc_data(path, method_name):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Normalise key names across different saving conventions
        for task in ['AR', 'VA']:
            if task in data:
                td = data[task]
                if 'y_true' in td and 'true' not in td:
                    td['true'] = td['y_true']
                if 'y_pred_probs' in td and 'probs' not in td:
                    td['probs'] = td['y_pred_probs']
        print(f'  ✓ {method_name}')
        return data
    except FileNotFoundError:
        print(f'  ✗ {method_name}: file not found at {path}')
        return None
    except Exception as e:
        print(f'  ✗ {method_name}: {e}')
        return None


def safe_roc_auc(y_true, y_probs, name=''):
    y_true  = np.array(y_true,  dtype=int)
    y_probs = np.array(y_probs, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None, None, np.nan
    if np.any(np.isnan(y_probs)) or np.any(np.isinf(y_probs)):
        return None, None, np.nan
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    return fpr, tpr, auc(fpr, tpr)


def plot_roc_comparison(all_data, task, save_path, prefix):
    plt.figure(figsize=(12, 9))
    results = []

    for method_name in METHOD_ORDER:
        if method_name not in all_data or all_data[method_name] is None:
            continue
        roc_data = all_data[method_name]
        if task not in roc_data:
            continue
        td = roc_data[task]
        fpr, tpr, auc_val = safe_roc_auc(td['true'], td['probs'], method_name)
        if fpr is None or np.isnan(auc_val):
            continue

        marker_idx = np.arange(0, len(fpr), max(1, len(fpr) // 10))
        plt.plot(fpr, tpr,
                 label=f'{method_name} (AUC = {auc_val:.4f})',
                 linewidth=2.5,
                 color=COLORS.get(method_name, 'black'),
                 linestyle=LINE_STYLES.get(method_name, '-'),
                 marker=MARKERS.get(method_name, 'o'),
                 markevery=marker_idx,
                 markersize=6,
                 alpha=0.85)
        print(f'  [{task}] {method_name}: AUC = {auc_val:.4f} (n={len(td["true"])})')
        results.append({'method': method_name, 'auc': auc_val, 'n_samples': len(td['true'])})

    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2.5,
             label='Random Classifier', alpha=0.5)
    task_full = 'Arousal' if task == 'AR' else 'Valence'
    plt.xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=15, fontweight='bold')
    plt.title(f'ROC Curve Comparison — {task_full} ({prefix})',
              fontsize=17, fontweight='bold', pad=20)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right', fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  ✓ Saved: {save_path}')
    return results


def plot_auc_bar(ar_results, va_results, save_path, prefix):
    ar_dict = {r['method']: r['auc'] for r in ar_results}
    va_dict = {r['method']: r['auc'] for r in va_results}
    methods = [m for m in METHOD_ORDER if m in ar_dict and m in va_dict]
    ar_scores = [ar_dict[m] for m in methods]
    va_scores = [va_dict[m] for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width / 2, ar_scores, width, label='Arousal (AR)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width / 2, va_scores, width, label='Valence (VA)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
    ax.set_title(f'AUC Comparison — All Methods ({prefix})', fontsize=16,
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  ✓ Saved: {save_path}')


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']

    ROC_DATA_PATHS = build_roc_paths(prefix)

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', 'ROC_Comparisons_All')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print(f'LOADING ROC DATA  ({prefix})')
    print('=' * 70)
    all_roc_data = {}
    for method_name, path in ROC_DATA_PATHS.items():
        data = load_roc_data(path, method_name)
        if data is not None:
            all_roc_data[method_name] = data

    if not all_roc_data:
        print('\nNo data loaded — exiting.')
        raise SystemExit(1)

    print(f'\nLoaded {len(all_roc_data)} / {len(ROC_DATA_PATHS)} methods.')

    print('\n' + '=' * 70)
    print('AROUSAL ROC')
    print('=' * 70)
    ar_results = plot_roc_comparison(
        all_roc_data, 'AR',
        os.path.join(output_dir, 'ROC_Comparison_AR_All.png'),
        prefix)

    print('\n' + '=' * 70)
    print('VALENCE ROC')
    print('=' * 70)
    va_results = plot_roc_comparison(
        all_roc_data, 'VA',
        os.path.join(output_dir, 'ROC_Comparison_VA_All.png'),
        prefix)

    print('\n' + '=' * 70)
    print('AUC BAR CHART')
    print('=' * 70)
    plot_auc_bar(ar_results, va_results,
                 os.path.join(output_dir, 'AUC_Bar_Comparison_All.png'),
                 prefix)

    # Summary CSV
    rows = ([{**r, 'Task': 'AR'} for r in ar_results] +
            [{**r, 'Task': 'VA'} for r in va_results])
    summary_df = pd.DataFrame(rows).rename(
        columns={'method': 'Method', 'auc': 'AUC', 'n_samples': 'N_Samples'})
    csv_path = os.path.join(output_dir, 'ROC_AUC_Summary_All.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f'\n✓ Saved: {csv_path}')

    # Statistics
    for task_label, res in [('AROUSAL', ar_results), ('VALENCE', va_results)]:
        aucs = [r['auc'] for r in res]
        best  = res[int(np.argmax(aucs))]
        worst = res[int(np.argmin(aucs))]
        print(f'\n{task_label}:')
        print(f'  Best:  {best["auc"]:.4f} ({best["method"]})')
        print(f'  Worst: {worst["auc"]:.4f} ({worst["method"]})')
        print(f'  Mean:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
        print(f'  Range: {max(aucs) - min(aucs):.4f}')

    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)
    print(f'\nAll outputs saved to: {output_dir}')
