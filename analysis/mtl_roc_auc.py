"""
MTL Baselines ROC-AUC
Loads per-participant predictions from all 5 MTL baseline PKL files and
plots combined ROC curves for arousal and valence.

Usage
-----
    python mtl_roc_auc.py                  # runs on VREED (default)
    python mtl_roc_auc.py --dataset dssn_eq
    python mtl_roc_auc.py --dataset dssn_em
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
    p = argparse.ArgumentParser(description='MTL baselines ROC-AUC')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'],
                   help='Dataset to analyse (default: vreed)')
    return p.parse_args()


COLORS = {
    'P-STL':       '#1f77b4',
    'STL':         '#2ca02c',
    'MTL':         '#d62728',
    'MTL+UW':      '#ff7f0e',
    'MTL+PCGrad':  '#9467bd',
}

LINE_STYLES = {
    'P-STL':       ':',
    'STL':         '--',
    'MTL':         '-',
    'MTL+UW':      '-.',
    'MTL+PCGrad':  (0, (3, 1, 1, 1)),
}


def get_model_dirs(prefix):
    """Return dict of model name → result directory, parameterised by prefix."""
    mtl_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL')
    return {
        'P-STL':      os.path.join(mtl_dir, f'{prefix}_pstl_results'),
        'STL':        os.path.join(mtl_dir, f'{prefix}_stl_results'),
        'MTL':        os.path.join(mtl_dir, f'{prefix}_hps_results'),
        'MTL+UW':     os.path.join(mtl_dir, f'{prefix}_hps_uw_results'),
        'MTL+PCGrad': os.path.join(mtl_dir, f'{prefix}_hps_pcgrad_results'),
    }


PKL_MAP = {
    'P-STL':      'pstl_results.pkl',
    'STL':        'stl_tuned_results.pkl',
    'MTL':        'hps_tuned_results.pkl',
    'MTL+UW':     'hps_uw_results.pkl',
    'MTL+PCGrad': 'hps_pcgrad_results.pkl',
}


def load_predictions(model_name, model_dir):
    """Return {'ar': {'y_true': ..., 'y_probs': ...}, 'va': {...}} or None."""
    preds = {'ar': None, 'va': None}

    pkl_path = os.path.join(model_dir, PKL_MAP[model_name])
    if not os.path.exists(pkl_path):
        print(f'  ✗ Not found: {pkl_path}')
        return preds

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    print(f'  ✓ Loaded: {pkl_path}')

    all_ar_true, all_ar_probs = [], []
    all_va_true, all_va_probs = [], []

    for task in results.get('per_participant', []):
        all_ar_true.extend(task['y_true_ar'])
        all_ar_probs.extend(task['y_pred_probs_ar'])
        all_va_true.extend(task['y_true_va'])
        all_va_probs.extend(task['y_pred_probs_va'])

    if all_ar_true:
        preds['ar'] = {'y_true': np.array(all_ar_true), 'y_probs': np.array(all_ar_probs)}
    if all_va_true:
        preds['va'] = {'y_true': np.array(all_va_true), 'y_probs': np.array(all_va_probs)}

    return preds


def compute_auc(data):
    if data is None or len(np.unique(data['y_true'])) < 2:
        return None, None, None
    fpr, tpr, _ = roc_curve(data['y_true'], data['y_probs'])
    return fpr, tpr, auc(fpr, tpr)


def plot_roc(all_predictions, task, save_path, prefix):
    plt.figure(figsize=(10, 8))
    summary = []

    for model_name in COLORS:
        pred = all_predictions.get(model_name, {}).get(task.lower())
        fpr, tpr, roc_auc = compute_auc(pred)
        if fpr is None:
            print(f'  Skipping {model_name} {task} (no valid data)')
            continue
        plt.plot(fpr, tpr,
                 label=f'{model_name} (AUC = {roc_auc:.3f})',
                 color=COLORS[model_name],
                 linestyle=LINE_STYLES[model_name],
                 linewidth=2.5)
        print(f'  {model_name} {task} AUC: {roc_auc:.4f}')
        summary.append({'Model': model_name, 'AUC': roc_auc,
                        'N': len(pred['y_true'])})

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.3, label='Random')
    task_full = 'Arousal' if task == 'AR' else 'Valence'
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curves — {task_full} ({prefix})', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  ✓ Saved: {save_path}')
    return summary


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']

    MODEL_DIRS = get_model_dirs(prefix)

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_combined_roc_plots')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print(f'LOADING PREDICTIONS  ({prefix})')
    print('=' * 60)
    all_predictions = {}
    for name, path in MODEL_DIRS.items():
        print(f'\n{name}:')
        all_predictions[name] = load_predictions(name, path)

    print('\n' + '=' * 60)
    print('AROUSAL ROC')
    print('=' * 60)
    ar_summary = plot_roc(all_predictions, 'AR',
                          os.path.join(output_dir, 'combined_roc_arousal.png'),
                          prefix)

    print('\n' + '=' * 60)
    print('VALENCE ROC')
    print('=' * 60)
    va_summary = plot_roc(all_predictions, 'VA',
                          os.path.join(output_dir, 'combined_roc_valence.png'),
                          prefix)

    # AUC summary CSV
    rows = ([{**r, 'Task': 'AR'} for r in ar_summary] +
            [{**r, 'Task': 'VA'} for r in va_summary])
    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'auc_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f'\nAUC Summary:\n{summary_df.to_string(index=False)}')
    print(f'\n✓ Saved: {csv_path}')
