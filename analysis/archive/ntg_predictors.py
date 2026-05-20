"""
What predicts per-user Negative Transfer Gap (NTG)?

This consolidates the "why does NTG happen" question into ONE reproducible
script so the correlations can be inspected and re-run, rather than computed
ad hoc. For each dataset and affective dimension it correlates per-user NTG
against several candidate per-user predictors and prints/【saves a table.

NTG (paper convention)
----------------------
    NTG_u = Acc_MTL,u - Acc_STL,u        (positive = MTL benefits user)
An AUC-based NTG is also computed where AUC is defined.

Candidate predictors (all read from existing result CSVs; no torch needed)
--------------------------------------------------------------------------
    stl_acc        per-user STL accuracy            -> task difficulty proxy
    class_imbalance max(pos,neg)/total from STL conf-matrix (0.5=balanced,1=single class)
    n_windows      total test windows               -> data-volume proxy

Degenerate users (STL acc == 0 or 1, i.e. single-class test sets) are excluded
from the correlations because their NTG is mechanically 0 and uninformative.

Reads
-----
    results/{prefix}_MTL/{prefix}_stl_results/per_participant_results.csv
    results/{prefix}_MTL/{prefix}_hps_results/per_participant_results.csv

Writes
------
    results/{prefix}_MTL/analysis/ntg_predictors_{prefix}.csv   (per-user table)
    results/{prefix}_MTL/analysis/ntg_predictors_summary_{prefix}.csv (correlations)

Usage
-----
    python ntg_predictors.py --dataset vreed
    python ntg_predictors.py --dataset dssn_eq
    python ntg_predictors.py --dataset dssn_em
"""
import argparse
import os, sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config import get_dataset_config, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description='What predicts per-user NTG?')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def corr(x, y):
    """Pearson and Spearman on the finite, paired entries."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return dict(n=int(m.sum()), pearson_r=np.nan, pearson_p=np.nan,
                    spearman_rho=np.nan, spearman_p=np.nan)
    r, pr = pearsonr(x[m], y[m])
    rho, ps = spearmanr(x[m], y[m])
    return dict(n=int(m.sum()), pearson_r=r, pearson_p=pr,
                spearman_rho=rho, spearman_p=ps)


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']
    mtl_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTL')
    out_dir = os.path.join(mtl_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    stl = pd.read_csv(os.path.join(mtl_dir, f'{prefix}_stl_results',
                                   'per_participant_results.csv')).set_index('Participant ID')
    hps = pd.read_csv(os.path.join(mtl_dir, f'{prefix}_hps_results',
                                   'per_participant_results.csv')).set_index('Participant ID')
    pid = list(stl.index)

    rows, summ = [], []
    print(f"\n================ {prefix} ================")
    for dim in ['AR', 'VA']:
        acc, auc = f'{dim} Acc', f'{dim} AUC'
        tn, tp, fn, fp = f'{dim} TN', f'{dim} TP', f'{dim} FN', f'{dim} FP'

        stl_acc = stl[acc].reindex(pid).values
        ntg_acc = (hps[acc] - stl[acc]).reindex(pid).values
        ntg_auc = (hps[auc] - stl[auc]).reindex(pid).values

        pos = (stl[tp] + stl[fn]).reindex(pid).values
        neg = (stl[tn] + stl[fp]).reindex(pid).values
        tot = pos + neg
        n_windows = tot.astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            class_imbalance = np.maximum(pos, neg) / np.where(tot == 0, np.nan, tot)

        degenerate = (stl_acc == 1.0) | (stl_acc == 0.0)
        keep = ~degenerate

        for pidx, a, ni, ci, na, nu, dg in zip(
                pid, stl_acc, n_windows, class_imbalance, ntg_acc, ntg_auc, degenerate):
            rows.append({'participant_id': pidx, 'dim': dim, 'stl_acc': a,
                         'n_windows': ni, 'class_imbalance': ci,
                         'ntg_acc': na, 'ntg_auc': nu, 'degenerate': bool(dg)})

        print(f"\n[{dim}]  non-degenerate users: {keep.sum()}/{len(pid)}")
        predictors = {'stl_acc(difficulty)': stl_acc,
                      'class_imbalance': class_imbalance,
                      'n_windows(data)': n_windows}
        for pname, pvals in predictors.items():
            for tgt_name, tgt in [('NTG_acc', ntg_acc), ('NTG_auc', ntg_auc)]:
                c = corr(pvals[keep], tgt[keep])
                print(f"  {pname:24s} vs {tgt_name}:  "
                      f"rho={c['spearman_rho']:+.3f} p={c['spearman_p']:.3f}  "
                      f"(pearson r={c['pearson_r']:+.3f} p={c['pearson_p']:.3f}, n={c['n']})")
                summ.append({'dim': dim, 'predictor': pname, 'target': tgt_name, **c})

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, f'ntg_predictors_{prefix}.csv'), index=False)
    pd.DataFrame(summ).to_csv(
        os.path.join(out_dir, f'ntg_predictors_summary_{prefix}.csv'), index=False)
    print(f"\nWrote ntg_predictors_{prefix}.csv and ntg_predictors_summary_{prefix}.csv")
