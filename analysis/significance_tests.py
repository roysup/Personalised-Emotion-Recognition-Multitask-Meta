"""
Significance test for the core MTL personalization claim.

Comparison (paired across participants, two-sided Wilcoxon signed-rank):
    MTL-HPS  vs  STL   — does the shared-backbone MTL model beat the
                         fully personalized per-user STL baseline?

Tested on per-participant ACCURACY across the 6 cells
(3 datasets × 2 dimensions). No multiple-comparison correction is
applied: AR and VA are conceptually distinct affective constructs
(arousal vs valence) and each (dataset, dimension) is treated as an
independent primary endpoint, not a redundant test of the same
hypothesis. Effect size reported as rank-biserial correlation,
computed directly from signed ranks.

Inputs
------
results/{DATASET}_MTL/{DATASET}_{stl,hps}_results/per_participant_results.csv
    columns: 'Participant ID', 'AR Acc', 'VA Acc'

Outputs
-------
results/significance_tests.csv
    One row per (dataset, dimension) with the p-value, rank-biserial
    effect size, medians, and N.

Usage
-----
    python analysis/significance_tests.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
from config import RESULTS_DIR

DATASETS  = ['VREED', 'DSSN_EQ', 'DSSN_EM']
DIMS      = ['AR', 'VA']


# ----------------------------------------------------------------------
# Loader — return {participant_id: accuracy} for one (dataset, dimension)
# ----------------------------------------------------------------------
def _load_mtl_csv(dataset, model_tag, dim):
    """model_tag in {'stl', 'hps'}. Returns dict pid -> accuracy."""
    csv_path = os.path.join(
        RESULTS_DIR, f'{dataset}_MTL',
        f'{dataset}_{model_tag}_results', 'per_participant_results.csv')
    df = pd.read_csv(csv_path)
    col = f'{dim} Acc'
    return dict(zip(df['Participant ID'].astype(int), df[col].astype(float)))


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------
def _paired_wilcoxon(a, b):
    """
    Two-sided Wilcoxon signed-rank with rank-biserial effect size.

    Returns (n_effective, statistic_W, p_value, rank_biserial).
    Zero differences are dropped (wilcoxon default zero_method='wilcox').
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    nz = diffs[diffs != 0.0]
    n_eff = len(nz)
    if n_eff < 1:
        return 0, np.nan, np.nan, np.nan

    ranks  = rankdata(np.abs(nz))
    W_plus  = ranks[nz > 0].sum()
    W_minus = ranks[nz < 0].sum()
    total  = W_plus + W_minus
    rb = (W_plus - W_minus) / total if total > 0 else 0.0

    try:
        res = wilcoxon(a, b, zero_method='wilcox',
                       alternative='two-sided', method='auto')
        stat, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        stat, p = np.nan, np.nan
    return n_eff, stat, p, rb


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------
def _run_mtl_hps_vs_stl():
    rows = []
    for ds in DATASETS:
        for dim in DIMS:
            try:
                a_map = _load_mtl_csv(ds, 'hps', dim)
                b_map = _load_mtl_csv(ds, 'stl', dim)
            except FileNotFoundError as e:
                rows.append({'dataset': ds, 'dim': dim,
                             'error': f'missing: {e.filename}'})
                continue

            pids = sorted(set(a_map) & set(b_map))
            if not pids:
                rows.append({'dataset': ds, 'dim': dim,
                             'error': 'no participant overlap'})
                continue

            a = np.array([a_map[p] for p in pids])
            b = np.array([b_map[p] for p in pids])
            n_eff, W, p, rb = _paired_wilcoxon(a, b)

            rows.append({
                'dataset':     ds,
                'dim':         dim,
                'n_paired':    len(pids),
                'n_effective': n_eff,
                'median_MTL-HPS': float(np.median(a)),
                'median_STL':     float(np.median(b)),
                'median_diff':    float(np.median(a - b)),
                'wilcoxon_W':     W,
                'p_value':        p,
                'rank_biserial':  rb,
                'error':          '',
            })

    df = pd.DataFrame(rows)
    df['sig_0.05'] = (df['p_value'] < 0.05).where(df['p_value'].notna(), '')
    return df


def _format_console(df):
    if df.empty or 'p_value' not in df.columns:
        return df.to_string(index=False)
    show = df.copy()
    if 'p_value' in show:
        show['p_value'] = show['p_value'].map(
            lambda v: f'{v:.4f}' if pd.notna(v) else '—')
    for c in show.columns:
        if c.startswith('median_') or c == 'rank_biserial':
            show[c] = show[c].map(lambda v: f'{v:+.4f}' if pd.notna(v) else '—')
    return show.to_string(index=False)


if __name__ == '__main__':
    print('=' * 78)
    print('Paired Wilcoxon signed-rank test on per-participant accuracy')
    print('MTL-HPS vs STL  —  no multiple-comparison correction')
    print('(AR and VA are independent primary endpoints, not redundant tests)')
    print('=' * 78)

    df = _run_mtl_hps_vs_stl()
    print(_format_console(df))

    out_csv = os.path.join(RESULTS_DIR, 'significance_tests.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n[OK] Saved {out_csv}')

    print('\nInterpretation:')
    print('  rank_biserial > 0  -> MTL-HPS outperforms STL on most pairs')
    print('  p_value < 0.05     -> significant at conventional threshold')
