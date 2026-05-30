"""
Significance tests for the three headline claims of the paper.

Comparisons (paired across participants, two-sided Wilcoxon signed-rank):
    1. MTL-HPS  vs  STL      — does shared-backbone MTL beat per-user STL?
    2. MTML-MT  vs  MTML-ST  — do multi-task meta-episodes beat single-task?
    3. MTML-MI  vs  MTML-MT  — does MI-guided sampling beat uniform multi-task?

For each comparison, tests are run on macro-F1 across the 6 cells
(3 datasets × 2 dimensions). Within each comparison's family of 6 tests,
p-values are corrected with Holm. Effect size reported as rank-biserial
correlation, computed directly from signed ranks.

Inputs
------
results/{DATASET}_MTL/{DATASET}_{stl,hps}_results/per_participant_results.csv
    columns: 'Participant ID', 'AR Macro F1', 'VA Macro F1'
results/{DATASET}_MTML/{DATASET}_reptile_{st,mt,mi}/reptile_{st,mt,mi}_results.pkl
    keys: 'test_results_per_participant_ar', 'test_results_per_participant_va'
          — each a list of dicts with 'participant_id' and '{ar,va}_f1'.

Outputs
-------
results/significance_tests.csv
    One row per (comparison, dataset, dimension) with the raw and
    Holm-corrected p-values, rank-biserial effect size, medians, and N.

Usage
-----
    python analysis/significance_tests.py
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
from config import RESULTS_DIR

DATASETS  = ['VREED', 'DSSN_EQ', 'DSSN_EM']
DIMS      = ['AR', 'VA']


# ----------------------------------------------------------------------
# Loaders — return {participant_id: macro_f1} for one (dataset, dimension)
# ----------------------------------------------------------------------
def _load_mtl_csv(dataset, model_tag, dim):
    """model_tag in {'stl', 'hps'}. Returns dict pid -> macro-F1."""
    csv_path = os.path.join(
        RESULTS_DIR, f'{dataset}_MTL',
        f'{dataset}_{model_tag}_results', 'per_participant_results.csv')
    df = pd.read_csv(csv_path)
    col = f'{dim} Macro F1'
    return dict(zip(df['Participant ID'].astype(int), df[col].astype(float)))


def _load_mtml_pkl(dataset, variant, dim):
    """variant in {'st', 'mt', 'mi'}. Returns dict pid -> macro-F1."""
    pkl_path = os.path.join(
        RESULTS_DIR, f'{dataset}_MTML',
        f'{dataset}_reptile_{variant}', f'reptile_{variant}_results.pkl')
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    key  = f'test_results_per_participant_{dim.lower()}'
    fkey = f'{dim.lower()}_f1'
    return {int(r['participant_id']): float(r[fkey]) for r in d[key]}


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
    W_plus = ranks[nz > 0].sum()
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


def _holm(pvals):
    """Holm step-down correction. NaNs are passed through."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if not mask.any():
        return out
    pm = p[mask]
    order = np.argsort(pm)
    m = len(pm)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pm[idx])
        adj[idx] = min(1.0, running)
    out[mask] = adj
    return out


# ----------------------------------------------------------------------
# Comparison plumbing
# ----------------------------------------------------------------------
COMPARISONS = [
    # (name, loader_a, loader_b, label_a, label_b)
    ('MTL-HPS_vs_STL',
     lambda ds, dim: _load_mtl_csv(ds, 'hps', dim),
     lambda ds, dim: _load_mtl_csv(ds, 'stl', dim),
     'MTL-HPS', 'STL'),
    ('MTML-MT_vs_MTML-ST',
     lambda ds, dim: _load_mtml_pkl(ds, 'mt', dim),
     lambda ds, dim: _load_mtml_pkl(ds, 'st', dim),
     'MTML-MT', 'MTML-ST'),
    ('MTML-MI_vs_MTML-MT',
     lambda ds, dim: _load_mtml_pkl(ds, 'mi', dim),
     lambda ds, dim: _load_mtml_pkl(ds, 'mt', dim),
     'MTML-MI', 'MTML-MT'),
]


def _run_comparison(name, load_a, load_b, label_a, label_b):
    rows = []
    for ds in DATASETS:
        for dim in DIMS:
            try:
                a_map = load_a(ds, dim)
                b_map = load_b(ds, dim)
            except FileNotFoundError as e:
                rows.append({'comparison': name, 'dataset': ds, 'dim': dim,
                             'error': f'missing: {e.filename}'})
                continue

            pids = sorted(set(a_map) & set(b_map))
            if not pids:
                rows.append({'comparison': name, 'dataset': ds, 'dim': dim,
                             'error': 'no participant overlap'})
                continue

            a = np.array([a_map[p] for p in pids])
            b = np.array([b_map[p] for p in pids])
            n_eff, W, p, rb = _paired_wilcoxon(a, b)

            rows.append({
                'comparison': name,
                'dataset':    ds,
                'dim':        dim,
                'n_paired':   len(pids),
                'n_effective': n_eff,
                f'median_{label_a}': float(np.median(a)),
                f'median_{label_b}': float(np.median(b)),
                'median_diff':       float(np.median(a - b)),
                'wilcoxon_W':        W,
                'p_raw':             p,
                'rank_biserial':     rb,
                'error':             '',
            })

    df = pd.DataFrame(rows)
    # Holm-correct only the rows that produced a p-value (one family per
    # comparison, 6 cells max).
    df['p_holm'] = _holm(df['p_raw'].values) if 'p_raw' in df else np.nan
    df['sig_holm_0.05'] = (df['p_holm'] < 0.05).where(df['p_holm'].notna(), '')
    return df


def _format_console(df):
    if df.empty or 'p_raw' not in df.columns:
        return df.to_string(index=False)
    show = df.copy()
    for c in ('p_raw', 'p_holm'):
        if c in show:
            show[c] = show[c].map(lambda v: f'{v:.4f}' if pd.notna(v) else '—')
    for c in show.columns:
        if c.startswith('median_') or c == 'rank_biserial':
            show[c] = show[c].map(lambda v: f'{v:+.4f}' if pd.notna(v) else '—')
    return show.to_string(index=False)


if __name__ == '__main__':
    out_rows = []
    print('=' * 78)
    print('Paired Wilcoxon signed-rank tests on per-participant macro-F1')
    print('Holm correction applied within each comparison family (6 cells).')
    print('=' * 78)

    for name, load_a, load_b, label_a, label_b in COMPARISONS:
        print(f'\n>>> {name}  ({label_a}  vs  {label_b})')
        df = _run_comparison(name, load_a, load_b, label_a, label_b)
        print(_format_console(df))
        out_rows.append(df)

    final = pd.concat(out_rows, ignore_index=True)
    out_csv = os.path.join(RESULTS_DIR, 'significance_tests.csv')
    final.to_csv(out_csv, index=False)
    print(f'\n[OK] Saved {out_csv}')

    # Brief interpretation hint
    print('\nInterpretation:')
    print('  rank_biserial > 0  -> first method outperforms second on most pairs')
    print('  p_holm < 0.05      -> survives multiple-comparison correction')
