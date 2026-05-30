"""
Significance tests for the core MTL personalization claim:
MTL-HPS vs STL on the three datasets (VREED, DSSN_EQ, DSSN_EM)
and two affective dimensions (AR, VA).

Two complementary tests are run:

1. PRIMARY — Paired Wilcoxon signed-rank on per-participant accuracy.
   Each participant contributes one (accuracy_HPS, accuracy_STL) pair.
   This matches the personalization claim's natural unit of analysis:
   "does MTL help individual users?". Effect size: rank-biserial.

2. SECONDARY — McNemar's test on pooled window-level predictions.
   For each (dataset, dimension), all test-window predictions are
   pooled across participants into one 2x2 contingency table
   (HPS-correct/incorrect x STL-correct/incorrect) and McNemar's
   exact binomial test is applied. This matches the convention used
   in prior personalized affective-computing work (Taylor et al. 2020;
   Jaques et al.) and gives much higher statistical power, at the
   cost of ignoring within-participant clustering.

No multiple-comparison correction is applied: AR and VA are
conceptually distinct affective constructs and each (dataset,
dimension) is treated as an independent primary endpoint.

Inputs
------
results/{DATASET}_MTL/{DATASET}_{stl,hps}_results/per_participant_results.csv
    columns: 'Participant ID', 'AR Acc', 'VA Acc'              (Wilcoxon)
results/{DATASET}_MTL/{DATASET}_{stl,hps}_results/{stl,hps}_tuned_results.pkl
    key 'per_participant' -> list of dicts with y_true_{ar,va} /
    y_pred_{ar,va} arrays at window level                       (McNemar)

Outputs
-------
results/significance_tests.csv          — Wilcoxon (participant-level)
results/significance_tests_mcnemar.csv  — McNemar (window-level)

Usage
-----
    python analysis/significance_tests.py
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata, binomtest, chi2

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
from config import RESULTS_DIR

DATASETS  = ['VREED', 'DSSN_EQ', 'DSSN_EM']
DIMS      = ['AR', 'VA']


def _load_mtl_csv(dataset, model_tag, dim):
    csv_path = os.path.join(
        RESULTS_DIR, f'{dataset}_MTL',
        f'{dataset}_{model_tag}_results', 'per_participant_results.csv')
    df = pd.read_csv(csv_path)
    col = f'{dim} Acc'
    return dict(zip(df['Participant ID'].astype(int), df[col].astype(float)))


def _load_window_preds(dataset, model_tag, dim):
    pkl_path = os.path.join(
        RESULTS_DIR, f'{dataset}_MTL',
        f'{dataset}_{model_tag}_results', f'{model_tag}_tuned_results.pkl')
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    d_lower = dim.lower()
    out = {}
    for p in d['per_participant']:
        pid = int(p['participant_id'])
        out[pid] = (np.asarray(p[f'y_true_{d_lower}']).astype(int),
                    np.asarray(p[f'y_pred_{d_lower}']).astype(int))
    return out


def _paired_wilcoxon(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    diffs = a - b
    nz = diffs[diffs != 0.0]
    n_eff = len(nz)
    if n_eff < 1:
        return 0, np.nan, np.nan, np.nan
    ranks   = rankdata(np.abs(nz))
    W_plus  = ranks[nz > 0].sum()
    W_minus = ranks[nz < 0].sum()
    total   = W_plus + W_minus
    rb = (W_plus - W_minus) / total if total > 0 else 0.0
    try:
        res = wilcoxon(a, b, zero_method='wilcox',
                       alternative='two-sided', method='auto')
        stat, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        stat, p = np.nan, np.nan
    return n_eff, stat, p, rb


def _mcnemar(a_correct, b_correct):
    a = np.asarray(a_correct, dtype=bool); b = np.asarray(b_correct, dtype=bool)
    n = len(a)
    b_count = int(np.sum(~a &  b))   # HPS wrong, STL right
    c_count = int(np.sum( a & ~b))   # HPS right, STL wrong
    discordant = b_count + c_count
    if discordant == 0:
        return dict(n_total=n, b=b_count, c=c_count, discordant=0,
                    p_value=1.0, odds_ratio=float('nan'))
    p_exact = binomtest(min(b_count, c_count), discordant, p=0.5,
                        alternative='two-sided').pvalue
    odds = (c_count / b_count) if b_count > 0 else float('inf')
    return dict(n_total=n, b=b_count, c=c_count, discordant=discordant,
                p_value=float(p_exact), odds_ratio=odds)


def _run_wilcoxon():
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
                'dataset':        ds,
                'dim':            dim,
                'n_paired':       len(pids),
                'n_effective':    n_eff,
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


def _run_mcnemar():
    rows = []
    for ds in DATASETS:
        for dim in DIMS:
            try:
                hps = _load_window_preds(ds, 'hps', dim)
                stl = _load_window_preds(ds, 'stl', dim)
            except FileNotFoundError as e:
                rows.append({'dataset': ds, 'dim': dim,
                             'error': f'missing: {e.filename}'})
                continue
            pids = sorted(set(hps) & set(stl))
            a_correct_all, b_correct_all = [], []
            for pid in pids:
                yt_h, yp_h = hps[pid]
                yt_s, yp_s = stl[pid]
                if len(yt_h) != len(yt_s):
                    continue
                a_correct_all.append(yp_h == yt_h)
                b_correct_all.append(yp_s == yt_s)
            if not a_correct_all:
                rows.append({'dataset': ds, 'dim': dim,
                             'error': 'no aligned window predictions'})
                continue
            a_correct = np.concatenate(a_correct_all)
            b_correct = np.concatenate(b_correct_all)
            res = _mcnemar(a_correct, b_correct)
            rows.append({
                'dataset':        ds,
                'dim':            dim,
                'n_participants': len(pids),
                'n_windows':      res['n_total'],
                'acc_MTL-HPS':    float(np.mean(a_correct)),
                'acc_STL':        float(np.mean(b_correct)),
                'b_HPSwrong_STLright': res['b'],
                'c_HPSright_STLwrong': res['c'],
                'discordant':     res['discordant'],
                'p_value':        res['p_value'],
                'odds_ratio':     res['odds_ratio'],
                'error':          '',
            })
    df = pd.DataFrame(rows)
    df['sig_0.05'] = (df['p_value'] < 0.05).where(df['p_value'].notna(), '')
    return df


def _format(df, float_cols, signed_cols=()):
    if df.empty:
        return df.to_string(index=False)
    show = df.copy()
    for c in float_cols:
        if c in show:
            show[c] = show[c].map(lambda v: f'{v:.4f}' if pd.notna(v) else '-')
    for c in signed_cols:
        if c in show:
            show[c] = show[c].map(lambda v: f'{v:+.4f}' if pd.notna(v) else '-')
    return show.to_string(index=False)


if __name__ == '__main__':
    print('=' * 78)
    print('PRIMARY  - Paired Wilcoxon signed-rank on per-participant accuracy')
    print('           MTL-HPS vs STL.  Unit: participants.')
    print('           No multiple-comparison correction (AR/VA independent).')
    print('=' * 78)
    w_df = _run_wilcoxon()
    print(_format(w_df, ['p_value'],
                  ['median_MTL-HPS', 'median_STL', 'median_diff',
                   'rank_biserial']))
    out_w = os.path.join(RESULTS_DIR, 'significance_tests.csv')
    w_df.to_csv(out_w, index=False)
    print(f'\n[OK] Saved {out_w}')

    print('\n' + '=' * 78)
    print('SECONDARY - McNemar exact test on pooled window-level predictions')
    print('            MTL-HPS vs STL.  Unit: test windows.')
    print('            Convention used by Taylor et al. (2020) and Jaques et al.')
    print('            NOTE: ignores within-participant clustering.')
    print('=' * 78)
    m_df = _run_mcnemar()
    print(_format(m_df, ['p_value'], ['acc_MTL-HPS', 'acc_STL', 'odds_ratio']))
    out_m = os.path.join(RESULTS_DIR, 'significance_tests_mcnemar.csv')
    m_df.to_csv(out_m, index=False)
    print(f'\n[OK] Saved {out_m}')

    print('\nInterpretation:')
    print('  Wilcoxon rank_biserial > 0   -> MTL-HPS wins for more participants')
    print('  McNemar  odds_ratio   > 1    -> MTL-HPS correct on more windows')
    print('  p_value  < 0.05              -> significant at conventional alpha')
