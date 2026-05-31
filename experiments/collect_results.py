"""
collect_results.py
==================
Reads all experiment .pkl result files across 3 datasets × 13 scripts and
compiles a single CSV with mean ± std for every metric, matching the
DETERMINISM VERIFICATION block printed at the end of each experiment.

Run from the repo root:
    python experiments/collect_results.py
    python experiments/collect_results.py --results_dir /custom/path/results

Output: results_summary.csv  (placed in RESULTS_DIR by default)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo-root detection (works whether run from repo root or experiments/)
# ---------------------------------------------------------------------------
_THIS = os.path.abspath(__file__)
# Try two levels up from the script location
_REPO = os.path.dirname(os.path.dirname(_THIS))
if not os.path.isdir(os.path.join(_REPO, 'src')):
    _REPO = os.path.dirname(_THIS)

sys.path.insert(0, os.path.join(_REPO, 'src'))

# ---------------------------------------------------------------------------
# Map: (dataset_prefix, script_name) → relative path to the pkl result file
# ---------------------------------------------------------------------------
# fmt: {prefix} is substituted at runtime (VREED / DSSN_EQ / DSSN_EM)

MTL_SCRIPTS = [
    # (friendly_name,    subfolder_template,            pkl_filename)
    ('P-STL',           '{p}_MTL/{p}_pstl_results',    'pstl_results.pkl'),
    ('STL',             '{p}_MTL/{p}_stl_results',     'stl_tuned_results.pkl'),
    ('MTL-HPS',         '{p}_MTL/{p}_hps_results',     'hps_tuned_results.pkl'),
    ('MTL-PCGrad',      '{p}_MTL/{p}_hps_pcgrad_results', 'hps_pcgrad_results.pkl'),
    ('MTL-UW',          '{p}_MTL/{p}_hps_uw_results',  'hps_uw_results.pkl'),
    ('MTL-TAG',         '{p}_TAG/{p}_tag_results',     'tag_results.pkl'),
]

MTML_SCRIPTS = [
    ('SI',              '{p}_MTML/{p}_SI',              'si_results.pkl'),
    ('TL-FT',           '{p}_MTML/{p}_TF',              'tlft_results.pkl'),
    ('Transfer-MTL',    '{p}_MTML/{p}_transfer_mtl',    'transfer_mtl_results.pkl'),
    ('MTL-Retrain',     '{p}_MTML/{p}_mtl_retrain',     'mtl_retrain_results.pkl'),
    ('Pure-Meta',       '{p}_MTML/{p}_pure_meta',       'pure_meta_results.pkl'),
    ('Reptile-ST',      '{p}_MTML/{p}_reptile_st',      'reptile_st_results.pkl'),
    ('Reptile-MT',      '{p}_MTML/{p}_reptile_mt',      'reptile_mt_results.pkl'),
    ('Reptile-MI',      '{p}_MTML/{p}_reptile_mi',      'reptile_mi_results.pkl'),
    ('Reptile-MI-Dyn',  '{p}_MTML/{p}_reptile_mi_dynamic', 'reptile_mi_dynamic_results.pkl'),
]

ALL_SCRIPTS = MTL_SCRIPTS + MTML_SCRIPTS

DATASETS = [
    ('VREED',   'VREED'),
    ('DSSN_EQ', 'DSSN_EQ'),
    ('DSSN_EM', 'DSSN_EM'),
]

METRICS = ['auc', 'acc', 'precision', 'recall', 'f1']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def extract_metrics(data: dict) -> dict | None:
    """
    Pull AR/VA mean + std from a loaded pkl dict.

    The pkl may store values as:
      ar_auc, ar_acc, ar_precision, ar_recall, ar_f1          (means)
      ar_auc_std, ar_acc_std, ar_precision_std, ...            (stds)
    """
    row = {}
    found_any = False
    for dim in ('ar', 'va'):
        for m in METRICS:
            mean_key = f'{dim}_{m}'
            std_key  = f'{dim}_{m}_std'
            mean_val = _safe_float(data.get(mean_key, np.nan))
            std_val  = _safe_float(data.get(std_key,  np.nan))
            row[f'{dim.upper()}_{m}_mean'] = mean_val
            row[f'{dim.upper()}_{m}_std']  = std_val
            if not np.isnan(mean_val):
                found_any = True
    return row if found_any else None


def load_pkl(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(results_dir: str) -> pd.DataFrame:
    rows = []
    missing = []

    for dataset_name, prefix in DATASETS:
        for friendly, subfolder_tmpl, pkl_name in ALL_SCRIPTS:
            subfolder = subfolder_tmpl.format(p=prefix)
            pkl_path  = os.path.join(results_dir, subfolder, pkl_name)

            data = load_pkl(pkl_path)
            if data is None:
                missing.append(f"  MISSING  [{dataset_name}] {friendly}")
                # Still add a row of NaNs so the table is complete
                row = {'Dataset': dataset_name, 'Method': friendly}
                for dim in ('AR', 'VA'):
                    for m in METRICS:
                        row[f'{dim}_{m}_mean'] = np.nan
                        row[f'{dim}_{m}_std']  = np.nan
                rows.append(row)
                continue

            metrics = extract_metrics(data)
            if metrics is None:
                missing.append(f"  NO_METRICS [{dataset_name}] {friendly}")
                row = {'Dataset': dataset_name, 'Method': friendly}
                for dim in ('AR', 'VA'):
                    for m in METRICS:
                        row[f'{dim}_{m}_mean'] = np.nan
                        row[f'{dim}_{m}_std']  = np.nan
                rows.append(row)
                continue

            row = {'Dataset': dataset_name, 'Method': friendly}
            row.update(metrics)
            rows.append(row)
            print(f"  OK  [{dataset_name}] {friendly}")

    if missing:
        print(f"\n{'='*60}")
        print(f"Missing / empty results ({len(missing)} entries):")
        for m in missing:
            print(m)
        print(f"{'='*60}\n")

    # Build ordered columns
    col_order = ['Dataset', 'Method']
    for dim in ('AR', 'VA'):
        for m in METRICS:
            col_order += [f'{dim}_{m}_mean', f'{dim}_{m}_std']

    df = pd.DataFrame(rows, columns=col_order)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Collect all experiment results into a CSV.')
    p.add_argument('--results_dir', type=str, default=None,
                   help='Path to the results/ directory (default: <repo_root>/results)')
    p.add_argument('--out', type=str, default=None,
                   help='Output CSV path (default: <results_dir>/results_summary.csv)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    results_dir = args.results_dir or os.path.join(_REPO, 'results')
    if not os.path.isdir(results_dir):
        print(f"[ERROR] results_dir not found: {results_dir}")
        sys.exit(1)

    out_path = args.out or os.path.join(results_dir, 'results_summary.csv')

    print(f"Scanning: {results_dir}\n")
    df = collect(results_dir)

    df.to_csv(out_path, index=False, float_format='%.8f')
    print(f"\n✓ Saved {len(df)} rows → {out_path}")

    # Pretty-print a pivot-style preview
    print("\nPreview (AR F1 mean ± std per method):")
    pivot = df.pivot_table(
        index='Method', columns='Dataset',
        values='AR_f1_mean', aggfunc='first'
    )
    print(pivot.to_string())
