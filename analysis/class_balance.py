"""
Class Balance Check
Prints AR and VA label distributions for the specified dataset.

Usage
-----
    python class_balance.py                  # runs on VREED (default)
    python class_balance.py --dataset dssn_eq
    python class_balance.py --dataset dssn_em
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import pandas as pd
from config import get_dataset_config


def parse_args():
    p = argparse.ArgumentParser(description='Class balance check')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'],
                   help='Dataset to analyse (default: vreed)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    df = pd.read_csv(cfg['csv_path'])

    # Apply column renames if any (e.g. DSSN datasets)
    if cfg['column_renames']:
        df = df.rename(columns=cfg['column_renames'])

    print(f"Dataset: {args.dataset}  ({cfg['results_prefix']})")
    print(f"Rows: {len(df)}")

    print('\nAR_Rating class balance:')
    print(df['AR_Rating'].value_counts(normalize=True))

    print('\nVA_Rating class balance:')
    print(df['VA_Rating'].value_counts(normalize=True))
