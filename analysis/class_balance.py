"""
Class Balance Check
Prints AR and VA label distributions for the VREED dataset.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import pandas as pd
from config import CSV_PATH


if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    print('AR_Rating class balance:')
    print(df['AR_Rating'].value_counts(normalize=True))

    print('\nVA_Rating class balance:')
    print(df['VA_Rating'].value_counts(normalize=True))
