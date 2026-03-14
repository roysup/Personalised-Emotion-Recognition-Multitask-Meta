"""
VREED dataset configuration.
Centralises the CSV path, column renames, and hardcoded train/test splits
so no script needs to duplicate this boilerplate.

Usage
-----
from dataset_configs.vreed import load_vreed_df, participant_ids, HARDCODED_SPLITS
df, participant_ids = load_vreed_df()
"""
import sys
import os
import pickle
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from config import HARDCODED_SPLITS
from paths import CSV_PATH, PKL_PATH

# Use the symlink created: ln -s "Phase A/data" data
# CSV_PATH = os.path.join(_REPO_ROOT, 'data', 'VREED_data_v2.csv')
# PKL_PATH = os.path.join(_REPO_ROOT, 'data', 'unique_id_trials_VREED_v2.pkl')

participant_ids = sorted(HARDCODED_SPLITS.keys())

def load_vreed_df(preserve_trial_order: bool = False) -> pd.DataFrame:
    """
    Load and clean the VREED CSV.

    Parameters
    ----------
    preserve_trial_order : bool
        If True, reorder rows using the canonical trial ordering from the pkl
        (required by pstl.py). If False (default), keep the CSV row order.

    Returns
    -------
    df : DataFrame with columns ECG, GSR, AR_Rating, VA_Rating, Trial, ID, ID_video, ...
    """
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['ECG', 'GSR', 'Unnamed: 0.1', 'Unnamed: 0', 'Trial'],
                 errors='ignore')
    df = df.rename(columns={'ECG_scaled': 'ECG', 'GSR_scaled': 'GSR',
                            'Num_Code': 'Trial'})

    if preserve_trial_order:
        with open(PKL_PATH, 'rb') as f:
            unique_id_trials = sorted(pickle.load(f))
        df['participant_trial_encoded'] = df['ID_video'].astype(str)
        reordered = pd.DataFrame()
        for id_trial in unique_id_trials:
            reordered = pd.concat([reordered, df[df['ID_video'] == id_trial]])
        df = reordered.reset_index(drop=True)
    
    return df


def load_vreed_df_mtml() -> pd.DataFrame:
    """
    Load and clean the VREED CSV for MTML scripts.

    Identical to load_vreed_df() but adds a 'Trial' column aliased from the
    numeric video code (Num_Code → video → Trial), which the MTML scripts
    require for trial-level indexing in create_sliding_windows.

    Returns
    -------
    df : DataFrame with columns ECG, GSR, AR_Rating, VA_Rating, Trial, ID, ID_video, ...
         sorted by ID then Trial for deterministic ordering.
    """
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['ECG', 'GSR', 'Unnamed: 0.1', 'Unnamed: 0', 'Trial'],
                 errors='ignore')
    df = df.rename(columns={'ECG_scaled': 'ECG', 'GSR_scaled': 'GSR',
                            'Num_Code': 'video'})
    df['Trial'] = df['video']
    df = df.sort_values(['ID', 'Trial']).reset_index(drop=True)
    return df
