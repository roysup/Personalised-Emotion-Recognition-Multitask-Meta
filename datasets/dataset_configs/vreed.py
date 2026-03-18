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

from config import HARDCODED_SPLITS, CSV_PATH, PKL_PATH

# Use the symlink created: ln -s "Phase A/data" data
# CSV_PATH = os.path.join(_REPO_ROOT, 'data', 'VREED_data_v2.csv')
# PKL_PATH = os.path.join(_REPO_ROOT, 'data', 'unique_id_trials_VREED_v2.pkl')

participant_ids = sorted(HARDCODED_SPLITS.keys())

def load_vreed_df(preserve_trial_order: bool = False,
                  mode: str = 'standard') -> pd.DataFrame:
    """
    Load and clean the VREED CSV.

    Parameters
    ----------
    preserve_trial_order : bool
        If True, reorder rows using the canonical trial ordering from the pkl
        (required by pstl.py and si.py). Only used when mode='standard'.
    mode : str
        'standard' — default. Returns ECG, GSR, AR/VA_Rating, Trial, ID, ID_video,
                      trial_global. Used by MTL baselines and SI/TL-FT scripts.
        'mtml'     — Returns the same columns plus Trial aliased from Num_Code,
                     sorted by ID then Trial. Used by MTML/meta-learning scripts.

    Returns
    -------
    df : cleaned DataFrame
    """
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['ECG', 'GSR', 'Unnamed: 0.1', 'Unnamed: 0', 'Trial'],
                 errors='ignore')
    df = df.rename(columns={'ECG_scaled': 'ECG', 'GSR_scaled': 'GSR',
                            'Num_Code': 'Trial'})

    if mode == 'mtml':
        df = df.sort_values(['ID', 'Trial']).reset_index(drop=True)
        return df

    # mode == 'standard'
    if preserve_trial_order:
        with open(PKL_PATH, 'rb') as f:
            unique_id_trials = sorted(pickle.load(f))
        df['participant_trial_encoded'] = df['ID_video'].astype(str)
        reordered = pd.DataFrame()
        for id_trial in unique_id_trials:
            reordered = pd.concat([reordered, df[df['ID_video'] == id_trial]])
        df = reordered.reset_index(drop=True)

    df['trial_global'] = df['ID'].astype(str) + '_' + df['Trial'].astype(str)
    return df
