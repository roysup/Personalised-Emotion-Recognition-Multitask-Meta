"""
VREED dataset configuration.
Centralises the CSV path, column renames, and hardcoded train/test splits
so no script needs to duplicate this boilerplate.

Usage
-----
from dataset_configs.vreed import load_vreed_df, participant_ids, HARDCODED_SPLITS
df, participant_ids = load_vreed_df()
"""
import pickle
import pandas as pd
from config import HARDCODED_SPLITS

# CSV_PATH = '/content/drive/MyDrive/Phase A/src_new/dataset_configs/VREED_data_v2.csv'
# PKL_PATH = '/content/drive/MyDrive/Phase A/src_new/dataset_configs/unique_id_trials_VREED_v2.pkl'

CSV_PATH = '/content/drive/MyDrive/Phase A/data/VREED_data_v2.csv'
PKL_PATH = '/content/drive/MyDrive/Phase A/data/unique_id_trials_VREED_v2.pkl'

# import os
# _DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.join(_DIR, 'VREED_data_v2.csv')
# PKL_PATH = os.path.join(_DIR, 'unique_id_trials_VREED_v2.pkl')

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
