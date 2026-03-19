"""
DSSN_EQ dataset configuration.
ECG 1, ECG 2, GSR — 3 channels, 34 participants
(IDs 2,5,8,9,10-39), 6 videos per participant.
Train: 5 videos, Test: 1 video per participant.
Window: 2560, Stride: 1280.

Usage
-----
from dataset_configs.dssn_eq import load_dssn_eq_df, DSSN_EQ_CONFIG
"""
import sys, os, pickle
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from config import get_dataset_config

DSSN_EQ_CONFIG   = get_dataset_config('dssn_eq')
DSSN_EQ_SPLITS   = DSSN_EQ_CONFIG['splits']
participant_ids  = sorted(DSSN_EQ_SPLITS.keys())


def load_dssn_eq_df(preserve_trial_order: bool = False,
                    mode: str = 'standard') -> pd.DataFrame:
    """
    Load and clean the DSSN_EQ CSV.

    Parameters
    ----------
    preserve_trial_order : bool
        If True, reorder rows using the canonical trial ordering from the pkl.
    mode : str
        'standard' — for MTL baselines. Returns feature columns + Trial + ID.
        'mtml'     — sorted by (ID, Trial) for meta-learning scripts.

    Returns
    -------
    df : cleaned DataFrame with columns: ECG 1, ECG 2, GSR,
         AR_Rating, VA_Rating, Trial, ID, ID_video
    """
    cfg = DSSN_EQ_CONFIG
    df = pd.read_csv(cfg['csv_path'])

    # Apply column renames
    if cfg['column_renames']:
        df = df.rename(columns=cfg['column_renames'])

    if mode == 'mtml':
        df = df.sort_values(['ID', 'Trial']).reset_index(drop=True)
        return df

    # mode == 'standard'
    if preserve_trial_order:
        with open(cfg['pkl_path'], 'rb') as f:
            unique_id_trials = sorted(pickle.load(f))
        df['participant_trial_encoded'] = df[cfg['id_trial_col']].astype(str)
        reordered = pd.DataFrame()
        for id_trial in unique_id_trials:
            reordered = pd.concat([reordered,
                                   df[df[cfg['id_trial_col']] == id_trial]])
        df = reordered.reset_index(drop=True)

    df['trial_global'] = df['ID'].astype(str) + '_' + df['Trial'].astype(str)
    return df
