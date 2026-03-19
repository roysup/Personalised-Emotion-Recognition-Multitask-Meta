"""
DSSN_EM dataset configuration.
eda_values, bvp_values, heart_rate — 3 channels, 30 participants (IDs 10–39),
6 videos per participant. Train: 5, Test: 1.
Window: 640, Stride: 320.

Usage
-----
from dataset_configs.dssn_em import load_dssn_em_df, DSSN_EM_CONFIG
"""
import sys, os, pickle
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from config import get_dataset_config

DSSN_EM_CONFIG   = get_dataset_config('dssn_em')
DSSN_EM_SPLITS   = DSSN_EM_CONFIG['splits']
participant_ids  = sorted(DSSN_EM_SPLITS.keys())


def load_dssn_em_df(preserve_trial_order: bool = False,
                    mode: str = 'standard') -> pd.DataFrame:
    """
    Load and clean the DSSN_EM CSV.

    Parameters
    ----------
    preserve_trial_order : bool
        If True, reorder rows using the canonical trial ordering from the pkl.
    mode : str
        'standard' — for MTL baselines.
        'mtml'     — sorted by (ID, Trial) for meta-learning scripts.

    Returns
    -------
    df : cleaned DataFrame with columns: eda_values, bvp_values, heart_rate,
         AR_Rating, VA_Rating, Trial, ID, ID_video
    """
    cfg = DSSN_EM_CONFIG
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
