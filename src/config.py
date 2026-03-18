# =============================
# ENVIRONMENT  (must come before any torch import)
# =============================
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

# =============================
# PATHS
# =============================
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(_REPO_ROOT, 'results')
CSV_PATH    = os.path.join(DATA_DIR, 'VREED_data_v2.csv')
PKL_PATH    = os.path.join(DATA_DIR, 'unique_id_trials_VREED_v2.pkl')

# =============================
# TRAINING CONSTANTS  (VREED defaults — backward-compatible)
# =============================
SEED        = 42
WINDOW_SIZE = 2560
STRIDE      = 1280
N_FOLDS     = 5
MAX_NORM    = 1.0
EPOCHS      = 30
FT_EPOCHS   = 10   # fine-tuning epochs (transfer_mtl.py, tlft.py)

# =============================
# BATCH SIZES
# =============================
PSTL_BATCH_SIZE = 32
STL_BATCH_SIZE  = 8
MTL_BATCH_SIZE  = 26

# =============================
# LEARNING RATES / L2
# =============================
MTL_SHARED_LR = 3e-4
MTL_TASK_LR   = 1e-4
L2_TASK       = 1e-5
L2_SHARED     = 0.0
L2_LAMBDA     = 1e-5   # alias used by pstl.py

# =============================
# UW LOG-VARIANCE LEARNING RATES
# =============================
UW_LOG_VAR_LR_AR = 4e-3
UW_LOG_VAR_LR_VA = 1e-3

# =============================
# META-LEARNING DEFAULTS
# =============================
META_STEPS   = 50
META_LR      = 0.01
INNER_STEPS  = 10
INNER_LR     = 1e-3
EPISODE_SIZE = 5

# =============================
# SCRIPT-SPECIFIC LR DEFAULTS
# =============================
TF_LR_PRE          = 1e-3
TF_LR_FT           = 1e-3
TRANSFER_MTL_LR_PT = 1e-4
TRANSFER_MTL_LR_FT = 5e-5

# =============================
# MTML TEST/TRAIN SPLIT
# =============================
# VREED: 6 of 26 participants held out
TEST_PARTICIPANTS       = [105, 109, 112, 125, 131, 132]
TRAIN_PARTICIPANTS      = None  # derived per-script

# DSSN_EQ: 6 of 30 participants held out (80/20 split)
DSSN_EQ_TEST_PARTICIPANTS = [28, 32, 36, 17, 39, 25]

# DSSN_EM: same 6 as DSSN_EQ (same participant pool, pending confirmation)
DSSN_EM_TEST_PARTICIPANTS = [17, 25, 28, 32, 36, 39]

# =============================
# VREED 10/2 TRAIN/TEST SPLITS  (26 participants, IDs 104–134)
# =============================
VREED_SPLITS = {
    107: {'train': [5, 9, 10, 2, 3, 4, 11, 6, 7, 12], 'test': [1, 8]},
    122: {'train': [4, 12, 5, 11, 9, 10, 7, 3, 8, 2], 'test': [1, 6]},
    109: {'train': [12, 8, 11, 10, 6, 2, 1, 4, 5, 7], 'test': [3, 9]},
    106: {'train': [4, 1, 3, 12, 10, 9, 5, 2, 8, 11], 'test': [6, 7]},
    124: {'train': [12, 5, 9, 10, 7, 2, 4, 1, 11, 6], 'test': [3, 8]},
    112: {'train': [10, 5, 4, 9, 1, 11, 3, 6, 2, 8], 'test': [7, 12]},
    126: {'train': [7, 12, 5, 11, 10, 3, 1, 2, 8, 6], 'test': [9, 4]},
    132: {'train': [1, 7, 9, 5, 8, 4, 10, 6, 12, 2], 'test': [11, 3]},
    129: {'train': [3, 5, 1, 8, 6, 12, 9, 11, 10, 7], 'test': [2, 4]},
    134: {'train': [7, 2, 12, 3, 8, 9, 10, 11, 6, 4], 'test': [5, 1]},
    105: {'train': [3, 2, 10, 6, 11, 12, 1, 8, 9, 5], 'test': [7, 4]},
    104: {'train': [12, 9, 6, 4, 7, 1, 3, 11, 5, 8], 'test': [10, 2]},
    111: {'train': [10, 6, 8, 9, 7, 2, 3, 4, 5, 11], 'test': [1, 12]},
    114: {'train': [4, 3, 7, 12, 1, 11, 2, 8, 9, 5], 'test': [10, 6]},
    128: {'train': [6, 11, 10, 7, 9, 3, 12, 2, 8, 4], 'test': [5, 1]},
    133: {'train': [7, 5, 10, 6, 8, 3, 1, 12, 2, 9], 'test': [11, 4]},
    131: {'train': [11, 3, 10, 1, 4, 9, 2, 8, 7, 5], 'test': [12, 6]},
    123: {'train': [10, 1, 8, 11, 2, 5, 6, 7, 4, 3], 'test': [12, 9]},
    110: {'train': [4, 6, 12, 3, 10, 7, 2, 1, 8, 11], 'test': [9, 5]},
    125: {'train': [4, 12, 3, 8, 6, 7, 10, 2, 5, 11], 'test': [1, 9]},
    127: {'train': [6, 1, 8, 7, 2, 5, 9, 10, 4, 12], 'test': [3, 11]},
    113: {'train': [10, 1, 6, 7, 4, 5, 11, 2, 9, 12], 'test': [8, 3]},
    120: {'train': [1, 3, 12, 9, 2, 6, 4, 7, 10, 5], 'test': [11, 8]},
    117: {'train': [8, 10, 5, 3, 7, 6, 1, 11, 2, 9], 'test': [4, 12]},
    108: {'train': [5, 8, 10, 1, 7, 4, 2, 3, 9, 12], 'test': [11, 6]},
    116: {'train': [2, 10, 12, 1, 6, 5, 7, 3, 9, 4], 'test': [8, 11]},
}

# Backward-compatible alias — existing VREED scripts import HARDCODED_SPLITS
HARDCODED_SPLITS = VREED_SPLITS

# =============================
# DSSN_EQ 5/1 TRAIN/TEST SPLITS  (30 participants, IDs 10–39)
# =============================
DSSN_EQ_SPLITS = hardcoded_splits = {
        2: {'train': [3, 5, 1, 6, 4], 'test': [2]},
        5: {'train': [6, 2, 5, 1, 4], 'test': [3]},
        8: {'train': [6, 5, 3, 4, 2], 'test': [1]},
        9: {'train': [2, 6, 3, 4, 5], 'test': [1]},
        10: {'train': [5, 2, 3, 1, 4], 'test': [6]},
        11: {'train': [2, 1, 5, 3, 6], 'test': [4]},
        12: {'train': [2, 4, 3, 5, 1], 'test': [6]},
        13: {'train': [2, 6, 4, 1, 3], 'test': [5]},
        14: {'train': [1, 6, 2, 3, 4], 'test': [5]},
        15: {'train': [4, 1, 6, 3, 2], 'test': [5]},
        16: {'train': [2, 4, 5, 1, 3], 'test': [6]},
        17: {'train': [4, 5, 1, 3, 6], 'test': [2]},
        18: {'train': [2, 1, 6, 4, 5], 'test': [3]},
        19: {'train': [2, 5, 4, 1, 3], 'test': [6]},
        20: {'train': [6, 1, 3, 5, 4], 'test': [2]},
        21: {'train': [5, 6, 4, 3, 2], 'test': [1]},
        22: {'train': [4, 2, 1, 6, 5], 'test': [3]},
        23: {'train': [1, 5, 4, 6, 3], 'test': [2]},
        24: {'train': [1, 3, 4, 5, 6], 'test': [2]},
        25: {'train': [6, 4, 1, 2, 3], 'test': [5]},
        26: {'train': [3, 4, 5, 6, 2], 'test': [1]},
        27: {'train': [6, 2, 4, 5, 3], 'test': [1]},
        28: {'train': [1, 3, 4, 6, 5], 'test': [2]},
        29: {'train': [2, 1, 5, 3, 6], 'test': [4]},
        30: {'train': [6, 3, 5, 4, 2], 'test': [1]},
        31: {'train': [4, 5, 2, 3, 1], 'test': [6]},
        32: {'train': [5, 2, 4, 1, 6], 'test': [3]}, 
        33: {'train': [5, 1, 3, 2, 6], 'test': [4]},
        34: {'train': [6, 5, 2, 3, 4], 'test': [1]},
        35: {'train': [4, 6, 3, 1, 2], 'test': [5]},
        36: {'train': [4, 3, 6, 2, 5], 'test': [1]},
        37: {'train': [5, 6, 1, 2, 4], 'test': [3]},
        38: {'train': [4, 2, 5, 1, 6], 'test': [3]},
        39: {'train': [3, 5, 6, 2, 1], 'test': [4]}
}

# =============================
# DSSN_EM 5/1 TRAIN/TEST SPLITS  (30 participants, IDs 10–39)
# =============================
DSSN_EM_SPLITS = {
    10: {'train': [5, 2, 3, 1, 4], 'test': [6]},
    11: {'train': [2, 1, 5, 3, 6], 'test': [4]},
    12: {'train': [2, 4, 3, 5, 1], 'test': [6]},
    13: {'train': [2, 6, 4, 1, 3], 'test': [5]},
    14: {'train': [1, 6, 2, 3, 4], 'test': [5]},
    15: {'train': [4, 1, 6, 3, 2], 'test': [5]},
    16: {'train': [2, 4, 5, 1, 3], 'test': [6]},
    17: {'train': [4, 5, 1, 3, 6], 'test': [2]},
    18: {'train': [2, 1, 6, 4, 5], 'test': [3]},
    19: {'train': [6, 5, 4, 1, 2], 'test': [3]},
    20: {'train': [6, 1, 3, 5, 4], 'test': [2]},
    21: {'train': [5, 6, 4, 3, 2], 'test': [1]},
    22: {'train': [4, 2, 1, 6, 5], 'test': [3]},
    23: {'train': [1, 5, 4, 6, 3], 'test': [2]},
    24: {'train': [1, 3, 4, 5, 6], 'test': [2]},
    25: {'train': [6, 4, 1, 2, 3], 'test': [5]},
    26: {'train': [3, 4, 5, 6, 2], 'test': [1]},
    27: {'train': [6, 2, 4, 5, 3], 'test': [1]},
    28: {'train': [1, 3, 4, 6, 5], 'test': [2]},
    29: {'train': [2, 1, 5, 3, 6], 'test': [4]},
    30: {'train': [6, 3, 5, 4, 2], 'test': [1]},
    31: {'train': [4, 5, 2, 3, 1], 'test': [6]},
    32: {'train': [5, 2, 3, 1, 4], 'test': [6]},
    33: {'train': [5, 1, 3, 2, 6], 'test': [4]},
    34: {'train': [6, 5, 2, 3, 4], 'test': [1]},
    35: {'train': [4, 6, 3, 1, 2], 'test': [5]},
    36: {'train': [4, 3, 6, 2, 5], 'test': [1]},
    37: {'train': [5, 6, 1, 2, 4], 'test': [3]},
    38: {'train': [4, 2, 5, 1, 6], 'test': [3]},
    39: {'train': [3, 5, 6, 2, 1], 'test': [4]},
}


# ==============================================================================
# DATASET REGISTRY
# ==============================================================================
# Each dataset config is a dict with all parameters that differ across datasets.
# Experiment scripts call get_dataset_config(name) instead of hard-wiring values.
# VREED scripts that import bare names (WINDOW_SIZE, HARDCODED_SPLITS, etc.)
# continue to work unchanged — those module-level constants are the VREED defaults.
# ==============================================================================

_DATASET_REGISTRY = {
    'vreed': {
        'csv_path':       os.path.join(DATA_DIR, 'VREED_data_v2.csv'),
        'pkl_path':       os.path.join(DATA_DIR, 'unique_id_trials_VREED_v2.pkl'),
        'feature_cols':   ['ECG', 'GSR'],
        'input_dim':      2,
        'window_size':    2560,
        'stride':         1280,
        'pstl_batch':     32,
        'stl_batch':      8,
        'mtl_batch':      26,
        'splits':         VREED_SPLITS,
        'test_participants': TEST_PARTICIPANTS,
        'results_prefix': 'VREED',
        'column_renames': {},
        'id_trial_col':   'ID_video',
        'trial_col':      'Trial',
        'uw_logvar_lr_ar': 4e-3,
        'uw_logvar_lr_va': 1e-3,
    },
    'dssn_eq': {
        'csv_path':       os.path.join(DATA_DIR, 'DSSN_EQ_data_v3.csv'),
        'pkl_path':       os.path.join(DATA_DIR, 'unique_id_trials_DSSN_EQ_v2.pkl'),
        'feature_cols':   ['ECG 1', 'ECG 2', 'GSR'],
        'input_dim':      3,
        'window_size':    2560,
        'stride':         1280,
        'pstl_batch':     32,
        'stl_batch':      8,
        'mtl_batch':      34,
        'splits':         DSSN_EQ_SPLITS,
        'test_participants': DSSN_EQ_TEST_PARTICIPANTS,
        'results_prefix': 'DSSN_EQ',
        'column_renames': {'ID_Trial': 'ID_video'},
        'id_trial_col':   'ID_video',
        'trial_col':      'Trial',
        'uw_logvar_lr_ar': 4e-3,
        'uw_logvar_lr_va': 1e-3,
    },
    'dssn_em': {
        'csv_path':       os.path.join(DATA_DIR, 'DSSN_EM_data_v2.csv'),
        'pkl_path':       os.path.join(DATA_DIR, 'unique_id_trials_DSSN_EM_v2.pkl'),
        'feature_cols':   ['eda_values', 'bvp_values', 'heart_rate'],
        'input_dim':      3,
        'window_size':    640,
        'stride':         320,
        'pstl_batch':     32,
        'stl_batch':      8,
        'mtl_batch':      28,
        'splits':         DSSN_EM_SPLITS,
        'test_participants': DSSN_EM_TEST_PARTICIPANTS,
        'results_prefix': 'DSSN_EM',
        'column_renames': {'ID_Trial': 'ID_video'},
        'id_trial_col':   'ID_video',
        'trial_col':      'Trial',
        'uw_logvar_lr_ar': 4e-3,
        'uw_logvar_lr_va': 1e-3,
    },
}


def get_dataset_config(name: str) -> dict:
    """
    Return the full configuration dict for a dataset.

    Parameters
    ----------
    name : str — one of 'vreed', 'dssn_eq', 'dssn_em'

    Returns
    -------
    dict with keys: csv_path, pkl_path, feature_cols, input_dim,
        window_size, stride, pstl_batch, stl_batch, mtl_batch,
        splits, results_prefix, column_renames,
        id_trial_col, trial_col, uw_logvar_lr_ar, uw_logvar_lr_va
    """
    key = name.lower().replace('-', '_')
    if key not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(_DATASET_REGISTRY.keys())}")
    return dict(_DATASET_REGISTRY[key])   # return a copy
