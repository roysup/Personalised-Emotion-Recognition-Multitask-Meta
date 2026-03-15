# =============================
# ENVIRONMENT  (must come before any torch import)
# =============================
import os
import sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

# =============================
# SHARED IMPORTS
# =============================
import gc
import copy
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score,
    mutual_info_score,
)

# =============================
# PATHS
# =============================
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(_REPO_ROOT, 'results')
CSV_PATH    = os.path.join(DATA_DIR, 'VREED_data_v2.csv')
PKL_PATH    = os.path.join(DATA_DIR, 'unique_id_trials_VREED_v2.pkl')

# =============================
# TRAINING CONSTANTS
# =============================
SEED        = 42
WINDOW_SIZE = 2560
STRIDE      = 1280
N_FOLDS     = 5
MAX_NORM    = 1.0
EPOCHS      = 30
FT_EPOCHS      = 10   # fine-tuning epochs (transfer_mtl.py)
FT_BATCH_SIZE  = 32   # fine-tuning batch size (transfer_mtl.py)

# =============================
# BATCH SIZES
# =============================
PSTL_BATCH_SIZE = 32
STL_BATCH_SIZE  = 8
MTL_BATCH_SIZE  = 26
SI_BATCH_SIZE   = 32

# =============================
# MTL LEARNING RATES / L2
# =============================
MTL_SHARED_LR = 3e-4
MTL_TASK_LR   = 1e-4
MTL_L2_TASK   = 1e-5
MTL_L2_SHARED = 0.0

# Aliases used by reptile scripts (backbone L2 is always 0, task L2 always 1e-5)
L2_SHARED = MTL_L2_SHARED  # 0.0
L2_TASK   = MTL_L2_TASK    # 1e-5

# =============================
# META-LEARNING DEFAULTS
# =============================
META_STEPS   = 50
META_LR      = 0.01
INNER_STEPS  = 10
INNER_LR     = 1e-3
EPISODE_SIZE = 5
L2_LAMBDA    = 1e-5

# =============================
# SCRIPT-SPECIFIC LR DEFAULTS
# =============================
SI_LR              = 3e-4    # si.py pretrain lr (= MTL_SHARED_LR)
SI_L2              = 1e-5    # si.py l2 (= L2_LAMBDA)
RETRAIN_LR         = 1e-4    # mtl_retrain.py
TF_LR_PRE          = 1e-3    # tlft.py pretrain lr
TF_LR_FT           = 1e-3    # tlft.py finetune lr
TF_L2              = 1e-5    # tlft.py l2 (= L2_LAMBDA)
TRANSFER_MTL_LR_PT = 1e-4    # transfer_mtl.py pretrain lr
TRANSFER_MTL_LR_FT = 5e-5    # transfer_mtl.py finetune lr

# =============================
# MTML TEST/TRAIN SPLIT
# =============================
TEST_PARTICIPANTS  = [105, 109, 112, 125, 131, 132]
TRAIN_PARTICIPANTS = None  # derived per-script: sorted([p for p in participant_ids if p not in TEST_PARTICIPANTS])

# =============================
# HARDCODED 10/2 TRAIN/TEST SPLITS
# =============================
HARDCODED_SPLITS = {
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
