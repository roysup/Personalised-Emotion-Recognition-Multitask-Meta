"""
Model definitions for all training scripts.

MTL_baselines
-------------
PSTLModel      — population single-task model (used by pstl.py)
STLModel       — per-participant single-task model (used by stl.py)
MTLModel       — Hard Parameter Sharing (used by mtl_hps.py and mtl_pcgrad.py)
MTLModelUW     — HPS + learnable log_vars per task (used by mtl_uw.py)

MTML_baselines
--------------
BaseFeatureExtractor  — shared CNN+LSTM backbone (used by MTML scripts)
TaskHead              — per-participant dense head (used by MTML scripts)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SHARED INIT HELPER
# ============================================================

def _xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================================
# MTL_baselines — single-task
# ============================================================

class SingleTaskModel(nn.Module):
    """
    Single-task CNN+LSTM model used by pstl.py and stl.py.
    Input: (batch, 2, window_size) — ECG + GSR channels.
    Output: (batch, 1) logit.
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv1d(2, 128, kernel_size=2, padding=0)
        self.bn1    = nn.BatchNorm1d(128)
        self.pool1  = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2  = nn.Conv1d(128, 64, kernel_size=1, padding=0)
        self.bn2    = nn.BatchNorm1d(64)
        self.pool2  = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm   = nn.LSTM(64, 64, batch_first=True)
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 64)
        self.out    = nn.Linear(64, 1)
        self.apply(_xavier_init)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.out(x)


# ============================================================
# MTL_baselines — multi-task
# ============================================================

class MTLModel(nn.Module):
    """
    Shared CNN+LSTM backbone with per-participant dense output heads.
    Used by mtl_hps.py and mtl_pcgrad.py.

    Parameters
    ----------
    num_tasks : int — number of participants (26 for VREED)
    """
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks

        # Shared layers
        self.conv1 = nn.Conv1d(2, 128, kernel_size=2, padding=0)
        self.bn1   = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=1, padding=0)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm  = nn.LSTM(64, 64, batch_first=True)

        # Task-specific heads
        self.task_dense1 = nn.ModuleList([nn.Linear(64, 128) for _ in range(num_tasks)])
        self.task_dense2 = nn.ModuleList([nn.Linear(128, 64) for _ in range(num_tasks)])
        self.task_out    = nn.ModuleList([nn.Linear(64, 1)   for _ in range(num_tasks)])

        self.apply(_xavier_init)

    def shared_forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return torch.mean(x, dim=1)

    def forward(self, x, task_ids):
        shared  = self.shared_forward(x)
        outputs = torch.zeros(len(x), 1, device=x.device)
        for t in torch.unique(task_ids):
            t    = t.item()
            mask = (task_ids == t)
            h    = F.relu(self.task_dense1[t](shared[mask]))
            h    = F.relu(self.task_dense2[t](h))
            outputs[mask] = self.task_out[t](h)
        return outputs

    def shared_parameters(self):
        return (list(self.conv1.parameters()) + list(self.bn1.parameters()) +
                list(self.conv2.parameters()) + list(self.bn2.parameters()) +
                list(self.lstm.parameters()))

    def task_specific_parameters(self):
        return (list(self.task_dense1.parameters()) +
                list(self.task_dense2.parameters()) +
                list(self.task_out.parameters()))


class MTLModelUW(MTLModel):
    """
    MTLModel extended with per-task learnable log-uncertainty weights.
    Used by mtl_uw.py.

    Extra attribute
    ---------------
    log_vars : nn.Parameter of shape (num_tasks,) — initialised to zeros
    """
    def __init__(self, num_tasks: int):
        super().__init__(num_tasks)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))


# ============================================================
# MTML_baselines
# ============================================================

class BaseFeatureExtractor(nn.Module):
    """
    Shared CNN+LSTM backbone for MTML scripts (no task-specific heads).
    Output: (batch, 64) feature vector.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 128, kernel_size=2, padding=0)
        self.bn1   = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=1, padding=0)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm  = nn.LSTM(64, 64, batch_first=True)
        self.apply(_xavier_init)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return torch.mean(x, dim=1)


class TaskHead(nn.Module):
    """
    Per-participant dense head for MTML scripts.
    Input: (batch, 64) from BaseFeatureExtractor.
    Output: (batch, 1) logit.
    """
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 64)
        self.out    = nn.Linear(64, 1)
        self.apply(_xavier_init)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.out(x)
