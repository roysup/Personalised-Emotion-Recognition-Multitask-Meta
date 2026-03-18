"""
Model definitions for all training scripts.
All models accept input_dim (number of signal channels) — defaults to 2 for VREED.

MTL_baselines
-------------
SingleTaskModel  — per-participant or population single-task model (pstl.py, stl.py)
MTLModel         — Hard Parameter Sharing with compute_l2() (mtl_hps.py, mtl_pcgrad.py)
MTLModelUW       — HPS + learnable log_vars per task (mtl_uw.py)

MTML_baselines
--------------
BaseFeatureExtractor  — shared CNN+LSTM backbone (MTML scripts)
TaskHead              — per-participant dense head (MTML scripts)
MTLTransferModel      — HPS backbone + Sequential task heads + add_task_head()
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
    Input: (batch, n_channels, window_size) — permuted internally from (batch, window, channels).
    Output: (batch, 1) logit.

    Parameters
    ----------
    input_dim : int — number of input channels (default 2 for VREED: ECG + GSR)
    """
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.conv1  = nn.Conv1d(input_dim, 128, kernel_size=2, padding=0)
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
    num_tasks : int — number of participants
    input_dim : int — number of input channels (default 2)
    """
    def __init__(self, num_tasks: int, input_dim: int = 2):
        super().__init__()
        self.num_tasks = num_tasks

        # Shared layers
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=2, padding=0)
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

    def compute_l2(self, l2_shared: float = 0.0, l2_task: float = 1e-5) -> torch.Tensor:
        ls = l2_shared * sum(p.norm(2) ** 2
                             for p in self.shared_parameters() if p.requires_grad)
        lt = l2_task   * sum(p.norm(2) ** 2
                             for p in self.task_specific_parameters() if p.requires_grad)
        return ls + lt


class MTLModelUW(MTLModel):
    """
    MTLModel extended with per-task learnable log-uncertainty weights.

    Parameters
    ----------
    num_tasks : int
    input_dim : int — number of input channels (default 2)
    """
    def __init__(self, num_tasks: int, input_dim: int = 2):
        super().__init__(num_tasks, input_dim)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))


# ============================================================
# MTML_baselines — backbone + head
# ============================================================

class BaseFeatureExtractor(nn.Module):
    """
    Shared CNN+LSTM backbone for MTML scripts (no task-specific heads).
    Output: (batch, 64) feature vector.

    Parameters
    ----------
    input_dim : int — number of input channels (default 2)
    """
    def __init__(self, input_dim: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=2, padding=0)
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
    """Per-participant dense head. Input: (batch, 64). Output: (batch, 1) logit."""
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


# ============================================================
# MTML_baselines — retrain / transfer model
# ============================================================

class MTLTransferModel(nn.Module):
    """
    Shared BaseFeatureExtractor backbone with per-participant Sequential task heads.

    Parameters
    ----------
    num_tasks : int
    hidden    : int — backbone output width (default 64)
    input_dim : int — number of input channels (default 2)
    """
    def __init__(self, num_tasks: int, hidden: int = 64, input_dim: int = 2):
        super().__init__()
        self.backbone  = BaseFeatureExtractor(input_dim)
        self.head1     = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, 128), nn.ReLU()) for _ in range(num_tasks)])
        self.head2     = nn.ModuleList(
            [nn.Sequential(nn.Linear(128, 64), nn.ReLU()) for _ in range(num_tasks)])
        self.out       = nn.ModuleList(
            [nn.Linear(64, 1) for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.apply(_xavier_init)

    def forward(self, x, task_ids):
        feats = self.backbone(x)
        out   = torch.zeros(feats.size(0), 1, device=x.device)
        for t in torch.unique(task_ids):
            mask = (task_ids == t)
            ti   = int(t.item())
            if ti < 0 or ti >= self.num_tasks or feats[mask].size(0) == 0:
                continue
            h = self.head1[ti](feats[mask])
            h = self.head2[ti](h)
            out[mask] = self.out[ti](h)
        return out

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def task_specific_parameters(self):
        return (list(self.head1.parameters()) +
                list(self.head2.parameters()) +
                list(self.out.parameters()))

    def compute_l2(self, l2_shared: float = 0.0, l2_task: float = 1e-5) -> torch.Tensor:
        ls = l2_shared * sum(p.norm(2) ** 2
                             for p in self.backbone_parameters() if p.requires_grad)
        lt = l2_task   * sum(p.norm(2) ** 2
                             for p in self.task_specific_parameters() if p.requires_grad)
        return ls + lt

    def add_task_head(self) -> int:
        device = next(self.parameters()).device
        self.head1.append(nn.Sequential(nn.Linear(64, 128), nn.ReLU()).to(device))
        self.head2.append(nn.Sequential(nn.Linear(128, 64), nn.ReLU()).to(device))
        self.out.append(nn.Linear(64, 1).to(device))
        self.num_tasks += 1
        return self.num_tasks - 1
