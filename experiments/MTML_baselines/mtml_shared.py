"""
Shared utilities for MTML_baselines scripts.
Import with: from mtml_shared import (create_sliding_windows, build_support_query,
                                       BaseFeatureExtractor, TaskHead, adapt_inner_loop,
                                       make_kfolds, final_meta_test_user, aggregate_results)
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score

from config import SEED, MAX_NORM
from utils import compute_metrics_from_cm


WINDOW_SIZE = 2560
STRIDE      = 1280


# =============================
# DATA
# =============================
def create_sliding_windows(data, window_size=WINDOW_SIZE, stride=STRIDE, task_id=None):
    X, y_ar, y_va, task_IDS, trial_ids = [], [], [], [], []
    for t in sorted(data['Trial'].unique()):
        d = data[data['Trial'] == t].reset_index(drop=True)
        orig = len(d)
        if orig < window_size:
            pad = pd.DataFrame({'ECG': [0]*(window_size-orig), 'GSR': [0]*(window_size-orig),
                                 'AR_Rating': [d['AR_Rating'].iloc[-1]]*(window_size-orig),
                                 'VA_Rating': [d['VA_Rating'].iloc[-1]]*(window_size-orig),
                                 'Trial': [t]*(window_size-orig)})
            d = pd.concat([d, pad], ignore_index=True)
        T = len(d); last = 0
        for i in range(0, T - window_size + 1, stride):
            X.append(d[['ECG', 'GSR']].iloc[i:i+window_size].values.astype(np.float32))
            idx = min(i + window_size - 1, orig - 1)
            y_ar.append(d['AR_Rating'].iloc[idx]); y_va.append(d['VA_Rating'].iloc[idx])
            task_IDS.append(task_id if task_id is not None else -1)
            trial_ids.append(t); last = i
        nxt = last + stride
        if nxt < T:
            valid = d[['ECG', 'GSR']].iloc[nxt:].values
            pad = window_size - len(valid)
            padded = np.pad(valid, ((0, pad), (0, 0))).astype(np.float32)
            X.append(padded); y_ar.append(d['AR_Rating'].iloc[orig-1])
            y_va.append(d['VA_Rating'].iloc[orig-1])
            task_IDS.append(task_id if task_id is not None else -1); trial_ids.append(t)
    return (np.array(X), np.array(y_ar, dtype=np.float32), np.array(y_va, dtype=np.float32),
            np.array(task_IDS, dtype=np.int64), np.array(trial_ids, dtype=np.int64))


def build_support_query(task_df, support_trials, query_trials, ar_or_va='ar',
                        seed=SEED, window_size=WINDOW_SIZE, stride=STRIDE):
    sup_df = task_df[task_df['Trial'].isin(sorted(support_trials))]
    Xs, yas, yvs, _, _ = create_sliding_windows(sup_df, window_size, stride)
    qry_df = task_df[task_df['Trial'].isin(sorted(query_trials))] if query_trials else pd.DataFrame()
    if len(qry_df) > 0:
        Xq, yar, yvr, _, _ = create_sliding_windows(qry_df, window_size, stride)
    else:
        Xq = np.empty((0, window_size, 2)); yar = np.empty((0,)); yvr = np.empty((0,))

    X_sup = torch.tensor(Xs).float().permute(0, 2, 1)
    X_q   = torch.tensor(Xq).float().permute(0, 2, 1)
    y_sup = torch.tensor(yas if ar_or_va=='ar' else yvs).float().unsqueeze(1)
    y_q   = torch.tensor(yar if ar_or_va=='ar' else yvr).float().unsqueeze(1)

    g = torch.Generator(); g.manual_seed(seed)
    sup_loader = DataLoader(TensorDataset(X_sup, y_sup), batch_size=8,
                            shuffle=True, generator=g, num_workers=0)
    q_loader   = DataLoader(TensorDataset(X_q, y_q), batch_size=32,
                            shuffle=False, num_workers=0)
    return sup_loader, q_loader


# =============================
# MODELS
# =============================
def _xavier_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None: nn.init.zeros_(m.bias)


class BaseFeatureExtractor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 128, kernel_size=2); self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2, stride=2, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=1); self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.lstm  = nn.LSTM(64, hidden, batch_first=True)
        self.apply(_xavier_init)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = x.permute(0, 2, 1); out, _ = self.lstm(x)
        return torch.mean(out, dim=1)


class TaskHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(hidden, 128); self.fc2 = nn.Linear(128, 64); self.out = nn.Linear(64, 1)
        self.apply(_xavier_init)

    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.out(x)


# =============================
# INNER LOOP
# =============================
def compute_l2_split(shared_params, task_params, l2_shared=0.0, l2_task=1e-5):
    reg = torch.tensor(0.0)
    if l2_shared > 0:
        for p in shared_params: reg = reg + l2_shared * torch.sum(p**2)
    if l2_task > 0:
        for p in task_params: reg = reg + l2_task * torch.sum(p**2)
    return reg


def adapt_inner_loop(base_model, head, sup_loader, ar_or_va, inner_steps, inner_lr,
                     device, l2_shared=0.0, l2_task=1e-5):
    adapted_base = copy.deepcopy(base_model).to(device)
    adapted_head = copy.deepcopy(head).to(device)
    adapted_base.train(); adapted_head.train()
    sp = list(adapted_base.parameters()); tp = list(adapted_head.parameters())
    opt = optim.Adam(sp + tp, lr=inner_lr)
    sched = ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    for step in range(inner_steps):
        ep_loss = 0.0; nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(adapted_head(adapted_base(Xb)), yb)
            loss = loss + compute_l2_split(sp, tp, l2_shared, l2_task).to(device)
            if not torch.isnan(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(sp+tp, MAX_NORM); opt.step()
            ep_loss += loss.item(); nb += 1
        if nb > 0: sched.step(ep_loss / nb)
    return adapted_base, adapted_head


# =============================
# K-FOLDS
# =============================
def make_kfolds(ids, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation([int(x) for x in ids]).tolist()
    folds, start = [], 0
    for i in range(k):
        size = len(perm)//k + (1 if i < len(perm)%k else 0)
        folds.append(perm[start:start+size]); start += size
    return folds


# =============================
# EVALUATION
# =============================
def evaluate_test_user(base_model, head, test_df, splits, uid, ar_or_va, device,
                       inner_steps, inner_lr, l2_shared=0.0, l2_task=1e-5):
    sup_loader, q_loader = build_support_query(
        test_df, splits[uid]['train'], splits[uid]['test'], ar_or_va)
    if len(q_loader.dataset) == 0: return None
    new_head = TaskHead().to(device)
    adapted_base, adapted_head = adapt_inner_loop(
        base_model, new_head, sup_loader, ar_or_va, inner_steps, inner_lr, device, l2_shared, l2_task)
    adapted_base.eval(); adapted_head.eval()
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in q_loader:
            probs.extend(torch.sigmoid(adapted_head(adapted_base(Xb.to(device)))).cpu().numpy().flatten())
            labels.extend(yb.numpy().flatten())
    y_true = np.array(labels).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)
    return {'participant_id': uid, 'y_true': y_true, 'y_pred': y_pred,
            'y_pred_probs': y_prob, 'cm': cm,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
