"""
MTL Retrain — Hard Parameter Sharing retrained on all participants.
Pre-tuned on train participants, then retrained on train+test for final evaluation.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

import gc
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Sampler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

from config import HARDCODED_SPLITS, SEED, MAX_NORM, RETRAIN_LR, L2_SHARED, L2_TASK
from config import EPOCHS, WINDOW_SIZE, STRIDE, N_FOLDS, RESULTS_DIR
from utils import set_all_seeds, compute_metrics_from_cm, safe_roc_auc, make_kfolds
from data import create_sliding_windows, BalancedSampler
from dataset_configs.vreed import load_vreed_df
from models import BaseFeatureExtractor

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir = os.path.join(BASE_OUTPUT_DIR, 'VREED_MTL_retrain')
model_dir  = os.path.join(output_dir, 'models')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

learning_rates = [RETRAIN_LR]

set_all_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df(mode='mtml')

participant_ids = sorted([p for p in df['ID'].unique() if p in hardcoded_splits])
test_participants  = [105, 109, 112, 125, 131, 132]
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {len(train_participants)}  Test: {len(test_participants)}")


def make_combined_loader(tasks_dict, user_list, label_type, split='train'):
    all_X, all_y, all_tids, all_vids = [], [], [], []
    local_map, spt = {}, {}
    for lt, uid in enumerate(sorted(user_list)):
        d = tasks_dict[uid]
        X, y_ar, y_va, tids, vids = create_sliding_windows(d, WINDOW_SIZE, STRIDE, task_id=lt)
        if X.shape[0] == 0: continue
        y = y_ar if label_type == 'ar' else y_va
        all_X.append(X); all_y.append(y); all_tids.append(tids); all_vids.append(vids)
        local_map[lt] = uid; spt[lt] = X.shape[0]
    if not all_X:
        return DataLoader(TensorDataset(torch.empty(0,WINDOW_SIZE,2),
                                        torch.empty(0,1),torch.empty(0,dtype=torch.long),
                                        torch.empty(0,dtype=torch.long)), batch_size=1), 0, local_map
    X = np.concatenate(all_X); y = np.concatenate(all_y)
    tids = np.concatenate(all_tids); vids = np.concatenate(all_vids)
    X_t = torch.tensor(X); y_t = torch.tensor(y).unsqueeze(1)  # (N, window, channels) — model permutes internally
    dataset = TensorDataset(X_t, y_t, torch.tensor(tids), torch.tensor(vids))
    sampler = BalancedSampler(tids, list(local_map.keys()), spt, seed=SEED)
    loader = DataLoader(dataset, batch_size=len(local_map), sampler=sampler, num_workers=0)
    print(f"[{split}/{label_type}] users={len(local_map)} samples={len(dataset)}")
    return loader, len(dataset), local_map


# =============================
# MODELS
# =============================
class SharedBackbone(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 128, 2); self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2, 2, 1)
        self.conv2 = nn.Conv1d(128, 64, 1); self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.lstm  = nn.LSTM(64, hidden, batch_first=True)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, window, channels) → (batch, channels, window)
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = x.permute(0,2,1); out, _ = self.lstm(x)
        return torch.mean(out, dim=1)


class MultiTaskModel(nn.Module):
    def __init__(self, num_tasks, hidden=64):
        super().__init__()
        self.backbone = SharedBackbone(hidden)
        self.dense1 = nn.ModuleList([nn.Sequential(nn.Linear(hidden,128),nn.ReLU()) for _ in range(num_tasks)])
        self.dense2 = nn.ModuleList([nn.Sequential(nn.Linear(128,64),nn.ReLU())     for _ in range(num_tasks)])
        self.out    = nn.ModuleList([nn.Linear(64,1)                                  for _ in range(num_tasks)])
        self.num_tasks = num_tasks
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, task_ids):
        feats = self.backbone(x)
        out = torch.zeros(feats.size(0), 1, device=x.device)
        for t in torch.unique(task_ids):
            mask = (task_ids == t); ti = int(t.item())
            if ti < 0 or ti >= self.num_tasks or feats[mask].size(0) == 0: continue
            h = self.dense1[ti](feats[mask])
            h = self.dense2[ti](h)
            out[mask] = self.out[ti](h)
        return out


def compute_l2(model):
    l2s = L2_SHARED * sum(p.norm(2)**2 for p in model.backbone.parameters() if p.requires_grad)
    l2t = L2_TASK * (sum(p.norm(2)**2 for m in model.dense1 for p in m.parameters() if p.requires_grad) +
                     sum(p.norm(2)**2 for m in model.dense2 for p in m.parameters() if p.requires_grad) +
                     sum(p.norm(2)**2 for m in model.out     for p in m.parameters() if p.requires_grad))
    return l2s + l2t


# =============================
# HYPERPARAMETER TUNING
# =============================
def make_folds(user_ids, k=5):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(user_ids).tolist()
    folds, start = [], 0
    for i in range(k):
        size = len(perm)//k + (1 if i < len(perm)%k else 0)
        folds.append(sorted(perm[start:start+size])); start += size
    return folds


def train_fold(model, loader, lr, epochs):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    for ep in range(epochs):
        model.train(); run = 0.0
        for Xb, yb, tids, _ in loader:
            Xb, yb, tids = Xb.to(device), yb.to(device), tids.to(device)
            opt.zero_grad()
            loss = loss_fn(model(Xb, tids), yb).squeeze(-1).mean() + compute_l2(model)
            if torch.isnan(loss): return None
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
        sched.step(run / max(1, len(loader)))
    return model


def eval_fold(model, loader, local_map):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for Xb, yb, tids, _ in loader:
            Xb, yb, tids = Xb.to(device), yb.to(device), tids.to(device)
            pr = (torch.sigmoid(model(Xb, tids)) > 0.5).int().cpu().numpy().flatten()
            preds.extend(pr); labels.extend(yb.int().cpu().numpy().flatten())
    return f1_score(labels, preds, average='macro', zero_division=0)


def hyperparameter_tuning(label_type='ar'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] MTL-Retrain\n{'='*60}")
    results = []; train_folds = make_folds(train_participants)
    for lr in learning_rates:
        fold_f1s = []
        for fold_i in range(N_FOLDS):
            val_ps = train_folds[fold_i]
            tr_ps  = [p for j,f in enumerate(train_folds) if j != fold_i for p in f]
            tr_tasks = {uid: df[df['ID']==uid][df['Trial'].isin(hardcoded_splits[uid]['train'])].reset_index(drop=True) for uid in tr_ps}
            va_tasks = {uid: df[df['ID']==uid][df['Trial'].isin(hardcoded_splits[uid]['train'])].reset_index(drop=True) for uid in val_ps}
            tr_loader, _, tr_map = make_combined_loader(tr_tasks, tr_ps, label_type, 'train')
            va_loader, _, va_map = make_combined_loader(va_tasks, val_ps, label_type, 'val')
            if not tr_map or not va_map: continue
            model = MultiTaskModel(len(tr_map)).to(device)
            model = train_fold(model, tr_loader, lr, EPOCHS)
            if model is None: continue
            val_f1 = eval_fold(model, va_loader, va_map)
            fold_f1s.append(val_f1)
            print(f"  fold {fold_i+1}: f1={val_f1:.4f}")
            del model; torch.cuda.empty_cache(); gc.collect()
        if not fold_f1s: continue
        avg = np.mean(fold_f1s)
        results.append({'lr': lr, 'avg_f1': avg, 'std_f1': np.std(fold_f1s)}); print(f"  avg={avg:.4f}")
    if not results: return 1e-4
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type}_tuning_results_mtl.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr']


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    best_lr_ar = hyperparameter_tuning('ar')
    best_lr_va = hyperparameter_tuning('va')

    all_users = sorted(train_participants + test_participants)
    user_frames = {uid: {'train': df[df['ID']==uid][df['Trial'].isin(hardcoded_splits[uid]['train'])].reset_index(drop=True),
                         'test':  df[df['ID']==uid][df['Trial'].isin(hardcoded_splits[uid]['test'])].reset_index(drop=True)}
                   for uid in all_users}
    retrain_tasks = {uid: user_frames[uid]['train'] for uid in all_users}

    tr_loader_ar, _, ar_map = make_combined_loader(retrain_tasks, all_users, 'ar', 'FINAL-ar')
    tr_loader_va, _, va_map = make_combined_loader(retrain_tasks, all_users, 'va', 'FINAL-va')

    def train_final(loader, lr, label_type, local_map):
        set_all_seeds(SEED)
        model = MultiTaskModel(len(local_map)).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        best_loss = float('inf'); best_state = None
        for ep in range(1, EPOCHS+1):
            model.train(); run = 0.0; preds, labels = [], []
            for Xb, yb, tids, _ in loader:
                Xb, yb, tids = Xb.to(device), yb.to(device), tids.to(device)
                opt.zero_grad()
                loss = loss_fn(model(Xb, tids), yb).squeeze(-1).mean() + compute_l2(model)
                if torch.isnan(loss): raise ValueError(f"NaN epoch {ep}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                opt.step(); run += loss.item()
                with torch.no_grad():
                    preds.extend((torch.sigmoid(model(Xb, tids))>0.5).int().cpu().numpy().flatten())
                    labels.extend(yb.int().cpu().numpy().flatten())
            avg = run / max(1, len(loader))
            if ep % 5 == 0 or ep == 1:
                f1 = f1_score(labels, preds, average='macro', zero_division=0)
                print(f"  [{label_type.upper()}] Epoch {ep}/{EPOCHS} loss={avg:.4f} f1={f1:.4f}")
            sched.step(avg)
            if avg < best_loss: best_loss = avg; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state); return model

    print('\n' + '='*60 + '\nTRAINING FINAL AR\n' + '='*60)
    model_ar = train_final(tr_loader_ar, best_lr_ar, 'ar', ar_map)
    torch.save(model_ar.state_dict(), os.path.join(model_dir, 'mtl_final_best_ar.pt'))

    print('\n' + '='*60 + '\nTRAINING FINAL VA\n' + '='*60)
    model_va = train_final(tr_loader_va, best_lr_va, 'va', va_map)
    torch.save(model_va.state_dict(), os.path.join(model_dir, 'mtl_final_best_va.pt'))

    model_ar.eval(); model_va.eval()

    def get_local_idx(uid, local_map):
        for li, ru in local_map.items():
            if ru == uid: return li
        return None

    results_ar, results_va = [], []
    with torch.no_grad():
        for uid in sorted(test_participants):
            X, y_ar, y_va, _, _ = create_sliding_windows(user_frames[uid]['test'], WINDOW_SIZE, STRIDE, task_id=0)
            if X.shape[0] == 0: continue
            X_t = torch.tensor(X).to(device)  # (N, window, channels) — model permutes internally
            li_ar = get_local_idx(uid, ar_map); li_va = get_local_idx(uid, va_map)
            if li_ar is None or li_va is None: continue
            tids_ar = torch.full((X_t.size(0),), li_ar, dtype=torch.long, device=device)
            tids_va = torch.full((X_t.size(0),), li_va, dtype=torch.long, device=device)
            for model, tids, y_true, results, label in [
                (model_ar, tids_ar, y_ar, results_ar, 'ar'),
                (model_va, tids_va, y_va, results_va, 'va')]:
                probs = torch.sigmoid(model(X_t, tids)).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int); y_int = y_true.astype(int)
                cm = confusion_matrix(y_int, preds, labels=[0,1])
                acc, prec, rec, f1 = compute_metrics_from_cm(cm)
                results.append({'participant_id': uid, 'y_true': y_int, 'y_pred': preds,
                                 'y_pred_probs': probs, 'cm': cm,
                                 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
            print(f"  Participant {uid}: AR acc={results_ar[-1]['accuracy']:.4f} | VA acc={results_va[-1]['accuracy']:.4f}")

    def aggregate(results, label):
        all_true  = np.concatenate([r['y_true'] for r in results])
        all_pred  = np.concatenate([r['y_pred'] for r in results])
        all_probs = np.concatenate([r['y_pred_probs'] for r in results])
        cm = confusion_matrix(all_true, all_pred, labels=[0,1])
        acc, prec, rec, f1 = compute_metrics_from_cm(cm)
        auc_val, fpr, tpr = safe_roc_auc(all_true, all_probs)
        print(f"\n{label}: Acc={acc:.4f} F1={f1:.4f} AUC={auc_val:.4f}")
        return all_true, all_probs, cm, acc, prec, rec, f1, auc_val, fpr, tpr

    all_true_ar, all_probs_ar, cm_AR, ar_acc, ar_prec, ar_rec, ar_f1, ar_auc, ar_fpr, ar_tpr = aggregate(results_ar, 'AR')
    all_true_va, all_probs_va, cm_VA, va_acc, va_prec, va_rec, va_f1, va_auc, va_fpr, va_tpr = aggregate(results_va, 'VA')

    roc_data = {'AR': {'true': all_true_ar, 'probs': all_probs_ar},
                'VA': {'true': all_true_va, 'probs': all_probs_va}}
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    final_results = {
        'train_participants': train_participants, 'test_participants': test_participants,
        'best_hyperparameters': {'AR': {'lr': best_lr_ar, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK},
                                  'VA': {'lr': best_lr_va, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK}},
        'ar_acc': ar_acc, 'ar_precision': ar_prec, 'ar_recall': ar_rec, 'ar_f1': ar_f1, 'ar_auc': ar_auc,
        'va_acc': va_acc, 'va_precision': va_prec, 'va_recall': va_rec, 'va_f1': va_f1, 'va_auc': va_auc,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': cm_AR, 'cm_va': cm_VA,
    }
    with open(os.path.join(output_dir, 'mtl_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    # =============================
    # DETERMINISM SUMMARY
    # =============================
    from utils import compute_per_participant_stds, print_determinism_summary

    def _prefix(results, prefix):
        return [{f"{prefix}_acc": r["accuracy"], f"{prefix}_precision": r["precision"],
                 f"{prefix}_recall": r["recall"], f"{prefix}_f1": r["f1"],
                 f"y_true_{prefix}": r["y_true"], f"y_pred_probs_{prefix}": r["y_pred_probs"]}
                for r in results]

    ar_stds = compute_per_participant_stds(_prefix(results_ar, "ar"), "ar")
    va_stds = compute_per_participant_stds(_prefix(results_va, "va"), "va")
    print_determinism_summary(
        {f"ar_{k}": final_results[f"ar_{k}"] for k in ["auc", "acc", "precision", "recall", "f1"]},
        {f"va_{k}": final_results[f"va_{k}"] for k in ["auc", "acc", "precision", "recall", "f1"]},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")