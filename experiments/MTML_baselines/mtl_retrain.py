"""
MTL Retrain — Hard Parameter Sharing retrained on all participants.
Pre-tuned on train participants, then retrained on train+test for final evaluation.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

# config MUST be imported first — it sets CUBLAS/PYTHONHASHSEED before torch loads
from config import (HARDCODED_SPLITS, SEED, MAX_NORM, MTL_TASK_LR,
                    EPOCHS, WINDOW_SIZE, STRIDE, N_FOLDS, RESULTS_DIR,
                    L2_SHARED, L2_TASK, TEST_PARTICIPANTS)

import gc
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score

from utils import (set_all_seeds, compute_metrics_from_cm, safe_roc_auc,
                   make_kfolds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from data import create_sliding_windows, BalancedSampler
from models import MTLTransferModel
from dataset_configs.vreed import load_vreed_df

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir       = os.path.join(BASE_OUTPUT_DIR, 'VREED_MTL_retrain')
model_dir        = os.path.join(output_dir, 'models')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir,  exist_ok=True)

learning_rates = [MTL_TASK_LR]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_all_seeds(SEED)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df(mode='mtml')

participant_ids   = sorted([p for p in df['ID'].unique() if p in hardcoded_splits])
test_participants  = list(TEST_PARTICIPANTS)
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {len(train_participants)}  Test: {len(test_participants)}")

# =============================
# LOADER BUILDER
# =============================
def make_combined_loader(tasks_dict, user_list, label_type, split='train'):
    all_X, all_y, all_tids, all_vids = [], [], [], []
    local_map, spt = {}, {}
    for lt, uid in enumerate(sorted(user_list)):
        d = tasks_dict[uid]
        X, y_ar, y_va, tids, vids = create_sliding_windows(d, WINDOW_SIZE, STRIDE, task_id=lt)
        if X.shape[0] == 0:
            continue
        y = y_ar if label_type == 'ar' else y_va
        all_X.append(X); all_y.append(y)
        all_tids.append(tids); all_vids.append(vids)
        local_map[lt] = uid; spt[lt] = X.shape[0]
    if not all_X:
        from torch.utils.data import TensorDataset, DataLoader
        return (DataLoader(TensorDataset(torch.empty(0, WINDOW_SIZE, 2),
                                         torch.empty(0, 1),
                                         torch.empty(0, dtype=torch.long),
                                         torch.empty(0, dtype=torch.long)),
                           batch_size=1),
                0, local_map)
    from torch.utils.data import TensorDataset, DataLoader
    X_arr = np.concatenate(all_X); y_arr = np.concatenate(all_y)
    tids  = np.concatenate(all_tids); vids  = np.concatenate(all_vids)
    X_t   = torch.tensor(X_arr, dtype=torch.float32)
    y_t   = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t, torch.tensor(tids), torch.tensor(vids))
    sampler = BalancedSampler(tids, list(local_map.keys()), spt, seed=SEED)
    loader  = DataLoader(dataset, batch_size=len(local_map), sampler=sampler, num_workers=0)
    print(f"[{split}/{label_type}] users={len(local_map)} samples={len(dataset)}")
    return loader, len(dataset), local_map

# =============================
# TRAINING HELPERS
# =============================
def _train_fold(model, loader, lr, epochs):
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    for ep in range(epochs):
        model.train(); run = 0.0
        for Xb, yb, tids, _ in loader:
            Xb, yb, tids = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), tids.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = (loss_fn(model(Xb, tids), yb).squeeze(-1).mean()
                    + model.compute_l2(L2_SHARED, L2_TASK))
            if torch.isnan(loss):
                raise ValueError(f"NaN in _train_fold [ep {ep+1}]")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
        sched.step(run / max(1, len(loader)))
    return model

def _eval_fold(model, loader):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for Xb, yb, tids, _ in loader:
            Xb, tids = Xb.to(device, non_blocking=True), tids.to(device, non_blocking=True)
            pr = (torch.sigmoid(model(Xb, tids)) > 0.5).int().cpu().numpy().flatten()
            preds.extend(pr)
            labels.extend(yb.int().numpy().flatten())
    return f1_score(labels, preds, average='macro', zero_division=0)

def train_final(loader, lr, label_type, local_map):
    """Train the final model on all participants (train + test) for evaluation."""
    set_all_seeds(SEED)
    model   = MTLTransferModel(len(local_map)).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_loss  = float('inf')
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train(); run = 0.0; preds, labels_list = [], []
        for Xb, yb, tids, _ in loader:
            Xb, yb, tids = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), tids.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = (loss_fn(model(Xb, tids), yb).squeeze(-1).mean()
                    + model.compute_l2(L2_SHARED, L2_TASK))
            if torch.isnan(loss):
                raise ValueError(f"NaN epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
            with torch.no_grad():
                preds.extend((torch.sigmoid(model(Xb, tids)) > 0.5)
                              .int().cpu().numpy().flatten())
                labels_list.extend(yb.int().cpu().numpy().flatten())
        avg = run / max(1, len(loader))
        if ep % 5 == 0 or ep == 1:
            f1 = f1_score(labels_list, preds, average='macro', zero_division=0)
            print(f"  [{label_type.upper()}] Epoch {ep}/{EPOCHS} loss={avg:.4f} f1={f1:.4f}")
        sched.step(avg)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model

# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type='ar'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] MTL-Retrain\n{'='*60}")
    results     = []
    train_folds = make_kfolds(train_participants)
    for lr in learning_rates:
        fold_f1s = []
        for fold_i in range(N_FOLDS):
            val_ps = train_folds[fold_i]
            tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
            tr_tasks = {
                uid: df[(df['ID'] == uid) &
                        (df['Trial'].isin(hardcoded_splits[uid]['train']))].reset_index(drop=True)
                for uid in tr_ps}
            va_tasks = {
                uid: df[(df['ID'] == uid) &
                        (df['Trial'].isin(hardcoded_splits[uid]['train']))].reset_index(drop=True)
                for uid in val_ps}
            tr_loader, _, tr_map = make_combined_loader(tr_tasks, tr_ps, label_type, 'train')
            va_loader, _, va_map = make_combined_loader(va_tasks, val_ps, label_type, 'val')
            if not tr_map or not va_map:
                continue
            model = MTLTransferModel(len(tr_map)).to(device)
            model = _train_fold(model, tr_loader, lr, EPOCHS)
            if model is None:
                continue
            val_f1 = _eval_fold(model, va_loader)
            fold_f1s.append(val_f1)
            print(f"  fold {fold_i+1}: f1={val_f1:.4f}")
            del model; torch.cuda.empty_cache(); gc.collect()
        if not fold_f1s:
            continue
        avg = np.mean(fold_f1s)
        results.append({'lr': lr, 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
        print(f"  avg={avg:.4f}")
    if not results:
        return 1e-4
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type}_tuning_results_mtl.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr']

# =============================
# LOCAL INDEX LOOKUP
# =============================
def _get_local_idx(uid, local_map):
    for li, ru in local_map.items():
        if ru == uid:
            return li
    return None

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    experiment_t0 = time.time()

    best_lr_ar = hyperparameter_tuning('ar')
    best_lr_va = hyperparameter_tuning('va')

    all_users   = sorted(train_participants + test_participants)
    user_frames = {
        uid: {
            'train': df[(df['ID'] == uid) &
                        (df['Trial'].isin(hardcoded_splits[uid]['train']))].reset_index(drop=True),
            'test':  df[(df['ID'] == uid) &
                        (df['Trial'].isin(hardcoded_splits[uid]['test']))].reset_index(drop=True),
        }
        for uid in all_users
    }
    retrain_tasks = {uid: user_frames[uid]['train'] for uid in all_users}

    tr_loader_ar, _, ar_map = make_combined_loader(retrain_tasks, all_users, 'ar', 'FINAL-ar')
    tr_loader_va, _, va_map = make_combined_loader(retrain_tasks, all_users, 'va', 'FINAL-va')

    print('\n' + '='*60 + '\nTRAINING FINAL AR\n' + '='*60)
    train_t0 = time.time()
    model_ar = train_final(tr_loader_ar, best_lr_ar, 'ar', ar_map)
    print(f"  AR training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_ar.state_dict(), os.path.join(model_dir, 'mtl_final_best_ar.pt'))

    print('\n' + '='*60 + '\nTRAINING FINAL VA\n' + '='*60)
    train_t0 = time.time()
    model_va = train_final(tr_loader_va, best_lr_va, 'va', va_map)
    print(f"  VA training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_va.state_dict(), os.path.join(model_dir, 'mtl_final_best_va.pt'))

    model_ar.eval(); model_va.eval()

    results_ar, results_va = [], []
    with torch.no_grad():
        for uid in sorted(test_participants):
            X, y_ar, y_va, _, _ = create_sliding_windows(
                user_frames[uid]['test'], WINDOW_SIZE, STRIDE, task_id=0)
            if X.shape[0] == 0:
                continue
            X_t   = torch.tensor(X, dtype=torch.float32).to(device, non_blocking=True)
            li_ar = _get_local_idx(uid, ar_map)
            li_va = _get_local_idx(uid, va_map)
            if li_ar is None or li_va is None:
                continue
            tids_ar = torch.full((X_t.size(0),), li_ar, dtype=torch.long, device=device)
            tids_va = torch.full((X_t.size(0),), li_va, dtype=torch.long, device=device)

            for model, tids, y_true, results_list, label in [
                (model_ar, tids_ar, y_ar, results_ar, 'ar'),
                (model_va, tids_va, y_va, results_va, 'va'),
            ]:
                probs = torch.sigmoid(model(X_t, tids)).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                y_int = y_true.astype(int)
                cm    = confusion_matrix(y_int, preds, labels=[0, 1])
                acc, prec, rec, f1 = compute_metrics_from_cm(cm)
                p = label
                results_list.append({
                    'participant_id':       uid,
                    'cm':                   cm,
                    f'{p}_acc':             acc,
                    f'{p}_precision':       prec,
                    f'{p}_recall':          rec,
                    f'{p}_f1':              f1,
                    f'y_true_{p}':          y_int,
                    f'y_pred_{p}':          preds,
                    f'y_pred_probs_{p}':    probs,
                })
            print(f"  Participant {uid}: "
                  f"AR acc={results_ar[-1]['ar_acc']:.4f} | "
                  f"VA acc={results_va[-1]['va_acc']:.4f}")

    agg = aggregate_mtml_results(results_ar, results_va)

    roc_data = {
        'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
        'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']},
    }
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')

    final_results = {
        'train_participants': train_participants,
        'test_participants':  test_participants,
        'best_hyperparameters': {
            'AR': {'lr': best_lr_ar, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK},
            'VA': {'lr': best_lr_va, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc', 'precision', 'recall', 'f1', 'auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc', 'precision', 'recall', 'f1', 'auc']},
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'mtl_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    print_determinism_summary(
        {f'ar_{k}': final_results[f'ar_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        {f'va_{k}': final_results[f'va_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
