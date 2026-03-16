"""
Transfer-MTL — MTL pretrain on train participants, add new head per test participant, fine-tune.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

# config MUST be imported first — it sets CUBLAS/PYTHONHASHSEED before torch loads
from config import (HARDCODED_SPLITS, SEED, MAX_NORM,
                    TRANSFER_MTL_LR_PT, TRANSFER_MTL_LR_FT,
                    L2_SHARED, L2_TASK, EPOCHS, FT_EPOCHS, FT_BATCH_SIZE,
                    WINDOW_SIZE, STRIDE, N_FOLDS, TEST_PARTICIPANTS, RESULTS_DIR)

import gc
import copy
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import TensorDataset, DataLoader

from utils import (set_all_seeds, compute_metrics_from_cm, safe_roc_auc,
                   make_kfolds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary,
                   prefix_results)
from data import create_sliding_windows, BalancedSampler
from models import MTLTransferModel
from dataset_configs.vreed import load_vreed_df

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir       = os.path.join(BASE_OUTPUT_DIR, 'VREED_TransferMTL')
model_dir        = os.path.join(output_dir, 'models')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir,  exist_ok=True)

learning_rates_pt = [TRANSFER_MTL_LR_PT]
learning_rates_ft = [TRANSFER_MTL_LR_FT]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\nOutput: {output_dir}")
set_all_seeds(SEED)

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
        X, y_ar, y_va, tids, vids = create_sliding_windows(
            tasks_dict[uid], WINDOW_SIZE, STRIDE, lt)
        if X.shape[0] == 0:
            continue
        y = y_ar if label_type == 'ar' else y_va
        all_X.append(X); all_y.append(y)
        all_tids.append(tids); all_vids.append(vids)
        local_map[lt] = uid; spt[lt] = X.shape[0]
    if not all_X:
        return (DataLoader(TensorDataset(torch.empty(0, WINDOW_SIZE, 2, dtype=torch.float32),
                                         torch.empty(0, 1, dtype=torch.float32),
                                         torch.empty(0, dtype=torch.long),
                                         torch.empty(0, dtype=torch.long)),
                           batch_size=1),
                0, local_map)
    X_arr = np.concatenate(all_X); y_arr = np.concatenate(all_y)
    tids  = np.concatenate(all_tids); vids  = np.concatenate(all_vids)
    X_t   = torch.tensor(X_arr, dtype=torch.float32)
    y_t   = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t, torch.tensor(tids), torch.tensor(vids))
    sampler = BalancedSampler(tids, list(local_map.keys()), spt, SEED)
    loader  = DataLoader(dataset, batch_size=len(local_map), sampler=sampler, num_workers=0)
    print(f"[{split}/{label_type}] users={len(local_map)} samples={len(dataset)}")
    return loader, len(dataset), local_map


# =============================
# TRAINING HELPERS
# =============================
def pretrain_mtl(loader, local_map, lr, label_type):
    model   = MTLTransferModel(len(local_map)).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_loss  = float('inf')
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train(); run = 0.0
        for Xb, yb, tids, _ in loader:
            Xb, yb, tids = Xb.to(device), yb.to(device), tids.to(device)
            opt.zero_grad()
            loss = (loss_fn(model(Xb, tids), yb).squeeze(-1).mean()
                    + model.compute_l2(L2_SHARED, L2_TASK))
            if torch.isnan(loss):
                raise ValueError(f"NaN epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
        avg = run / max(1, len(loader))
        if ep % 5 == 0 or ep == 1:
            print(f"  [{label_type.upper()}-PT] Epoch {ep} loss={avg:.4f}")
        sched.step(avg)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def finetune_user(base_model, X, y, lr, pid):
    """
    Deep-copy the pre-trained model, add a new task head for this participant,
    and fine-tune all parameters on the participant's training data.

    Returns (fine-tuned model, local head index) or (None, None) on NaN.
    """
    model     = copy.deepcopy(base_model).to(device)
    local_idx = model.add_task_head()   # appends head, returns its index

    g = torch.Generator()
    g.manual_seed(SEED + pid)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32).unsqueeze(1)),
        batch_size=FT_BATCH_SIZE, shuffle=True, generator=g, num_workers=0)

    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(FT_EPOCHS):
        model.train(); run = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            tids   = torch.full((Xb.size(0),), local_idx, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = loss_fn(model(Xb, tids), yb)
            if torch.isnan(loss):
                return None, None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
        sched.step(run / max(1, len(loader)))

    return model, local_idx


def eval_user(model, local_idx, X, y):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32).unsqueeze(1)),
        batch_size=FT_BATCH_SIZE, shuffle=False, num_workers=0)
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            tids_b = torch.full((Xb.size(0),), local_idx, dtype=torch.long, device=device)
            probs.extend(torch.sigmoid(model(Xb.to(device), tids_b))
                         .cpu().numpy().flatten())
            labels.extend(yb.numpy().flatten())
    y_true = np.array(labels).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)
    return {'y_true': y_true, 'y_pred': y_pred, 'y_pred_probs': y_prob, 'cm': cm,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type='ar'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Transfer-MTL\n{'='*60}")
    results     = []
    train_folds = make_kfolds(train_participants)
    for lr_pt in learning_rates_pt:
        for lr_ft in learning_rates_ft:
            fold_f1s = []
            for fold_i in range(N_FOLDS):
                val_ps = train_folds[fold_i]
                tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
                tr_tasks = {
                    uid: df[(df['ID'] == uid) &
                            (df['Trial'].isin(hardcoded_splits[uid]['train']))].reset_index(drop=True)
                    for uid in tr_ps}
                loader, _, lmap = make_combined_loader(tr_tasks, tr_ps, label_type, 'train')
                if not lmap:
                    continue
                try:
                    base = pretrain_mtl(loader, lmap, lr_pt, label_type)
                except Exception:
                    continue
                val_f1s = []
                for uid in val_ps:
                    u_df = df[df['ID'] == uid].reset_index(drop=True)
                    Xft, y_ar, y_va, _, _ = create_sliding_windows(
                        u_df[u_df['Trial'].isin(hardcoded_splits[uid]['train'])],
                        WINDOW_SIZE, STRIDE)
                    Xte, yar_te, yva_te, _, _ = create_sliding_windows(
                        u_df[u_df['Trial'].isin(hardcoded_splits[uid]['test'])],
                        WINDOW_SIZE, STRIDE)
                    yft = y_ar if label_type == 'ar' else y_va
                    yte = yar_te if label_type == 'ar' else yva_te
                    if len(Xft) == 0 or len(Xte) == 0:
                        continue
                    ft_model, li = finetune_user(base, Xft, yft, lr_ft, uid)
                    if ft_model is None:
                        continue
                    r = eval_user(ft_model, li, Xte, yte)
                    val_f1s.append(r['f1'])
                if val_f1s:
                    fold_f1s.append(np.mean(val_f1s))
                    print(f"  fold {fold_i+1}: f1={fold_f1s[-1]:.4f}")
                del base; torch.cuda.empty_cache(); gc.collect()
            if not fold_f1s:
                continue
            avg = np.mean(fold_f1s)
            results.append({'lr_pt': lr_pt, 'lr_ft': lr_ft,
                             'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
            print(f"  avg f1={avg:.4f}")
    if not results:
        return 1e-4, 5e-5
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type}_tuning_results_transfermtl.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr_pt'], best['lr_ft']


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    best_lr_pt_ar, best_lr_ft_ar = hyperparameter_tuning('ar')
    best_lr_pt_va, best_lr_ft_va = hyperparameter_tuning('va')

    tr_tasks = {
        uid: df[(df['ID'] == uid) &
                (df['Trial'].isin(hardcoded_splits[uid]['train']))].reset_index(drop=True)
        for uid in train_participants}

    print('\n' + '='*60 + '\nPRETRAINING FINAL AR MTL\n' + '='*60)
    loader_ar, _, map_ar = make_combined_loader(tr_tasks, train_participants, 'ar', 'FINAL-ar')
    set_all_seeds(SEED)
    base_ar = pretrain_mtl(loader_ar, map_ar, best_lr_pt_ar, 'ar')
    torch.save(base_ar.state_dict(), os.path.join(model_dir, 'base_model_ar_final.pth'))

    print('\n' + '='*60 + '\nPRETRAINING FINAL VA MTL\n' + '='*60)
    loader_va, _, map_va = make_combined_loader(tr_tasks, train_participants, 'va', 'FINAL-va')
    set_all_seeds(SEED)
    base_va = pretrain_mtl(loader_va, map_va, best_lr_pt_va, 'va')
    torch.save(base_va.state_dict(), os.path.join(model_dir, 'base_model_va_final.pth'))

    results_ar, results_va = [], []
    for pid in sorted(test_participants):
        u_df = df[df['ID'] == pid].reset_index(drop=True)
        Xft, y_ar_ft, y_va_ft, _, _ = create_sliding_windows(
            u_df[u_df['Trial'].isin(hardcoded_splits[pid]['train'])], WINDOW_SIZE, STRIDE)
        Xte, y_ar_te, y_va_te, _, _ = create_sliding_windows(
            u_df[u_df['Trial'].isin(hardcoded_splits[pid]['test'])],  WINDOW_SIZE, STRIDE)

        print(f"\nParticipant {pid}: fine-tuning")
        for base, lr_ft, y_ft, y_te, results, label in [
            (base_ar, best_lr_ft_ar, y_ar_ft, y_ar_te, results_ar, 'ar'),
            (base_va, best_lr_ft_va, y_va_ft, y_va_te, results_va, 'va'),
        ]:
            ft_model, li = finetune_user(base, Xft, y_ft, lr_ft, pid)
            if ft_model is None:
                continue
            r = eval_user(ft_model, li, Xte, y_te)
            r['participant_id'] = pid
            results.append(r)
            print(f"  {label.upper()}: Acc={r['accuracy']:.4f} F1={r['f1']:.4f}")

    agg = aggregate_mtml_results(results_ar, results_va)

    for cm, label, fname in [(agg['cm_ar'], 'AR', 'ar_cm_transfermtl.png'),
                              (agg['cm_va'], 'VA', 'va_cm_transfermtl.png')]:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'{label}=0', f'{label}=1'],
                    yticklabels=[f'{label}=0', f'{label}=1'])
        plt.title(f'CM {label} (Transfer-MTL)')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, fname)); plt.close()

    roc_data = {
        'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
        'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']},
    }
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    final_results = {
        'train_participants': train_participants,
        'test_participants':  test_participants,
        'best_hyperparameters': {
            'AR': {'lr_pt': best_lr_pt_ar, 'lr_ft': best_lr_ft_ar},
            'VA': {'lr_pt': best_lr_pt_va, 'lr_ft': best_lr_ft_va}},
        'ar_acc': agg['ar_acc'], 'ar_precision': agg['ar_precision'],
        'ar_recall': agg['ar_recall'], 'ar_f1': agg['ar_f1'], 'ar_auc': agg['ar_auc'],
        'va_acc': agg['va_acc'], 'va_precision': agg['va_precision'],
        'va_recall': agg['va_recall'], 'va_f1': agg['va_f1'], 'va_auc': agg['va_auc'],
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'transfermtl_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    ar_stds = compute_per_participant_stds(prefix_results(results_ar, 'ar'), 'ar')
    va_stds = compute_per_participant_stds(prefix_results(results_va, 'va'), 'va')
    print_determinism_summary(
        {f'ar_{k}': final_results[f'ar_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        {f'va_{k}': final_results[f'va_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
