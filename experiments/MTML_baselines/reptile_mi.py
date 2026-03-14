"""
Reptile Meta-MTL — MI-Guided Multi-Task Episodes
Uses mutual information between participant physiological-affective signatures
to select episodes with a mix of similar and diverse tasks.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

from collections import OrderedDict
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, mutual_info_score

from config import HARDCODED_SPLITS, SEED
from utils import set_all_seeds, compute_metrics_from_cm, safe_roc_auc
from models import BaseFeatureExtractor, TaskHead
from training import adapt_inner_loop, evaluate_test_user
from data import build_support_query
from paths import CSV_PATH, RESULTS_DIR

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir = os.path.join(BASE_OUTPUT_DIR, 'VREED_ReptileMeta_MI_episode')
os.makedirs(output_dir, exist_ok=True)

WINDOW_SIZE = 2560; STRIDE = 1280
L2_SHARED = 0.0; L2_TASK = 1e-5
META_STEPS = 50; META_LR = 0.01
INNER_STEPS = 10; INNER_LR = 1e-3
EPISODE_SIZE = 5

set_all_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = pd.read_csv(CSV_PATH)
df = df.drop(columns=['ECG','GSR','Unnamed: 0.1','Unnamed: 0','Trial'], errors='ignore')
df = df.rename(columns={'ECG_scaled':'ECG','GSR_scaled':'GSR','Num_Code':'video'})
df['Trial'] = df['video']
df = df.sort_values(['ID','Trial']).reset_index(drop=True)

participant_ids   = sorted([p for p in df['ID'].unique() if p in hardcoded_splits])
test_participants  = [105,109,112,125,131,132]
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {len(train_participants)}  Test: {len(test_participants)}")


# =============================
# MI SIGNATURE AND MATRIX
# =============================
def _digitize(x, n_bins=16):
    x = np.asarray(x).reshape(-1)
    if len(x) == 0: return np.zeros(1, dtype=int)
    lo, hi = x.min(), x.max()
    if np.isclose(lo, hi): return np.zeros_like(x, dtype=int)
    bins = np.linspace(lo, hi, n_bins + 1)
    return np.digitize(x, bins[1:-1], right=False).astype(int)


def _task_signature(task_df, label_type, splits, uid, max_pts=20000):
    train_df = task_df[task_df['Trial'].isin(splits[uid]['train'])]
    if len(train_df) > max_pts:
        idx = np.linspace(0, len(train_df)-1, max_pts).astype(int)
        train_df = train_df.iloc[idx]
    ecg = _digitize(train_df['ECG'].values)
    gsr = _digitize(train_df['GSR'].values)
    y   = train_df['AR_Rating' if label_type=='ar' else 'VA_Rating'].astype(int).values
    return np.stack([ecg, gsr, y], axis=1).astype(int)


def build_mi_matrix(tasks_data, label_type, splits):
    uids = sorted(tasks_data.keys())
    sigs = {uid: _task_signature(tasks_data[uid], label_type, splits, uid) for uid in uids}
    mi = {uid: {} for uid in uids}
    for i, ui in enumerate(uids):
        for j, uj in enumerate(uids):
            if j < i:
                mi[ui][uj] = mi[uj][ui]; continue
            m = min(len(sigs[ui]), len(sigs[uj]))
            if m == 0:
                mi[ui][uj] = 0.0; continue
            v = (mutual_info_score(sigs[ui][:m,0], sigs[uj][:m,0]) +
                 mutual_info_score(sigs[ui][:m,1], sigs[uj][:m,1]) +
                 mutual_info_score(sigs[ui][:m,2], sigs[uj][:m,2])) / 3.0
            mi[ui][uj] = float(v); mi[uj][ui] = float(v)
    return mi


def mi_guided_episode(task_ids, mi_matrix, rng, size=5, n_sim=2, n_div=2):
    task_ids = list(task_ids)
    if len(task_ids) <= size: return task_ids
    anchor  = int(rng.choice(task_ids))
    others  = [p for p in task_ids if p != anchor]
    ranked  = sorted(others, key=lambda p: mi_matrix[anchor][p], reverse=True)
    similar = ranked[:min(n_sim, len(ranked))]
    remain  = [p for p in others if p not in similar]
    ranked_low = sorted(remain, key=lambda p: mi_matrix[anchor][p])
    diverse = ranked_low[:min(n_div, len(ranked_low))]
    selected = [anchor] + similar + diverse
    unused = [p for p in task_ids if p not in selected]
    needed = max(0, size - len(selected))
    if needed and unused:
        selected += list(rng.choice(unused, min(needed, len(unused)), replace=False))
    return selected[:size]


# =============================
# META TRAINING — MI-GUIDED EPISODES
# =============================
def reptile_train_mi(train_users, label_type, seed):
    mi_matrix = build_mi_matrix(train_users, label_type, hardcoded_splits)
    base  = BaseFeatureExtractor().to(device)
    heads = {uid: TaskHead().to(device) for uid in train_users}
    rng   = np.random.default_rng(seed)
    uids  = sorted(train_users.keys())
    for step in range(META_STEPS):
        selected = mi_guided_episode(uids, mi_matrix, rng, EPISODE_SIZE)
        deltas = []
        for uid in selected:
            sup_loader, _ = build_support_query(train_users[uid],
                                                 hardcoded_splits[uid]['train'], [],
                                                 label_type, seed=SEED)
            adapted_base, adapted_head = adapt_inner_loop(
                base, heads[uid], sup_loader, label_type,
                INNER_STEPS, INNER_LR, device, L2_SHARED, L2_TASK)
            heads[uid] = adapted_head
            delta = OrderedDict()
            for (name, p0), (_, pa) in zip(base.named_parameters(), adapted_base.named_parameters()):
                delta[name] = pa.data - p0.data
            deltas.append(delta)
        mean_delta = {name: torch.stack([d[name] for d in deltas]).mean(0) for name in deltas[0]}
        with torch.no_grad():
            for name, p in base.named_parameters():
                p.data.add_(META_LR * mean_delta[name])
    return base


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    train_users = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in train_participants}
    test_users  = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in test_participants}

    print('\n' + '='*60 + '\nTRAINING FINAL AR\n' + '='*60)
    set_all_seeds(SEED)
    model_ar = reptile_train_mi(train_users, 'ar', SEED)
    torch.save(model_ar.state_dict(), os.path.join(output_dir, 'reptile_mi_model_ar_final.pth'))

    print('\n' + '='*60 + '\nTRAINING FINAL VA\n' + '='*60)
    set_all_seeds(SEED)
    model_va = reptile_train_mi(train_users, 'va', SEED)
    torch.save(model_va.state_dict(), os.path.join(output_dir, 'reptile_mi_model_va_final.pth'))

    results_ar, results_va = [], []
    for model, results, label in [
        (model_ar, results_ar, 'ar'),
        (model_va, results_va, 'va')]:
        print(f'\n' + '='*60 + f'\nEVALUATION {label.upper()}\n' + '='*60)
        for uid in sorted(test_participants):
            print(f"  Participant {uid}: adapting")
            r = evaluate_test_user(model, TaskHead(), test_users[uid], hardcoded_splits,
                                    uid, label, device, INNER_STEPS, INNER_LR, L2_SHARED, L2_TASK)
            if r is not None:
                results.append(r)
                print(f"  Participant {uid}: Acc={r['accuracy']:.4f} F1={r['f1']:.4f}")

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

    global_roc = {'AR': {'fpr': ar_fpr, 'tpr': ar_tpr, 'auc': ar_auc, 'y_true': all_true_ar, 'y_pred_probs': all_probs_ar},
                  'VA': {'fpr': va_fpr, 'tpr': va_tpr, 'auc': va_auc, 'y_true': all_true_va, 'y_pred_probs': all_probs_va}}
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(global_roc, f)

    final_results = {
        'train_participants': train_participants, 'test_participants': test_participants,
        'best_hyperparameters': {
            'AR': {'meta_steps': META_STEPS, 'meta_lr': META_LR, 'inner_steps': INNER_STEPS,
                   'inner_lr': INNER_LR, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK},
            'VA': {'meta_steps': META_STEPS, 'meta_lr': META_LR, 'inner_steps': INNER_STEPS,
                   'inner_lr': INNER_LR, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK}},
        'ar_acc': ar_acc, 'ar_precision': ar_prec, 'ar_recall': ar_rec, 'ar_f1': ar_f1, 'ar_auc': ar_auc,
        'va_acc': va_acc, 'va_precision': va_prec, 'va_recall': va_rec, 'va_f1': va_f1, 'va_auc': va_auc,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': cm_AR, 'cm_va': cm_VA,
    }
    with open(os.path.join(output_dir, 'reptile_mi_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    print(f"\n✓ All results saved to: {output_dir}")
