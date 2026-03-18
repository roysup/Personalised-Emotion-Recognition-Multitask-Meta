"""
Reptile Meta-MTL — MI-Guided Multi-Task Episodes
Uses mutual information between participant physiological-affective signatures
to select episodes with a mix of similar and diverse tasks.
"""
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
from config import (SEED,
                    META_STEPS, META_LR, INNER_STEPS, INNER_LR, EPISODE_SIZE,
                    L2_SHARED, L2_TASK, HARDCODED_SPLITS, TEST_PARTICIPANTS,
                    RESULTS_DIR)
import numpy as np
import pickle
import torch
from sklearn.metrics import mutual_info_score
from utils import (set_all_seeds,
                   aggregate_mtml_results, compute_per_participant_stds,
                   print_determinism_summary)
from models import BaseFeatureExtractor, TaskHead
from training import adapt_inner_loop, evaluate_test_user, reptile_outer_update
from data import build_support_query
from dataset_configs.vreed import load_vreed_df

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir = os.path.join(BASE_OUTPUT_DIR, 'VREED_ReptileMeta_MI_episode')
os.makedirs(output_dir, exist_ok=True)

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
        adapted_bases = []
        for uid in selected:
            sup_loader, _ = build_support_query(train_users[uid],
                                                 hardcoded_splits[uid]['train'], [],
                                                 label_type, seed=SEED)
            adapted_base, adapted_head = adapt_inner_loop(
                base, heads[uid], sup_loader, label_type,
                INNER_STEPS, INNER_LR, device, L2_SHARED, L2_TASK)
            heads[uid] = adapted_head
            adapted_bases.append(adapted_base)
        reptile_outer_update(base, adapted_bases, META_LR)
    return base

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    experiment_t0 = time.time()

    train_users = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in train_participants}
    test_users  = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in test_participants}

    print('\n' + '='*60 + '\nTRAINING FINAL AR\n' + '='*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_ar = reptile_train_mi(train_users, 'ar', SEED)
    print(f"  AR training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_ar.state_dict(), os.path.join(output_dir, 'reptile_mi_model_ar_final.pth'))

    print('\n' + '='*60 + '\nTRAINING FINAL VA\n' + '='*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_va = reptile_train_mi(train_users, 'va', SEED)
    print(f"  VA training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_va.state_dict(), os.path.join(output_dir, 'reptile_mi_model_va_final.pth'))

    results_ar, results_va = [], []
    for model, results_list, label in [
        (model_ar, results_ar, 'ar'),
        (model_va, results_va, 'va')]:
        print(f'\n' + '='*60 + f'\nEVALUATION {label.upper()}\n' + '='*60)
        for uid in sorted(test_participants):
            print(f"  Participant {uid}: adapting")
            r = evaluate_test_user(model, TaskHead(), test_users[uid], hardcoded_splits,
                                    uid, label, device, INNER_STEPS, INNER_LR, L2_SHARED, L2_TASK)
            if r is not None:
                results_list.append(r)
                print(f"  Participant {uid}: Acc={r[f'{label}_acc']:.4f} F1={r[f'{label}_f1']:.4f}")

    agg = aggregate_mtml_results(results_ar, results_va)

    global_roc = {
        'AR': {'fpr': agg['fpr_ar'], 'tpr': agg['tpr_ar'], 'auc': agg['ar_auc'],
               'y_true': agg['all_true_ar'], 'y_pred_probs': agg['all_probs_ar']},
        'VA': {'fpr': agg['fpr_va'], 'tpr': agg['tpr_va'], 'auc': agg['va_auc'],
               'y_true': agg['all_true_va'], 'y_pred_probs': agg['all_probs_va']},
    }
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(global_roc, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')

    final_results = {
        'train_participants': train_participants, 'test_participants': test_participants,
        'best_hyperparameters': {
            'AR': {'meta_steps': META_STEPS, 'meta_lr': META_LR, 'inner_steps': INNER_STEPS,
                   'inner_lr': INNER_LR, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK},
            'VA': {'meta_steps': META_STEPS, 'meta_lr': META_LR, 'inner_steps': INNER_STEPS,
                   'inner_lr': INNER_LR, 'l2_shared': L2_SHARED, 'l2_task': L2_TASK}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'reptile_mi_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    print_determinism_summary(
        {f'ar_{k}': final_results[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final_results[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
