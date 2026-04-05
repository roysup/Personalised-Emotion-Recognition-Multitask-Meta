# """
# Reptile MI — Mutual-Information-guided episode sampling.
# Computes pairwise MI between participants using discretised ECG/GSR/label
# distributions, then selects episodes with a mix of similar and diverse
# participants relative to an anchor. Multi-participant episodes (EPISODE_SIZE),
# backbone-only outer update. Heads are kept per-participant and updated
# individually.

# Usage
# -----
#     python reptile_mi.py                  # runs on VREED (default)
#     python reptile_mi.py --dataset dssn_eq
# """
# import argparse
# import os, sys, time
# _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

# from config import (SEED, META_STEPS, META_LR,
#                     INNER_STEPS, INNER_LR, EPISODE_SIZE,
#                     L2_SHARED, L2_TASK, RESULTS_DIR)
# import numpy as np
# import pickle
# import torch
# from sklearn.metrics import mutual_info_score
# from data import build_support_query
# from dataset_configs.loader import load_dataset
# from models import BaseFeatureExtractor, TaskHead
# from utils import (set_all_seeds, aggregate_mtml_results,
#                    compute_per_participant_stds, print_determinism_summary)
# from training import adapt_inner_loop, evaluate_test_user, reptile_outer_update


# def parse_args():
#     p = argparse.ArgumentParser(description='Reptile MI')
#     p.add_argument('--dataset', type=str, default='vreed',
#                    choices=['vreed', 'dssn_eq', 'dssn_em'])
#     return p.parse_args()


# # ================================================================
# # MI-GUIDED TASK EPISODES
# # ================================================================

# def _digitize_series(x, n_bins=16):
#     """Discretize a continuous 1D signal into bins for MI computation."""
#     x = np.asarray(x).reshape(-1)
#     if len(x) == 0:
#         return np.zeros(1, dtype=int)
#     x_min, x_max = np.min(x), np.max(x)
#     if np.isclose(x_min, x_max):
#         return np.zeros_like(x, dtype=int)
#     bins = np.linspace(x_min, x_max, n_bins + 1)
#     return np.digitize(x, bins[1:-1], right=False).astype(int)


# def _compute_task_mi_signature(task_df, label_type='ar', feature_cols=None,
#                                 max_points=20000):
#     """Build a discretised task-level signature for MI computation."""
#     if feature_cols is None:
#         feature_cols = ['ECG', 'GSR']

#     df_local = task_df.copy()
#     if len(df_local) > max_points:
#         idx = np.linspace(0, len(df_local) - 1, max_points).astype(int)
#         df_local = df_local.iloc[idx].reset_index(drop=True)

#     disc_cols = []
#     for col in feature_cols:
#         disc_cols.append(_digitize_series(df_local[col].values, n_bins=16))

#     y_col = 'AR_Rating' if label_type == 'ar' else 'VA_Rating'
#     y_disc = df_local[y_col].astype(int).values

#     return np.stack(disc_cols + [y_disc], axis=1).astype(int)


# def build_task_mi_matrix(tasks_data, splits, label_type='ar',
#                          feature_cols=None, use_train_trials_only=True):
#     """Compute pairwise MI between tasks/users."""
#     task_ids = sorted(list(tasks_data.keys()))
#     signatures = {}

#     for pid in task_ids:
#         task_df = tasks_data[pid]
#         if use_train_trials_only and pid in splits:
#             task_df = task_df[task_df['Trial'].isin(
#                 splits[pid]['train'])].reset_index(drop=True)
#         signatures[pid] = _compute_task_mi_signature(
#             task_df, label_type=label_type, feature_cols=feature_cols)

#     n_sig_cols = signatures[task_ids[0]].shape[1]
#     mi_matrix = {pid: {} for pid in task_ids}

#     for i, pid_i in enumerate(task_ids):
#         sig_i = signatures[pid_i]
#         for j, pid_j in enumerate(task_ids):
#             if j < i:
#                 mi_matrix[pid_i][pid_j] = mi_matrix[pid_j][pid_i]
#                 continue
#             sig_j = signatures[pid_j]
#             m = min(len(sig_i), len(sig_j))
#             if m == 0:
#                 mi_val = 0.0
#             else:
#                 mi_per_col = []
#                 for c in range(n_sig_cols):
#                     mi_per_col.append(
#                         mutual_info_score(sig_i[:m, c], sig_j[:m, c]))
#                 mi_val = np.mean(mi_per_col)
#             mi_matrix[pid_i][pid_j] = float(mi_val)
#             mi_matrix[pid_j][pid_i] = float(mi_val)

#     return mi_matrix


# def sample_mi_guided_episode(task_ids, mi_matrix, rng, episode_size=5,
#                               n_similar=2, n_diverse=2):
#     """MI-guided episodic sampler: anchor + similar + diverse participants."""
#     task_ids = list(task_ids)
#     if len(task_ids) <= episode_size:
#         return list(task_ids)

#     anchor = int(rng.choice(task_ids))
#     others = [pid for pid in task_ids if pid != anchor]

#     ranked = sorted(others, key=lambda pid: mi_matrix[anchor][pid], reverse=True)
#     similar = ranked[:min(n_similar, len(ranked))]

#     remaining = [pid for pid in others if pid not in similar]
#     ranked_low = sorted(remaining, key=lambda pid: mi_matrix[anchor][pid])
#     diverse = ranked_low[:min(n_diverse, len(ranked_low))]

#     selected = [anchor] + similar + diverse
#     unused = [pid for pid in task_ids if pid not in selected]
#     needed = max(0, episode_size - len(selected))
#     if needed > 0 and len(unused) > 0:
#         extra = list(rng.choice(unused, size=min(needed, len(unused)),
#                                 replace=False))
#         selected.extend(extra)

#     return selected[:episode_size]


# # ================================================================
# # META-TRAINING
# # ================================================================

# def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
#                    meta_lr, inner_lr, l2_shared, l2_task):
#     """Reptile-MI: MI-guided episodes, backbone-only outer update."""
#     base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
#     heads = {pid: TaskHead().to(device) for pid in train_ps}
#     rng   = np.random.default_rng(SEED)

#     # Build MI matrix from train participants
#     tasks_data = {uid: df[df['ID'] == uid].reset_index(drop=True)
#                   for uid in train_ps}
#     mi_matrix = build_task_mi_matrix(
#         tasks_data, splits, label_type=label_type,
#         feature_cols=cfg['feature_cols'], use_train_trials_only=True)

#     task_ids = list(train_ps)
#     print(f"\n[FINAL-{label_type.upper()}] Meta-training on "
#           f"{len(task_ids)} train participants")

#     for step in range(META_STEPS):
#         # MI-guided multi-user episode (only difference from MT)
#         selected_pids = sample_mi_guided_episode(
#             task_ids, mi_matrix, rng,
#             episode_size=min(EPISODE_SIZE, len(task_ids)),
#             n_similar=2, n_diverse=2)

#         if step % 20 == 0:
#             anchor = selected_pids[0]
#             mi_vals = [mi_matrix[anchor][p] for p in selected_pids[1:]]
#             print(f"    Episode MI: {np.round(mi_vals, 3)}")

#         adapted_bases = []
#         for pid in selected_pids:
#             p_df = tasks_data[pid]
#             sup_loader, _ = build_support_query(
#                 p_df, splits[pid]['train'], [],
#                 ar_or_va=label_type,
#                 window_size=cfg['window_size'], stride=cfg['stride'],
#                 feature_cols=cfg['feature_cols'])

#             adapted_base, adapted_head = adapt_inner_loop(
#                 base, heads[pid], sup_loader, label_type,
#                 INNER_STEPS, inner_lr, device,
#                 l2_shared=l2_shared, l2_task=l2_task)
#             adapted_bases.append(adapted_base)

#             # Keep updated head per-participant
#             heads[pid] = adapted_head

#         # Outer update on backbone only
#         reptile_outer_update(base, adapted_bases, meta_lr)

#         if (step + 1) % 10 == 0 or step == 0:
#             print(f"  [{label_type.upper()}] Reptile-MI step {step+1}/{META_STEPS}")

#     torch.save(base.state_dict(),
#                os.path.join(output_dir, f'reptile_mi_base_{label_type}.pth'))
#     return base


# # ================================================================
# # MAIN
# # ================================================================

# if __name__ == '__main__':
#     args = parse_args()
#     experiment_t0 = time.time()

#     df, cfg = load_dataset(args.dataset, mode='mtml')
#     splits   = cfg['splits']
#     prefix   = cfg['results_prefix']
#     train_ps = cfg['train_participants']
#     test_ps  = cfg['test_participants']

#     output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_reptile_mi')
#     os.makedirs(output_dir, exist_ok=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     set_all_seeds(SEED)
#     if device.type == 'cuda':
#         torch.backends.cudnn.benchmark = True
#     print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")

#     for lt in ['ar', 'va']:
#         print(f"\n{'='*60}\nREPTILE-MI META-TRAINING {lt.upper()}\n{'='*60}")
#         set_all_seeds(SEED)
#         base = _reptile_train(lt, df, splits, train_ps, cfg, device, output_dir,
#                               meta_lr=META_LR, inner_lr=INNER_LR,
#                               l2_shared=L2_SHARED, l2_task=L2_TASK)

#         print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()}\n{'='*60}")
#         results = []
#         for uid in sorted(test_ps):
#             if uid not in splits:
#                 continue
#             t_df = df[df['ID'] == uid].reset_index(drop=True)
#             head = TaskHead().to(device)
#             r = evaluate_test_user(
#                 base, head, t_df, splits, uid, lt, device,
#                 inner_steps=INNER_STEPS, inner_lr=INNER_LR,
#                 l2_shared=L2_SHARED, l2_task=L2_TASK,
#                 window_size=cfg['window_size'], stride=cfg['stride'],
#                 feature_cols=cfg['feature_cols'])
#             if r is not None:
#                 results.append(r)
#                 print(f"  P{uid}: {lt.upper()} Acc={r[f'{lt}_acc']:.4f} "
#                       f"F1={r[f'{lt}_f1']:.4f}")

#         if lt == 'ar':
#             results_ar = results
#         else:
#             results_va = results

#     agg = aggregate_mtml_results(results_ar, results_va)
#     with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
#         pickle.dump({'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
#                      'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']}}, f)

#     ar_stds = compute_per_participant_stds(results_ar, 'ar')
#     va_stds = compute_per_participant_stds(results_va, 'va')
#     final = {
#         'train_participants': train_ps, 'test_participants': test_ps,
#         **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **ar_stds, **va_stds,
#         'test_results_per_participant_ar': results_ar,
#         'test_results_per_participant_va': results_va,
#         'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
#     }
#     with open(os.path.join(output_dir, 'reptile_mi_results.pkl'), 'wb') as f:
#         pickle.dump(final, f)

#     print_determinism_summary(
#         {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         ar_stds, va_stds)

#     print(f"\n✓ All results saved to: {output_dir}")
#     print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")


"""
Reptile MI — Mutual-Information-guided episode sampling.
Computes pairwise MI between participants using discretised ECG/GSR/label
distributions, then selects episodes with a mix of similar and diverse
participants relative to an anchor. Sequential adaptation within each
episode: a single clone of the backbone is adapted across all sampled
users in order, then the Reptile outer update moves the meta-parameters
toward the result. Heads are kept per-participant and updated individually.

Supports optional balanced k-shot support sampling via K_PER_CLASS:
  None  = use all available support windows (default)
  int   = subsample k windows per class (e.g. 20 → 40 total)

Usage
-----
    python reptile_mi.py                  # runs on VREED (default)
    python reptile_mi.py --dataset dssn_eq
"""
import argparse
import copy
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, META_STEPS, META_LR, MAX_NORM,
                    INNER_STEPS, INNER_LR, EPISODE_SIZE,
                    L2_SHARED, L2_TASK, K_PER_CLASS, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mutual_info_score
from data import build_support_query
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, TaskHead
from utils import (set_all_seeds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import evaluate_test_user, reptile_outer_update


def parse_args():
    p = argparse.ArgumentParser(description='Reptile MI')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


# ================================================================
# MI-GUIDED TASK EPISODES
# ================================================================

def _digitize_series(x, n_bins=16):
    """Discretize a continuous 1D signal into bins for MI computation."""
    x = np.asarray(x).reshape(-1)
    if len(x) == 0:
        return np.zeros(1, dtype=int)
    x_min, x_max = np.min(x), np.max(x)
    if np.isclose(x_min, x_max):
        return np.zeros_like(x, dtype=int)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    return np.digitize(x, bins[1:-1], right=False).astype(int)


def _compute_task_mi_signature(task_df, label_type='ar', feature_cols=None,
                                max_points=20000):
    """Build a discretised task-level signature for MI computation."""
    if feature_cols is None:
        feature_cols = ['ECG', 'GSR']

    df_local = task_df.copy()
    if len(df_local) > max_points:
        idx = np.linspace(0, len(df_local) - 1, max_points).astype(int)
        df_local = df_local.iloc[idx].reset_index(drop=True)

    disc_cols = []
    for col in feature_cols:
        disc_cols.append(_digitize_series(df_local[col].values, n_bins=16))

    y_col = 'AR_Rating' if label_type == 'ar' else 'VA_Rating'
    y_disc = df_local[y_col].astype(int).values

    return np.stack(disc_cols + [y_disc], axis=1).astype(int)


def build_task_mi_matrix(tasks_data, splits, label_type='ar',
                         feature_cols=None, use_train_trials_only=True):
    """Compute pairwise MI between tasks/users."""
    task_ids = sorted(list(tasks_data.keys()))
    signatures = {}

    for pid in task_ids:
        task_df = tasks_data[pid]
        if use_train_trials_only and pid in splits:
            task_df = task_df[task_df['Trial'].isin(
                splits[pid]['train'])].reset_index(drop=True)
        signatures[pid] = _compute_task_mi_signature(
            task_df, label_type=label_type, feature_cols=feature_cols)

    n_sig_cols = signatures[task_ids[0]].shape[1]
    mi_matrix = {pid: {} for pid in task_ids}

    for i, pid_i in enumerate(task_ids):
        sig_i = signatures[pid_i]
        for j, pid_j in enumerate(task_ids):
            if j < i:
                mi_matrix[pid_i][pid_j] = mi_matrix[pid_j][pid_i]
                continue
            sig_j = signatures[pid_j]
            m = min(len(sig_i), len(sig_j))
            if m == 0:
                mi_val = 0.0
            else:
                mi_per_col = []
                for c in range(n_sig_cols):
                    mi_per_col.append(
                        mutual_info_score(sig_i[:m, c], sig_j[:m, c]))
                mi_val = np.mean(mi_per_col)
            mi_matrix[pid_i][pid_j] = float(mi_val)
            mi_matrix[pid_j][pid_i] = float(mi_val)

    return mi_matrix


def sample_mi_guided_episode(task_ids, mi_matrix, rng, episode_size=5,
                              n_similar=2, n_diverse=2):
    """MI-guided episodic sampler: anchor + similar + diverse participants."""
    task_ids = list(task_ids)
    if len(task_ids) <= episode_size:
        return list(task_ids)

    anchor = int(rng.choice(task_ids))
    others = [pid for pid in task_ids if pid != anchor]

    ranked = sorted(others, key=lambda pid: mi_matrix[anchor][pid], reverse=True)
    similar = ranked[:min(n_similar, len(ranked))]

    remaining = [pid for pid in others if pid not in similar]
    ranked_low = sorted(remaining, key=lambda pid: mi_matrix[anchor][pid])
    diverse = ranked_low[:min(n_diverse, len(ranked_low))]

    selected = [anchor] + similar + diverse
    unused = [pid for pid in task_ids if pid not in selected]
    needed = max(0, episode_size - len(selected))
    if needed > 0 and len(unused) > 0:
        extra = list(rng.choice(unused, size=min(needed, len(unused)),
                                replace=False))
        selected.extend(extra)

    return selected[:episode_size]


# ================================================================
# SEQUENTIAL EPISODE ADAPTATION
# ================================================================
def _adapt_episode_step(episode_base, head, sup_loader, ar_or_va,
                        inner_steps, inner_lr, device,
                        l2_shared=0.0, l2_task=1e-5):
    """
    Adapt episode_base IN PLACE on one user's support data.
    The head is deep-copied so each user gets their own head,
    but the backbone is shared and mutated across users within
    the episode (sequential Reptile).

    Returns
    -------
    adapted_head : the updated head for this user
    """
    adapted_head = copy.deepcopy(head).to(device)
    episode_base.train()
    adapted_head.train()

    sp = list(episode_base.parameters())
    tp = list(adapted_head.parameters())
    opt   = torch.optim.Adam(sp + tp, lr=inner_lr)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(inner_steps):
        ep_loss = 0.0
        nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(adapted_head(episode_base(Xb)), yb)
            #loss = loss + (l2_shared * sum(p.norm(2)**2 for p in sp if p.requires_grad) +
                           #l2_task   * sum(p.norm(2)**2 for p in tp if p.requires_grad))
            loss = loss + (l2_shared * sum(p.norm(2)**2 for p in sp if p.requires_grad and p.ndim >= 2) + l2_task* sum(p.norm(2)**2 for p in tp if p.requires_grad and p.ndim >= 2))
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sp + tp, max_norm=MAX_NORM)
                opt.step()
            ep_loss += loss.item()
            nb += 1
        if nb > 0:
            sched.step(ep_loss / nb)

    return adapted_head


# ================================================================
# META-TRAINING
# ================================================================

def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
                   meta_lr, inner_lr, l2_shared, l2_task,
                   balanced_k_per_class=None):
    """Reptile-MI: MI-guided episodes, sequential adaptation, backbone-only outer update."""
    base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    heads = {pid: TaskHead().to(device) for pid in train_ps}
    rng   = np.random.default_rng(SEED)

    # Build MI matrix from train participants
    tasks_data = {uid: df[df['ID'] == uid].reset_index(drop=True)
                  for uid in train_ps}
    mi_matrix = build_task_mi_matrix(
        tasks_data, splits, label_type=label_type,
        feature_cols=cfg['feature_cols'], use_train_trials_only=True)

    task_ids = list(train_ps)
    print(f"\n[FINAL-{label_type.upper()}] Meta-training on "
          f"{len(task_ids)} train participants")

    if balanced_k_per_class is not None:
        print(f"  [{label_type.upper()}] Balanced k-shot: {balanced_k_per_class} per class")
    else:
        print(f"  [{label_type.upper()}] Using all support windows")

    for step in range(META_STEPS):
        # MI-guided multi-user episode (only difference from MT)
        selected_pids = sample_mi_guided_episode(
            task_ids, mi_matrix, rng,
            episode_size=min(EPISODE_SIZE, len(task_ids)),
            n_similar=2, n_diverse=2)

        if step % 20 == 0:
            anchor = selected_pids[0]
            mi_vals = [mi_matrix[anchor][p] for p in selected_pids[1:]]
            print(f"    Episode MI: {np.round(mi_vals, 3)}")

        # Single clone for the whole episode — adapted sequentially
        episode_base = copy.deepcopy(base).to(device)

        for pid in selected_pids:
            p_df = tasks_data[pid]
            sup_loader, _ = build_support_query(
                p_df, splits[pid]['train'], [],
                ar_or_va=label_type,
                window_size=cfg['window_size'], stride=cfg['stride'],
                feature_cols=cfg['feature_cols'],
                balanced_k_per_class=balanced_k_per_class)

            # Adapt episode_base in place; only the head is copied
            adapted_head = _adapt_episode_step(
                episode_base, heads[pid], sup_loader, label_type,
                INNER_STEPS, inner_lr, device,
                l2_shared=l2_shared, l2_task=l2_task)

            # Keep updated head per-participant
            heads[pid] = adapted_head

        # Outer update: move backbone toward the end of the sequential trajectory
        reptile_outer_update(base, [episode_base], meta_lr)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{label_type.upper()}] Reptile-MI step {step+1}/{META_STEPS}")

    torch.save(base.state_dict(),
               os.path.join(output_dir, f'reptile_mi_base_{label_type}.pth'))
    return base


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    args = parse_args()
    experiment_t0 = time.time()

    df, cfg = load_dataset(args.dataset, mode='mtml')
    splits   = cfg['splits']
    prefix   = cfg['results_prefix']
    train_ps = cfg['train_participants']
    test_ps  = cfg['test_participants']

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_reptile_mi')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"K_PER_CLASS: {K_PER_CLASS}")

    for lt in ['ar', 'va']:
        print(f"\n{'='*60}\nREPTILE-MI META-TRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base = _reptile_train(lt, df, splits, train_ps, cfg, device, output_dir,
                              meta_lr=META_LR, inner_lr=INNER_LR,
                              l2_shared=L2_SHARED, l2_task=L2_TASK,
                              balanced_k_per_class=K_PER_CLASS)

        print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()}\n{'='*60}")
        results = []
        for uid in sorted(test_ps):
            if uid not in splits:
                continue
            t_df = df[df['ID'] == uid].reset_index(drop=True)
            head = TaskHead().to(device)
            r = evaluate_test_user(
                base, head, t_df, splits, uid, lt, device,
                inner_steps=INNER_STEPS, inner_lr=INNER_LR,
                l2_shared=L2_SHARED, l2_task=L2_TASK,
                window_size=cfg['window_size'], stride=cfg['stride'],
                feature_cols=cfg['feature_cols'],
                balanced_k_per_class=K_PER_CLASS)
            if r is not None:
                results.append(r)
                print(f"  P{uid}: {lt.upper()} Acc={r[f'{lt}_acc']:.4f} "
                      f"F1={r[f'{lt}_f1']:.4f}")

        if lt == 'ar':
            results_ar = results
        else:
            results_va = results

    agg = aggregate_mtml_results(results_ar, results_va)
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump({'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
                     'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']}}, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')
    final = {
        'train_participants': train_ps, 'test_participants': test_ps,
        'k_per_class': K_PER_CLASS,
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'reptile_mi_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")