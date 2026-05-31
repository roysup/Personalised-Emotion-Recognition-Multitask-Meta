"""
Reptile MI-Dynamic — Ablation: MI matrix recomputed from learned representations.

This is an ablation of reptile_mi.py. The static-MI version computes the
participant-similarity matrix once before meta-training from discretized raw
signal-label sequences, and that matrix never updates as the backbone evolves.

This script keeps the rest of the MTML-MI pipeline identical but periodically
re-estimates the MI matrix from the *current* shared backbone's representations
of each participant's training windows. The intent is to test whether the
benefits of MI-guided sampling stem from a fixed signal-level prior (as in the
paper) or whether allowing the similarity measure to co-adapt with the learned
representation produces different (better/worse) generalization.

Differences from reptile_mi.py:
  1. Initial MI matrix is still built from raw signals (same warm start).
  2. Every `mi_refresh_every` meta-steps, the MI matrix is recomputed by:
       - encoding each train participant's support-pool windows through the
         current backbone to obtain (N_u, 64) feature matrices;
       - discretizing each of the 64 feature dimensions into B bins;
       - appending the binarized affect label as a final column;
       - estimating pairwise MI as the mean per-column MI score across
         dimensions, exactly mirroring the static formulation.
  3. Sampling, episode adaptation, outer-loop update, and test-time evaluation
     are unchanged.

Result file: reptile_mi_dynamic_results.pkl
Backbone:    reptile_mi_dynamic_base_{ar,va}.pth
"""
import argparse
import copy
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, META_STEPS, META_LR, META_HEAD_LR, MAX_NORM,
                    INNER_STEPS, INNER_LR, EPISODE_SIZE,
                    L2_SHARED, L2_TASK, K_PER_CLASS, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mutual_info_score
from data import build_support_query, create_sliding_windows
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, TaskHead
from utils import (set_all_seeds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import evaluate_test_user, reptile_outer_update

# Re-use the static helpers so behaviour at step 0 matches reptile_mi.py exactly.
from reptile_mi import (
    _digitize_series,
    _compute_task_mi_signature,
    build_task_mi_matrix,
    sample_mi_guided_episode,
    _split_similar_diverse,
    _adapt_episode_step,
)


# ================================================================
# DYNAMIC MI — recompute from current backbone representations
# ================================================================

# How often (in meta-steps) to recompute the MI matrix from the live backbone.
# Recomputation is O(U * windows_per_user) backbone forwards + O(U^2 * D) MI
# scores, so it is non-trivial; we keep it relatively infrequent.
MI_REFRESH_EVERY = 20

# Number of feature dimensions used when discretizing learned representations.
# The backbone emits 64-D vectors; using all 64 inflates MI estimation noise on
# small support pools, so we average MI across the top-variance dims selected
# from the pooled training set. This mirrors the per-channel averaging of the
# static MI formulation (which averages across 2-3 signal channels + 1 label).
DYNAMIC_MI_TOP_DIMS = 8

DYNAMIC_MI_BINS = 18


def _encode_user_windows(base, X_user, device, batch_size=64):
    """Run the current shared backbone on a participant's windowed inputs.

    Parameters
    ----------
    base : BaseFeatureExtractor (already on device)
    X_user : np.ndarray (N, window_size, n_channels)

    Returns
    -------
    feats : np.ndarray (N, 64)
    """
    if len(X_user) == 0:
        return np.zeros((0, 64), dtype=np.float32)

    was_training = base.training
    base.eval()
    feats = []
    X_t = torch.tensor(X_user).float()
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(device, non_blocking=True)
            zb = base(xb)
            feats.append(zb.detach().cpu().numpy())
    if was_training:
        base.train()
    return np.concatenate(feats, axis=0).astype(np.float32)


def _precompute_user_windows(tasks_data, splits, label_type, cfg):
    """Window each train participant's training trials once and cache.

    Returns dict[pid] -> (X_user, y_user) with X shape (N, win, ch)
    and y shape (N,), restricted to that user's training trials.
    """
    cache = {}
    for pid, p_df in tasks_data.items():
        if pid not in splits:
            continue
        train_trials = splits[pid]['train']
        sub_df = p_df[p_df['Trial'].isin(train_trials)].reset_index(drop=True)
        if len(sub_df) == 0:
            cache[pid] = (np.zeros((0, cfg['window_size'], cfg['input_dim']),
                                   dtype=np.float32),
                          np.zeros((0,), dtype=np.float32))
            continue
        X, y_ar, y_va, _, _ = create_sliding_windows(
            sub_df, cfg['window_size'], cfg['stride'],
            feature_cols=cfg['feature_cols'])
        y_user = y_ar if label_type == 'ar' else y_va
        cache[pid] = (X.astype(np.float32), y_user.astype(np.float32))
    return cache


def _pick_top_variance_dims(feature_cache, top_k):
    """Select feature dims with highest pooled across-user variance.

    Stacks all users' features and ranks the 64 dims by variance, returning
    the indices of the top_k most-variable dims. This keeps the MI estimate
    focused on the part of the representation that actually varies, similar
    in spirit to picking informative signal channels.
    """
    pooled = []
    for _pid, (feats, _y) in feature_cache.items():
        if len(feats) > 0:
            pooled.append(feats)
    if not pooled:
        return np.arange(min(top_k, 64))
    F = np.concatenate(pooled, axis=0)
    variances = F.var(axis=0)
    order = np.argsort(-variances)
    return order[:min(top_k, F.shape[1])]


def build_dynamic_mi_matrix(base, tasks_data, splits, label_type, cfg, device,
                            top_k=DYNAMIC_MI_TOP_DIMS, n_bins=DYNAMIC_MI_BINS):
    """Recompute the inter-participant MI matrix from current backbone reps.

    Mirrors `build_task_mi_matrix` in reptile_mi.py but uses learned features
    instead of raw discretized signals. Pairwise MI is computed column-wise
    on aligned-length signatures and averaged.
    """
    # Window every user once (could be cached across refreshes — kept simple
    # here to make ablation behaviour easy to reason about).
    window_cache = _precompute_user_windows(tasks_data, splits, label_type, cfg)

    # Encode each user's windows with the current backbone.
    feature_cache = {}
    for pid, (X_user, y_user) in window_cache.items():
        feats = _encode_user_windows(base, X_user, device)
        feature_cache[pid] = (feats, y_user)

    top_dims = _pick_top_variance_dims(feature_cache, top_k)

    # Build per-user signature: columns = discretized features + label.
    signatures = {}
    for pid, (feats, y_user) in feature_cache.items():
        if len(feats) == 0:
            signatures[pid] = np.zeros((1, top_dims.size + 1), dtype=int)
            continue
        disc_cols = [_digitize_series(feats[:, d], n_bins=n_bins)
                     for d in top_dims]
        disc_cols.append(y_user.astype(int))
        signatures[pid] = np.stack(disc_cols, axis=1).astype(int)

    task_ids = sorted(signatures.keys())
    n_cols = signatures[task_ids[0]].shape[1]
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
                for c in range(n_cols):
                    mi_per_col.append(
                        mutual_info_score(sig_i[:m, c], sig_j[:m, c]))
                mi_val = float(np.mean(mi_per_col))
            mi_matrix[pid_i][pid_j] = mi_val
            mi_matrix[pid_j][pid_i] = mi_val

    return mi_matrix


def parse_args():
    p = argparse.ArgumentParser(description='Reptile MI (dynamic ablation)')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    p.add_argument('--mi_refresh_every', type=int, default=MI_REFRESH_EVERY,
                   help='Recompute MI matrix every N meta-steps.')
    return p.parse_args()


# ================================================================
# META-TRAINING
# ================================================================

def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
                   meta_lr, meta_head_lr, inner_lr, l2_shared, l2_task,
                   meta_steps, inner_steps, episode_size,
                   mi_refresh_every,
                   balanced_k_per_class=None):
    """Reptile-MI-Dynamic: MI-guided episodes with backbone-conditioned MI."""
    base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    heads = {pid: TaskHead().to(device) for pid in train_ps}
    rng   = np.random.default_rng(SEED)

    tasks_data = {uid: df[df['ID'] == uid].reset_index(drop=True)
                  for uid in train_ps}

    # Static warm-start MI: identical to reptile_mi.py at step 0.
    mi_matrix = build_task_mi_matrix(
        tasks_data, splits, label_type=label_type,
        feature_cols=cfg['feature_cols'], use_train_trials_only=True)

    task_ids = list(train_ps)
    eff_episode_size = min(episode_size, len(task_ids))
    n_similar, n_diverse = _split_similar_diverse(eff_episode_size)

    print(f"\n[FINAL-{label_type.upper()}] Meta-training on "
          f"{len(task_ids)} train participants  (MI-Dynamic ablation)")
    print(f"  META_STEPS={meta_steps}, META_LR={meta_lr}, "
          f"META_HEAD_LR={meta_head_lr}, "
          f"INNER_STEPS={inner_steps}, INNER_LR={inner_lr}")
    print(f"  EPISODE_SIZE={eff_episode_size} (n_similar={n_similar}, n_diverse={n_diverse})")
    print(f"  L2_SHARED={l2_shared}, L2_TASK={l2_task}")
    print(f"  MI_REFRESH_EVERY={mi_refresh_every}  "
          f"TOP_DIMS={DYNAMIC_MI_TOP_DIMS}  BINS={DYNAMIC_MI_BINS}")

    if balanced_k_per_class is not None:
        print(f"  [{label_type.upper()}] Balanced k-shot: {balanced_k_per_class} per class")
    else:
        print(f"  [{label_type.upper()}] Using all support windows")

    mi_refresh_count = 0
    for step in range(meta_steps):
        # Refresh MI matrix from learned representations every N steps
        # (skip step 0 — already warm-started from static MI).
        if step > 0 and (step % mi_refresh_every == 0):
            t_mi = time.time()
            mi_matrix = build_dynamic_mi_matrix(
                base, tasks_data, splits, label_type, cfg, device)
            mi_refresh_count += 1
            print(f"  [{label_type.upper()}] MI refresh #{mi_refresh_count} "
                  f"at step {step}  ({time.time() - t_mi:.1f}s)")

        selected_pids = sample_mi_guided_episode(
            task_ids, mi_matrix, rng,
            episode_size=eff_episode_size,
            n_similar=n_similar, n_diverse=n_diverse)

        if step % 20 == 0 and len(selected_pids) > 1:
            anchor = selected_pids[0]
            mi_vals = [mi_matrix[anchor][p] for p in selected_pids[1:]]
            print(f"    Episode MI: {np.round(mi_vals, 3)}")

        episode_base = copy.deepcopy(base).to(device)

        for pid in selected_pids:
            p_df = tasks_data[pid]
            sup_loader, _ = build_support_query(
                p_df, splits[pid]['train'], [],
                ar_or_va=label_type,
                window_size=cfg['window_size'], stride=cfg['stride'],
                feature_cols=cfg['feature_cols'],
                balanced_k_per_class=balanced_k_per_class)

            adapted_head = _adapt_episode_step(
                episode_base, heads[pid], sup_loader, label_type,
                inner_steps, inner_lr, device,
                l2_shared=l2_shared, l2_task=l2_task)

            with torch.no_grad():
                for p_persistent, p_adapted in zip(heads[pid].parameters(),
                                                    adapted_head.parameters()):
                    p_persistent.data.add_(meta_head_lr * (p_adapted.data - p_persistent.data))

        reptile_outer_update(base, [episode_base], meta_lr)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{label_type.upper()}] Reptile-MI-Dynamic step {step+1}/{meta_steps}")

    torch.save(base.state_dict(),
               os.path.join(output_dir, f'reptile_mi_dynamic_base_{label_type}.pth'))
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

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML',
                              f'{prefix}_reptile_mi_dynamic')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"K_PER_CLASS: {K_PER_CLASS}")
    print(f"MI refresh every {args.mi_refresh_every} meta-steps")

    hp = {
        'ar': {
            'meta_lr':      cfg.get('reptile_meta_lr_ar',      META_LR),
            'meta_head_lr': cfg.get('meta_head_lr_ar',         META_HEAD_LR),
            'inner_lr':     cfg.get('reptile_inner_lr_ar',     INNER_LR),
            'inner_steps':  cfg.get('reptile_inner_steps_ar',  INNER_STEPS),
            'episode_size': cfg.get('reptile_episode_size_ar', EPISODE_SIZE),
            'l2_shared':    cfg.get('reptile_l2_shared_ar',    L2_SHARED),
            'l2_task':      cfg.get('reptile_l2_task_ar',      L2_TASK),
            'meta_steps':   cfg.get('reptile_meta_steps_ar',   META_STEPS),
        },
        'va': {
            'meta_lr':      cfg.get('reptile_meta_lr_va',      META_LR),
            'meta_head_lr': cfg.get('meta_head_lr_va',         META_HEAD_LR),
            'inner_lr':     cfg.get('reptile_inner_lr_va',     INNER_LR),
            'inner_steps':  cfg.get('reptile_inner_steps_va',  INNER_STEPS),
            'episode_size': cfg.get('reptile_episode_size_va', EPISODE_SIZE),
            'l2_shared':    cfg.get('reptile_l2_shared_va',    L2_SHARED),
            'l2_task':      cfg.get('reptile_l2_task_va',      L2_TASK),
            'meta_steps':   cfg.get('reptile_meta_steps_va',   META_STEPS),
        },
    }

    for lt in ['ar', 'va']:
        h = hp[lt]
        print(f"\n{'='*60}\nREPTILE-MI-DYNAMIC META-TRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base = _reptile_train(
            lt, df, splits, train_ps, cfg, device, output_dir,
            meta_lr=h['meta_lr'], meta_head_lr=h['meta_head_lr'],
            inner_lr=h['inner_lr'],
            l2_shared=h['l2_shared'], l2_task=h['l2_task'],
            meta_steps=h['meta_steps'], inner_steps=h['inner_steps'],
            episode_size=h['episode_size'],
            mi_refresh_every=args.mi_refresh_every,
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
                inner_steps=h['inner_steps'], inner_lr=h['inner_lr'],
                l2_shared=h['l2_shared'], l2_task=h['l2_task'],
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
        'mi_refresh_every': args.mi_refresh_every,
        'mi_top_dims': DYNAMIC_MI_TOP_DIMS,
        'mi_bins': DYNAMIC_MI_BINS,
        'hyperparameters': hp,
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'reptile_mi_dynamic_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
