"""
Reptile MMD-Window — anchor-similar-divergent episode sampling guided by
MMD² between participants' window-sets.

Each participant is represented as a set of overlapping windowed segments
(flattened multichannel time × channel vectors). Pairwise MMD²(X_u, X_v)
is computed with a Gaussian kernel (median-heuristic bandwidth) on
subsampled windows. Similarity = -MMD² so that highest = most similar,
matching the anchor-high/low convention from reptile_mi.

The episode adaptation and Reptile outer-loop update are reused from
reptile_mi unchanged — only the cross-participant similarity matrix
construction differs.
"""
import argparse
import copy
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, META_STEPS, META_LR, META_HEAD_LR,
                    INNER_STEPS, INNER_LR, EPISODE_SIZE,
                    L2_SHARED, L2_TASK, K_PER_CLASS, RESULTS_DIR)
import numpy as np
import pickle
import torch
from data import build_support_query
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, TaskHead
from utils import (set_all_seeds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import evaluate_test_user, reptile_outer_update

from reptile_mi import (sample_mi_guided_episode, _split_similar_diverse,
                        _adapt_episode_step)
from mmd_utils import mmd_similarity


def parse_args():
    p = argparse.ArgumentParser(description='Reptile MMD-Window')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


# ================================================================
# MMD-GUIDED TASK EPISODES (raw windowed signals)
# ================================================================

def _build_windows(task_df, feature_cols, window_size, stride):
    """
    Build overlapping windows from a participant's raw multichannel
    signal. Returns a (n_windows, window_size * n_channels) float32
    array of flattened windows.
    """
    X = task_df[feature_cols].values.astype(np.float32)   # (N, C)
    N = len(X)
    if N < window_size:
        return np.empty((0, window_size * len(feature_cols)),
                        dtype=np.float32)
    n_windows = (N - window_size) // stride + 1
    out = np.empty((n_windows, window_size, len(feature_cols)),
                   dtype=np.float32)
    for i in range(n_windows):
        s = i * stride
        out[i] = X[s:s + window_size]
    return out.reshape(n_windows, -1)


def build_task_mmd_window_matrix(tasks_data, splits, cfg,
                                  feature_cols=None, max_windows=150,
                                  use_train_trials_only=True, seed=42):
    """
    Pairwise similarity = -MMD² between participants on raw windowed
    signals. Each participant is a sample of flattened windows.

    `max_windows` caps the per-participant sample size: MMD kernel is
    O(N^2) in N, and the windows are high-dimensional (window_size * C),
    so a sample of ~100–200 windows keeps the matrix build tractable.
    """
    if feature_cols is None:
        feature_cols = ['ECG', 'GSR']
    window_size = cfg['window_size']
    stride = cfg['stride']
    rng = np.random.default_rng(seed)
    task_ids = sorted(list(tasks_data.keys()))

    windows_by_pid = {}
    for pid in task_ids:
        task_df = tasks_data[pid]
        if use_train_trials_only and pid in splits:
            task_df = task_df[task_df['Trial'].isin(
                splits[pid]['train'])].reset_index(drop=True)
        W = _build_windows(task_df, feature_cols, window_size, stride)
        if len(W) > max_windows:
            idx = rng.choice(len(W), max_windows, replace=False)
            W = W[idx]
        windows_by_pid[pid] = W

    sim_matrix = {pid: {} for pid in task_ids}
    for i, pid_i in enumerate(task_ids):
        Xi = windows_by_pid[pid_i]
        for j, pid_j in enumerate(task_ids):
            if j < i:
                sim_matrix[pid_i][pid_j] = sim_matrix[pid_j][pid_i]
                continue
            if i == j:
                sim_matrix[pid_i][pid_j] = 0.0
                continue
            Yj = windows_by_pid[pid_j]
            if len(Xi) == 0 or len(Yj) == 0:
                sim_matrix[pid_i][pid_j] = 0.0
                sim_matrix[pid_j][pid_i] = 0.0
                continue
            s = mmd_similarity(Xi, Yj, rng=rng)
            sim_matrix[pid_i][pid_j] = float(s)
            sim_matrix[pid_j][pid_i] = float(s)

    return sim_matrix


# ================================================================
# META-TRAINING
# ================================================================
def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
                   meta_lr, meta_head_lr, inner_lr, l2_shared, l2_task,
                   meta_steps, inner_steps, episode_size,
                   balanced_k_per_class=None):
    """Reptile-MMD-Window: MMD-guided episodes on raw windowed signals."""
    base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    heads = {pid: TaskHead().to(device) for pid in train_ps}
    rng   = np.random.default_rng(SEED)

    tasks_data = {uid: df[df['ID'] == uid].reset_index(drop=True)
                  for uid in train_ps}
    sim_matrix = build_task_mmd_window_matrix(
        tasks_data, splits, cfg,
        feature_cols=cfg['feature_cols'],
        use_train_trials_only=True, seed=SEED)

    task_ids = list(train_ps)
    eff_episode_size = min(episode_size, len(task_ids))
    n_similar, n_diverse = _split_similar_diverse(eff_episode_size)

    print(f"\n[FINAL-{label_type.upper()}] Meta-training on "
          f"{len(task_ids)} train participants (MMD-Window)")
    print(f"  META_STEPS={meta_steps}, META_LR={meta_lr}, "
          f"META_HEAD_LR={meta_head_lr}, "
          f"INNER_STEPS={inner_steps}, INNER_LR={inner_lr}")
    print(f"  EPISODE_SIZE={eff_episode_size} "
          f"(n_similar={n_similar}, n_diverse={n_diverse})")
    print(f"  L2_SHARED={l2_shared}, L2_TASK={l2_task}")
    if balanced_k_per_class is not None:
        print(f"  [{label_type.upper()}] Balanced k-shot: "
              f"{balanced_k_per_class} per class")
    else:
        print(f"  [{label_type.upper()}] Using all support windows")

    for step in range(meta_steps):
        selected_pids = sample_mi_guided_episode(
            task_ids, sim_matrix, rng,
            episode_size=eff_episode_size,
            n_similar=n_similar, n_diverse=n_diverse)

        if step % 20 == 0 and len(selected_pids) > 1:
            anchor = selected_pids[0]
            sims = [sim_matrix[anchor][p] for p in selected_pids[1:]]
            print(f"    Episode sim (-MMD²): {np.round(sims, 4)}")

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
                for p_persistent, p_adapted in zip(
                        heads[pid].parameters(), adapted_head.parameters()):
                    p_persistent.data.add_(
                        meta_head_lr * (p_adapted.data - p_persistent.data))

        reptile_outer_update(base, [episode_base], meta_lr)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{label_type.upper()}] Reptile-MMD-Window "
                  f"step {step+1}/{meta_steps}")

    torch.save(base.state_dict(),
               os.path.join(output_dir,
                            f'reptile_mmd_window_base_{label_type}.pth'))
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
                              f'{prefix}_reptile_mmd_window')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"K_PER_CLASS: {K_PER_CLASS}")

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
        print(f"\n{'='*60}\nREPTILE-MMD-WINDOW META-TRAINING "
              f"{lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base = _reptile_train(
            lt, df, splits, train_ps, cfg, device, output_dir,
            meta_lr=h['meta_lr'], meta_head_lr=h['meta_head_lr'],
            inner_lr=h['inner_lr'],
            l2_shared=h['l2_shared'], l2_task=h['l2_task'],
            meta_steps=h['meta_steps'], inner_steps=h['inner_steps'],
            episode_size=h['episode_size'],
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
                print(f"  P{uid}: {lt.upper()} "
                      f"Acc={r[f'{lt}_acc']:.4f} "
                      f"F1={r[f'{lt}_f1']:.4f}")

        if lt == 'ar':
            results_ar = results
        else:
            results_va = results

    agg = aggregate_mtml_results(results_ar, results_va)
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump({'AR': {'true': agg['all_true_ar'],
                            'probs': agg['all_probs_ar']},
                     'VA': {'true': agg['all_true_va'],
                            'probs': agg['all_probs_va']}}, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')
    final = {
        'train_participants': train_ps, 'test_participants': test_ps,
        'k_per_class': K_PER_CLASS,
        'hyperparameters': hp,
        **{f'ar_{k}': agg[f'ar_{k}']
           for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}']
           for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir,
                           'reptile_mmd_window_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}']
         for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}']
         for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
