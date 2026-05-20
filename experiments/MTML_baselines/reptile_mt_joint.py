"""
Reptile Multi-Task — JOINT episode variant (ablation)
=====================================================
Ablation of reptile_mt.py. Identical in every respect EXCEPT the inner-loop
adaptation within an episode:

  * reptile_mt.py (sequential): a single backbone clone is adapted to each
    episode user one-after-another, carrying parameters forward, then the
    Reptile outer update moves toward the end of that trajectory.

  * reptile_mt_joint.py (this file, JOINT): within an episode, all K users'
    support sets are optimized SIMULTANEOUSLY — at each inner step the combined
    (mean) loss over the K users is back-propagated in a single update to the
    shared backbone and the K per-user heads. This mirrors the joint multi-task
    episode construction used by Upadhyay et al. (MTML), where episode tasks are
    co-trained with a combined (optionally uncertainty-weighted) objective.

Everything else — episode sampling, balanced k-shot support sets, per-user
heads, the Reptile outer update, meta-test single-user adaptation, and the
sign-flip diagnostic — is unchanged, so this is a controlled comparison of
SEQUENTIAL vs JOINT episode adaptation.

Note on deployment alignment: at meta-test a single unseen user is still adapted
with a fresh head. This ablation therefore tests whether a backbone meta-trained
via joint multi-user episodes yields a better single-user-adaptable initialization
than one trained via sequential episodes.

Optional --uw flag adds per-episode uncertainty weighting (Kendall et al.) over
the K users' losses, matching Upadhyay's loss-combination choice. Default is
plain mean (equal weighting), which is appropriate here because all per-user
losses are the same binary cross-entropy on the same scale.

Usage
-----
    python reptile_mt_joint.py --dataset vreed
    python reptile_mt_joint.py --dataset dssn_eq --uw
"""
import argparse
import copy
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, META_STEPS, META_LR, META_HEAD_LR, MAX_NORM,
                    INNER_STEPS, INNER_LR, EPISODE_SIZE,
                    N_FOLDS, L2_SHARED, L2_TASK, K_PER_CLASS,
                    RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score
from data import build_support_query
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, TaskHead
from utils import (set_all_seeds, make_kfolds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import evaluate_test_user, reptile_outer_update


def parse_args():
    p = argparse.ArgumentParser(description='Reptile MT (joint episode ablation)')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    p.add_argument('--uw', action='store_true',
                   help='Use per-episode uncertainty weighting over the K users '
                        '(Kendall et al.). Default: plain mean.')
    return p.parse_args()


# =============================
# PER-LABEL-TYPE HPARAM LOOKUP
# =============================
def _get_reptile_hparams(cfg, lt):
    return {
        'meta_lr':       cfg.get(f'reptile_meta_lr_{lt}',      META_LR),
        'meta_head_lr':  cfg.get(f'meta_head_lr_{lt}',         META_HEAD_LR),
        'inner_lr':      cfg.get(f'reptile_inner_lr_{lt}',     INNER_LR),
        'inner_steps':   cfg.get(f'reptile_inner_steps_{lt}',  INNER_STEPS),
        'episode_size':  cfg.get(f'reptile_episode_size_{lt}', EPISODE_SIZE),
        'l2_shared':     cfg.get(f'reptile_l2_shared_{lt}',    L2_SHARED),
        'l2_task':       cfg.get(f'reptile_l2_task_{lt}',      L2_TASK),
        'meta_steps':    cfg.get(f'reptile_meta_steps_{lt}',   META_STEPS),
    }


# =============================
# SIGN-FLIP DIAGNOSTIC (identical to reptile_mt.py)
# =============================
def _check_sign_flip(results, label_type):
    print(f"\n  {'-'*50}")
    print(f"  SIGN-FLIP DIAGNOSTIC [{label_type.upper()}]")
    print(f"  {'-'*50}")
    per_pid = []
    auc_normal_list, auc_flipped_list, n_flipped = [], [], 0
    for r in results:
        y_true = r[f'y_true_{label_type}']
        y_prob = r[f'y_pred_probs_{label_type}']
        pid = r.get('participant_id', '?')
        if len(np.unique(y_true)) < 2:
            print(f"    P{pid}: (single-class in test, skipped)")
            per_pid.append({'participant_id': pid, 'auc_normal': np.nan,
                            'auc_flipped': np.nan, 'flipped': False})
            continue
        auc_n = roc_auc_score(y_true, y_prob)
        auc_f = roc_auc_score(y_true, 1 - y_prob)
        flipped = auc_f > auc_n
        n_flipped += int(flipped)
        auc_normal_list.append(auc_n); auc_flipped_list.append(auc_f)
        per_pid.append({'participant_id': pid, 'auc_normal': auc_n,
                        'auc_flipped': auc_f, 'flipped': flipped})
        print(f"    P{pid}: AUC={auc_n:.3f}  flipped={auc_f:.3f}"
              f"{'  <- FLIPPED' if flipped else ''}")
    summary = {}
    if auc_normal_list:
        mean_n = float(np.mean(auc_normal_list))
        mean_f = float(np.mean(auc_flipped_list))
        n_total = len(auc_normal_list)
        print(f"\n  Mean AUC (normal):  {mean_n:.4f}")
        print(f"  Mean AUC (flipped): {mean_f:.4f}")
        print(f"  Flipped: {n_flipped}/{n_total} participants")
        if mean_n < 0.45 and mean_f > 0.55:
            verdict = "SYSTEMATIC SIGN FLIP"
        elif 0.45 <= mean_n <= 0.55 and 0.45 <= mean_f <= 0.55:
            verdict = "NO USABLE SIGNAL"
        elif n_flipped >= n_total / 2 and abs(mean_n - mean_f) < 0.05:
            verdict = "PER-PARTICIPANT INSTABILITY"
        else:
            verdict = "NO SIGN FLIP"
        print(f"  Verdict: {verdict}")
        summary = {'mean_auc_normal': mean_n, 'mean_auc_flipped': mean_f,
                   'n_flipped': n_flipped, 'n_total': n_total, 'verdict': verdict}
    return {'per_participant': per_pid, 'summary': summary}


# =============================
# JOINT EPISODE ADAPTATION
# =============================
def _materialize_support(sup_loader, device):
    """Collect a user's full (small) support set into one tensor pair."""
    Xs, ys = [], []
    for Xb, yb in sup_loader:
        Xs.append(Xb); ys.append(yb)
    if not Xs:
        return None
    return torch.cat(Xs).to(device), torch.cat(ys).to(device)


def _adapt_episode_joint(episode_base, head_clones, user_batches, label_type,
                         inner_steps, inner_lr, device,
                         l2_shared=0.0, l2_task=1e-5, use_uw=False):
    """
    Adapt episode_base IN PLACE jointly across all episode users.

    At each inner step the combined (mean, or UW-weighted) loss over all K users
    is back-propagated in a single optimizer update to the shared backbone and
    all K per-user heads. This is the joint counterpart to reptile_mt's
    sequential _adapt_episode_step.

    Parameters
    ----------
    episode_base : backbone clone (mutated in place)
    head_clones  : dict {pid: head clone} — one per episode user (mutated)
    user_batches : dict {pid: (X, y)} — each user's materialized support set
    use_uw       : if True, add per-episode Kendall uncertainty weighting

    Returns
    -------
    head_clones : the adapted per-user heads (for the EMA head update)
    """
    episode_base.train()
    for h in head_clones.values():
        h.train()

    pids = list(user_batches.keys())
    sp = list(episode_base.parameters())
    tp = [p for pid in pids for p in head_clones[pid].parameters()]

    params = sp + tp
    log_vars = None
    if use_uw:
        # one log-variance per episode user (reset each episode)
        log_vars = torch.zeros(len(pids), device=device, requires_grad=True)
        params = params + [log_vars]

    opt   = torch.optim.Adam(params, lr=inner_lr)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(inner_steps):
        opt.zero_grad(set_to_none=True)
        per_user_losses = []
        for pid, (Xb, yb) in user_batches.items():
            logit = head_clones[pid](episode_base(Xb))
            per_user_losses.append(loss_fn(logit, yb))

        losses = torch.stack(per_user_losses)            # (K,)
        if use_uw:
            precision = torch.exp(-log_vars)
            combined = (precision * losses + 0.5 * log_vars).mean()
        else:
            combined = losses.mean()                     # joint, equal weighting

        l2 = (l2_shared * sum(p.norm(2) ** 2 for p in sp
                              if p.requires_grad and p.ndim >= 2) +
              l2_task   * sum(p.norm(2) ** 2 for p in tp
                              if p.requires_grad and p.ndim >= 2))
        total = combined + l2

        if not torch.isnan(total):
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=MAX_NORM)
            opt.step()
        sched.step(total.item())

    return head_clones


# =============================
# META-TRAINING (joint episodes)
# =============================
def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
                   meta_lr, meta_head_lr, inner_lr, l2_shared, l2_task,
                   inner_steps, episode_size, meta_steps,
                   balanced_k_per_class=None, use_uw=False):
    base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    heads = {pid: TaskHead().to(device) for pid in train_ps}
    rng   = np.random.default_rng(SEED)

    print(f"  [{label_type.upper()}] JOINT episodes | "
          f"{'UW-weighted' if use_uw else 'mean'} loss combination")
    if balanced_k_per_class is not None:
        print(f"  [{label_type.upper()}] Balanced k-shot: {balanced_k_per_class} per class")
    else:
        print(f"  [{label_type.upper()}] Using all support windows")
    print(f"  [{label_type.upper()}] meta_lr={meta_lr}, meta_head_lr={meta_head_lr}")

    for step in range(meta_steps):
        episode_ps = rng.choice(train_ps, size=min(episode_size, len(train_ps)),
                                replace=False).tolist()

        # Build each episode user's support set once
        user_batches = {}
        for pid in episode_ps:
            p_df = df[df['ID'] == pid].reset_index(drop=True)
            sup_loader, _ = build_support_query(
                p_df, splits[pid]['train'], [],
                ar_or_va=label_type,
                window_size=cfg['window_size'], stride=cfg['stride'],
                feature_cols=cfg['feature_cols'],
                balanced_k_per_class=balanced_k_per_class)
            sb = _materialize_support(sup_loader, device)
            if sb is not None:
                user_batches[pid] = sb
        if not user_batches:
            continue

        # Single backbone clone, adapted JOINTLY across the episode's users
        episode_base = copy.deepcopy(base).to(device)
        head_clones  = {pid: copy.deepcopy(heads[pid]).to(device)
                        for pid in user_batches}

        head_clones = _adapt_episode_joint(
            episode_base, head_clones, user_batches, label_type,
            inner_steps, inner_lr, device,
            l2_shared=l2_shared, l2_task=l2_task, use_uw=use_uw)

        # EMA-style update on persistent heads (1.0 = full replace)
        with torch.no_grad():
            for pid in user_batches:
                for p_persistent, p_adapted in zip(heads[pid].parameters(),
                                                    head_clones[pid].parameters()):
                    p_persistent.data.add_(meta_head_lr * (p_adapted.data - p_persistent.data))

        # Outer Reptile update toward the jointly-adapted backbone
        reptile_outer_update(base, [episode_base], meta_lr)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{label_type.upper()}] Reptile-MT-joint step {step+1}/{meta_steps}")

    tag = 'reptile_mt_joint'
    torch.save(base.state_dict(),
               os.path.join(output_dir, f'{tag}_base_{label_type}.pth'))
    return base


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    args = parse_args()
    experiment_t0 = time.time()

    df, cfg = load_dataset(args.dataset, mode='mtml')
    splits   = cfg['splits']
    prefix   = cfg['results_prefix']
    train_ps = cfg['train_participants']
    test_ps  = cfg['test_participants']

    suffix = '_uw' if args.uw else ''
    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML',
                              f'{prefix}_reptile_mt_joint{suffix}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"K_PER_CLASS: {K_PER_CLASS} | UW: {args.uw}")

    hp_ar = _get_reptile_hparams(cfg, 'ar')
    hp_va = _get_reptile_hparams(cfg, 'va')
    print(f"\n  [AR] Hyperparameters: {hp_ar}")
    print(f"  [VA] Hyperparameters: {hp_va}")

    sign_flip_diag = {}
    results_ar, results_va = [], []

    for lt in ['ar', 'va']:
        hp = hp_ar if lt == 'ar' else hp_va

        print(f"\n{'='*60}\nREPTILE-MT-JOINT META-TRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base = _reptile_train(lt, df, splits, train_ps, cfg, device, output_dir,
                              meta_lr=hp['meta_lr'],
                              meta_head_lr=hp['meta_head_lr'],
                              inner_lr=hp['inner_lr'],
                              l2_shared=hp['l2_shared'],
                              l2_task=hp['l2_task'],
                              inner_steps=hp['inner_steps'],
                              episode_size=hp['episode_size'],
                              meta_steps=hp['meta_steps'],
                              balanced_k_per_class=K_PER_CLASS,
                              use_uw=args.uw)

        print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()}\n{'='*60}")
        results = []
        for uid in sorted(test_ps):
            if uid not in splits:
                continue
            t_df = df[df['ID'] == uid].reset_index(drop=True)
            head = TaskHead().to(device)
            r = evaluate_test_user(
                base, head, t_df, splits, uid, lt, device,
                inner_steps=hp['inner_steps'], inner_lr=hp['inner_lr'],
                l2_shared=hp['l2_shared'], l2_task=hp['l2_task'],
                window_size=cfg['window_size'], stride=cfg['stride'],
                feature_cols=cfg['feature_cols'],
                balanced_k_per_class=K_PER_CLASS)
            if r is not None:
                results.append(r)
                print(f"  P{uid}: {lt.upper()} Acc={r[f'{lt}_acc']:.4f} "
                      f"F1={r[f'{lt}_f1']:.4f}")

        sign_flip_diag[lt] = _check_sign_flip(results, lt)
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
        'variant': 'reptile_mt_joint', 'uw': args.uw,
        'train_participants': train_ps, 'test_participants': test_ps,
        'best_hyperparameters': {'AR': hp_ar, 'VA': hp_va},
        'k_per_class': K_PER_CLASS,
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
        'sign_flip_diagnostic_ar': sign_flip_diag.get('ar', {}),
        'sign_flip_diagnostic_va': sign_flip_diag.get('va', {}),
    }
    with open(os.path.join(output_dir, 'reptile_mt_joint_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n{'='*60}\nSIGN-FLIP SUMMARY\n{'='*60}")
    for lt in ['ar', 'va']:
        s = sign_flip_diag.get(lt, {}).get('summary', {})
        if s:
            print(f"  [{lt.upper()}] mean AUC normal={s['mean_auc_normal']:.4f}, "
                  f"flipped={s['mean_auc_flipped']:.4f}, "
                  f"flipped {s['n_flipped']}/{s['n_total']}")
            print(f"         {s['verdict']}")

    print(f"\nAll results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
