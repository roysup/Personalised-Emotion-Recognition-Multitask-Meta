"""
Reptile Multi-Task (MT)
Shared backbone + per-participant heads during meta-training.
Reptile outer-loop updates backbone only.
At test time: fresh head per test participant, adapt both.

Usage
-----
    python reptile_mt.py                  # runs on VREED (default)
    python reptile_mt.py --dataset dssn_eq
"""
import argparse
import os, sys, time, copy, random
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, MAX_NORM, META_STEPS, META_LR,
                    INNER_STEPS, INNER_LR, EPISODE_SIZE,
                    L2_SHARED, L2_TASK, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from data import build_support_query
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, TaskHead
from utils import (set_all_seeds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import evaluate_test_user, reptile_outer_update


def parse_args():
    p = argparse.ArgumentParser(description='Reptile MT')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def _inner_adapt_mt(base, head, sup_loader, label_type, device):
    """Inner-loop adaptation: adapt both backbone copy and head copy."""
    ab = copy.deepcopy(base).to(device)
    ah = copy.deepcopy(head).to(device)
    ab.train(); ah.train()

    sp = list(ab.parameters())
    tp = list(ah.parameters())
    opt     = optim.Adam(sp + tp, lr=INNER_LR)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(INNER_STEPS):
        ep_loss = 0.0; nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(ah(ab(Xb)), yb)
            l2 = (L2_SHARED * sum(p.norm(2)**2 for p in sp if p.requires_grad) +
                  L2_TASK * sum(p.norm(2)**2 for p in tp if p.requires_grad))
            total = loss + l2
            if not torch.isnan(total):
                total.backward()
                torch.nn.utils.clip_grad_norm_(sp + tp, MAX_NORM)
                opt.step()
            ep_loss += total.item(); nb += 1
        if nb > 0: sched.step(ep_loss / nb)
    return ab


def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir):
    """Reptile-MT meta-training: outer update on backbone only."""
    base = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    # Per-participant heads (keyed by participant id)
    heads = {pid: TaskHead().to(device) for pid in train_ps}
    rng = random.Random(SEED)

    for step in range(META_STEPS):
        episode_ps = rng.sample(train_ps, min(EPISODE_SIZE, len(train_ps)))
        adapted_bases = []

        for pid in episode_ps:
            p_df = df[df['ID'] == pid].reset_index(drop=True)
            sup_loader, _ = build_support_query(
                p_df, splits[pid]['train'], [],
                ar_or_va=label_type,
                window_size=cfg['window_size'], stride=cfg['stride'],
                feature_cols=cfg['feature_cols'])

            ab = _inner_adapt_mt(base, heads[pid], sup_loader, label_type, device)
            adapted_bases.append(ab)

        # Outer update on backbone only
        reptile_outer_update(base, adapted_bases, META_LR)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{label_type.upper()}] Reptile-MT step {step+1}/{META_STEPS}")

    torch.save(base.state_dict(), os.path.join(output_dir, f'reptile_mt_base_{label_type}.pth'))
    return base


if __name__ == '__main__':
    args = parse_args()
    experiment_t0 = time.time()

    df, cfg = load_dataset(args.dataset, mode='mtml')
    splits   = cfg['splits']
    prefix   = cfg['results_prefix']
    train_ps = cfg['train_participants']
    test_ps  = cfg['test_participants']

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_reptile_mt')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")

    for lt in ['ar', 'va']:
        print(f"\n{'='*60}\nREPTILE-MT META-TRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base = _reptile_train(lt, df, splits, train_ps, cfg, device, output_dir)

        print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()}\n{'='*60}")
        results = []
        for uid in sorted(test_ps):
            if uid not in splits: continue
            t_df = df[df['ID'] == uid].reset_index(drop=True)
            head = TaskHead().to(device)  # fresh head per test participant
            r = evaluate_test_user(
                base, head, t_df, splits, uid, lt, device,
                inner_steps=INNER_STEPS, inner_lr=INNER_LR,
                l2_shared=L2_SHARED, l2_task=L2_TASK,
                window_size=cfg['window_size'], stride=cfg['stride'],
                feature_cols=cfg['feature_cols'])
            if r is not None:
                results.append(r)
                print(f"  P{uid}: {lt.upper()} Acc={r[f'{lt}_acc']:.4f} F1={r[f'{lt}_f1']:.4f}")

        if lt == 'ar': results_ar = results
        else: results_va = results

    agg = aggregate_mtml_results(results_ar, results_va)
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump({'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
                     'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']}}, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')
    final = {
        'train_participants': train_ps, 'test_participants': test_ps,
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'reptile_mt_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
