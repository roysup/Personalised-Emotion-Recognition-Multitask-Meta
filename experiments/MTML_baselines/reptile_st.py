"""
Reptile Single-Task (ST)
Shared backbone + per-participant heads during meta-training.
Samples 1 participant per meta-step.
Reptile outer-loop updates backbone only.
Heads are kept per-participant and updated individually.
At test time: fresh head per test participant, adapt both.

Supports optional balanced k-shot support sampling via K_PER_CLASS:
  None  = use all available support windows (default)
  int   = subsample k windows per class (e.g. 20 → 40 total)

Usage
-----
    python reptile_st.py                  # runs on VREED (default)
    python reptile_st.py --dataset dssn_eq
"""
import argparse
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, META_STEPS, META_LR,
                    INNER_STEPS, INNER_LR,
                    N_FOLDS, L2_SHARED, L2_TASK, K_PER_CLASS,
                    RESULTS_DIR)
import numpy as np
import pickle
import torch
from sklearn.metrics import f1_score
from data import build_support_query
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, TaskHead
from utils import (set_all_seeds, make_kfolds,
                   aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import adapt_inner_loop, evaluate_test_user, reptile_outer_update


def parse_args():
    p = argparse.ArgumentParser(description='Reptile ST')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
                   meta_lr, inner_lr, l2_shared, l2_task,
                   balanced_k_per_class=None):
    """Reptile-ST: per-participant heads, 1 participant per step, backbone-only outer update."""
    base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    heads = {pid: TaskHead().to(device) for pid in train_ps}
    rng   = np.random.default_rng(SEED)

    if balanced_k_per_class is not None:
        print(f"  [{label_type.upper()}] Balanced k-shot: {balanced_k_per_class} per class")
    else:
        print(f"  [{label_type.upper()}] Using all support windows")

    for step in range(META_STEPS):
        pid = int(rng.choice(train_ps))
        p_df = df[df['ID'] == pid].reset_index(drop=True)

        sup_loader, _ = build_support_query(
            p_df, splits[pid]['train'], [],
            ar_or_va=label_type,
            window_size=cfg['window_size'], stride=cfg['stride'],
            feature_cols=cfg['feature_cols'],
            balanced_k_per_class=balanced_k_per_class)

        adapted_base, adapted_head = adapt_inner_loop(
            base, heads[pid], sup_loader, label_type,
            INNER_STEPS, inner_lr, device,
            l2_shared=l2_shared, l2_task=l2_task)

        # Reptile outer update — backbone only
        reptile_outer_update(base, [adapted_base], meta_lr)

        # Keep updated head per-participant
        heads[pid] = adapted_head

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{label_type.upper()}] Reptile-ST step {step+1}/{META_STEPS}")

    torch.save(base.state_dict(),
               os.path.join(output_dir, f'reptile_st_base_{label_type}.pth'))
    return base


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, df, splits, train_ps, cfg, device,
                          output_dir, balanced_k_per_class=None):
    """K-fold CV on train participants to validate Reptile-ST hyperparameters."""
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Reptile-ST"
          f"  ({cfg['results_prefix']})\n{'='*60}")
    print(f"  META_LR={META_LR}, INNER_LR={INNER_LR}, "
          f"INNER_STEPS={INNER_STEPS}")
    print(f"  L2: Shared={L2_SHARED}, Task={L2_TASK}")
    print(f"  K_PER_CLASS={balanced_k_per_class}")

    results = []
    train_folds = make_kfolds(train_ps, seed=SEED)

    for meta_lr in [META_LR]:
        for inner_lr in [INNER_LR]:
            for l2_s in [L2_SHARED]:
                for l2_t in [L2_TASK]:
                    fold_f1s = []
                    for fold_i in range(N_FOLDS):
                        val_ps = train_folds[fold_i]
                        tr_ps  = [p for j, f in enumerate(train_folds)
                                  if j != fold_i for p in f]

                        # Meta-train on fold's train participants
                        set_all_seeds(SEED)
                        base  = BaseFeatureExtractor(
                            input_dim=cfg['input_dim']).to(device)
                        heads = {pid: TaskHead().to(device) for pid in tr_ps}
                        rng   = np.random.default_rng(SEED)

                        for step in range(META_STEPS):
                            pid = int(rng.choice(tr_ps))
                            p_df = df[df['ID'] == pid].reset_index(drop=True)
                            sup_loader, _ = build_support_query(
                                p_df, splits[pid]['train'], [],
                                ar_or_va=label_type,
                                window_size=cfg['window_size'],
                                stride=cfg['stride'],
                                feature_cols=cfg['feature_cols'],
                                balanced_k_per_class=balanced_k_per_class)
                            adapted_base, adapted_head = adapt_inner_loop(
                                base, heads[pid], sup_loader, label_type,
                                INNER_STEPS, inner_lr, device,
                                l2_shared=l2_s, l2_task=l2_t)
                            reptile_outer_update(base, [adapted_base], meta_lr)
                            heads[pid] = adapted_head

                        # Adapt + evaluate on fold's val participants
                        val_f1s = []
                        for uid in sorted(val_ps):
                            if uid not in splits:
                                continue
                            t_df = df[df['ID'] == uid].reset_index(drop=True)
                            head = TaskHead().to(device)
                            r = evaluate_test_user(
                                base, head, t_df, splits, uid, label_type,
                                device,
                                inner_steps=INNER_STEPS, inner_lr=inner_lr,
                                l2_shared=l2_s, l2_task=l2_t,
                                window_size=cfg['window_size'],
                                stride=cfg['stride'],
                                feature_cols=cfg['feature_cols'],
                                balanced_k_per_class=balanced_k_per_class)
                            if r is not None:
                                y_true = r[f'y_true_{label_type}']
                                y_pred = r[f'y_pred_{label_type}']
                                val_f1s.append(f1_score(
                                    y_true, y_pred,
                                    average='macro', zero_division=0))

                        if val_f1s:
                            fold_f1s.append(np.mean(val_f1s))
                            print(f"  Fold {fold_i + 1}/{N_FOLDS}: "
                                  f"Val F1 = {fold_f1s[-1]:.4f}")

                    if not fold_f1s:
                        continue
                    avg = np.mean(fold_f1s)
                    results.append({
                        'meta_lr': meta_lr, 'inner_lr': inner_lr,
                        'l2_shared': l2_s, 'l2_task': l2_t,
                        'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
                    print(f"  Average F1: {avg:.4f}")

    if not results:
        return META_LR, INNER_LR, L2_SHARED, L2_TASK
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir,
              f'{label_type}_tuning_results_reptile_st.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['meta_lr'], best['inner_lr'], best['l2_shared'], best['l2_task']


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

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_reptile_st')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"K_PER_CLASS: {K_PER_CLASS}")

    # Hyperparameter tuning on train participants only
    # best_meta_lr_ar, best_inner_lr_ar, best_l2s_ar, best_l2t_ar = \
    #     hyperparameter_tuning('ar', df, splits, train_ps, cfg, device, output_dir,
    #                           balanced_k_per_class=K_PER_CLASS)
    # best_meta_lr_va, best_inner_lr_va, best_l2s_va, best_l2t_va = \
    #     hyperparameter_tuning('va', df, splits, train_ps, cfg, device, output_dir,
    #                           balanced_k_per_class=K_PER_CLASS)
    
    best_meta_lr_ar = best_meta_lr_va = META_LR
    best_inner_lr_ar = best_inner_lr_va = INNER_LR
    best_l2s_ar = best_l2s_va = L2_SHARED
    best_l2t_ar = best_l2t_va = L2_TASK

    for lt in ['ar', 'va']:
        if lt == 'ar':
            meta_lr, inner_lr = best_meta_lr_ar, best_inner_lr_ar
            l2_s, l2_t = best_l2s_ar, best_l2t_ar
        else:
            meta_lr, inner_lr = best_meta_lr_va, best_inner_lr_va
            l2_s, l2_t = best_l2s_va, best_l2t_va

        print(f"\n{'='*60}\nREPTILE-ST META-TRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base = _reptile_train(lt, df, splits, train_ps, cfg, device, output_dir,
                              meta_lr=meta_lr, inner_lr=inner_lr,
                              l2_shared=l2_s, l2_task=l2_t,
                              balanced_k_per_class=K_PER_CLASS)

        print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()}\n{'='*60}")
        results = []
        for uid in sorted(test_ps):
            if uid not in splits:
                continue
            t_df = df[df['ID'] == uid].reset_index(drop=True)
            head = TaskHead().to(device)  # fresh head per test participant
            r = evaluate_test_user(
                base, head, t_df, splits, uid, lt, device,
                inner_steps=INNER_STEPS, inner_lr=inner_lr,
                l2_shared=l2_s, l2_task=l2_t,
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
        'best_hyperparameters': {
            'AR': {'meta_lr': best_meta_lr_ar, 'inner_lr': best_inner_lr_ar,
                   'l2_shared': best_l2s_ar, 'l2_task': best_l2t_ar},
            'VA': {'meta_lr': best_meta_lr_va, 'inner_lr': best_inner_lr_va,
                   'l2_shared': best_l2s_va, 'l2_task': best_l2t_va}},
        'k_per_class': K_PER_CLASS,
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'reptile_st_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")