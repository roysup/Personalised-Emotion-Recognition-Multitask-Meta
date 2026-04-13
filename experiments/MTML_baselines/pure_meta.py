"""
Pure Meta-Learning (Reptile-style)
Single monolithic model (no backbone/head split). Samples 1 participant per
meta-step. L2 applied uniformly to all parameters. At test time: deep-copy
model, adapt per test participant via inner-loop, then evaluate.

Supports optional balanced k-shot support sampling via K_PER_CLASS:
  None  = use all available support windows (default)
  int   = subsample k windows per class (e.g. 20 → 40 total)

Usage
-----
    python pure_meta.py                  # runs on VREED (default)
    python pure_meta.py --dataset dssn_eq
"""
import argparse
import os, sys, time, copy
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, MAX_NORM, META_STEPS, META_LR,
                    INNER_STEPS, INNER_LR,
                    N_FOLDS, L2_TASK, K_PER_CLASS, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score
from data import build_support_query
from dataset_configs.loader import load_dataset
from models import SingleTaskModel
from utils import (set_all_seeds, compute_metrics_from_cm,
                   aggregate_mtml_results, make_kfolds,
                   compute_per_participant_stds,
                   print_determinism_summary)


def parse_args():
    p = argparse.ArgumentParser(description='Pure Meta-Learning')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


# =============================
# INNER-LOOP ADAPTATION (single model, uniform L2)
# =============================
def _adapt_single_model(model, sup_loader, inner_steps, inner_lr,
                        l2_lambda, device):
    """Adapt a deep copy of the full model on support data."""
    adapted = copy.deepcopy(model).to(device)
    adapted.train()

    opt     = optim.Adam(adapted.parameters(), lr=inner_lr)
    sched   = ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(inner_steps):
        ep_loss = 0.0
        nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(adapted(Xb), yb)
            # Uniform L2 on ALL parameters
            #l2_reg = l2_lambda * sum(torch.sum(p ** 2) for p in adapted.parameters())
            l2_reg = l2_lambda * sum(torch.sum(p ** 2) for p in adapted.parameters() if p.ndim >= 2)
            total = loss + l2_reg
            if not torch.isnan(total):
                total.backward()
                torch.nn.utils.clip_grad_norm_(adapted.parameters(), MAX_NORM)
                opt.step()
            ep_loss += total.item()
            nb += 1
        if nb > 0:
            sched.step(ep_loss / nb)

    return adapted


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, df, splits, train_ps, cfg, device,
                          output_dir, balanced_k_per_class=None):
    """K-fold CV on train participants to validate meta-learning hyperparameters."""
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Pure Meta"
          f"  ({cfg['results_prefix']})\n{'='*60}")
    print(f"  META_LR={META_LR}, INNER_LR={INNER_LR}, "
          f"INNER_STEPS={INNER_STEPS}, L2={L2_TASK}")
    print(f"  K_PER_CLASS={balanced_k_per_class}")

    results = []
    train_folds = make_kfolds(train_ps, seed=SEED)

    for meta_lr in [META_LR]:
        for inner_lr in [INNER_LR]:
            for l2 in [L2_TASK]:
                fold_f1s = []
                for fold_i in range(N_FOLDS):
                    val_ps = train_folds[fold_i]
                    tr_ps  = [p for j, f in enumerate(train_folds)
                              if j != fold_i for p in f]

                    # Meta-train on fold's train participants
                    set_all_seeds(SEED)
                    model = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
                    rng = np.random.default_rng(SEED)

                    for step in range(META_STEPS):
                        uid = int(rng.choice(tr_ps))
                        p_df = df[df['ID'] == uid].reset_index(drop=True)
                        sup_loader, _ = build_support_query(
                            p_df, splits[uid]['train'], [],
                            ar_or_va=label_type,
                            window_size=cfg['window_size'],
                            stride=cfg['stride'],
                            feature_cols=cfg['feature_cols'],
                            balanced_k_per_class=balanced_k_per_class)
                        adapted = _adapt_single_model(
                            model, sup_loader, INNER_STEPS, inner_lr,
                            l2, device)
                        with torch.no_grad():
                            for p, p_new in zip(model.parameters(),
                                                adapted.parameters()):
                                p.data.add_(meta_lr * (p_new.data - p.data))

                    # Adapt + evaluate on fold's val participants
                    val_f1s = []
                    for uid in sorted(val_ps):
                        if uid not in splits:
                            continue
                        t_df = df[df['ID'] == uid].reset_index(drop=True)
                        sup_loader, q_loader = build_support_query(
                            t_df, splits[uid]['train'], splits[uid]['test'],
                            ar_or_va=label_type,
                            window_size=cfg['window_size'],
                            stride=cfg['stride'],
                            feature_cols=cfg['feature_cols'],
                            balanced_k_per_class=balanced_k_per_class)
                        if len(q_loader.dataset) == 0:
                            continue
                        adapted = _adapt_single_model(
                            model, sup_loader, INNER_STEPS, inner_lr,
                            l2, device)
                        adapted.eval()
                        probs, labels = [], []
                        with torch.no_grad():
                            for Xb, yb in q_loader:
                                probs.extend(torch.sigmoid(
                                    adapted(Xb.to(device))
                                ).cpu().numpy().flatten())
                                labels.extend(yb.numpy().flatten())
                        y_true = np.array(labels).astype(int)
                        y_pred = (np.array(probs) > 0.5).astype(int)
                        val_f1s.append(f1_score(y_true, y_pred,
                                                average='macro',
                                                zero_division=0))

                    if val_f1s:
                        fold_f1s.append(np.mean(val_f1s))
                        print(f"  Fold {fold_i + 1}/{N_FOLDS}: "
                              f"Val F1 = {fold_f1s[-1]:.4f}")

                if not fold_f1s:
                    continue
                avg = np.mean(fold_f1s)
                results.append({
                    'meta_lr': meta_lr, 'inner_lr': inner_lr, 'l2': l2,
                    'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
                print(f"  Average F1: {avg:.4f}")

    if not results:
        return META_LR, INNER_LR, L2_TASK
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir,
              f'{label_type}_tuning_results_pure_meta.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['meta_lr'], best['inner_lr'], best['l2']


# =============================
# META-TRAINING (1 participant per step, Reptile outer update)
# =============================
def _meta_train(label_type, df, splits, train_ps, cfg, device, output_dir,
                meta_lr, inner_lr, l2, balanced_k_per_class=None):
    """Reptile-style meta-training with 1 participant per step."""
    model = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
    rng = np.random.default_rng(SEED)

    if balanced_k_per_class is not None:
        print(f"  [{label_type.upper()}] Balanced k-shot: {balanced_k_per_class} per class")
    else:
        print(f"  [{label_type.upper()}] Using all support windows")

    for step in range(META_STEPS):
        uid = int(rng.choice(train_ps))
        p_df = df[df['ID'] == uid].reset_index(drop=True)

        sup_loader, _ = build_support_query(
            p_df, splits[uid]['train'], [],
            ar_or_va=label_type,
            window_size=cfg['window_size'], stride=cfg['stride'],
            feature_cols=cfg['feature_cols'],
            balanced_k_per_class=balanced_k_per_class)

        adapted = _adapt_single_model(
            model, sup_loader, INNER_STEPS, inner_lr, l2, device)

        # Reptile outer update
        with torch.no_grad():
            for p, p_new in zip(model.parameters(), adapted.parameters()):
                p.data.add_(meta_lr * (p_new.data - p.data))

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  [{label_type.upper()}] Meta-step {step+1}/{META_STEPS}")

    torch.save(model.state_dict(),
               os.path.join(output_dir, f'meta_model_{label_type}_final.pth'))
    return model


# =============================
# TEST USER EVALUATION
# =============================
def _evaluate_test_user(model, test_df, splits, uid, ar_or_va, cfg, device,
                        inner_lr, l2, balanced_k_per_class=None):
    """Adapt and evaluate on one test participant."""
    sup_loader, q_loader = build_support_query(
        test_df, splits[uid]['train'], splits[uid]['test'],
        ar_or_va=ar_or_va,
        window_size=cfg['window_size'], stride=cfg['stride'],
        feature_cols=cfg['feature_cols'],
        balanced_k_per_class=balanced_k_per_class)

    if len(q_loader.dataset) == 0:
        return None

    adapted = _adapt_single_model(
        model, sup_loader, INNER_STEPS, inner_lr, l2, device)

    adapted.eval()
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in q_loader:
            probs.extend(torch.sigmoid(adapted(Xb.to(device))).cpu().numpy().flatten())
            labels.extend(yb.numpy().flatten())

    y_true = np.array(labels).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)

    p = ar_or_va
    return {
        'participant_id': uid, 'cm': cm,
        f'{p}_acc': acc, f'{p}_precision': prec,
        f'{p}_recall': rec, f'{p}_f1': f1,
        f'y_true_{p}': y_true, f'y_pred_{p}': y_pred,
        f'y_pred_probs_{p}': y_prob,
    }


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

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_pure_meta')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"Train: {train_ps}\nTest:  {test_ps}")
    print(f"K_PER_CLASS: {K_PER_CLASS}")

    # Hyperparameter tuning on train participants only
    best_meta_lr_ar, best_inner_lr_ar, best_l2_ar = hyperparameter_tuning(
        'ar', df, splits, train_ps, cfg, device, output_dir,
        balanced_k_per_class=K_PER_CLASS)
    best_meta_lr_va, best_inner_lr_va, best_l2_va = hyperparameter_tuning(
        'va', df, splits, train_ps, cfg, device, output_dir,
        balanced_k_per_class=K_PER_CLASS)

    for lt in ['ar', 'va']:
        if lt == 'ar':
            meta_lr, inner_lr, l2 = best_meta_lr_ar, best_inner_lr_ar, best_l2_ar
        else:
            meta_lr, inner_lr, l2 = best_meta_lr_va, best_inner_lr_va, best_l2_va

        print(f"\n{'='*60}\nMETA-TRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        model = _meta_train(lt, df, splits, train_ps, cfg, device, output_dir,
                            meta_lr=meta_lr, inner_lr=inner_lr, l2=l2,
                            balanced_k_per_class=K_PER_CLASS)

        print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()} — TEST PARTICIPANTS\n{'='*60}")
        results = []
        for uid in sorted(test_ps):
            if uid not in splits:
                continue
            t_df = df[df['ID'] == uid].reset_index(drop=True)
            r = _evaluate_test_user(model, t_df, splits, uid, lt, cfg, device,
                                    inner_lr=inner_lr, l2=l2,
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
            'AR': {'meta_lr': best_meta_lr_ar, 'inner_lr': best_inner_lr_ar, 'l2': best_l2_ar},
            'VA': {'meta_lr': best_meta_lr_va, 'inner_lr': best_inner_lr_va, 'l2': best_l2_va}},
        'k_per_class': K_PER_CLASS,
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'pure_meta_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")