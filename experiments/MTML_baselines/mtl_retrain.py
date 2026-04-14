"""
MTL Retrain — For each test participant, train a fresh MTLModel from scratch
on all train participants + that one test participant (using train videos).
Evaluate the test participant using their task-specific head on held-out test videos.

Usage
-----
    python mtl_retrain.py                  # runs on VREED (default)
    python mtl_retrain.py --dataset dssn_eq
"""
import argparse
import os, sys, time, gc
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, EPOCHS, MAX_NORM, N_FOLDS,
                    MTL_SHARED_LR,
                    L2_SHARED, L2_TASK, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from data import create_sliding_windows, make_mtl_loader
from dataset_configs.loader import load_dataset
from models import MTLModel
from utils import (set_all_seeds, compute_metrics_from_cm, make_kfolds,
                   aggregate_mtml_results, compute_per_participant_stds,
                   print_determinism_summary)


def parse_args():
    p = argparse.ArgumentParser(description='MTL Retrain')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, df, cfg, device, output_dir):
    """K-fold CV on train participants to find best LR.
    
    Mirrors the final training approach: for each fold, train an MTL model
    on tr_ps + val_ps jointly (using train videos), then evaluate val
    participants via their own task heads on their held-out test videos.
    """
    splits   = cfg['splits']
    train_ps = cfg['train_participants']
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] MTL-Retrain"
          f"  ({cfg['results_prefix']})\n{'='*60}")
    print(f"  L2: Shared={L2_SHARED}, Task={L2_TASK}")

    results = []
    train_folds = make_kfolds(train_ps, seed=SEED)

    for lr in [MTL_SHARED_LR]:
        fold_f1s = []
        print(f"\nTesting LR={lr}")

        for fold_i in range(N_FOLDS):
            print(f"  Processing Fold {fold_i + 1}/{N_FOLDS}")
            val_ps = train_folds[fold_i]
            tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]

            # Build combined data dict: tr_ps then val_ps, all using train videos
            all_fold_ps = sorted(tr_ps) + sorted(val_ps)
            all_data = {}
            pid_to_task = {}
            for idx, pid in enumerate(all_fold_ps):
                p_df = df[df['ID'] == pid].reset_index(drop=True)
                all_data[idx] = p_df[p_df['Trial'].isin(
                    splits[pid]['train'])].reset_index(drop=True)
                pid_to_task[pid] = idx

            loader, _, _ = make_mtl_loader(
                all_data, cfg['window_size'], cfg['stride'],
                label_type=label_type, batch_size=len(all_data),
                seed=SEED, feature_cols=cfg['feature_cols'])

            model = MTLModel(len(all_data),
                             input_dim=cfg['input_dim']).to(device)
            opt   = optim.Adam(model.parameters(), lr=lr)
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')

            for ep in range(EPOCHS):
                model.train()
                run = 0.0
                for batch in loader:
                    X_b, y_b, tids, _ = [b.to(device) for b in batch]
                    opt.zero_grad(set_to_none=True)
                    loss_vec = loss_fn(model(X_b, tids), y_b).squeeze(-1)
                    l2 = model.compute_l2(L2_SHARED, L2_TASK)
                    total = loss_vec.mean() + l2
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                    opt.step()
                    run += total.item()
                sched.step(run / len(loader))

            # Evaluate val participants using their own heads on TEST videos
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for pid in sorted(val_ps):
                    task_idx = pid_to_task[pid]
                    test_df = df[(df['ID'] == pid) & df['Trial'].isin(
                        splits[pid]['test'])].reset_index(drop=True)
                    if len(test_df) == 0:
                        continue
                    X_v, y_ar_v, y_va_v, _, _ = create_sliding_windows(
                        test_df, cfg['window_size'], cfg['stride'],
                        task_id=task_idx, feature_cols=cfg['feature_cols'])
                    if len(X_v) == 0:
                        continue
                    y_v = y_ar_v if label_type == 'ar' else y_va_v
                    X_t = torch.tensor(X_v, dtype=torch.float32).to(device)
                    tids = torch.full((len(X_v),), task_idx, dtype=torch.long).to(device)
                    probs = torch.sigmoid(model(X_t, tids)).cpu().numpy().flatten()
                    all_preds.extend((probs > 0.5).astype(int))
                    all_labels.extend(y_v.astype(int))

            if len(all_labels) > 0:
                f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            else:
                f1 = 0.0
            fold_f1s.append(f1)
            print(f"  Fold {fold_i + 1}: Val F1 = {f1:.4f}")

            del model, loader
            torch.cuda.empty_cache()
            gc.collect()

        avg = np.mean(fold_f1s)
        results.append({'lr': lr, 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
        print(f"  Average F1: {avg:.4f}")

    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type}_tuning_results_mtl.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr']


# =============================
# TRAINING
# =============================
def _train_mtl(label_type, lr, data_dict, cfg, device, output_dir, ckpt_tag):
    """Train MTLModel on the given participants jointly."""
    num_tasks = len(data_dict)
    loader, _, _ = make_mtl_loader(
        data_dict, cfg['window_size'], cfg['stride'],
        label_type=label_type, batch_size=num_tasks,
        seed=SEED, feature_cols=cfg['feature_cols'])

    model   = MTLModel(num_tasks, input_dim=cfg['input_dim']).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    ckpt = os.path.join(output_dir, f'mtl_best_{label_type}_{ckpt_tag}.pt')

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in loader:
            X_b, y_b, tids, _ = [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss_vec = loss_fn(model(X_b, tids), y_b).squeeze(-1)
            l2 = model.compute_l2(L2_SHARED, L2_TASK)
            total = loss_vec.mean() + l2
            if torch.isnan(total):
                raise ValueError(f"NaN at epoch {epoch+1} [{label_type.upper()}]")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
            running += total.item()

        avg = running / len(loader)
        sched.step(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    [{label_type.upper()}] Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    # Clean up checkpoint
    if os.path.exists(ckpt):
        os.remove(ckpt)
    return model


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

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_mtl_retrain')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"Train: {train_ps}\nTest:  {test_ps}")
    print(f"\nL2: Shared={L2_SHARED}, Task={L2_TASK}")

    # Hyperparameter tuning on train participants only
    # best_lr_ar = hyperparameter_tuning('ar', df, cfg, device, output_dir)
    # best_lr_va = hyperparameter_tuning('va', df, cfg, device, output_dir)
    best_lr_ar = best_lr_va = MTL_SHARED_LR

    # Build base train data (train participants' train videos)
    base_train_data = {}
    for idx, pid in enumerate(sorted(train_ps)):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        base_train_data[idx] = p_df[p_df['Trial'].isin(
            splits[pid]['train'])].reset_index(drop=True)
    num_base = len(base_train_data)

    # ===== AR PASS =====
    print(f"\n{'='*60}\nTRAINING + EVALUATION AR (per test participant)\n{'='*60}")
    set_all_seeds(SEED)
    results_ar = []

    for uid in sorted(test_ps):
        if uid not in splits:
            continue
        print(f"\n--- Test Participant {uid} [AR] ---")

        data_dict = dict(base_train_data)
        test_task_idx = num_base
        p_df = df[df['ID'] == uid].reset_index(drop=True)
        data_dict[test_task_idx] = p_df[p_df['Trial'].isin(
            splits[uid]['train'])].reset_index(drop=True)

        model_ar = _train_mtl('ar', best_lr_ar, data_dict, cfg, device,
                              output_dir, ckpt_tag=f'p{uid}')
        model_ar.eval()
        with torch.no_grad():
            test_df = p_df[p_df['Trial'].isin(splits[uid]['test'])].reset_index(drop=True)
            X, y_ar, y_va, _, _ = create_sliding_windows(
                test_df, cfg['window_size'], cfg['stride'],
                task_id=test_task_idx, feature_cols=cfg['feature_cols'])
            if len(X) > 0:
                X_t  = torch.tensor(X, dtype=torch.float32).to(device)
                tids = torch.full((len(X),), test_task_idx, dtype=torch.long).to(device)
                probs_ar = torch.sigmoid(model_ar(X_t, tids)).cpu().numpy().flatten()
                preds_ar = (probs_ar > 0.5).astype(int)
                labels_ar = y_ar.astype(int)
                cm_ar = confusion_matrix(labels_ar, preds_ar, labels=[0, 1])
                acc_ar, prec_ar, rec_ar, f1_ar = compute_metrics_from_cm(cm_ar)
                results_ar.append({
                    'participant_id': uid, 'cm': cm_ar,
                    'ar_acc': acc_ar, 'ar_precision': prec_ar,
                    'ar_recall': rec_ar, 'ar_f1': f1_ar,
                    'y_true_ar': labels_ar, 'y_pred_ar': preds_ar,
                    'y_pred_probs_ar': probs_ar,
                })
                print(f"  AR Acc={acc_ar:.4f} F1={f1_ar:.4f}")

        del model_ar
        torch.cuda.empty_cache()
        gc.collect()

    # ===== VA PASS =====
    print(f"\n{'='*60}\nTRAINING + EVALUATION VA (per test participant)\n{'='*60}")
    set_all_seeds(SEED)
    results_va = []

    for uid in sorted(test_ps):
        if uid not in splits:
            continue
        print(f"\n--- Test Participant {uid} [VA] ---")

        data_dict = dict(base_train_data)
        test_task_idx = num_base
        p_df = df[df['ID'] == uid].reset_index(drop=True)
        data_dict[test_task_idx] = p_df[p_df['Trial'].isin(
            splits[uid]['train'])].reset_index(drop=True)

        model_va = _train_mtl('va', best_lr_va, data_dict, cfg, device,
                              output_dir, ckpt_tag=f'p{uid}')
        model_va.eval()
        with torch.no_grad():
            test_df = p_df[p_df['Trial'].isin(splits[uid]['test'])].reset_index(drop=True)
            X, y_ar, y_va, _, _ = create_sliding_windows(
                test_df, cfg['window_size'], cfg['stride'],
                task_id=test_task_idx, feature_cols=cfg['feature_cols'])
            if len(X) > 0:
                X_t  = torch.tensor(X, dtype=torch.float32).to(device)
                tids = torch.full((len(X),), test_task_idx, dtype=torch.long).to(device)
                probs_va = torch.sigmoid(model_va(X_t, tids)).cpu().numpy().flatten()
                preds_va = (probs_va > 0.5).astype(int)
                labels_va = y_va.astype(int)
                cm_va = confusion_matrix(labels_va, preds_va, labels=[0, 1])
                acc_va, prec_va, rec_va, f1_va = compute_metrics_from_cm(cm_va)
                results_va.append({
                    'participant_id': uid, 'cm': cm_va,
                    'va_acc': acc_va, 'va_precision': prec_va,
                    'va_recall': rec_va, 'va_f1': f1_va,
                    'y_true_va': labels_va, 'y_pred_va': preds_va,
                    'y_pred_probs_va': probs_va,
                })
                print(f"  VA Acc={acc_va:.4f} F1={f1_va:.4f}")

        del model_va
        torch.cuda.empty_cache()
        gc.collect()

    # Aggregate
    agg = aggregate_mtml_results(results_ar, results_va)
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump({'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
                     'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']}}, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')

    final = {
        'train_participants': train_ps, 'test_participants': test_ps,
        'best_hyperparameters': {'AR': {'lr': best_lr_ar}, 'VA': {'lr': best_lr_va}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'mtl_retrain_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")