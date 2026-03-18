"""
MTL Retrain — Pre-train MTL on train participants, then retrain fresh
task heads per test participant while keeping the shared backbone frozen.

Usage
-----
    python mtl_retrain.py                  # runs on VREED (default)
    python mtl_retrain.py --dataset dssn_eq
"""
import argparse
import os, sys, time, copy
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, EPOCHS, MAX_NORM, FT_EPOCHS,
                    MTL_SHARED_LR, MTL_TASK_LR,
                    L2_SHARED, L2_TASK, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from data import make_mtl_loader, build_support_query
from dataset_configs.loader import load_dataset
from models import MTLModel, TaskHead, BaseFeatureExtractor
from utils import (set_all_seeds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import evaluate_test_user


def parse_args():
    p = argparse.ArgumentParser(description='MTL Retrain')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def _pretrain_mtl(label_type, train_data_dict, cfg, device, output_dir):
    """Pre-train MTL model on train participants only."""
    num_train = len(train_data_dict)
    loader, _, _ = make_mtl_loader(
        train_data_dict, cfg['window_size'], cfg['stride'],
        label_type=label_type, batch_size=cfg['mtl_batch'], seed=SEED,
        feature_cols=cfg['feature_cols'])

    model   = MTLModel(num_train, input_dim=cfg['input_dim']).to(device)
    opt     = optim.Adam([
        {'params': model.shared_parameters(),        'lr': MTL_SHARED_LR},
        {'params': model.task_specific_parameters(), 'lr': MTL_TASK_LR},
    ])
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    ckpt = os.path.join(output_dir, f'pretrain_{label_type}.pt')

    for epoch in range(EPOCHS):
        model.train(); running = 0.0
        for batch in loader:
            X_b, y_b, task_ids, _ = [b.to(device, non_blocking=True) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_b, task_ids), y_b)
            total = loss + model.compute_l2(L2_SHARED, L2_TASK)
            if torch.isnan(total): raise ValueError(f"NaN ep {epoch+1} [{label_type}]")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); running += total.item()
        avg = running / len(loader)
        sched.step(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label_type.upper()}] Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg; torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))

    # Extract shared backbone as BaseFeatureExtractor
    base = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    base.conv1.load_state_dict(model.conv1.state_dict())
    base.bn1.load_state_dict(model.bn1.state_dict())
    base.conv2.load_state_dict(model.conv2.state_dict())
    base.bn2.load_state_dict(model.bn2.state_dict())
    base.lstm.load_state_dict(model.lstm.state_dict())
    return base


def _retrain_head_and_eval(base_model, test_df, splits, uid, label_type,
                           cfg, device, inner_steps, inner_lr):
    """Freeze backbone, train a new TaskHead, evaluate."""
    sup_loader, q_loader = build_support_query(
        test_df, splits[uid]['train'], splits[uid]['test'], label_type,
        window_size=cfg['window_size'], stride=cfg['stride'],
        feature_cols=cfg['feature_cols'])
    if len(q_loader.dataset) == 0:
        return None

    frozen_base = copy.deepcopy(base_model).to(device)
    frozen_base.eval()
    for p in frozen_base.parameters():
        p.requires_grad = False

    head    = TaskHead().to(device)
    opt     = optim.Adam(head.parameters(), lr=inner_lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(inner_steps):
        head.train(); ep_loss = 0.0; nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                feat = frozen_base(Xb)
            loss = loss_fn(head(feat), yb)
            l2 = L2_TASK * sum(p.norm(2)**2 for p in head.parameters() if p.requires_grad)
            total = loss + l2
            total.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), MAX_NORM)
            opt.step(); ep_loss += total.item(); nb += 1
        if nb > 0: sched.step(ep_loss / nb)

    frozen_base.eval(); head.eval()
    from sklearn.metrics import confusion_matrix
    from utils import compute_metrics_from_cm
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in q_loader:
            out = head(frozen_base(Xb.to(device, non_blocking=True)))
            probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
            labels.extend(yb.numpy().flatten())
    y_true = np.array(labels).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)
    p = label_type
    return {
        'participant_id': uid, 'cm': cm,
        f'{p}_acc': acc, f'{p}_precision': prec,
        f'{p}_recall': rec, f'{p}_f1': f1,
        f'y_true_{p}': y_true, f'y_pred_{p}': y_pred,
        f'y_pred_probs_{p}': y_prob,
    }


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
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"Train: {train_ps}\nTest:  {test_ps}")

    # Build train data dict keyed by task_idx
    train_data = {}
    for idx, pid in enumerate(sorted(train_ps)):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        train_data[idx] = p_df[p_df['Trial'].isin(splits[pid]['train'])].reset_index(drop=True)

    for lt in ['ar', 'va']:
        print(f"\n{'='*60}\nPRETRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base = _pretrain_mtl(lt, train_data, cfg, device, output_dir)
        torch.save(base.state_dict(), os.path.join(output_dir, f'base_{lt}.pth'))

        results = []
        for uid in sorted(test_ps):
            if uid not in splits: continue
            t_df = df[df['ID'] == uid].reset_index(drop=True)
            r = _retrain_head_and_eval(base, t_df, splits, uid, lt, cfg, device,
                                       inner_steps=FT_EPOCHS, inner_lr=MTL_TASK_LR)
            if r is not None:
                results.append(r)
                print(f"  P{uid}: {lt.upper()} Acc={r[f'{lt}_acc']:.4f} F1={r[f'{lt}_f1']:.4f}")

        if lt == 'ar':
            results_ar = results
        else:
            results_va = results

    agg = aggregate_mtml_results(results_ar, results_va)
    roc_data = {'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
                'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']}}
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

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
    with open(os.path.join(output_dir, 'mtl_retrain_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
