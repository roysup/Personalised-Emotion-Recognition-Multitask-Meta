"""
TAG — MTL training with per-batch inter-task affinity probing.

Trains the same MTL-HPS model as mtl_hps.py while collecting one affinity
matrix per training batch. Saves the mean matrix across all batches plus
per-participant affinity scores, alongside the standard MTL evaluation
artefacts so this is a complete experiment, not a probe-only side-run.

The probe reuses the main training optimizer with full save/restore around
each probe step, so the temporary Adam update inherits the accumulated
moments from training. Probes run BEFORE each training step, measuring
affinities at the current base parameters theta_t.

Outputs (results/{prefix}_TAG/{prefix}_tag_results/):
    {ar,va}_final_affinity_matrix.npy
    affinity_scores_per_participant.csv
    + per_participant_results.csv, ar_cm.png, va_cm.png, ar_roc.png, va_roc.png,
      tag_results.pkl, {prefix}_tag_misclassification_rates.csv

Usage
-----
    python tag_train.py                  # VREED (default)
    python tag_train.py --dataset dssn_eq
    python tag_train.py --dataset dssn_em
"""
import argparse
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from config import (SEED, EPOCHS, MAX_NORM,
                    MTL_SHARED_LR, MTL_TASK_LR,
                    L2_SHARED, L2_TASK, RESULTS_DIR)
from data import make_mtl_loader
from dataset_configs.loader import load_dataset
from models import MTLModel
from utils import set_all_seeds, aggregate_results
from training import save_all_results, evaluate_mtl_all
from tag import compute_inter_task_affinity, extract_per_participant_scores


def parse_args():
    p = argparse.ArgumentParser(description='TAG inter-task affinity analysis')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def _train_with_affinity(label_type, lr_shared, lr_task, l2_task,
                         train_data, cfg, device, output_dir):
    """Train one MTL model (AR or VA) while collecting per-batch affinity.

    The affinity probe reuses the main training optimizer (saving/restoring
    its state around each probe) so the temporary update inherits the
    accumulated Adam moments. Probes run BEFORE each training step.
    """
    num_tasks = cfg['num_tasks']

    loader, _, _ = make_mtl_loader(
        train_data, cfg['window_size'], cfg['stride'],
        label_type=label_type, batch_size=cfg['mtl_batch'], seed=SEED,
        feature_cols=cfg['feature_cols'])

    model = MTLModel(num_tasks, input_dim=cfg['input_dim']).to(device)
    shared_params = list(model.shared_parameters())
    task_params   = list(model.task_specific_parameters())

    main_opt = optim.Adam([
        {'params': shared_params, 'lr': lr_shared},
        {'params': task_params,   'lr': lr_task},
    ])
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        main_opt, mode='min', factor=0.1, patience=3)

    train_loss_fn    = nn.BCEWithLogitsLoss()
    affinity_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    best_loss = float('inf')
    ckpt = os.path.join(output_dir, f'best_model_{label_type}_tag.pt')
    affinity_storage = []

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in loader:
            X_b, y_b, task_ids, _ = [b.to(device, non_blocking=True) for b in batch]

            # Affinity probe BEFORE the real training update — measures
            # affinities at the current base parameters theta_t.
            aff = compute_inter_task_affinity(
                model, X_b, y_b, task_ids,
                main_opt, shared_params, task_params,
                affinity_loss_fn, num_tasks)
            affinity_storage.append(aff)

            # Standard training step
            main_opt.zero_grad(set_to_none=True)
            loss  = train_loss_fn(model(X_b, task_ids), y_b)
            total = loss + model.compute_l2(L2_SHARED, l2_task)
            if torch.isnan(total):
                raise ValueError(f"NaN at epoch {epoch+1} [{label_type.upper()}]")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            main_opt.step()
            running += total.item()

        avg = running / len(loader)
        sched.step(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label_type.upper()}] Epoch {epoch+1}/{EPOCHS}  "
                  f"loss={avg:.4f}  affinity_batches={len(affinity_storage)}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt)

    if not affinity_storage:
        raise RuntimeError(f"No affinity matrices collected for {label_type.upper()}")

    affinity_arr = np.stack(affinity_storage)
    final_matrix = affinity_arr.mean(axis=0)
    np.save(os.path.join(output_dir, f'{label_type}_final_affinity_matrix.npy'),
            final_matrix)
    print(f"  [{label_type.upper()}] Aggregated {len(affinity_storage)} matrices")

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    return model, final_matrix


if __name__ == '__main__':
    args = parse_args()
    t0 = time.time()

    df, cfg = load_dataset(args.dataset)
    splits = cfg['splits']
    p_ids  = cfg['participant_ids']
    prefix = cfg['results_prefix']

    OUTPUT_DIR = os.path.join(RESULTS_DIR, f'{prefix}_TAG', f'{prefix}_tag_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {OUTPUT_DIR}")

    # Same per-dataset hyperparameter routing as mtl_hps.py
    sh_ar = cfg.get('mtl_shared_lr_ar', MTL_SHARED_LR)
    tk_ar = cfg.get('mtl_task_lr_ar',   MTL_TASK_LR)
    l2_ar = cfg.get('l2_task_ar',       L2_TASK)
    sh_va = cfg.get('mtl_shared_lr_va', MTL_SHARED_LR)
    tk_va = cfg.get('mtl_task_lr_va',   MTL_TASK_LR)
    l2_va = cfg.get('l2_task_va',       L2_TASK)

    train_data, test_data = {}, {}
    for task_idx, pid in enumerate(p_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        train_data[task_idx] = p_df[p_df['Trial'].isin(splits[pid]['train'])].reset_index(drop=True)
        test_data[task_idx]  = p_df[p_df['Trial'].isin(splits[pid]['test'])].reset_index(drop=True)

    print("\n" + "="*60 + "\nTRAINING AR (with affinity probing)\n" + "="*60)
    set_all_seeds(SEED)
    model_ar, mat_ar = _train_with_affinity('ar', sh_ar, tk_ar, l2_ar,
                                             train_data, cfg, device, OUTPUT_DIR)

    print("\n" + "="*60 + "\nTRAINING VA (with affinity probing)\n" + "="*60)
    set_all_seeds(SEED)
    model_va, mat_va = _train_with_affinity('va', sh_va, tk_va, l2_va,
                                             train_data, cfg, device, OUTPUT_DIR)

    # Per-participant affinity scores
    ar_scores = extract_per_participant_scores(mat_ar, p_ids)
    va_scores = extract_per_participant_scores(mat_va, p_ids)
    pd.DataFrame({
        'participant_id':    p_ids,
        'ar_affinity_score': [ar_scores[p] for p in p_ids],
        'va_affinity_score': [va_scores[p] for p in p_ids],
    }).to_csv(os.path.join(OUTPUT_DIR, 'affinity_scores_per_participant.csv'),
              index=False)

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    results = evaluate_mtl_all(model_ar, model_va, test_data, p_ids, device,
                               cfg['window_size'], cfg['stride'],
                               feature_cols=cfg['feature_cols'])
    agg = aggregate_results(results)

    results_df, ar_stds, va_stds = save_all_results(
        results, agg, OUTPUT_DIR,
        method_name='MTL-TAG',
        misclassification_csv=f'{prefix}_tag_misclassification_rates.csv')

    with open(os.path.join(OUTPUT_DIR, 'tag_results.pkl'), 'wb') as f:
        pickle.dump({**agg,
                     'per_participant':         results,
                     'per_participant_table':   results_df,
                     **ar_stds, **va_stds,
                     'ar_final_affinity_matrix': mat_ar,
                     'va_final_affinity_matrix': mat_va,
                     'ar_affinity_scores':       ar_scores,
                     'va_affinity_scores':       va_scores,
                     'participant_ids':          p_ids}, f)

    print(f"\n✓ All outputs in: {OUTPUT_DIR}")
    print(f"Total experiment time: {time.time() - t0:.1f}s")
