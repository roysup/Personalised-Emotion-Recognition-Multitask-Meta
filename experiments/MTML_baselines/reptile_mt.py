# """
# Reptile Multi-Task (MT)
# Shared backbone + per-participant heads during meta-training.
# Samples EPISODE_SIZE participants per meta-step.
# Reptile outer-loop updates backbone only.
# Heads are kept per-participant and updated individually.
# At test time: fresh head per test participant, adapt both.

# Usage
# -----
#     python reptile_mt.py                  # runs on VREED (default)
#     python reptile_mt.py --dataset dssn_eq
# """
# import argparse
# import os, sys, time
# _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

# from config import (SEED, META_STEPS, META_LR,
#                     INNER_STEPS, INNER_LR, EPISODE_SIZE,
#                     N_FOLDS, L2_SHARED, L2_TASK, RESULTS_DIR)
# import numpy as np
# import pickle
# import torch
# from sklearn.metrics import f1_score
# from data import build_support_query
# from dataset_configs.loader import load_dataset
# from models import BaseFeatureExtractor, TaskHead
# from utils import (set_all_seeds, make_kfolds, aggregate_mtml_results,
#                    compute_per_participant_stds, print_determinism_summary)
# from training import adapt_inner_loop, evaluate_test_user, reptile_outer_update


# def parse_args():
#     p = argparse.ArgumentParser(description='Reptile MT')
#     p.add_argument('--dataset', type=str, default='vreed',
#                    choices=['vreed', 'dssn_eq', 'dssn_em'])
#     return p.parse_args()


# def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
#                    meta_lr, inner_lr, l2_shared, l2_task):
#     """Reptile-MT meta-training: multi-participant episodes, backbone-only outer update."""
#     base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
#     heads = {pid: TaskHead().to(device) for pid in train_ps}
#     rng   = np.random.default_rng(SEED)

#     for step in range(META_STEPS):
#         episode_ps = rng.choice(train_ps, size=min(EPISODE_SIZE, len(train_ps)),
#                                 replace=False).tolist()
#         adapted_bases = []

#         for pid in episode_ps:
#             p_df = df[df['ID'] == pid].reset_index(drop=True)
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
#             print(f"  [{label_type.upper()}] Reptile-MT step {step+1}/{META_STEPS}")

#     torch.save(base.state_dict(),
#                os.path.join(output_dir, f'reptile_mt_base_{label_type}.pth'))
#     return base


# # =============================
# # HYPERPARAMETER TUNING
# # =============================
# def hyperparameter_tuning(label_type, df, splits, train_ps, cfg, device, output_dir):
#     """K-fold CV on train participants to validate Reptile-MT hyperparameters."""
#     print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Reptile-MT"
#           f"  ({cfg['results_prefix']})\n{'='*60}")
#     print(f"  META_LR={META_LR}, INNER_LR={INNER_LR}, "
#           f"INNER_STEPS={INNER_STEPS}, EPISODE_SIZE={EPISODE_SIZE}")
#     print(f"  L2: Shared={L2_SHARED}, Task={L2_TASK}")

#     results = []
#     train_folds = make_kfolds(train_ps, seed=SEED)

#     for meta_lr in [META_LR]:
#         for inner_lr in [INNER_LR]:
#             for l2_s in [L2_SHARED]:
#                 for l2_t in [L2_TASK]:
#                     fold_f1s = []
#                     for fold_i in range(N_FOLDS):
#                         val_ps = train_folds[fold_i]
#                         tr_ps  = [p for j, f in enumerate(train_folds)
#                                   if j != fold_i for p in f]

#                         # Meta-train on fold's train participants
#                         set_all_seeds(SEED)
#                         base  = BaseFeatureExtractor(
#                             input_dim=cfg['input_dim']).to(device)
#                         heads = {pid: TaskHead().to(device) for pid in tr_ps}
#                         rng   = np.random.default_rng(SEED)

#                         for step in range(META_STEPS):
#                             episode_ps = rng.choice(
#                                 tr_ps,
#                                 size=min(EPISODE_SIZE, len(tr_ps)),
#                                 replace=False).tolist()
#                             adapted_bases = []

#                             for pid in episode_ps:
#                                 p_df = df[df['ID'] == pid].reset_index(
#                                     drop=True)
#                                 sup_loader, _ = build_support_query(
#                                     p_df, splits[pid]['train'], [],
#                                     ar_or_va=label_type,
#                                     window_size=cfg['window_size'],
#                                     stride=cfg['stride'],
#                                     feature_cols=cfg['feature_cols'])
#                                 adapted_base, adapted_head = adapt_inner_loop(
#                                     base, heads[pid], sup_loader, label_type,
#                                     INNER_STEPS, inner_lr, device,
#                                     l2_shared=l2_s, l2_task=l2_t)
#                                 adapted_bases.append(adapted_base)
#                                 heads[pid] = adapted_head

#                             reptile_outer_update(base, adapted_bases, meta_lr)

#                         # Adapt + evaluate on fold's val participants
#                         val_f1s = []
#                         for uid in sorted(val_ps):
#                             if uid not in splits:
#                                 continue
#                             t_df = df[df['ID'] == uid].reset_index(drop=True)
#                             head = TaskHead().to(device)
#                             r = evaluate_test_user(
#                                 base, head, t_df, splits, uid, label_type,
#                                 device,
#                                 inner_steps=INNER_STEPS, inner_lr=inner_lr,
#                                 l2_shared=l2_s, l2_task=l2_t,
#                                 window_size=cfg['window_size'],
#                                 stride=cfg['stride'],
#                                 feature_cols=cfg['feature_cols'])
#                             if r is not None:
#                                 y_true = r[f'y_true_{label_type}']
#                                 y_pred = r[f'y_pred_{label_type}']
#                                 val_f1s.append(f1_score(
#                                     y_true, y_pred,
#                                     average='macro', zero_division=0))

#                         if val_f1s:
#                             fold_f1s.append(np.mean(val_f1s))
#                             print(f"  Fold {fold_i + 1}/{N_FOLDS}: "
#                                   f"Val F1 = {fold_f1s[-1]:.4f}")

#                     if not fold_f1s:
#                         continue
#                     avg = np.mean(fold_f1s)
#                     results.append({
#                         'meta_lr': meta_lr, 'inner_lr': inner_lr,
#                         'l2_shared': l2_s, 'l2_task': l2_t,
#                         'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
#                     print(f"  Average F1: {avg:.4f}")

#     if not results:
#         return META_LR, INNER_LR, L2_SHARED, L2_TASK
#     best = max(results, key=lambda x: x['avg_f1'])
#     with open(os.path.join(output_dir,
#               f'{label_type}_tuning_results_reptile_mt.pkl'), 'wb') as f:
#         pickle.dump({'all': results, 'best': best}, f)
#     return best['meta_lr'], best['inner_lr'], best['l2_shared'], best['l2_task']


# # =============================
# # MAIN
# # =============================
# if __name__ == '__main__':
#     args = parse_args()
#     experiment_t0 = time.time()

#     df, cfg = load_dataset(args.dataset, mode='mtml')
#     splits   = cfg['splits']
#     prefix   = cfg['results_prefix']
#     train_ps = cfg['train_participants']
#     test_ps  = cfg['test_participants']

#     output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_reptile_mt')
#     os.makedirs(output_dir, exist_ok=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     set_all_seeds(SEED)
#     if device.type == 'cuda':
#         torch.backends.cudnn.benchmark = True
#     print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")

#     # Hyperparameter tuning on train participants only
#     best_meta_lr_ar, best_inner_lr_ar, best_l2s_ar, best_l2t_ar = \
#         hyperparameter_tuning('ar', df, splits, train_ps, cfg, device, output_dir)
#     best_meta_lr_va, best_inner_lr_va, best_l2s_va, best_l2t_va = \
#         hyperparameter_tuning('va', df, splits, train_ps, cfg, device, output_dir)

#     for lt in ['ar', 'va']:
#         if lt == 'ar':
#             meta_lr, inner_lr = best_meta_lr_ar, best_inner_lr_ar
#             l2_s, l2_t = best_l2s_ar, best_l2t_ar
#         else:
#             meta_lr, inner_lr = best_meta_lr_va, best_inner_lr_va
#             l2_s, l2_t = best_l2s_va, best_l2t_va

#         print(f"\n{'='*60}\nREPTILE-MT META-TRAINING {lt.upper()}\n{'='*60}")
#         set_all_seeds(SEED)
#         base = _reptile_train(lt, df, splits, train_ps, cfg, device, output_dir,
#                               meta_lr=meta_lr, inner_lr=inner_lr,
#                               l2_shared=l2_s, l2_task=l2_t)

#         print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()}\n{'='*60}")
#         results = []
#         for uid in sorted(test_ps):
#             if uid not in splits:
#                 continue
#             t_df = df[df['ID'] == uid].reset_index(drop=True)
#             head = TaskHead().to(device)
#             r = evaluate_test_user(
#                 base, head, t_df, splits, uid, lt, device,
#                 inner_steps=INNER_STEPS, inner_lr=inner_lr,
#                 l2_shared=l2_s, l2_task=l2_t,
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
#         'best_hyperparameters': {
#             'AR': {'meta_lr': best_meta_lr_ar, 'inner_lr': best_inner_lr_ar,
#                    'l2_shared': best_l2s_ar, 'l2_task': best_l2t_ar},
#             'VA': {'meta_lr': best_meta_lr_va, 'inner_lr': best_inner_lr_va,
#                    'l2_shared': best_l2s_va, 'l2_task': best_l2t_va}},
#         **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **ar_stds, **va_stds,
#         'test_results_per_participant_ar': results_ar,
#         'test_results_per_participant_va': results_va,
#         'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
#     }
#     with open(os.path.join(output_dir, 'reptile_mt_results.pkl'), 'wb') as f:
#         pickle.dump(final, f)

#     print_determinism_summary(
#         {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         ar_stds, va_stds)

#     print(f"\n✓ All results saved to: {output_dir}")
#     print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")


# ------------------------------------------------------------------------------------------------------------------

# """
# Reptile Multi-Task (MT)
# Shared backbone + per-participant heads during meta-training.
# Samples EPISODE_SIZE participants per meta-step.
# Sequential adaptation within each episode: a single clone of the
# backbone is adapted across all sampled users in order, then the
# Reptile outer update moves the meta-parameters toward the result.
# Heads are kept per-participant and updated individually.
# At test time: fresh head per test participant, adapt both.

# Usage
# -----
#     python reptile_mt.py                  # runs on VREED (default)
#     python reptile_mt.py --dataset dssn_eq
# """
# import argparse
# import copy
# import os, sys, time
# _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

# from config import (SEED, META_STEPS, META_LR, MAX_NORM,
#                     INNER_STEPS, INNER_LR, EPISODE_SIZE,
#                     N_FOLDS, L2_SHARED, L2_TASK, RESULTS_DIR)
# import numpy as np
# import pickle
# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.metrics import f1_score
# from data import build_support_query
# from dataset_configs.loader import load_dataset
# from models import BaseFeatureExtractor, TaskHead
# from utils import (set_all_seeds, make_kfolds, aggregate_mtml_results,
#                    compute_per_participant_stds, print_determinism_summary)
# from training import adapt_inner_loop, evaluate_test_user, reptile_outer_update


# def parse_args():
#     p = argparse.ArgumentParser(description='Reptile MT')
#     p.add_argument('--dataset', type=str, default='vreed',
#                    choices=['vreed', 'dssn_eq', 'dssn_em'])
#     return p.parse_args()


# # =============================
# # SEQUENTIAL EPISODE ADAPTATION
# # =============================
# def _adapt_episode_step(episode_base, head, sup_loader, ar_or_va,
#                         inner_steps, inner_lr, device,
#                         l2_shared=0.0, l2_task=1e-5):
#     """
#     Adapt episode_base IN PLACE on one user's support data.
#     The head is deep-copied so each user gets their own head,
#     but the backbone is shared and mutated across users within
#     the episode (sequential Reptile).

#     Returns
#     -------
#     adapted_head : the updated head for this user
#     """
#     adapted_head = copy.deepcopy(head).to(device)
#     episode_base.train()
#     adapted_head.train()

#     sp = list(episode_base.parameters())
#     tp = list(adapted_head.parameters())
#     opt   = torch.optim.Adam(sp + tp, lr=inner_lr)
#     sched = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
#     loss_fn = nn.BCEWithLogitsLoss()

#     for step in range(inner_steps):
#         ep_loss = 0.0
#         nb = 0
#         for Xb, yb in sup_loader:
#             Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
#             opt.zero_grad(set_to_none=True)
#             loss = loss_fn(adapted_head(episode_base(Xb)), yb)
#             loss = loss + (l2_shared * sum(p.norm(2)**2 for p in sp if p.requires_grad) +
#                            l2_task   * sum(p.norm(2)**2 for p in tp if p.requires_grad))
#             if not torch.isnan(loss):
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(sp + tp, max_norm=MAX_NORM)
#                 opt.step()
#             ep_loss += loss.item()
#             nb += 1
#         if nb > 0:
#             sched.step(ep_loss / nb)

#     return adapted_head


# # =============================
# # META-TRAINING
# # =============================
# def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
#                    meta_lr, inner_lr, l2_shared, l2_task):
#     """Reptile-MT meta-training: sequential adaptation within multi-participant episodes."""
#     base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
#     heads = {pid: TaskHead().to(device) for pid in train_ps}
#     rng   = np.random.default_rng(SEED)

#     for step in range(META_STEPS):
#         episode_ps = rng.choice(train_ps, size=min(EPISODE_SIZE, len(train_ps)),
#                                 replace=False).tolist()

#         # Single clone for the whole episode — adapted sequentially
#         episode_base = copy.deepcopy(base).to(device)

#         for pid in episode_ps:
#             p_df = df[df['ID'] == pid].reset_index(drop=True)
#             sup_loader, _ = build_support_query(
#                 p_df, splits[pid]['train'], [],
#                 ar_or_va=label_type,
#                 window_size=cfg['window_size'], stride=cfg['stride'],
#                 feature_cols=cfg['feature_cols'])

#             # Adapt episode_base in place; only the head is copied
#             adapted_head = _adapt_episode_step(
#                 episode_base, heads[pid], sup_loader, label_type,
#                 INNER_STEPS, inner_lr, device,
#                 l2_shared=l2_shared, l2_task=l2_task)

#             # Keep updated head per-participant
#             heads[pid] = adapted_head

#         # Outer update: move backbone toward the end of the sequential trajectory
#         reptile_outer_update(base, [episode_base], meta_lr)

#         if (step + 1) % 10 == 0 or step == 0:
#             print(f"  [{label_type.upper()}] Reptile-MT step {step+1}/{META_STEPS}")

#     torch.save(base.state_dict(),
#                os.path.join(output_dir, f'reptile_mt_base_{label_type}.pth'))
#     return base


# # =============================
# # HYPERPARAMETER TUNING
# # =============================
# def hyperparameter_tuning(label_type, df, splits, train_ps, cfg, device, output_dir):
#     """K-fold CV on train participants to validate Reptile-MT hyperparameters."""
#     print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Reptile-MT"
#           f"  ({cfg['results_prefix']})\n{'='*60}")
#     print(f"  META_LR={META_LR}, INNER_LR={INNER_LR}, "
#           f"INNER_STEPS={INNER_STEPS}, EPISODE_SIZE={EPISODE_SIZE}")
#     print(f"  L2: Shared={L2_SHARED}, Task={L2_TASK}")

#     results = []
#     train_folds = make_kfolds(train_ps, seed=SEED)

#     for meta_lr in [META_LR]:
#         for inner_lr in [INNER_LR]:
#             for l2_s in [L2_SHARED]:
#                 for l2_t in [L2_TASK]:
#                     fold_f1s = []
#                     for fold_i in range(N_FOLDS):
#                         val_ps = train_folds[fold_i]
#                         tr_ps  = [p for j, f in enumerate(train_folds)
#                                   if j != fold_i for p in f]

#                         # Meta-train on fold's train participants
#                         set_all_seeds(SEED)
#                         base  = BaseFeatureExtractor(
#                             input_dim=cfg['input_dim']).to(device)
#                         heads = {pid: TaskHead().to(device) for pid in tr_ps}
#                         rng   = np.random.default_rng(SEED)

#                         for step in range(META_STEPS):
#                             episode_ps = rng.choice(
#                                 tr_ps,
#                                 size=min(EPISODE_SIZE, len(tr_ps)),
#                                 replace=False).tolist()

#                             # Sequential episode adaptation
#                             episode_base = copy.deepcopy(base).to(device)

#                             for pid in episode_ps:
#                                 p_df = df[df['ID'] == pid].reset_index(
#                                     drop=True)
#                                 sup_loader, _ = build_support_query(
#                                     p_df, splits[pid]['train'], [],
#                                     ar_or_va=label_type,
#                                     window_size=cfg['window_size'],
#                                     stride=cfg['stride'],
#                                     feature_cols=cfg['feature_cols'])
#                                 adapted_head = _adapt_episode_step(
#                                     episode_base, heads[pid], sup_loader,
#                                     label_type,
#                                     INNER_STEPS, inner_lr, device,
#                                     l2_shared=l2_s, l2_task=l2_t)
#                                 heads[pid] = adapted_head

#                             reptile_outer_update(base, [episode_base], meta_lr)

#                         # Adapt + evaluate on fold's val participants
#                         val_f1s = []
#                         for uid in sorted(val_ps):
#                             if uid not in splits:
#                                 continue
#                             t_df = df[df['ID'] == uid].reset_index(drop=True)
#                             head = TaskHead().to(device)
#                             r = evaluate_test_user(
#                                 base, head, t_df, splits, uid, label_type,
#                                 device,
#                                 inner_steps=INNER_STEPS, inner_lr=inner_lr,
#                                 l2_shared=l2_s, l2_task=l2_t,
#                                 window_size=cfg['window_size'],
#                                 stride=cfg['stride'],
#                                 feature_cols=cfg['feature_cols'])
#                             if r is not None:
#                                 y_true = r[f'y_true_{label_type}']
#                                 y_pred = r[f'y_pred_{label_type}']
#                                 val_f1s.append(f1_score(
#                                     y_true, y_pred,
#                                     average='macro', zero_division=0))

#                         if val_f1s:
#                             fold_f1s.append(np.mean(val_f1s))
#                             print(f"  Fold {fold_i + 1}/{N_FOLDS}: "
#                                   f"Val F1 = {fold_f1s[-1]:.4f}")

#                     if not fold_f1s:
#                         continue
#                     avg = np.mean(fold_f1s)
#                     results.append({
#                         'meta_lr': meta_lr, 'inner_lr': inner_lr,
#                         'l2_shared': l2_s, 'l2_task': l2_t,
#                         'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
#                     print(f"  Average F1: {avg:.4f}")

#     if not results:
#         return META_LR, INNER_LR, L2_SHARED, L2_TASK
#     best = max(results, key=lambda x: x['avg_f1'])
#     with open(os.path.join(output_dir,
#               f'{label_type}_tuning_results_reptile_mt.pkl'), 'wb') as f:
#         pickle.dump({'all': results, 'best': best}, f)
#     return best['meta_lr'], best['inner_lr'], best['l2_shared'], best['l2_task']


# # =============================
# # MAIN
# # =============================
# if __name__ == '__main__':
#     args = parse_args()
#     experiment_t0 = time.time()

#     df, cfg = load_dataset(args.dataset, mode='mtml')
#     splits   = cfg['splits']
#     prefix   = cfg['results_prefix']
#     train_ps = cfg['train_participants']
#     test_ps  = cfg['test_participants']

#     output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_reptile_mt')
#     os.makedirs(output_dir, exist_ok=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     set_all_seeds(SEED)
#     if device.type == 'cuda':
#         torch.backends.cudnn.benchmark = True
#     print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")

#     # Hyperparameter tuning on train participants only
#     best_meta_lr_ar, best_inner_lr_ar, best_l2s_ar, best_l2t_ar = \
#         hyperparameter_tuning('ar', df, splits, train_ps, cfg, device, output_dir)
#     best_meta_lr_va, best_inner_lr_va, best_l2s_va, best_l2t_va = \
#         hyperparameter_tuning('va', df, splits, train_ps, cfg, device, output_dir)

#     for lt in ['ar', 'va']:
#         if lt == 'ar':
#             meta_lr, inner_lr = best_meta_lr_ar, best_inner_lr_ar
#             l2_s, l2_t = best_l2s_ar, best_l2t_ar
#         else:
#             meta_lr, inner_lr = best_meta_lr_va, best_inner_lr_va
#             l2_s, l2_t = best_l2s_va, best_l2t_va

#         print(f"\n{'='*60}\nREPTILE-MT META-TRAINING {lt.upper()}\n{'='*60}")
#         set_all_seeds(SEED)
#         base = _reptile_train(lt, df, splits, train_ps, cfg, device, output_dir,
#                               meta_lr=meta_lr, inner_lr=inner_lr,
#                               l2_shared=l2_s, l2_task=l2_t)

#         print(f"\n{'='*60}\nADAPT + EVAL {lt.upper()}\n{'='*60}")
#         results = []
#         for uid in sorted(test_ps):
#             if uid not in splits:
#                 continue
#             t_df = df[df['ID'] == uid].reset_index(drop=True)
#             head = TaskHead().to(device)
#             r = evaluate_test_user(
#                 base, head, t_df, splits, uid, lt, device,
#                 inner_steps=INNER_STEPS, inner_lr=inner_lr,
#                 l2_shared=l2_s, l2_task=l2_t,
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
#         'best_hyperparameters': {
#             'AR': {'meta_lr': best_meta_lr_ar, 'inner_lr': best_inner_lr_ar,
#                    'l2_shared': best_l2s_ar, 'l2_task': best_l2t_ar},
#             'VA': {'meta_lr': best_meta_lr_va, 'inner_lr': best_inner_lr_va,
#                    'l2_shared': best_l2s_va, 'l2_task': best_l2t_va}},
#         **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **ar_stds, **va_stds,
#         'test_results_per_participant_ar': results_ar,
#         'test_results_per_participant_va': results_va,
#         'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
#     }
#     with open(os.path.join(output_dir, 'reptile_mt_results.pkl'), 'wb') as f:
#         pickle.dump(final, f)

#     print_determinism_summary(
#         {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         ar_stds, va_stds)

#     print(f"\n✓ All results saved to: {output_dir}")
#     print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")

# ------------------------------------------------------------------------------------------------------------------

"""
Reptile Multi-Task (MT)
Shared backbone + per-participant heads during meta-training.
Samples EPISODE_SIZE participants per meta-step.
Sequential adaptation within each episode: a single clone of the
backbone is adapted across all sampled users in order, then the
Reptile outer update moves the meta-parameters toward the result.
Heads are kept per-participant and updated individually.
At test time: fresh head per test participant, adapt both.

Supports optional balanced k-shot support sampling via K_PER_CLASS:
  None  = use all available support windows (default)
  int   = subsample k windows per class (e.g. 20 → 40 total)

Usage
-----
    python reptile_mt.py                  # runs on VREED (default)
    python reptile_mt.py --dataset dssn_eq
"""
import argparse
import copy
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, META_STEPS, META_LR, MAX_NORM,
                    INNER_STEPS, INNER_LR, EPISODE_SIZE,
                    N_FOLDS, L2_SHARED, L2_TASK, K_PER_CLASS,
                    RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from data import build_support_query
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, TaskHead
from utils import (set_all_seeds, make_kfolds, aggregate_mtml_results,
                   compute_per_participant_stds, print_determinism_summary)
from training import adapt_inner_loop, evaluate_test_user, reptile_outer_update


def parse_args():
    p = argparse.ArgumentParser(description='Reptile MT')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


# =============================
# SEQUENTIAL EPISODE ADAPTATION
# =============================
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
            loss = loss + (l2_shared * sum(p.norm(2)**2 for p in sp if p.requires_grad) +
                           l2_task   * sum(p.norm(2)**2 for p in tp if p.requires_grad))
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sp + tp, max_norm=MAX_NORM)
                opt.step()
            ep_loss += loss.item()
            nb += 1
        if nb > 0:
            sched.step(ep_loss / nb)

    return adapted_head


# =============================
# META-TRAINING
# =============================
def _reptile_train(label_type, df, splits, train_ps, cfg, device, output_dir,
                   meta_lr, inner_lr, l2_shared, l2_task,
                   balanced_k_per_class=None):
    """Reptile-MT meta-training: sequential adaptation within multi-participant episodes."""
    base  = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
    heads = {pid: TaskHead().to(device) for pid in train_ps}
    rng   = np.random.default_rng(SEED)

    if balanced_k_per_class is not None:
        print(f"  [{label_type.upper()}] Balanced k-shot: {balanced_k_per_class} per class")
    else:
        print(f"  [{label_type.upper()}] Using all support windows")

    for step in range(META_STEPS):
        episode_ps = rng.choice(train_ps, size=min(EPISODE_SIZE, len(train_ps)),
                                replace=False).tolist()

        # Single clone for the whole episode — adapted sequentially
        episode_base = copy.deepcopy(base).to(device)

        for pid in episode_ps:
            p_df = df[df['ID'] == pid].reset_index(drop=True)
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
            print(f"  [{label_type.upper()}] Reptile-MT step {step+1}/{META_STEPS}")

    torch.save(base.state_dict(),
               os.path.join(output_dir, f'reptile_mt_base_{label_type}.pth'))
    return base


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, df, splits, train_ps, cfg, device,
                          output_dir, balanced_k_per_class=None):
    """K-fold CV on train participants to validate Reptile-MT hyperparameters."""
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Reptile-MT"
          f"  ({cfg['results_prefix']})\n{'='*60}")
    print(f"  META_LR={META_LR}, INNER_LR={INNER_LR}, "
          f"INNER_STEPS={INNER_STEPS}, EPISODE_SIZE={EPISODE_SIZE}")
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
                            episode_ps = rng.choice(
                                tr_ps,
                                size=min(EPISODE_SIZE, len(tr_ps)),
                                replace=False).tolist()

                            # Sequential episode adaptation
                            episode_base = copy.deepcopy(base).to(device)

                            for pid in episode_ps:
                                p_df = df[df['ID'] == pid].reset_index(
                                    drop=True)
                                sup_loader, _ = build_support_query(
                                    p_df, splits[pid]['train'], [],
                                    ar_or_va=label_type,
                                    window_size=cfg['window_size'],
                                    stride=cfg['stride'],
                                    feature_cols=cfg['feature_cols'],
                                    balanced_k_per_class=balanced_k_per_class)
                                adapted_head = _adapt_episode_step(
                                    episode_base, heads[pid], sup_loader,
                                    label_type,
                                    INNER_STEPS, inner_lr, device,
                                    l2_shared=l2_s, l2_task=l2_t)
                                heads[pid] = adapted_head

                            reptile_outer_update(base, [episode_base], meta_lr)

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
              f'{label_type}_tuning_results_reptile_mt.pkl'), 'wb') as f:
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

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_reptile_mt')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"K_PER_CLASS: {K_PER_CLASS}")

    # Hyperparameter tuning on train participants only
    best_meta_lr_ar, best_inner_lr_ar, best_l2s_ar, best_l2t_ar = \
        hyperparameter_tuning('ar', df, splits, train_ps, cfg, device, output_dir,
                              balanced_k_per_class=K_PER_CLASS)
    best_meta_lr_va, best_inner_lr_va, best_l2s_va, best_l2t_va = \
        hyperparameter_tuning('va', df, splits, train_ps, cfg, device, output_dir,
                              balanced_k_per_class=K_PER_CLASS)

    for lt in ['ar', 'va']:
        if lt == 'ar':
            meta_lr, inner_lr = best_meta_lr_ar, best_inner_lr_ar
            l2_s, l2_t = best_l2s_ar, best_l2t_ar
        else:
            meta_lr, inner_lr = best_meta_lr_va, best_inner_lr_va
            l2_s, l2_t = best_l2s_va, best_l2t_va

        print(f"\n{'='*60}\nREPTILE-MT META-TRAINING {lt.upper()}\n{'='*60}")
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
            head = TaskHead().to(device)
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
    with open(os.path.join(output_dir, 'reptile_mt_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")