# """
# Transfer MTL — Pre-train MTLTransferModel on train participants, then for each
# test participant: deep-copy the model, add a new task head via add_task_head(),
# fine-tune the ENTIRE model (backbone + new head) with standard Adam + BCEWithLogitsLoss.

# Usage
# -----
#     python transfer_mtl.py                  # runs on VREED (default)
#     python transfer_mtl.py --dataset dssn_eq
# """
# import argparse
# import os, sys, time, copy, gc
# _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
# sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

# from config import (SEED, EPOCHS, MAX_NORM, FT_EPOCHS, N_FOLDS,
#                     TRANSFER_MTL_LR_PT, TRANSFER_MTL_LR_FT,
#                     MTL_TASK_LR,
#                     L2_SHARED, L2_TASK, RESULTS_DIR)
# import numpy as np
# import pickle
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import confusion_matrix, f1_score
# from data import create_sliding_windows, make_mtl_loader, arrays_to_loader
# from dataset_configs.loader import load_dataset
# from models import MTLTransferModel
# from utils import (set_all_seeds, compute_metrics_from_cm, make_kfolds,
#                    aggregate_mtml_results, compute_per_participant_stds,
#                    print_determinism_summary)


# def parse_args():
#     p = argparse.ArgumentParser(description='Transfer MTL')
#     p.add_argument('--dataset', type=str, default='vreed',
#                    choices=['vreed', 'dssn_eq', 'dssn_em'])
#     return p.parse_args()


# def _pretrain_mtl(label_type, train_data_dict, lr_pt, lr_task,
#                   cfg, device, output_dir):
#     """Pre-train MTLTransferModel on train participants."""
#     num_train = len(train_data_dict)
#     loader, _, _ = make_mtl_loader(
#         train_data_dict, cfg['window_size'], cfg['stride'],
#         label_type=label_type, batch_size=num_train,
#         seed=SEED, feature_cols=cfg['feature_cols'])

#     model   = MTLTransferModel(num_train, input_dim=cfg['input_dim']).to(device)
#     opt     = optim.Adam([
#         {'params': model.backbone_parameters(),      'lr': lr_pt},
#         {'params': model.task_specific_parameters(), 'lr': lr_task},
#     ])
#     sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
#     loss_fn = nn.BCEWithLogitsLoss(reduction='none')
#     best_loss = float('inf')
#     ckpt = os.path.join(output_dir, f'pretrain_{label_type}.pt')

#     for epoch in range(EPOCHS):
#         model.train()
#         running = 0.0
#         for batch in loader:
#             X_b, y_b, task_ids, _ = [b.to(device) for b in batch]
#             opt.zero_grad(set_to_none=True)
#             loss_vec = loss_fn(model(X_b, task_ids), y_b).squeeze(-1)
#             l2 = model.compute_l2(L2_SHARED, L2_TASK)
#             total = loss_vec.mean() + l2
#             if torch.isnan(total):
#                 raise ValueError(f"NaN at ep {epoch+1} [{label_type}]")
#             total.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
#             opt.step()
#             running += total.item()

#         avg = running / len(loader)
#         sched.step(avg)
#         if (epoch + 1) % 5 == 0 or epoch == 0:
#             print(f"  [{label_type.upper()}] Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
#         if avg < best_loss:
#             best_loss = avg
#             torch.save(model.state_dict(), ckpt)

#     model.load_state_dict(torch.load(ckpt, weights_only=True))
#     return model


# def _finetune_user(base_model, user_train_df, label_type, lr_ft, uid,
#                    cfg, device):
#     """Deep-copy model, add new head, fine-tune entire model."""
#     model = copy.deepcopy(base_model).to(device)
#     local_idx = model.add_task_head()

#     X, y_ar, y_va, _, _ = create_sliding_windows(
#         user_train_df, cfg['window_size'], cfg['stride'],
#         feature_cols=cfg['feature_cols'])
#     if len(X) == 0:
#         return model, local_idx

#     y = y_ar if label_type == 'ar' else y_va

#     loader = arrays_to_loader(X, y, batch_size=32, shuffle=True, seed=SEED)

#     opt     = optim.Adam(model.parameters(), lr=lr_ft)
#     sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
#     loss_fn = nn.BCEWithLogitsLoss()

#     for ep in range(FT_EPOCHS):
#         model.train()
#         ep_loss = 0.0
#         nb = 0
#         for X_b, y_b in loader:
#             X_b, y_b = X_b.to(device), y_b.to(device)
#             tids = torch.full((X_b.size(0),), local_idx,
#                               dtype=torch.long, device=device)
#             opt.zero_grad(set_to_none=True)
#             loss = loss_fn(model(X_b, tids), y_b)
#             l2 = model.compute_l2(L2_SHARED, L2_TASK)
#             total = loss + l2
#             if torch.isnan(total):
#                 raise ValueError(f"NaN finetune [pid {uid}, ep {ep+1}]")
#             total.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
#             opt.step()
#             ep_loss += total.item()
#             nb += 1
#         if nb > 0:
#             sched.step(ep_loss / nb)

#     return model, local_idx


# def _eval_user(model, user_test_df, local_idx, label_type, cfg, device):
#     """Evaluate fine-tuned model on test data."""
#     X, y_ar, y_va, _, _ = create_sliding_windows(
#         user_test_df, cfg['window_size'], cfg['stride'],
#         feature_cols=cfg['feature_cols'])
#     if len(X) == 0:
#         return None

#     y = y_ar if label_type == 'ar' else y_va
#     X_t  = torch.tensor(X, dtype=torch.float32).to(device)
#     tids = torch.full((len(X),), local_idx, dtype=torch.long, device=device)

#     model.eval()
#     with torch.no_grad():
#         probs = torch.sigmoid(model(X_t, tids)).cpu().numpy().flatten()

#     y_true = y.astype(int)
#     y_pred = (probs > 0.5).astype(int)
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     acc, prec, rec, f1 = compute_metrics_from_cm(cm)

#     p = label_type
#     return {
#         'participant_id': None,  # filled by caller
#         'cm': cm,
#         f'{p}_acc': acc, f'{p}_precision': prec,
#         f'{p}_recall': rec, f'{p}_f1': f1,
#         f'y_true_{p}': y_true, f'y_pred_{p}': y_pred,
#         f'y_pred_probs_{p}': probs,
#     }


# # =============================
# # HYPERPARAMETER TUNING
# # =============================
# def hyperparameter_tuning(label_type, df, cfg, device, output_dir):
#     """K-fold CV on train participants to validate hyperparameters.

#     For each fold: pre-train on fold's train participants, then for each
#     val participant: deep-copy, add head, fine-tune on their train videos,
#     evaluate on their test videos.
#     """
#     splits   = cfg['splits']
#     train_ps = cfg['train_participants']
#     print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Transfer MTL"
#           f"  ({cfg['results_prefix']})\n{'='*60}")
#     print(f"  LR_PT={TRANSFER_MTL_LR_PT}, LR_FT={TRANSFER_MTL_LR_FT}, "
#           f"LR_TASK={MTL_TASK_LR}")
#     print(f"  L2: Shared={L2_SHARED}, Task={L2_TASK}")

#     results = []
#     train_folds = make_kfolds(train_ps, seed=SEED)

#     for lr_pt in [TRANSFER_MTL_LR_PT]:
#         for lr_ft in [TRANSFER_MTL_LR_FT]:
#             fold_f1s = []
#             print(f"\nTesting LR_PT={lr_pt}, LR_FT={lr_ft}")

#             for fold_i in range(N_FOLDS):
#                 print(f"  Processing Fold {fold_i + 1}/{N_FOLDS}")
#                 val_ps = train_folds[fold_i]
#                 tr_ps  = [p for j, f in enumerate(train_folds)
#                           if j != fold_i for p in f]

#                 # Build train data for this fold
#                 fold_train_data = {}
#                 for idx, pid in enumerate(sorted(tr_ps)):
#                     p_df = df[df['ID'] == pid].reset_index(drop=True)
#                     fold_train_data[idx] = p_df[p_df['Trial'].isin(
#                         splits[pid]['train'])].reset_index(drop=True)

#                 # Pre-train on fold's train participants
#                 set_all_seeds(SEED)
#                 base_model = _pretrain_mtl(label_type, fold_train_data,
#                                            lr_pt, MTL_TASK_LR,
#                                            cfg, device, output_dir)

#                 # Fine-tune + evaluate on each val participant
#                 val_f1s = []
#                 for uid in sorted(val_ps):
#                     if uid not in splits:
#                         continue
#                     user_train_df = df[(df['ID'] == uid) & df['Trial'].isin(
#                         splits[uid]['train'])].reset_index(drop=True)
#                     user_test_df = df[(df['ID'] == uid) & df['Trial'].isin(
#                         splits[uid]['test'])].reset_index(drop=True)

#                     ft_model, local_idx = _finetune_user(
#                         base_model, user_train_df, label_type,
#                         lr_ft, uid, cfg, device)
#                     r = _eval_user(ft_model, user_test_df, local_idx,
#                                    label_type, cfg, device)
#                     if r is not None:
#                         y_true = r[f'y_true_{label_type}']
#                         y_pred = r[f'y_pred_{label_type}']
#                         val_f1s.append(f1_score(y_true, y_pred,
#                                                 average='macro',
#                                                 zero_division=0))
#                     del ft_model

#                 if val_f1s:
#                     fold_f1s.append(np.mean(val_f1s))
#                     print(f"  Fold {fold_i + 1}: Val F1 = {fold_f1s[-1]:.4f}")

#                 del base_model
#                 torch.cuda.empty_cache()
#                 gc.collect()

#             if not fold_f1s:
#                 continue
#             avg = np.mean(fold_f1s)
#             results.append({'lr_pt': lr_pt, 'lr_ft': lr_ft,
#                             'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
#             print(f"  Average F1: {avg:.4f}")

#     if not results:
#         return TRANSFER_MTL_LR_PT, TRANSFER_MTL_LR_FT
#     best = max(results, key=lambda x: x['avg_f1'])
#     with open(os.path.join(output_dir,
#               f'{label_type}_tuning_results_transfer_mtl.pkl'), 'wb') as f:
#         pickle.dump({'all': results, 'best': best}, f)
#     return best['lr_pt'], best['lr_ft']


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

#     output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_transfer_mtl')
#     os.makedirs(output_dir, exist_ok=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     set_all_seeds(SEED)
#     if device.type == 'cuda':
#         torch.backends.cudnn.benchmark = True
#     print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
#     print(f"Train: {train_ps}\nTest:  {test_ps}")
#     print(f"\nL2: Shared={L2_SHARED}, Task={L2_TASK}")

#     # Hyperparameter tuning on train participants only
#     best_lr_pt_ar, best_lr_ft_ar = hyperparameter_tuning(
#         'ar', df, cfg, device, output_dir)
#     best_lr_pt_va, best_lr_ft_va = hyperparameter_tuning(
#         'va', df, cfg, device, output_dir)

#     # Build train data dict
#     train_data = {}
#     for idx, pid in enumerate(sorted(train_ps)):
#         p_df = df[df['ID'] == pid].reset_index(drop=True)
#         train_data[idx] = p_df[p_df['Trial'].isin(
#             splits[pid]['train'])].reset_index(drop=True)

#     for lt in ['ar', 'va']:
#         if lt == 'ar':
#             lr_pt, lr_ft = best_lr_pt_ar, best_lr_ft_ar
#         else:
#             lr_pt, lr_ft = best_lr_pt_va, best_lr_ft_va

#         print(f"\n{'='*60}\nPRETRAINING {lt.upper()}\n{'='*60}")
#         set_all_seeds(SEED)
#         base_model = _pretrain_mtl(lt, train_data, lr_pt, MTL_TASK_LR,
#                                    cfg, device, output_dir)
#         torch.save(base_model.state_dict(),
#                    os.path.join(output_dir, f'base_model_{lt}_final.pth'))

#         print(f"\n{'='*60}\nFINE-TUNING + EVAL {lt.upper()} — TEST PARTICIPANTS\n{'='*60}")
#         results = []
#         for uid in sorted(test_ps):
#             if uid not in splits:
#                 continue

#             user_train_df = df[(df['ID'] == uid) & df['Trial'].isin(
#                 splits[uid]['train'])].reset_index(drop=True)
#             user_test_df = df[(df['ID'] == uid) & df['Trial'].isin(
#                 splits[uid]['test'])].reset_index(drop=True)

#             print(f"\n  Participant {uid}: Fine-tuning {lt.upper()} model")
#             ft_model, local_idx = _finetune_user(
#                 base_model, user_train_df, lt,
#                 lr_ft, uid, cfg, device)

#             r = _eval_user(ft_model, user_test_df, local_idx, lt, cfg, device)
#             if r is not None:
#                 r['participant_id'] = uid
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
#             'AR': {'lr_pt': best_lr_pt_ar, 'lr_ft': best_lr_ft_ar},
#             'VA': {'lr_pt': best_lr_pt_va, 'lr_ft': best_lr_ft_va}},
#         **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
#         **ar_stds, **va_stds,
#         'test_results_per_participant_ar': results_ar,
#         'test_results_per_participant_va': results_va,
#         'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
#     }
#     with open(os.path.join(output_dir, 'transfer_mtl_results.pkl'), 'wb') as f:
#         pickle.dump(final, f)

#     print_determinism_summary(
#         {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
#         ar_stds, va_stds)

#     print(f"\n✓ All results saved to: {output_dir}")
#     print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")

"""
Transfer MTL — Pre-train MTLTransferModel on train participants, then for each
test participant: deep-copy the model, add a new task head via add_task_head(),
fine-tune backbone + new head with standard Adam + BCEWithLogitsLoss.

Usage
-----
    python transfer_mtl.py                  # runs on VREED (default)
    python transfer_mtl.py --dataset dssn_eq
"""
import argparse
import os, sys, time, copy, gc
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, EPOCHS, MAX_NORM, FT_EPOCHS, N_FOLDS,
                    TRANSFER_MTL_LR_PT, TRANSFER_MTL_LR_FT,
                    MTL_TASK_LR, TF_EPOCHS_PRE, 
                    L2_SHARED, L2_TASK, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from data import create_sliding_windows, make_mtl_loader, arrays_to_loader
from dataset_configs.loader import load_dataset
from models import MTLTransferModel
from utils import (set_all_seeds, compute_metrics_from_cm, make_kfolds,
                   aggregate_mtml_results, compute_per_participant_stds,
                   print_determinism_summary)


def parse_args():
    p = argparse.ArgumentParser(description='Transfer MTL')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def _pretrain_mtl(label_type, train_data_dict, lr_pt, lr_task,
                  cfg, device, output_dir):
    """Pre-train MTLTransferModel on train participants."""
    num_train = len(train_data_dict)
    loader, _, _ = make_mtl_loader(
        train_data_dict, cfg['window_size'], cfg['stride'],
        label_type=label_type, batch_size=num_train,
        seed=SEED, feature_cols=cfg['feature_cols'])

    model   = MTLTransferModel(num_train, input_dim=cfg['input_dim']).to(device)
    opt     = optim.Adam([
        {'params': model.backbone_parameters(),      'lr': lr_pt},
        {'params': model.task_specific_parameters(), 'lr': lr_task},
    ])
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    ckpt = os.path.join(output_dir, f'pretrain_{label_type}.pt')

    for epoch in range(TF_EPOCHS_PRE):
        model.train()
        running = 0.0
        for batch in loader:
            X_b, y_b, task_ids, _ = [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss_vec = loss_fn(model(X_b, task_ids), y_b).squeeze(-1)
            l2 = model.compute_l2(L2_SHARED, L2_TASK)
            total = loss_vec.mean() + l2
            if torch.isnan(total):
                raise ValueError(f"NaN at ep {epoch+1} [{label_type}]")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
            running += total.item()

        avg = running / len(loader)
        sched.step(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label_type.upper()}] Epoch {epoch+1}/{TF_EPOCHS_PRE}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    return model


def _finetune_user(base_model, user_train_df, label_type, lr_ft, uid,
                   cfg, device):
    """Deep-copy model, add new head, fine-tune backbone + new head only."""
    model = copy.deepcopy(base_model).to(device)
    local_idx = model.add_task_head()

    X, y_ar, y_va, _, _ = create_sliding_windows(
        user_train_df, cfg['window_size'], cfg['stride'],
        feature_cols=cfg['feature_cols'])
    if len(X) == 0:
        return model, local_idx

    y = y_ar if label_type == 'ar' else y_va

    loader = arrays_to_loader(X, y, batch_size=32, shuffle=True, seed=SEED)

    # Only optimise backbone + new head (old heads are frozen/ignored)
    backbone_params = list(model.backbone.parameters())
    new_head_params = (list(model.head1[local_idx].parameters()) +
                       list(model.head2[local_idx].parameters()) +
                       list(model.out[local_idx].parameters()))
    active_params = backbone_params + new_head_params

    opt     = optim.Adam(active_params, lr=lr_ft)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(FT_EPOCHS):
        model.train()
        ep_loss = 0.0
        nb = 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            tids = torch.full((X_b.size(0),), local_idx,
                              dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_b, tids), y_b)
            # L2 on backbone + new head only (not old pre-training heads)
            #l2 = (L2_SHARED * sum(p.norm(2)**2 for p in backbone_params if p.requires_grad) +
                  #L2_TASK   * sum(p.norm(2)**2 for p in new_head_params if p.requires_grad))
            l2 = (L2_SHARED * sum(p.norm(2)**2 for p in backbone_params if p.requires_grad and p.ndim >= 2) + 
                  L2_TASK   * sum(p.norm(2)**2 for p in new_head_params if p.requires_grad and p.ndim >= 2))
            
            total = loss + l2
            if torch.isnan(total):
                raise ValueError(f"NaN finetune [pid {uid}, ep {ep+1}]")
            total.backward()
            torch.nn.utils.clip_grad_norm_(active_params, MAX_NORM)
            opt.step()
            ep_loss += total.item()
            nb += 1
        if nb > 0:
            sched.step(ep_loss / nb)

    return model, local_idx


def _eval_user(model, user_test_df, local_idx, label_type, cfg, device):
    """Evaluate fine-tuned model on test data."""
    X, y_ar, y_va, _, _ = create_sliding_windows(
        user_test_df, cfg['window_size'], cfg['stride'],
        feature_cols=cfg['feature_cols'])
    if len(X) == 0:
        return None

    y = y_ar if label_type == 'ar' else y_va
    X_t  = torch.tensor(X, dtype=torch.float32).to(device)
    tids = torch.full((len(X),), local_idx, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t, tids)).cpu().numpy().flatten()

    y_true = y.astype(int)
    y_pred = (probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)

    p = label_type
    return {
        'participant_id': None,  # filled by caller
        'cm': cm,
        f'{p}_acc': acc, f'{p}_precision': prec,
        f'{p}_recall': rec, f'{p}_f1': f1,
        f'y_true_{p}': y_true, f'y_pred_{p}': y_pred,
        f'y_pred_probs_{p}': probs,
    }


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, df, cfg, device, output_dir):
    """K-fold CV on train participants to validate hyperparameters.

    For each fold: pre-train on fold's train participants, then for each
    val participant: deep-copy, add head, fine-tune on their train videos,
    evaluate on their test videos.
    """
    splits   = cfg['splits']
    train_ps = cfg['train_participants']
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Transfer MTL"
          f"  ({cfg['results_prefix']})\n{'='*60}")
    print(f"  LR_PT={TRANSFER_MTL_LR_PT}, LR_FT={TRANSFER_MTL_LR_FT}, "
          f"LR_TASK={MTL_TASK_LR}")
    print(f"  L2: Shared={L2_SHARED}, Task={L2_TASK}")

    results = []
    train_folds = make_kfolds(train_ps, seed=SEED)

    for lr_pt in [TRANSFER_MTL_LR_PT]:
        for lr_ft in [TRANSFER_MTL_LR_FT]:
            fold_f1s = []
            print(f"\nTesting LR_PT={lr_pt}, LR_FT={lr_ft}")

            for fold_i in range(N_FOLDS):
                print(f"  Processing Fold {fold_i + 1}/{N_FOLDS}")
                val_ps = train_folds[fold_i]
                tr_ps  = [p for j, f in enumerate(train_folds)
                          if j != fold_i for p in f]

                # Build train data for this fold
                fold_train_data = {}
                for idx, pid in enumerate(sorted(tr_ps)):
                    p_df = df[df['ID'] == pid].reset_index(drop=True)
                    fold_train_data[idx] = p_df[p_df['Trial'].isin(
                        splits[pid]['train'])].reset_index(drop=True)

                # Pre-train on fold's train participants
                set_all_seeds(SEED)
                base_model = _pretrain_mtl(label_type, fold_train_data,
                                           lr_pt, MTL_TASK_LR,
                                           cfg, device, output_dir)

                # Fine-tune + evaluate on each val participant
                val_f1s = []
                for uid in sorted(val_ps):
                    if uid not in splits:
                        continue
                    user_train_df = df[(df['ID'] == uid) & df['Trial'].isin(
                        splits[uid]['train'])].reset_index(drop=True)
                    user_test_df = df[(df['ID'] == uid) & df['Trial'].isin(
                        splits[uid]['test'])].reset_index(drop=True)

                    ft_model, local_idx = _finetune_user(
                        base_model, user_train_df, label_type,
                        lr_ft, uid, cfg, device)
                    r = _eval_user(ft_model, user_test_df, local_idx,
                                   label_type, cfg, device)
                    if r is not None:
                        y_true = r[f'y_true_{label_type}']
                        y_pred = r[f'y_pred_{label_type}']
                        val_f1s.append(f1_score(y_true, y_pred,
                                                average='macro',
                                                zero_division=0))
                    del ft_model

                if val_f1s:
                    fold_f1s.append(np.mean(val_f1s))
                    print(f"  Fold {fold_i + 1}: Val F1 = {fold_f1s[-1]:.4f}")

                del base_model
                torch.cuda.empty_cache()
                gc.collect()

            if not fold_f1s:
                continue
            avg = np.mean(fold_f1s)
            results.append({'lr_pt': lr_pt, 'lr_ft': lr_ft,
                            'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
            print(f"  Average F1: {avg:.4f}")

    if not results:
        return TRANSFER_MTL_LR_PT, TRANSFER_MTL_LR_FT
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir,
              f'{label_type}_tuning_results_transfer_mtl.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr_pt'], best['lr_ft']


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

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_transfer_mtl')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")
    print(f"Train: {train_ps}\nTest:  {test_ps}")
    print(f"\nL2: Shared={L2_SHARED}, Task={L2_TASK}")

    # Hyperparameter tuning on train participants only
    # best_lr_pt_ar, best_lr_ft_ar = hyperparameter_tuning(
    #     'ar', df, cfg, device, output_dir)
    # best_lr_pt_va, best_lr_ft_va = hyperparameter_tuning(
    #     'va', df, cfg, device, output_dir)
    
    best_lr_pt_ar = best_lr_pt_va = TRANSFER_MTL_LR_PT
    best_lr_ft_ar = best_lr_ft_va = TRANSFER_MTL_LR_FT

    # Build train data dict
    train_data = {}
    for idx, pid in enumerate(sorted(train_ps)):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        train_data[idx] = p_df[p_df['Trial'].isin(
            splits[pid]['train'])].reset_index(drop=True)

    for lt in ['ar', 'va']:
        if lt == 'ar':
            lr_pt, lr_ft = best_lr_pt_ar, best_lr_ft_ar
        else:
            lr_pt, lr_ft = best_lr_pt_va, best_lr_ft_va

        print(f"\n{'='*60}\nPRETRAINING {lt.upper()}\n{'='*60}")
        set_all_seeds(SEED)
        base_model = _pretrain_mtl(lt, train_data, lr_pt, MTL_TASK_LR,
                                   cfg, device, output_dir)
        torch.save(base_model.state_dict(),
                   os.path.join(output_dir, f'base_model_{lt}_final.pth'))

        print(f"\n{'='*60}\nFINE-TUNING + EVAL {lt.upper()} — TEST PARTICIPANTS\n{'='*60}")
        results = []
        for uid in sorted(test_ps):
            if uid not in splits:
                continue

            user_train_df = df[(df['ID'] == uid) & df['Trial'].isin(
                splits[uid]['train'])].reset_index(drop=True)
            user_test_df = df[(df['ID'] == uid) & df['Trial'].isin(
                splits[uid]['test'])].reset_index(drop=True)

            print(f"\n  Participant {uid}: Fine-tuning {lt.upper()} model")
            ft_model, local_idx = _finetune_user(
                base_model, user_train_df, lt,
                lr_ft, uid, cfg, device)

            r = _eval_user(ft_model, user_test_df, local_idx, lt, cfg, device)
            if r is not None:
                r['participant_id'] = uid
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
            'AR': {'lr_pt': best_lr_pt_ar, 'lr_ft': best_lr_ft_ar},
            'VA': {'lr_pt': best_lr_pt_va, 'lr_ft': best_lr_ft_va}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **ar_stds, **va_stds,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'transfer_mtl_results.pkl'), 'wb') as f:
        pickle.dump(final, f)

    print_determinism_summary(
        {f'ar_{k}': final[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")