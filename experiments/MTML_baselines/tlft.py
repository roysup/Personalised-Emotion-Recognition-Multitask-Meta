"""
Transfer Learning + Fine-Tuning (TL-FT)
Pre-train on train participants, fine-tune per test participant.

Usage
-----
    python tlft.py --dataset dssn_eq
"""
import argparse
import os, sys, time, copy
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

from config import (SEED, EPOCHS, MAX_NORM, N_FOLDS, PSTL_BATCH_SIZE,
                    TF_LR_PRE, TF_LR_FT, FT_EPOCHS, L2_TASK, RESULTS_DIR)
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from utils import (set_all_seeds, compute_metrics_from_cm,
                   aggregate_mtml_results, make_kfolds,
                   compute_per_participant_stds, print_determinism_summary)
from data import create_sliding_windows, arrays_to_loader
from dataset_configs.loader import load_dataset
from models import SingleTaskModel


def parse_args():
    p = argparse.ArgumentParser(description='TL-FT baseline')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    return p.parse_args()


def _get_windows(sub_df, label_type, cfg):
    X, y_ar, y_va, _, _ = create_sliding_windows(
        sub_df, cfg['window_size'], cfg['stride'],
        trial_col='participant_trial_encoded', feature_cols=cfg['feature_cols'])
    return X, (y_ar if label_type.upper() == 'AR' else y_va)


def pretrain(X, y, lr, l2_lambda, epochs, cfg, device):
    model   = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader  = arrays_to_loader(X, y, PSTL_BATCH_SIZE, shuffle=True, seed=SEED)
    for ep in range(epochs):
        model.train(); run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_b), y_b)
            if torch.isnan(loss): raise ValueError(f"NaN in pretrain [ep {ep+1}]")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
        sched.step(run / len(loader))
    return model


def finetune(base_model, X, y, lr, l2_lambda, epochs, pid, cfg, device):
    model = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader  = arrays_to_loader(X, y, PSTL_BATCH_SIZE, shuffle=True, seed=SEED + pid)
    for ep in range(epochs):
        model.train(); run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_b), y_b)
            if torch.isnan(loss): raise ValueError(f"NaN finetune [pid {pid}, ep {ep+1}]")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += loss.item()
        sched.step(run / len(loader))
    return model


def eval_model(model, X, y, device, cfg):
    model.eval()
    loader = arrays_to_loader(X, y, PSTL_BATCH_SIZE, shuffle=False)
    probs, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            probs.extend(torch.sigmoid(model(X_b.to(device, non_blocking=True))).cpu().numpy().flatten())
            trues.extend(y_b.numpy().flatten())
    y_true = np.array(trues).astype(int); y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)
    return {'y_true': y_true, 'y_pred': y_pred, 'y_pred_probs': y_prob,
            'cm': cm, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def _prefix_result(r_raw, pid, label):
    p = label.lower()
    return {
        'participant_id': pid, 'cm': r_raw['cm'],
        f'{p}_acc': r_raw['accuracy'], f'{p}_precision': r_raw['precision'],
        f'{p}_recall': r_raw['recall'], f'{p}_f1': r_raw['f1'],
        f'y_true_{p}': r_raw['y_true'], f'y_pred_{p}': r_raw['y_pred'],
        f'y_pred_probs_{p}': r_raw['y_pred_probs'],
    }


def hyperparameter_tuning(label_type, df, cfg, device, output_dir):
    splits = cfg['splits']
    train_ps = cfg['train_participants']
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type}] TL-FT"
          f"  ({cfg['results_prefix']})\n{'='*60}")
    results     = []
    train_folds = make_kfolds(train_ps, seed=SEED)
    for lr_pre in [TF_LR_PRE]:
        for lr_ft in [TF_LR_FT]:
            for l2 in [L2_TASK]:
                fold_f1s = []
                for fold_i in range(N_FOLDS):
                    val_ps = train_folds[fold_i]
                    tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
                    tr_pte = [f"{p}_{v}" for p in tr_ps if p in splits for v in splits[p]['train']]
                    tr_df  = df[df['participant_trial_encoded'].isin(tr_pte)].reset_index(drop=True)
                    Xpre, ypre = _get_windows(tr_df, label_type, cfg)
                    if len(Xpre) == 0: continue
                    base = pretrain(Xpre, ypre, lr_pre, l2, EPOCHS, cfg, device)
                    val_f1s = []
                    for pid in val_ps:
                        if pid not in splits: continue
                        tr_u = df[df['participant_trial_encoded'].isin(
                            [f"{pid}_{v}" for v in splits[pid]['train']])].reset_index(drop=True)
                        te_u = df[df['participant_trial_encoded'].isin(
                            [f"{pid}_{v}" for v in splits[pid]['test']])].reset_index(drop=True)
                        Xft, yft = _get_windows(tr_u, label_type, cfg)
                        Xte, yte = _get_windows(te_u, label_type, cfg)
                        if len(Xft) == 0 or len(Xte) == 0: continue
                        ft = finetune(base, Xft, yft, lr_ft, l2, FT_EPOCHS, pid, cfg, device)
                        r = eval_model(ft, Xte, yte, device, cfg)
                        val_f1s.append(f1_score(r['y_true'], r['y_pred'],
                                                average='macro', zero_division=0))
                    if val_f1s:
                        fold_f1s.append(np.mean(val_f1s))
                        print(f"  fold {fold_i+1}: f1={fold_f1s[-1]:.4f}")
                if not fold_f1s: continue
                avg = np.mean(fold_f1s)
                results.append({'lr_pre': lr_pre, 'lr_ft': lr_ft, 'l2': l2,
                                 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
                print(f"  avg f1={avg:.4f}")
    if not results: return 1e-3, 1e-3, 0.0
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type.lower()}_tuning_results_tlft.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr_pre'], best['lr_ft'], best['l2']


if __name__ == '__main__':
    args = parse_args()
    experiment_t0 = time.time()

    df, cfg = load_dataset(args.dataset)
    splits  = cfg['splits']
    prefix  = cfg['results_prefix']
    train_ps = cfg['train_participants']
    test_ps  = cfg['test_participants']

    output_dir = os.path.join(RESULTS_DIR, f'{prefix}_MTML', f'{prefix}_TF')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(SEED)
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    print(f"Device: {device}\nDataset: {args.dataset}\nOutput: {output_dir}")

    df['participant_trial_encoded'] = df[cfg['id_trial_col']].astype(str)

    best_lr_pre_ar, best_lr_ft_ar, best_l2_ar = hyperparameter_tuning('AR', df, cfg, device, output_dir)
    best_lr_pre_va, best_lr_ft_va, best_l2_va = hyperparameter_tuning('VA', df, cfg, device, output_dir)

    tr_pte      = [f"{p}_{v}" for p in train_ps if p in splits for v in splits[p]['train']]
    pretrain_df = df[df['participant_trial_encoded'].isin(tr_pte)].reset_index(drop=True)

    print('\n' + '='*60 + '\nPRETRAINING AR\n' + '='*60)
    Xpre_ar, ypre_ar = _get_windows(pretrain_df, 'AR', cfg)
    set_all_seeds(SEED)
    base_ar = pretrain(Xpre_ar, ypre_ar, best_lr_pre_ar, best_l2_ar, EPOCHS, cfg, device)
    torch.save(base_ar.state_dict(), os.path.join(output_dir, 'base_model_ar_final.pth'))

    print('\n' + '='*60 + '\nPRETRAINING VA\n' + '='*60)
    Xpre_va, ypre_va = _get_windows(pretrain_df, 'VA', cfg)
    set_all_seeds(SEED)
    base_va = pretrain(Xpre_va, ypre_va, best_lr_pre_va, best_l2_va, EPOCHS, cfg, device)
    torch.save(base_va.state_dict(), os.path.join(output_dir, 'base_model_va_final.pth'))

    results_ar, results_va = [], []
    for pid in sorted(test_ps):
        if pid not in splits: continue
        tr_u = df[df['participant_trial_encoded'].isin(
            [f"{pid}_{v}" for v in splits[pid]['train']])].reset_index(drop=True)
        te_u = df[df['participant_trial_encoded'].isin(
            [f"{pid}_{v}" for v in splits[pid]['test']])].reset_index(drop=True)
        Xft_ar, yft_ar = _get_windows(tr_u, 'AR', cfg)
        Xte_ar, yte_ar = _get_windows(te_u, 'AR', cfg)
        Xft_va, yft_va = _get_windows(tr_u, 'VA', cfg)
        Xte_va, yte_va = _get_windows(te_u, 'VA', cfg)

        print(f"\nParticipant {pid}: fine-tuning")
        ft_ar = finetune(base_ar, Xft_ar, yft_ar, best_lr_ft_ar, best_l2_ar, FT_EPOCHS, pid, cfg, device)
        ft_va = finetune(base_va, Xft_va, yft_va, best_lr_ft_va, best_l2_va, FT_EPOCHS, pid, cfg, device)
        r_ar = _prefix_result(eval_model(ft_ar, Xte_ar, yte_ar, device, cfg), pid, 'ar')
        r_va = _prefix_result(eval_model(ft_va, Xte_va, yte_va, device, cfg), pid, 'va')
        results_ar.append(r_ar); results_va.append(r_va)
        print(f"  AR Acc={r_ar['ar_acc']:.4f} F1={r_ar['ar_f1']:.4f} | "
              f"VA Acc={r_va['va_acc']:.4f} F1={r_va['va_f1']:.4f}")

    agg = aggregate_mtml_results(results_ar, results_va)
    roc_data = {'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
                'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']}}
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')
    final_results = {
        'train_participants': train_ps, 'test_participants': test_ps,
        'best_hyperparameters': {
            'AR': {'lr_pre': best_lr_pre_ar, 'lr_ft': best_lr_ft_ar, 'l2': best_l2_ar},
            'VA': {'lr_pre': best_lr_pre_va, 'lr_ft': best_lr_ft_va, 'l2': best_l2_va}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'tlft_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    print_determinism_summary(
        {f'ar_{k}': final_results[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final_results[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
