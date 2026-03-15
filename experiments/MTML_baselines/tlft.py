"""
Transfer Learning + Fine-Tuning (TL-FT)
Pre-train on 20 participants, fine-tune per test participant.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import HARDCODED_SPLITS, SEED, WINDOW_SIZE, STRIDE, MAX_NORM, EPOCHS
from utils import set_all_seeds, compute_metrics_from_cm, safe_roc_auc, create_kfold_splits, make_kfolds
from models import SingleTaskModel
from dataset_configs.vreed import load_vreed_df
from paths import RESULTS_DIR

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir = os.path.join(BASE_OUTPUT_DIR, 'VREED_TF')
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE     = 32
EPOCHS_PRETRAIN = EPOCHS
EPOCHS_FINETUNE = 10
N_FOLDS = 5
learning_rates_pre = [1e-3]
learning_rates_ft  = [1e-3]
l2_lambdas = [1e-5]

set_all_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df()
df['participant_trial_encoded'] = df['ID_video'].astype(str)

existing_ids = set(df['ID'].unique())
participant_ids = sorted([int(pid) for pid in hardcoded_splits.keys() if pid in existing_ids])

test_participants  = [105, 109, 112, 125, 131, 132]
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {train_participants}\nTest:  {test_participants}")


# =============================
# HELPERS
# =============================
def get_frames(XY, window_size, stride, label_type='AR'):
    frames, labels = [], []
    for pte in XY['participant_trial_encoded'].unique():
        cur = XY[XY['participant_trial_encoded'] == pte]
        if len(cur) == 0: continue
        ecg = cur['ECG'].values; gsr = cur['GSR'].values; orig_len = len(cur)
        if orig_len < window_size:
            pad = window_size - orig_len
            ecg = np.pad(ecg, (0, pad), constant_values=0)
            gsr = np.pad(gsr, (0, pad), constant_values=0)
        combined = np.stack([ecg, gsr], axis=1); T = len(combined); last_i = 0
        for i in range(0, T - window_size + 1, stride):
            frames.append(combined[i:i + window_size])
            labels.append(cur[f'{label_type}_Rating'].iloc[min(i + window_size - 1, orig_len - 1)])
            last_i = i
        next_i = last_i + stride
        if next_i < T:
            tail = combined[next_i:]
            pad = window_size - len(tail)
            if pad > 0: tail = np.pad(tail, ((0, pad), (0, 0)), constant_values=0)
            frames.append(tail); labels.append(cur[f'{label_type}_Rating'].iloc[-1])
    return np.array(frames), np.array(labels)


def make_loader(X, y, shuffle, seed=SEED):
    X_t = torch.tensor(X.astype('float32'))
    y_t = torch.tensor(y.astype('float32')).reshape(-1, 1)
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0, generator=g if shuffle else None)


def pretrain(X, y, lr, l2_lambda, epochs):
    set_all_seeds(SEED)
    model = SingleTaskModel().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = make_loader(X, y, shuffle=True)
    for ep in range(epochs):
        model.train(); run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X_b), y_b)
            l2 = l2_lambda * sum(p.norm(2)**2 for p in model.parameters() if p.requires_grad)
            total = loss + l2
            if torch.isnan(total): return None
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += total.item()
        sched.step(run / len(loader))
    return model


def finetune(base_model, X, y, lr, l2_lambda, epochs, pid):
    import copy
    model = SingleTaskModel().to(device)
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator(); g.manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.tensor(X.astype('float32')),
                                      torch.tensor(y.astype('float32')).reshape(-1, 1)),
                        batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0)
    for ep in range(epochs):
        model.train(); run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X_b), y_b)
            l2 = l2_lambda * sum(p.norm(2)**2 for p in model.parameters() if p.requires_grad)
            total = loss + l2
            if torch.isnan(total): return None
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += total.item()
        sched.step(run / len(loader))
    return model


def eval_model(model, X, y):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X.astype('float32')),
                                      torch.tensor(y.astype('float32')).reshape(-1, 1)),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    probs, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            probs.extend(torch.sigmoid(model(X_b.to(device))).cpu().numpy().flatten())
            trues.extend(y_b.numpy().flatten())
    y_true = np.array(trues).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)
    return {'y_true': y_true, 'y_pred': y_pred, 'y_pred_probs': y_prob,
            'cm': cm, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# =============================
# HYPERPARAMETER TUNING
# =============================
def create_participant_kfolds(participants, k=5):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(participants)
    folds, start = [], 0
    for i in range(k):
        size = len(perm) // k + (1 if i < len(perm) % k else 0)
        folds.append(list(perm[start:start + size])); start += size
    return folds


def hyperparameter_tuning(label_type='AR'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type}] TL-FT\n{'='*60}")
    results = []
    train_folds = create_participant_kfolds(train_participants)
    for lr_pre in learning_rates_pre:
        for lr_ft in learning_rates_ft:
            for l2 in l2_lambdas:
                fold_f1s = []
                for fold_i in range(N_FOLDS):
                    val_ps = train_folds[fold_i]
                    tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
                    tr_pte = [f"{p}_{v}" for p in tr_ps if p in hardcoded_splits
                              for v in hardcoded_splits[p]['train']]
                    tr_df = df[df['participant_trial_encoded'].isin(tr_pte)].reset_index(drop=True)
                    Xpre, ypre = get_frames(tr_df, WINDOW_SIZE, STRIDE, label_type)
                    if len(Xpre) == 0: continue
                    base = pretrain(Xpre, ypre, lr_pre, l2, EPOCHS_PRETRAIN)
                    if base is None: continue
                    val_f1s = []
                    for pid in val_ps:
                        if pid not in hardcoded_splits: continue
                        tr_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['train']]
                        te_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['test']]
                        u_tr = df[df['participant_trial_encoded'].isin(tr_pte_u)].reset_index(drop=True)
                        u_te = df[df['participant_trial_encoded'].isin(te_pte_u)].reset_index(drop=True)
                        Xft, yft = get_frames(u_tr, WINDOW_SIZE, STRIDE, label_type)
                        Xte, yte = get_frames(u_te, WINDOW_SIZE, STRIDE, label_type)
                        if len(Xft) == 0 or len(Xte) == 0: continue
                        ft = finetune(base, Xft, yft, lr_ft, l2, EPOCHS_FINETUNE, pid)
                        if ft is None: continue
                        r = eval_model(ft, Xte, yte)
                        val_f1s.append(f1_score(r['y_true'], r['y_pred'], average='macro', zero_division=0))
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


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    best_lr_pre_ar, best_lr_ft_ar, best_l2_ar = hyperparameter_tuning('AR')
    best_lr_pre_va, best_lr_ft_va, best_l2_va = hyperparameter_tuning('VA')

    # Build pretrain data from train participants
    tr_pte = [f"{p}_{v}" for p in train_participants if p in hardcoded_splits
              for v in hardcoded_splits[p]['train']]
    pretrain_df = df[df['participant_trial_encoded'].isin(tr_pte)].reset_index(drop=True)

    print('\n' + '='*60 + '\nPRETRAINING AR\n' + '='*60)
    Xpre_ar, ypre_ar = get_frames(pretrain_df, WINDOW_SIZE, STRIDE, 'AR')
    set_all_seeds(SEED)
    base_ar = pretrain(Xpre_ar, ypre_ar, best_lr_pre_ar, best_l2_ar, EPOCHS_PRETRAIN)
    torch.save(base_ar.state_dict(), os.path.join(output_dir, 'base_model_ar_final.pth'))

    print('\n' + '='*60 + '\nPRETRAINING VA\n' + '='*60)
    Xpre_va, ypre_va = get_frames(pretrain_df, WINDOW_SIZE, STRIDE, 'VA')
    set_all_seeds(SEED)
    base_va = pretrain(Xpre_va, ypre_va, best_lr_pre_va, best_l2_va, EPOCHS_PRETRAIN)
    torch.save(base_va.state_dict(), os.path.join(output_dir, 'base_model_va_final.pth'))

    results_ar, results_va = [], []
    for pid in sorted(test_participants):
        if pid not in hardcoded_splits: continue
        tr_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['train']]
        te_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['test']]
        u_tr = df[df['participant_trial_encoded'].isin(tr_pte_u)].reset_index(drop=True)
        u_te = df[df['participant_trial_encoded'].isin(te_pte_u)].reset_index(drop=True)

        Xft_ar, yft_ar = get_frames(u_tr, WINDOW_SIZE, STRIDE, 'AR')
        Xte_ar, yte_ar = get_frames(u_te, WINDOW_SIZE, STRIDE, 'AR')
        Xft_va, yft_va = get_frames(u_tr, WINDOW_SIZE, STRIDE, 'VA')
        Xte_va, yte_va = get_frames(u_te, WINDOW_SIZE, STRIDE, 'VA')

        print(f"\nParticipant {pid}: fine-tuning")
        ft_ar = finetune(base_ar, Xft_ar, yft_ar, best_lr_ft_ar, best_l2_ar, EPOCHS_FINETUNE, pid)
        ft_va = finetune(base_va, Xft_va, yft_va, best_lr_ft_va, best_l2_va, EPOCHS_FINETUNE, pid)

        r_ar = eval_model(ft_ar, Xte_ar, yte_ar); r_ar['participant_id'] = pid; results_ar.append(r_ar)
        r_va = eval_model(ft_va, Xte_va, yte_va); r_va['participant_id'] = pid; results_va.append(r_va)
        print(f"  AR Acc={r_ar['accuracy']:.4f} F1={r_ar['f1']:.4f} | "
              f"VA Acc={r_va['accuracy']:.4f} F1={r_va['f1']:.4f}")

    def aggregate_and_save(results, label, fpr_key, pkl_name):
        all_true  = np.concatenate([r['y_true'] for r in results])
        all_pred  = np.concatenate([r['y_pred'] for r in results])
        all_probs = np.concatenate([r['y_pred_probs'] for r in results])
        cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
        acc, prec, rec, f1 = compute_metrics_from_cm(cm)
        auc_val, fpr, tpr = safe_roc_auc(all_true, all_probs)
        print(f"\n{label}: Acc={acc:.4f} F1={f1:.4f} AUC={auc_val:.4f}")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'{label}=0', f'{label}=1'],
                    yticklabels=[f'{label}=0', f'{label}=1'])
        plt.title(f'CM {label} (TL-FT)'); plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, f'{label.lower()}_cm_tlft.png')); plt.close()
        return all_true, all_probs, cm, acc, prec, rec, f1, auc_val, fpr, tpr

    all_true_ar, all_probs_ar, cm_AR, ar_acc, ar_prec, ar_rec, ar_f1, ar_auc, ar_fpr, ar_tpr = aggregate_and_save(results_ar, 'AR', 'ar_fpr', 'ar')
    all_true_va, all_probs_va, cm_VA, va_acc, va_prec, va_rec, va_f1, va_auc, va_fpr, va_tpr = aggregate_and_save(results_va, 'VA', 'va_fpr', 'va')

    roc_data = {'AR': {'true': all_true_ar, 'probs': all_probs_ar},
                'VA': {'true': all_true_va, 'probs': all_probs_va}}
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    final_results = {
        'train_participants': train_participants, 'test_participants': test_participants,
        'best_hyperparameters': {
            'AR': {'lr_pre': best_lr_pre_ar, 'lr_ft': best_lr_ft_ar, 'l2': best_l2_ar},
            'VA': {'lr_pre': best_lr_pre_va, 'lr_ft': best_lr_ft_va, 'l2': best_l2_va}},
        'ar_acc': ar_acc, 'ar_precision': ar_prec, 'ar_recall': ar_rec, 'ar_f1': ar_f1, 'ar_auc': ar_auc,
        'va_acc': va_acc, 'va_precision': va_prec, 'va_recall': va_rec, 'va_f1': va_f1, 'va_auc': va_auc,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': cm_AR, 'cm_va': cm_VA,
    }
    with open(os.path.join(output_dir, 'tlft_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    print(f"\n✓ All results saved to: {output_dir}")
