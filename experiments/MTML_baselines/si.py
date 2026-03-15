"""
Subject-Independent (SI) Baseline
Train on 20 participants, evaluate on 6 held-out test participants.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

import numpy as np
import pickle  # still needed for saving results
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import HARDCODED_SPLITS, SEED, WINDOW_SIZE, STRIDE, MAX_NORM, EPOCHS
from utils import set_all_seeds, compute_metrics_from_cm, safe_roc_auc, F1Score, create_kfold_splits, make_kfolds
from models import SingleTaskModel
from dataset_configs.vreed import load_vreed_df
from paths import RESULTS_DIR

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir = os.path.join(BASE_OUTPUT_DIR, 'VREED_SI')
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE = 32
N_FOLDS = 5
learning_rates = [3e-4]
l2_lambdas = [1e-5]

set_all_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df(preserve_trial_order=True)
df['participant_trial_encoded'] = df['ID_video'].astype(str)

existing_ids = set(df['ID'].unique())
participant_ids = sorted([int(pid) for pid in hardcoded_splits.keys() if pid in existing_ids])
print(f"Total participants: {len(participant_ids)}")

test_participants  = [105, 109, 112, 125, 131, 132]
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {train_participants}\nTest:  {test_participants}")


# =============================
# HELPERS
# =============================
def get_frames(XY, window_size, stride, label_type='AR'):
    frames, labels, pte_list = [], [], []
    for pte in sorted(XY['participant_trial_encoded'].unique()):
        cur = XY[XY['participant_trial_encoded'] == pte]
        if cur.empty:
            continue
        ecg = cur['ECG'].values
        gsr = cur['GSR'].values
        orig_len = len(cur)
        if orig_len < window_size:
            pad = window_size - orig_len
            ecg = np.pad(ecg, (0, pad), constant_values=0)
            gsr = np.pad(gsr, (0, pad), constant_values=0)
        combined = np.stack([ecg, gsr], axis=1)
        T = len(combined)
        last_i = 0
        for i in range(0, T - window_size + 1, stride):
            frames.append(combined[i:i + window_size])
            labels.append(cur[f'{label_type}_Rating'].iloc[min(i + window_size - 1, orig_len - 1)])
            pte_list.append(pte)
            last_i = i
        next_i = last_i + stride
        if next_i < T:
            tail = combined[next_i:]
            pad = window_size - len(tail)
            if pad > 0:
                tail = np.pad(tail, ((0, pad), (0, 0)), constant_values=0)
            frames.append(tail)
            labels.append(cur[f'{label_type}_Rating'].iloc[-1])
            pte_list.append(pte)
    return np.array(frames), np.array(labels), pte_list


def make_loader(X, y, shuffle):
    X_t = torch.tensor(X.astype('float32'))
    y_t = torch.tensor(y.astype('float32')).reshape(-1, 1)
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0,
                      generator=g if shuffle else None)


def train_model(frames, labels, lr, l2_lambda, epochs=EPOCHS):
    set_all_seeds(SEED)
    model = SingleTaskModel().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loader = make_loader(frames, labels, shuffle=True)
    for epoch in range(epochs):
        model.train()
        run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X_b), y_b).mean()
            l2 = l2_lambda * sum(p.norm(2)**2 for p in model.parameters() if p.requires_grad)
            total = loss + l2
            if torch.isnan(total): return None
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += total.item()
        sched.step(run / len(loader))
    return model


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
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type}] SI\n{'='*60}")
    results = []
    train_folds = create_participant_kfolds(train_participants)
    for lr in learning_rates:
        for l2 in l2_lambdas:
            fold_f1s = []
            for fold_i in range(N_FOLDS):
                val_ps = train_folds[fold_i]
                tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
                tr_pte = [f"{p}_{v}" for p in tr_ps if p in hardcoded_splits
                          for v in hardcoded_splits[p]['train']]
                va_pte = [f"{p}_{v}" for p in val_ps if p in hardcoded_splits
                          for v in hardcoded_splits[p]['train']]
                tr_df = df[df['participant_trial_encoded'].isin(tr_pte)].reset_index(drop=True)
                va_df = df[df['participant_trial_encoded'].isin(va_pte)].reset_index(drop=True)
                Xtr, ytr, _ = get_frames(tr_df, WINDOW_SIZE, STRIDE, label_type)
                Xva, yva, _ = get_frames(va_df, WINDOW_SIZE, STRIDE, label_type)
                if len(Xtr) == 0 or len(Xva) == 0: fold_f1s.append(0.0); continue
                model = train_model(Xtr, ytr, lr, l2)
                if model is None: fold_f1s.append(0.0); continue
                model.eval()
                f1m = F1Score()
                loader = make_loader(Xva, yva, shuffle=False)
                with torch.no_grad():
                    for X_v, y_v in loader:
                        f1m.update_state(y_v.to(device), model(X_v.to(device)))
                fold_f1s.append(f1m.result())
                print(f"  fold {fold_i+1}: f1={fold_f1s[-1]:.4f}")
            avg = np.mean(fold_f1s)
            results.append({'lr': lr, 'l2': l2, 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
            print(f"  avg f1={avg:.4f}")
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type.lower()}_tuning_results_si.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr'], best['l2']


# =============================
# EVALUATION HELPER
# =============================
def evaluate_per_participant(model, test_participants, test_data, label_type):
    model.eval()
    results = []
    for pid in sorted(test_participants):
        test_trials = [f"{pid}_{v}" for v in hardcoded_splits[pid]['test']]
        p_df = test_data[test_data['participant_trial_encoded'].isin(test_trials)].reset_index(drop=True)
        if len(p_df) == 0: continue
        X, y, _ = get_frames(p_df, WINDOW_SIZE, STRIDE, label_type)
        if len(X) == 0: continue
        loader = make_loader(X, y, shuffle=False)
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
        results.append({'participant_id': pid, 'y_true': y_true, 'y_pred': y_pred,
                        'y_pred_probs': y_prob, 'cm': cm,
                        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
        print(f"  Participant {pid}: Acc={acc:.4f} F1={f1:.4f}")
    return results


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    best_lr_ar, best_l2_ar = hyperparameter_tuning('AR')
    best_lr_va, best_l2_va = hyperparameter_tuning('VA')

    # Build train/test datasets
    train_pte = [f"{p}_{v}" for p in train_participants if p in hardcoded_splits
                 for v in hardcoded_splits[p]['train']]
    test_pte  = [f"{p}_{v}" for p in test_participants if p in hardcoded_splits
                 for v in hardcoded_splits[p]['test']]
    train_data = df[df['participant_trial_encoded'].isin(train_pte)].reset_index(drop=True)
    test_data  = df[df['participant_trial_encoded'].isin(test_pte)].reset_index(drop=True)

    # Train AR
    print('\n' + '='*60 + '\nTRAINING AR\n' + '='*60)
    Xtr_ar, ytr_ar, _ = get_frames(train_data, WINDOW_SIZE, STRIDE, 'AR')
    set_all_seeds(SEED)
    model_ar = train_model(Xtr_ar, ytr_ar, best_lr_ar, best_l2_ar)
    torch.save(model_ar.state_dict(), os.path.join(output_dir, 'model_ar_si.pth'))

    # Train VA
    print('\n' + '='*60 + '\nTRAINING VA\n' + '='*60)
    Xtr_va, ytr_va, _ = get_frames(train_data, WINDOW_SIZE, STRIDE, 'VA')
    set_all_seeds(SEED)
    model_va = train_model(Xtr_va, ytr_va, best_lr_va, best_l2_va)
    torch.save(model_va.state_dict(), os.path.join(output_dir, 'model_va_si.pth'))

    # Evaluate
    print('\n' + '='*60 + '\nEVALUATION AR\n' + '='*60)
    results_ar = evaluate_per_participant(model_ar, test_participants, test_data, 'AR')
    print('\n' + '='*60 + '\nEVALUATION VA\n' + '='*60)
    results_va = evaluate_per_participant(model_va, test_participants, test_data, 'VA')

    # Aggregate
    def aggregate(results, label):
        all_true  = np.concatenate([r['y_true'] for r in results])
        all_pred  = np.concatenate([r['y_pred'] for r in results])
        all_probs = np.concatenate([r['y_pred_probs'] for r in results])
        cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
        acc, prec, rec, f1 = compute_metrics_from_cm(cm)
        auc_val, fpr, tpr = safe_roc_auc(all_true, all_probs)
        metric_lists = {k: [r[k] for r in results] for k in ['accuracy','precision','recall','f1']}
        auc_list = []
        for r in results:
            try: auc_list.append(roc_auc_score(r['y_true'], r['y_pred_probs']))
            except: auc_list.append(np.nan)
        stds = {k: np.std(v, ddof=1) for k, v in metric_lists.items()}
        stds['auc'] = np.std([x for x in auc_list if not np.isnan(x)], ddof=1)
        print(f"\n{label} Metrics:")
        print(f"  AUC={auc_val:.4f}±{stds['auc']:.4f}  Acc={acc:.4f}±{stds['accuracy']:.4f}  "
              f"F1={f1:.4f}±{stds['f1']:.4f}")
        return all_true, all_pred, all_probs, cm, acc, prec, rec, f1, auc_val, fpr, tpr, stds

    from sklearn.metrics import roc_auc_score
    all_true_ar, all_pred_ar, all_probs_ar, cm_AR, ar_acc, ar_prec, ar_rec, ar_f1, ar_auc, ar_fpr, ar_tpr, ar_stds = aggregate(results_ar, 'AR')
    all_true_va, all_pred_va, all_probs_va, cm_VA, va_acc, va_prec, va_rec, va_f1, va_auc, va_fpr, va_tpr, va_stds = aggregate(results_va, 'VA')

    # Save ROC data
    roc_data = {
        'AR': {'true': all_true_ar, 'probs': all_probs_ar},
        'VA': {'true': all_true_va, 'probs': all_probs_va},
    }
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    # Confusion matrix plots
    for cm, label, fname in [(cm_AR, 'AR', 'ar_cm_si.png'), (cm_VA, 'VA', 'va_cm_si.png')]:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'{label}=0', f'{label}=1'],
                    yticklabels=[f'{label}=0', f'{label}=1'])
        plt.title(f'Confusion Matrix {label} (SI)')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, fname)); plt.close()

    # Save results
    final_results = {
        'train_participants': train_participants, 'test_participants': test_participants,
        'best_hyperparameters': {'AR': {'lr': best_lr_ar, 'l2': best_l2_ar},
                                  'VA': {'lr': best_lr_va, 'l2': best_l2_va}},
        'ar_acc': ar_acc, 'ar_precision': ar_prec, 'ar_recall': ar_rec,
        'ar_f1': ar_f1,   'ar_auc': ar_auc,
        'va_acc': va_acc, 'va_precision': va_prec, 'va_recall': va_rec,
        'va_f1': va_f1,   'va_auc': va_auc,
        **{f'ar_{k}_std': v for k, v in ar_stds.items()},
        **{f'va_{k}_std': v for k, v in va_stds.items()},
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': cm_AR, 'cm_va': cm_VA,
    }
    with open(os.path.join(output_dir, 'si_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    # =============================
    # DETERMINISM SUMMARY
    # =============================
    from utils import compute_per_participant_stds, print_determinism_summary

    def _prefix(results, prefix):
        return [{f"{prefix}_acc": r["accuracy"], f"{prefix}_precision": r["precision"],
                 f"{prefix}_recall": r["recall"], f"{prefix}_f1": r["f1"],
                 f"y_true_{prefix}": r["y_true"], f"y_pred_probs_{prefix}": r["y_pred_probs"]}
                for r in results]

    ar_stds = compute_per_participant_stds(_prefix(results_ar, "ar"), "ar")
    va_stds = compute_per_participant_stds(_prefix(results_va, "va"), "va")
    print_determinism_summary(
        {f"ar_{k}": final_results[f"ar_{k}"] for k in ["auc", "acc", "precision", "recall", "f1"]},
        {f"va_{k}": final_results[f"va_{k}"] for k in ["auc", "acc", "precision", "recall", "f1"]},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
