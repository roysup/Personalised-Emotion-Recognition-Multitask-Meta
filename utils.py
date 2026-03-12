import os
import random
import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix


# =============================
# REPRODUCIBILITY
# =============================

# def set_all_seeds(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cudnn.allow_tf32 = False
#     torch.use_deterministic_algorithms(True)


# =============================
# METRICS
# =============================

# def safe_roc_auc(y_true, y_score):
#     """Return (auc, fpr, tpr) or (nan, None, None) if only one class present."""
#     if len(np.unique(y_true)) < 2:
#         return np.nan, None, None
#     fpr, tpr, _ = roc_curve(y_true, y_score)
#     return auc(fpr, tpr), fpr, tpr


# def compute_metrics_from_cm(cm):
#     """
#     From a 2x2 confusion matrix return (accuracy, macro_precision, macro_recall, macro_f1).
#     Convention: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP.
#     """
#     tn, fp, fn, tp = cm.ravel()

#     accuracy = (tp + tn) / (tp + tn + fp + fn)

#     precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
#     precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     macro_precision = (precision_0 + precision_1) / 2

#     recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#     recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     macro_recall = (recall_0 + recall_1) / 2

#     f1_0 = (2 * precision_0 * recall_0 / (precision_0 + recall_0)
#             if (precision_0 + recall_0) > 0 else 0.0)
#     f1_1 = (2 * precision_1 * recall_1 / (precision_1 + recall_1)
#             if (precision_1 + recall_1) > 0 else 0.0)
#     macro_f1 = (f1_0 + f1_1) / 2

#     return accuracy, macro_precision, macro_recall, macro_f1


# def print_metrics_detailed(label, acc, precision, recall, f1, auc_score=None):
#     print(f'\n--- {label} ---')
#     print(f'  Accuracy:         {acc:.4f}')
#     print(f'  Macro Precision:  {precision:.4f}')
#     print(f'  Macro Recall:     {recall:.4f}')
#     print(f'  Macro F1:         {f1:.4f}')
#     if auc_score is not None:
#         print(f'  AUC:              {auc_score:.4f}')


# =============================
# F1 METRIC (streaming, used during training)
# =============================

class F1Score:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = 0

    def update_state(self, y_true, y_pred_logits):
        y_pred = (torch.sigmoid(y_pred_logits) > self.threshold).float()
        y_true = y_true.float()
        self.tp += torch.sum(y_true * y_pred).item()
        self.fp += torch.sum((1 - y_true) * y_pred).item()
        self.fn += torch.sum(y_true * (1 - y_pred)).item()

    def result(self):
        p = self.tp / (self.tp + self.fp + 1e-7)
        r = self.tp / (self.tp + self.fn + 1e-7)
        return 2 * p * r / (p + r + 1e-7)

    def reset_state(self):
        self.tp = self.fp = self.fn = 0


# =============================
# CROSS-VALIDATION SPLITS
# =============================

def create_kfold_splits(train_videos, n_folds):
    """
    Divide train_videos (list of length 10) into n_folds folds.
    Each fold yields (train_fold_videos, val_fold_videos).
    Keeps folds deterministic: fold i uses videos[i*k : (i+1)*k] as val.
    """
    k = len(train_videos) // n_folds
    folds = []
    for i in range(n_folds):
        val = train_videos[i * k: (i + 1) * k]
        train = train_videos[: i * k] + train_videos[(i + 1) * k:]
        folds.append((train, val))
    return folds


# =============================
# RESULTS AGGREGATION
# =============================

def compute_per_participant_stds(results, key_prefix):
    """
    Given a list of per-participant result dicts and a key prefix ('ar' or 'va'),
    return a dict of {metric: std} across participants.
    Metrics: acc, precision, recall, f1, auc.
    """
    metrics = ['acc', 'precision', 'recall', 'f1']
    stds = {}
    for m in metrics:
        vals = [r[f'{key_prefix}_{m}'] for r in results]
        stds[f'{key_prefix}_{m}_std'] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

    auc_vals = []
    for r in results:
        v, _, _ = safe_roc_auc(r[f'y_true_{key_prefix}'], r[f'y_pred_probs_{key_prefix}'])
        auc_vals.append(v)
    clean = [x for x in auc_vals if not np.isnan(x)]
    stds[f'{key_prefix}_auc_std'] = np.std(clean, ddof=1) if len(clean) > 1 else 0.0
    return stds


def build_results_table(results):
    """
    Build a per-participant DataFrame from a list of result dicts.
    Each dict must contain: participant_id, cm_ar, cm_va,
    ar_acc, ar_precision, ar_recall, ar_f1,
    va_acc, va_precision, va_recall, va_f1,
    y_true_ar, y_pred_probs_ar, y_true_va, y_pred_probs_va.
    """
    rows = []
    for r in results:
        ar_tn, ar_fp, ar_fn, ar_tp = r['cm_ar'].ravel()
        va_tn, va_fp, va_fn, va_tp = r['cm_va'].ravel()
        ar_auc, _, _ = safe_roc_auc(r['y_true_ar'], r['y_pred_probs_ar'])
        va_auc, _, _ = safe_roc_auc(r['y_true_va'], r['y_pred_probs_va'])
        rows.append({
            'Participant ID':  r['participant_id'],
            'AR TN': ar_tn, 'AR TP': ar_tp, 'AR FN': ar_fn, 'AR FP': ar_fp,
            'AR Acc':     r['ar_acc'],
            'AR Macro P': r['ar_precision'],
            'AR Macro R': r['ar_recall'],
            'AR Macro F1':r['ar_f1'],
            'AR AUC':     ar_auc,
            'VA TN': va_tn, 'VA TP': va_tp, 'VA FN': va_fn, 'VA FP': va_fp,
            'VA Acc':     r['va_acc'],
            'VA Macro P': r['va_precision'],
            'VA Macro R': r['va_recall'],
            'VA Macro F1':r['va_f1'],
            'VA AUC':     va_auc,
        })
    return pd.DataFrame(rows)


def save_misclassification_rates(results, participant_ids_map, output_path):
    """
    Compute and save per-participant misclassification counts.
    participant_ids_map: dict {task_idx: participant_id} or list indexed by task_idx.
    results: list of dicts with task_idx (or participant_id), y_true_ar, y_pred_ar,
             y_true_va, y_pred_va.
    """
    rows = []
    for r in results:
        pid = (r['participant_id'] if 'participant_id' in r
               else participant_ids_map[r['task_idx']])
        ar_wrong = int(np.sum(r['y_true_ar'] != r['y_pred_ar']))
        va_wrong = int(np.sum(r['y_true_va'] != r['y_pred_va']))
        total    = len(r['y_true_ar'])
        rows.append({
            'participant_id':       pid,
            'ar_misclassified_rows': ar_wrong,
            'va_misclassified_rows': va_wrong,
            'total_rows':            total,
            'ar_accuracy':           1 - ar_wrong / total,
            'va_accuracy':           1 - va_wrong / total,
        })
        print(f"  Participant {pid}: AR misclassified={ar_wrong}/{total}, "
              f"VA misclassified={va_wrong}/{total}")
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved misclassification rates to: {output_path}")
    return df


def print_determinism_summary(ar_metrics, va_metrics, ar_stds, va_stds):
    """Print the mean ± std determinism verification block."""
    print("\n" + "=" * 60)
    print("DETERMINISM VERIFICATION (MEAN ± STD)")
    print("=" * 60)
    for label, m, s in [("AR", ar_metrics, ar_stds), ("VA", va_metrics, va_stds)]:
        print(f"\n{label} Metrics:")
        for key in ['auc', 'acc', 'precision', 'recall', 'f1']:
            prefix = label.lower()
            mean_val = m.get(f'{prefix}_{key}', m.get(key, float('nan')))
            std_val  = s.get(f'{prefix}_{key}_std', s.get(f'{key}_std', float('nan')))
            print(f"  {key.capitalize():<18}: {mean_val:.8f} ± {std_val:.8f}")


# =============================
# PLOTTING
# =============================

def save_confusion_matrix_plot(cm, title, filepath, cmap='Blues'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filepath)
    plt.close()


def save_roc_plot(fpr, tpr, roc_auc, title, filepath):
    if fpr is None or np.isnan(roc_auc):
        return
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})', color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath, dpi=300)
    plt.close()