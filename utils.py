import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def set_all_seeds(seed: int = 42):
    import os
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(True)

def safe_roc_auc(y_true, y_score):
    """
    Compute ROC-AUC only if both classes are present.
    Returns (auc, fpr, tpr) or (np.nan, None, None) if undefined.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan, None, None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr), fpr, tpr

# Custom F1 Score Metric
class F1Score:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update_state(self, y_true, y_pred):
        y_pred = (torch.sigmoid(y_pred) > self.threshold).float()
        y_true = y_true.float()
        tp = torch.sum(y_true * y_pred).item()
        fp = torch.sum((1 - y_true) * y_pred).item()
        fn = torch.sum(y_true * (1 - y_pred)).item()
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def result(self):
        p = self.tp / (self.tp + self.fp + 1e-7)
        r = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * p * r / (p + r + 1e-7)
        return f1

    def reset_state(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

# Time History Callback
class TimeHistory:
    def __init__(self):
        self.times = []

    def on_train_begin(self):
        self.times = []

    def on_epoch_begin(self):
        self.epoch_time_start = time.time()

    def on_epoch_end(self):
        self.times.append(time.time() - self.epoch_time_start)

# Plotting Function
def plot_performance(history, title, task='AR', fpr=None, tpr=None, roc_auc=None):
    epochs = range(1, len(history['accuracy']) + 1)
    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['accuracy'], 'bo-', label=f'Training {task} Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label=f'Validation {task} Accuracy')
    plt.title(f'{title} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['loss'], 'bo-', label=f'Training {task} Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label=f'Validation {task} Loss')
    plt.title(f'{title} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if fpr is not None and tpr is not None and roc_auc is not None:
        plt.subplot(1, 3, 3)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title} ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{title.lower().replace(" ", "_")}_performance.png'))
    plt.close()

# Function to compute metrics from confusion matrix
def compute_metrics_from_cm(cm):
    """
    Compute accuracy, macro precision, macro recall, and macro F1 from confusion matrix.
    For binary classification: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    """
    tn, fp, fn, tp = cm.ravel()

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision for each class
    precision_class_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    macro_precision = (precision_class_0 + precision_class_1) / 2

    # Recall for each class
    recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    macro_recall = (recall_class_0 + recall_class_1) / 2

    # F1 for each class
    f1_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0) if (precision_class_0 + recall_class_0) > 0 else 0
    f1_class_1 = 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) if (precision_class_1 + recall_class_1) > 0 else 0
    macro_f1 = (f1_class_0 + f1_class_1) / 2

    return accuracy, macro_precision, macro_recall, macro_f1

# Print Metrics
def print_metrics_detailed(label, acc, precision, recall, f1, auc_score=None):
    print(f'\n--- {label} Metrics ---')
    print(f'Accuracy:         {acc:.2%}')
    print(f'Macro Precision:  {precision:.2%}')
    print(f'Macro Recall:     {recall:.2%}')
    print(f'Macro F1 Score:   {f1:.2%}')
    if auc_score is not None:
        print(f'AUC:              {auc_score:.2%}')

def create_kfold_splits(train_videos):
    """
    Create 5 folds where each fold uses 2 videos as validation.
    train_videos should be a list of 10 video IDs.
    Returns list of (train_fold_videos, val_fold_videos) tuples.
    """
    folds = []
    for i in range(N_FOLDS):
        val_start_idx = i * 2
        val_end_idx = val_start_idx + 2
        val_videos = train_videos[val_start_idx:val_end_idx]
        train_videos_fold = train_videos[:val_start_idx] + train_videos[val_end_idx:]
        folds.append((train_videos_fold, val_videos))
    return folds