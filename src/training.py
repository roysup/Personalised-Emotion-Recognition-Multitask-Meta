import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from utils import compute_metrics_from_cm


# =============================
# PCGRAD GRADIENT MODIFIER
# =============================

def _pcgrad_project(grad_list):
    """Project gradients to remove conflicting components."""
    grads = grad_list.copy()
    for i in range(len(grads)):
        for j in range(len(grads)):
            if i == j:
                continue
            dot  = torch.dot(grads[i], grads[j])
            denom = torch.dot(grads[j], grads[j])
            if dot < 0 and denom > 0:
                grads[i] = grads[i] - (dot / denom) * grads[j]
    return torch.mean(torch.stack(grads), dim=0)


# =============================
# REPTILE OUTER UPDATE
# =============================

def reptile_outer_update(model, adapted_models, meta_lr):
    with torch.no_grad():
        for name, p in model.named_parameters():
            mean_adapted = torch.stack(
                [dict(m.named_parameters())[name].data for m in adapted_models]
            ).mean(0)
            p.data.add_(meta_lr * (mean_adapted - p.data))


# =============================
# EVALUATION — P-STL (DataLoader-based)
# =============================

def evaluate_per_participant(model, test_loaders_ar, test_loaders_va,
                             participant_ids, device, is_mtl=False):
    """Run inference over per-participant test loaders and collect results."""
    results = []

    if isinstance(model, tuple):
        model_ar, model_va = model
        model_ar.eval(); model_va.eval()
        use_pair = True
    elif isinstance(model, dict):
        use_pair = False; use_dict = True
    else:
        model.eval()
        use_pair = False; use_dict = False

    with torch.no_grad():
        for task_idx in sorted(test_loaders_ar.keys()):
            loader_ar = test_loaders_ar.get(task_idx)
            loader_va = test_loaders_va.get(task_idx)
            if loader_ar is None or loader_va is None:
                continue

            if use_pair:
                m_ar, m_va = model_ar, model_va
            elif use_dict:
                m_ar, m_va = model[task_idx]
                m_ar.eval(); m_va.eval()
            else:
                m_ar = m_va = model

            y_true_ar, y_pred_ar, y_probs_ar = [], [], []
            y_true_va, y_pred_va, y_probs_va = [], [], []

            for batch_ar, batch_va in zip(loader_ar, loader_va):
                X_ar = batch_ar[0].to(device, non_blocking=True); y_ar = batch_ar[1]
                X_va = batch_va[0].to(device, non_blocking=True); y_va = batch_va[1]

                if is_mtl:
                    out_ar = m_ar(X_ar, batch_ar[2].to(device, non_blocking=True))
                    out_va = m_va(X_va, batch_va[2].to(device, non_blocking=True))
                else:
                    out_ar = m_ar(X_ar)
                    out_va = m_va(X_va)

                prob_ar = torch.sigmoid(out_ar).cpu().numpy().flatten()
                prob_va = torch.sigmoid(out_va).cpu().numpy().flatten()
                y_true_ar.extend(y_ar.int().numpy().flatten())
                y_pred_ar.extend((prob_ar > 0.5).astype(int))
                y_probs_ar.extend(prob_ar)
                y_true_va.extend(y_va.int().numpy().flatten())
                y_pred_va.extend((prob_va > 0.5).astype(int))
                y_probs_va.extend(prob_va)

            y_true_ar = np.array(y_true_ar); y_pred_ar  = np.array(y_pred_ar)
            y_probs_ar= np.array(y_probs_ar)
            y_true_va = np.array(y_true_va); y_pred_va  = np.array(y_pred_va)
            y_probs_va= np.array(y_probs_va)

            cm_ar = confusion_matrix(y_true_ar, y_pred_ar, labels=[0, 1])
            cm_va = confusion_matrix(y_true_va, y_pred_va, labels=[0, 1])
            ar_acc, ar_prec, ar_rec, ar_f1 = compute_metrics_from_cm(cm_ar)
            va_acc, va_prec, va_rec, va_f1 = compute_metrics_from_cm(cm_va)

            pid = participant_ids[task_idx]
            print(f"  Participant {pid}: "
                  f"AR acc={ar_acc:.4f} f1={ar_f1:.4f} | "
                  f"VA acc={va_acc:.4f} f1={va_f1:.4f}")

            results.append({
                'task_idx': task_idx, 'participant_id': pid,
                'cm_ar': cm_ar, 'cm_va': cm_va,
                'ar_acc': ar_acc, 'ar_precision': ar_prec,
                'ar_recall': ar_rec, 'ar_f1': ar_f1,
                'va_acc': va_acc, 'va_precision': va_prec,
                'va_recall': va_rec, 'va_f1': va_f1,
                'y_true_ar': y_true_ar, 'y_pred_ar': y_pred_ar,
                'y_pred_probs_ar': y_probs_ar,
                'y_true_va': y_true_va, 'y_pred_va': y_pred_va,
                'y_pred_probs_va': y_probs_va,
            })

    return results


# =============================
# EVALUATION — STL (per-participant models, raw DataFrames)
# =============================

def evaluate_stl_all(models_ar, models_va, test_data_dict,
                     participant_ids, device,
                     window_size=2560, stride=1280, feature_cols=None):
    """Evaluate per-participant STL models over raw test DataFrames."""
    from data import create_sliding_windows

    results = []
    for task_idx, pid in enumerate(participant_ids):
        if task_idx not in models_ar or task_idx not in models_va:
            continue
        test_df = test_data_dict.get(task_idx)
        if test_df is None or len(test_df) == 0:
            continue
        X, y_ar, y_va, _, _ = create_sliding_windows(
            test_df, window_size, stride, task_id=task_idx,
            feature_cols=feature_cols)
        if len(X) == 0:
            continue

        X_t = torch.tensor(X, dtype=torch.float32).to(device, non_blocking=True)
        m_ar = models_ar[task_idx]; m_ar.eval()
        m_va = models_va[task_idx]; m_va.eval()

        with torch.no_grad():
            prob_ar = torch.sigmoid(m_ar(X_t)).cpu().numpy().flatten()
            prob_va = torch.sigmoid(m_va(X_t)).cpu().numpy().flatten()

        pred_ar = (prob_ar > 0.5).astype(int); y_ar_i = y_ar.astype(int)
        pred_va = (prob_va > 0.5).astype(int); y_va_i = y_va.astype(int)

        cm_ar = confusion_matrix(y_ar_i, pred_ar, labels=[0, 1])
        cm_va = confusion_matrix(y_va_i, pred_va, labels=[0, 1])
        ar_acc, ar_prec, ar_rec, ar_f1 = compute_metrics_from_cm(cm_ar)
        va_acc, va_prec, va_rec, va_f1 = compute_metrics_from_cm(cm_va)

        print(f"  Participant {pid}: AR acc={ar_acc:.4f} f1={ar_f1:.4f} | "
              f"VA acc={va_acc:.4f} f1={va_f1:.4f}")

        results.append({
            'task_idx': task_idx, 'participant_id': pid,
            'cm_ar': cm_ar, 'cm_va': cm_va,
            'ar_acc': ar_acc, 'ar_precision': ar_prec,
            'ar_recall': ar_rec, 'ar_f1': ar_f1,
            'va_acc': va_acc, 'va_precision': va_prec,
            'va_recall': va_rec, 'va_f1': va_f1,
            'y_true_ar': y_ar_i, 'y_pred_ar': pred_ar,
            'y_pred_probs_ar': prob_ar,
            'y_true_va': y_va_i, 'y_pred_va': pred_va,
            'y_pred_probs_va': prob_va,
        })
    return results


# =============================
# INNER-LOOP ADAPTATION  (Reptile / MAML style)
# =============================

def adapt_inner_loop(base_model, head, sup_loader, ar_or_va,
                     inner_steps, inner_lr, device,
                     l2_shared=0.0, l2_task=1e-5):
    import copy
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    adapted_base = copy.deepcopy(base_model).to(device)
    adapted_head = copy.deepcopy(head).to(device)
    adapted_base.train(); adapted_head.train()

    sp = list(adapted_base.parameters())
    tp = list(adapted_head.parameters())
    opt   = torch.optim.Adam(sp + tp, lr=inner_lr)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(inner_steps):
        ep_loss = 0.0; nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(adapted_head(adapted_base(Xb)), yb)
            loss = loss + (l2_shared * sum(p.norm(2)**2 for p in sp if p.requires_grad) +
                           l2_task   * sum(p.norm(2)**2 for p in tp if p.requires_grad))
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sp + tp, max_norm=1.0)
                opt.step()
            ep_loss += loss.item(); nb += 1
        if nb > 0:
            sched.step(ep_loss / nb)

    return adapted_base, adapted_head


def evaluate_test_user(base_model, head, test_df, splits, uid, ar_or_va,
                       device, inner_steps, inner_lr,
                       l2_shared=0.0, l2_task=1e-5,
                       window_size=2560, stride=1280, feature_cols=None):
    """Adapt and evaluate on one test participant."""
    from data import build_support_query

    sup_loader, q_loader = build_support_query(
        test_df, splits[uid]['train'], splits[uid]['test'], ar_or_va,
        window_size=window_size, stride=stride, feature_cols=feature_cols)

    if len(q_loader.dataset) == 0:
        return None

    adapted_base, adapted_head = adapt_inner_loop(
        base_model, head, sup_loader, ar_or_va,
        inner_steps, inner_lr, device, l2_shared, l2_task)

    adapted_base.eval(); adapted_head.eval()
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in q_loader:
            probs.extend(torch.sigmoid(adapted_head(adapted_base(Xb.to(device, non_blocking=True))))
                         .cpu().numpy().flatten())
            labels.extend(yb.numpy().flatten())

    y_true = np.array(labels).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)

    p = ar_or_va
    return {
        'participant_id':       uid,
        'cm':                   cm,
        f'{p}_acc':             acc,
        f'{p}_precision':       prec,
        f'{p}_recall':          rec,
        f'{p}_f1':              f1,
        f'y_true_{p}':          y_true,
        f'y_pred_{p}':          y_pred,
        f'y_pred_probs_{p}':    y_prob,
    }


# =============================
# MTL EVALUATION  (shared model, raw DataFrames)
# =============================

def evaluate_mtl_all(model_ar, model_va, test_data_dict,
                     participant_ids, device,
                     window_size=2560, stride=1280, feature_cols=None):
    """Evaluate a pair of MTL models over per-participant test sets."""
    from data import create_sliding_windows

    results = []
    for task_idx, pid in enumerate(participant_ids):
        test_df = test_data_dict.get(task_idx)
        if test_df is None or len(test_df) == 0:
            continue
        X, y_ar, y_va, _, _ = create_sliding_windows(
            test_df, window_size, stride, task_id=task_idx,
            feature_cols=feature_cols)
        if len(X) == 0:
            continue

        X_t    = torch.tensor(X, dtype=torch.float32).to(device, non_blocking=True)
        tids_t = torch.full((len(X),), task_idx, dtype=torch.long).to(device, non_blocking=True)

        model_ar.eval(); model_va.eval()
        with torch.no_grad():
            prob_ar = torch.sigmoid(model_ar(X_t, tids_t)).cpu().numpy().flatten()
            prob_va = torch.sigmoid(model_va(X_t, tids_t)).cpu().numpy().flatten()

        pred_ar = (prob_ar > 0.5).astype(int); y_ar_i = y_ar.astype(int)
        pred_va = (prob_va > 0.5).astype(int); y_va_i = y_va.astype(int)

        cm_ar = confusion_matrix(y_ar_i, pred_ar, labels=[0, 1])
        cm_va = confusion_matrix(y_va_i, pred_va, labels=[0, 1])
        ar_acc, ar_prec, ar_rec, ar_f1 = compute_metrics_from_cm(cm_ar)
        va_acc, va_prec, va_rec, va_f1 = compute_metrics_from_cm(cm_va)

        print(f"  Participant {pid}: AR acc={ar_acc:.4f} f1={ar_f1:.4f} | "
              f"VA acc={va_acc:.4f} f1={va_f1:.4f}")

        results.append({
            'task_idx': task_idx, 'participant_id': pid,
            'cm_ar': cm_ar, 'cm_va': cm_va,
            'ar_acc': ar_acc, 'ar_precision': ar_prec,
            'ar_recall': ar_rec, 'ar_f1': ar_f1,
            'va_acc': va_acc, 'va_precision': va_prec,
            'va_recall': va_rec, 'va_f1': va_f1,
            'y_true_ar': y_ar_i, 'y_pred_ar': pred_ar, 'y_pred_probs_ar': prob_ar,
            'y_true_va': y_va_i, 'y_pred_va': pred_va, 'y_pred_probs_va': prob_va,
        })
    return results


# =============================
# SAVE ALL RESULTS
# =============================

def save_all_results(results, agg, output_dir, method_name, misclassification_csv):
    import os
    from utils import (save_misclassification_rates, build_results_table,
                       compute_per_participant_stds, print_determinism_summary,
                       print_metrics_detailed, save_confusion_matrix_plot,
                       save_roc_plot)

    participant_ids_map = {
        r['task_idx']: r['participant_id']
        for r in results
        if 'task_idx' in r and 'participant_id' in r
    }

    save_misclassification_rates(
        results, participant_ids_map,
        os.path.join(output_dir, misclassification_csv))

    results_df = build_results_table(results)
    results_df.to_csv(os.path.join(output_dir, 'per_participant_results.csv'), index=False)
    print(results_df.to_string(index=False))

    ar_stds = compute_per_participant_stds(results, 'ar')
    va_stds = compute_per_participant_stds(results, 'va')

    save_confusion_matrix_plot(
        agg['cm_ar'], f'AR Confusion Matrix ({method_name})',
        os.path.join(output_dir, 'ar_cm.png'))
    save_confusion_matrix_plot(
        agg['cm_va'], f'VA Confusion Matrix ({method_name})',
        os.path.join(output_dir, 'va_cm.png'), cmap='Greens')
    save_roc_plot(agg['fpr_ar'], agg['tpr_ar'], agg['ar_auc'],
                  f'ROC AR ({method_name})', os.path.join(output_dir, 'ar_roc.png'))
    save_roc_plot(agg['fpr_va'], agg['tpr_va'], agg['va_auc'],
                  f'ROC VA ({method_name})', os.path.join(output_dir, 'va_roc.png'))

    print_metrics_detailed('AR', agg['ar_acc'], agg['ar_precision'],
                           agg['ar_recall'], agg['ar_f1'], agg['ar_auc'])
    print_metrics_detailed('VA', agg['va_acc'], agg['va_precision'],
                           agg['va_recall'], agg['va_f1'], agg['va_auc'])
    print_determinism_summary(
        {f'ar_{k}': agg[f'ar_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        {f'va_{k}': agg[f'va_{k}'] for k in ['auc', 'acc', 'precision', 'recall', 'f1']},
        ar_stds, va_stds)

    return results_df, ar_stds, va_stds
