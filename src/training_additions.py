# =============================
# MTL EVALUATION  (was _evaluate_all duplicated in mtl_hps / mtl_pcgrad / mtl_uw)
# =============================

def evaluate_mtl_all(model_ar, model_va, test_data_dict,
                     participant_ids, device,
                     window_size=2560, stride=1280):
    """
    Evaluate a pair of MTL models (one per label) over per-participant test sets.

    Parameters
    ----------
    model_ar / model_va : MTLModel (or MTLModelUW) — AR and VA models
    test_data_dict      : dict {task_idx: DataFrame}
    participant_ids     : list of participant IDs indexed by task_idx
    device              : torch.device
    window_size / stride: windowing parameters (default VREED values)

    Returns
    -------
    list of per-participant result dicts compatible with aggregate_results()
    and save_all_results().
    """
    from data import create_sliding_windows

    results = []
    for task_idx, pid in enumerate(participant_ids):
        test_df = test_data_dict.get(task_idx)
        if test_df is None or len(test_df) == 0:
            continue
        X, y_ar, y_va, _, _ = create_sliding_windows(
            test_df, window_size, stride, task_id=task_idx)
        if len(X) == 0:
            continue

        X_t    = torch.tensor(X, dtype=torch.float32).to(device)
        tids_t = torch.full((len(X),), task_idx, dtype=torch.long).to(device)

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
# MTML AGGREGATION  (was local aggregate() duplicated in si / pure_meta / reptile_*)
# =============================

def aggregate_mtml_results(results_ar, results_va):
    """
    Concatenate per-participant MTML results and return aggregate metrics for
    both AR and VA in one call.  Result dicts must have keys:
    y_true, y_pred, y_pred_probs, (optionally accuracy/precision/recall/f1).

    Returns
    -------
    dict with keys mirroring aggregate_results():
        cm_ar, cm_va,
        ar_acc, ar_precision, ar_recall, ar_f1, ar_auc, fpr_ar, tpr_ar,
        va_acc, va_precision, va_recall, va_f1, va_auc, fpr_va, tpr_va,
        all_true_ar, all_probs_ar, all_true_va, all_probs_va
    """
    def _agg_one(results, label):
        all_true  = np.concatenate([r['y_true']       for r in results])
        all_pred  = np.concatenate([r['y_pred']        for r in results])
        all_probs = np.concatenate([r['y_pred_probs']  for r in results])
        cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
        acc, prec, rec, f1 = compute_metrics_from_cm(cm)
        auc_val, fpr, tpr  = safe_roc_auc(all_true, all_probs)
        print(f"\n{label}: Acc={acc:.4f} F1={f1:.4f} AUC={auc_val:.4f}")
        return all_true, all_probs, cm, acc, prec, rec, f1, auc_val, fpr, tpr

    all_true_ar, all_probs_ar, cm_AR, ar_acc, ar_prec, ar_rec, ar_f1, ar_auc, ar_fpr, ar_tpr = _agg_one(results_ar, 'AR')
    all_true_va, all_probs_va, cm_VA, va_acc, va_prec, va_rec, va_f1, va_auc, va_fpr, va_tpr = _agg_one(results_va, 'VA')

    return {
        'cm_ar': cm_AR,         'cm_va': cm_VA,
        'ar_acc': ar_acc,       'ar_precision': ar_prec,
        'ar_recall': ar_rec,    'ar_f1': ar_f1,
        'ar_auc': ar_auc,       'fpr_ar': ar_fpr,  'tpr_ar': ar_tpr,
        'va_acc': va_acc,       'va_precision': va_prec,
        'va_recall': va_rec,    'va_f1': va_f1,
        'va_auc': va_auc,       'fpr_va': va_fpr,  'tpr_va': va_tpr,
        'all_true_ar': all_true_ar,   'all_probs_ar': all_probs_ar,
        'all_true_va': all_true_va,   'all_probs_va': all_probs_va,
    }
