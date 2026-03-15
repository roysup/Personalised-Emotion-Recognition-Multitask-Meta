# =============================
# MTML RESULT PREFIX HELPER  (was local _prefix() duplicated in 6 scripts)
# =============================

def prefix_results(results, prefix):
    """
    Re-key per-participant MTML result dicts so they match the key convention
    expected by compute_per_participant_stds and print_determinism_summary.

    Parameters
    ----------
    results : list of dicts with keys accuracy, precision, recall, f1,
              y_true, y_pred_probs
    prefix  : 'ar' or 'va'

    Returns
    -------
    list of dicts with keys  {prefix}_acc, {prefix}_precision, ...,
    y_true_{prefix}, y_pred_probs_{prefix}
    """
    return [
        {
            f'{prefix}_acc':        r['accuracy'],
            f'{prefix}_precision':  r['precision'],
            f'{prefix}_recall':     r['recall'],
            f'{prefix}_f1':         r['f1'],
            f'y_true_{prefix}':     r['y_true'],
            f'y_pred_probs_{prefix}': r['y_pred_probs'],
        }
        for r in results
    ]
