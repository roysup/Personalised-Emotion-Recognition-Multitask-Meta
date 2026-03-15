import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from utils import (compute_metrics_from_cm, safe_roc_auc,
                   aggregate_results, aggregate_mtml_results)


# =============================
# GENERIC TRAINING LOOP
# =============================

def train_model(model, train_loader, optimizer, scheduler,
                epochs, max_norm, l2_fn, device,
                label='', checkpoint_path=None,
                grad_modifier=None, seed_fn=None):
    """
    Generic training loop shared by all model types.

    Parameters
    ----------
    model            : nn.Module
    train_loader     : DataLoader — batches of (X, y, task_ids [, trial_ids])
    optimizer        : torch Optimizer
    scheduler        : LR scheduler (ReduceLROnPlateau)
    epochs           : int
    max_norm         : float — gradient clip norm
    l2_fn            : callable(model) -> scalar tensor — returns L2 reg term,
                       or None for no regularisation
    device           : torch.device
    label            : str — printed in progress lines
    checkpoint_path  : str or None — saves model at best train loss if provided
    grad_modifier    : callable(optimizer, per_task_losses, shared_params) -> None
                       for PCGrad; if None uses standard backward
    seed_fn          : callable() -> None — called once before the loop starts
                       (pass set_all_seeds(SEED) as a lambda)

    Returns
    -------
    history : dict {'train_loss': [...]}
    """
    if seed_fn is not None:
        seed_fn()

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    history = {'train_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            # Unpack — loaders may yield 3 or 4 tensors
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            task_ids = batch[2].to(device) if len(batch) > 2 else None

            optimizer.zero_grad(set_to_none=True)

            if task_ids is not None:
                outputs = model(X_batch, task_ids)
            else:
                outputs = model(X_batch)

            per_sample_loss = loss_fn(outputs, y_batch).squeeze(-1)

            if grad_modifier is not None:
                # PCGrad path: caller supplies the modifier
                grad_modifier(optimizer, per_sample_loss, task_ids, outputs)
                total_loss = per_sample_loss.mean()
                if l2_fn is not None:
                    total_loss = total_loss + l2_fn(model)
                total_loss.backward()
            else:
                total_loss = per_sample_loss.mean()
                if l2_fn is not None:
                    total_loss = total_loss + l2_fn(model)
                total_loss.backward()

            if torch.isnan(total_loss):
                raise ValueError(f"NaN loss at epoch {epoch + 1} [{label}]")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            running_loss += total_loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        history['train_loss'].append(avg_loss)
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        if checkpoint_path is not None and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)

    return history


# =============================
# UNCERTAINTY-WEIGHTED TRAINING LOOP
# =============================

def train_model_uw(model, train_loader, optimizer, scheduler,
                   epochs, max_norm, l2_fn, device,
                   label='', checkpoint_path=None, seed_fn=None):
    """
    Variant of train_model for uncertainty-weighted MTL.
    The model must expose a `log_vars` parameter (nn.Parameter of shape [num_tasks]).
    """
    if seed_fn is not None:
        seed_fn()

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    history = {'train_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            X_batch  = batch[0].to(device)
            y_batch  = batch[1].to(device)
            task_ids = batch[2].to(device)

            if len(torch.unique(task_ids)) != model.num_tasks:
                raise ValueError(
                    f"Batch missing tasks: found {len(torch.unique(task_ids))}, "
                    f"expected {model.num_tasks}")

            optimizer.zero_grad(set_to_none=True)

            outputs          = model(X_batch, task_ids)
            per_sample_loss  = loss_fn(outputs, y_batch).squeeze(-1)
            log_vars         = model.log_vars[task_ids]
            precision        = torch.exp(-log_vars)
            weighted_loss    = (precision * per_sample_loss + log_vars).mean()

            if l2_fn is not None:
                weighted_loss = weighted_loss + l2_fn(model)

            if torch.isnan(weighted_loss):
                raise ValueError(f"NaN loss at epoch {epoch + 1} [{label}]")

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            running_loss += weighted_loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        history['train_loss'].append(avg_loss)
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        if checkpoint_path is not None and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    Checkpoint saved (epoch {epoch+1})")

    return history


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


def make_pcgrad_modifier(shared_params):
    """
    Returns a grad_modifier compatible with train_model's signature.
    Computes per-task gradients on shared_params, projects them,
    then overwrites shared_params.grad after the caller's .backward().
    """
    def modifier(optimizer, per_sample_loss, task_ids, outputs):
        unique_tasks = torch.unique(task_ids)
        task_grads   = []
        for t in unique_tasks:
            mask   = (task_ids == t)
            loss_t = per_sample_loss[mask].mean()
            optimizer.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(
                loss_t, shared_params,
                retain_graph=True, create_graph=False, allow_unused=True)
            flat = []
            for p, g in zip(shared_params, grads):
                flat.append(g.contiguous().view(-1) if g is not None
                            else torch.zeros_like(p).view(-1))
            task_grads.append(torch.cat(flat))

        projected = _pcgrad_project(task_grads)

        # Will be applied after caller calls .backward() in train_model
        # Store on a side-channel so train_model can apply it post-backward.
        modifier._projected = projected
        modifier._params    = shared_params

    def post_backward_hook():
        if hasattr(modifier, '_projected'):
            offset = 0
            for p in modifier._params:
                n = p.numel()
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.copy_(modifier._projected[offset: offset + n].view_as(p))
                offset += n

    modifier.post_backward = post_backward_hook
    return modifier


# =============================
# EVALUATION
# =============================

def evaluate_per_participant(model, test_loaders_ar, test_loaders_va,
                             participant_ids, device, is_mtl=False):
    """
    Run inference over per-participant test loaders and collect results.

    Parameters
    ----------
    model            : nn.Module (for STL pass a dict {task_idx: (model_ar, model_va)})
    test_loaders_ar  : dict {task_idx: DataLoader}
    test_loaders_va  : dict {task_idx: DataLoader}
    participant_ids  : list of participant IDs indexed by task_idx
    device           : torch.device
    is_mtl           : bool — if True, model(X, task_ids) is called;
                       if False, model(X) is called.
                       For STL pass is_mtl=False and model as a 2-tuple (model_ar, model_va).

    Returns
    -------
    results : list of dicts, one per participant
    """
    results = []

    # Normalise model argument
    if isinstance(model, tuple):
        # STL: (model_ar, model_va)
        model_ar, model_va = model
        model_ar.eval()
        model_va.eval()
        use_pair = True
    elif isinstance(model, dict):
        # STL per-participant models: {task_idx: (model_ar, model_va)}
        use_pair = False
        use_dict = True
    else:
        model.eval()
        use_pair = False
        use_dict = False

    with torch.no_grad():
        for task_idx in sorted(test_loaders_ar.keys()):
            loader_ar = test_loaders_ar.get(task_idx)
            loader_va = test_loaders_va.get(task_idx)
            if loader_ar is None or loader_va is None:
                continue

            # Resolve models
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
                X_ar = batch_ar[0].to(device)
                y_ar = batch_ar[1]
                X_va = batch_va[0].to(device)
                y_va = batch_va[1]

                if is_mtl:
                    tids_ar = batch_ar[2].to(device)
                    tids_va = batch_va[2].to(device)
                    out_ar  = m_ar(X_ar, tids_ar)
                    out_va  = m_va(X_va, tids_va)
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

            y_true_ar = np.array(y_true_ar);  y_pred_ar = np.array(y_pred_ar)
            y_probs_ar= np.array(y_probs_ar)
            y_true_va = np.array(y_true_va);  y_pred_va = np.array(y_pred_va)
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
                'task_idx':        task_idx,
                'participant_id':  pid,
                'cm_ar':           cm_ar,
                'cm_va':           cm_va,
                'ar_acc':          ar_acc,
                'ar_precision':    ar_prec,
                'ar_recall':       ar_rec,
                'ar_f1':           ar_f1,
                'va_acc':          va_acc,
                'va_precision':    va_prec,
                'va_recall':       va_rec,
                'va_f1':           va_f1,
                'y_true_ar':       y_true_ar,
                'y_pred_ar':       y_pred_ar,
                'y_pred_probs_ar': y_probs_ar,
                'y_true_va':       y_true_va,
                'y_pred_va':       y_pred_va,
                'y_pred_probs_va': y_probs_va,
            })

    return results



# =============================
# SAVE ALL RESULTS
# =============================

# =============================
# META-LEARNING UTILITIES
# =============================

def compute_l2_split(shared_params, task_params, l2_shared=0.0, l2_task=1e-5):
    """
    Split L2 regularisation: separate lambda for shared backbone vs task head.

    Parameters
    ----------
    shared_params : iterable of parameters from the shared backbone
    task_params   : iterable of parameters from the task-specific head
    l2_shared     : float — lambda for shared params (typically 0.0)
    l2_task       : float — lambda for task head params (typically 1e-5)

    Returns
    -------
    Scalar tensor on CPU.
    """
    reg = torch.tensor(0.0)
    if l2_shared > 0:
        for p in shared_params:
            reg = reg + l2_shared * torch.sum(p ** 2)
    if l2_task > 0:
        for p in task_params:
            reg = reg + l2_task * torch.sum(p ** 2)
    return reg


def adapt_inner_loop(base_model, head, sup_loader, ar_or_va,
                     inner_steps, inner_lr, device,
                     l2_shared=0.0, l2_task=1e-5):
    """
    Reptile / MAML-style inner-loop adaptation for one participant.
    Deep-copies both backbone and head, adapts them on the support set,
    and returns the adapted copies (originals are unchanged).

    Parameters
    ----------
    base_model  : BaseFeatureExtractor — shared backbone
    head        : TaskHead — participant-specific head
    sup_loader  : DataLoader yielding (X, y) support batches
    ar_or_va    : 'ar' or 'va' — selects which label was loaded
    inner_steps : int — number of gradient steps
    inner_lr    : float — inner-loop learning rate
    device      : torch.device
    l2_shared   : float — L2 lambda for backbone params
    l2_task     : float — L2 lambda for head params

    Returns
    -------
    adapted_base : deep copy of backbone after adaptation
    adapted_head : deep copy of head after adaptation
    """
    import copy
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    adapted_base = copy.deepcopy(base_model).to(device)
    adapted_head = copy.deepcopy(head).to(device)
    adapted_base.train(); adapted_head.train()

    sp = list(adapted_base.parameters())
    tp = list(adapted_head.parameters())
    opt = torch.optim.Adam(sp + tp, lr=inner_lr)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(inner_steps):
        ep_loss = 0.0; nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(adapted_head(adapted_base(Xb)), yb)
            loss = loss + compute_l2_split(sp, tp, l2_shared, l2_task).to(device)
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
                       l2_shared=0.0, l2_task=1e-5):
    """
    Adapt the meta-learned backbone to one test participant and evaluate.
    Imports build_support_query from data at call time to avoid circular imports.

    Parameters
    ----------
    base_model  : BaseFeatureExtractor
    head        : TaskHead (freshly initialised)
    test_df     : DataFrame for this participant
    splits      : dict — hardcoded_splits with 'train'/'test' trial lists
    uid         : int — participant ID
    ar_or_va    : 'ar' or 'va'
    device      : torch.device
    inner_steps : int
    inner_lr    : float
    l2_shared   : float
    l2_task     : float

    Returns
    -------
    dict with participant_id, y_true, y_pred, y_pred_probs, cm,
    accuracy, precision, recall, f1  — or None if no query windows.
    """
    from data import build_support_query

    sup_loader, q_loader = build_support_query(
        test_df,
        splits[uid]['train'],
        splits[uid]['test'],
        ar_or_va)

    if len(q_loader.dataset) == 0:
        return None

    adapted_base, adapted_head = adapt_inner_loop(
        base_model, head, sup_loader, ar_or_va,
        inner_steps, inner_lr, device, l2_shared, l2_task)

    adapted_base.eval(); adapted_head.eval()
    probs, labels = [], []
    with torch.no_grad():
        for Xb, yb in q_loader:
            probs.extend(torch.sigmoid(adapted_head(adapted_base(Xb.to(device))))
                         .cpu().numpy().flatten())
            labels.extend(yb.numpy().flatten())

    y_true = np.array(labels).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)

    return {'participant_id': uid, 'y_true': y_true, 'y_pred': y_pred,
            'y_pred_probs': y_prob, 'cm': cm,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def save_all_results(results, agg, output_dir, method_name, misclassification_csv):
    """
    Save the full results bundle produced by any MTL/MTML training script.

    Parameters
    ----------
    results              : list of per-participant result dicts
    agg                  : dict returned by aggregate_results()
    output_dir           : str — directory to write all files into
    method_name          : str — used in plot titles, e.g. 'MTL-HPS'
    misclassification_csv: str — filename only (not full path), e.g.
                           'VREED_hps_misclassification_rates.csv'

    Writes
    ------
    per_participant_results.csv
    <misclassification_csv>
    ar_cm.png, va_cm.png
    ar_roc.png, va_roc.png
    (pickle saving is left to the caller because the pkl filename and any
     extra keys vary per script)
    """
    import os
    from utils import (save_misclassification_rates, build_results_table,
                       compute_per_participant_stds, print_determinism_summary,
                       print_metrics_detailed, save_confusion_matrix_plot,
                       save_roc_plot)

    save_misclassification_rates(
        results, [r['participant_id'] for r in results],
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

