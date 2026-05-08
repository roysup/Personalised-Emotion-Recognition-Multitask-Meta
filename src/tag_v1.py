"""
Task Affinity Grouping (TAG) — per-batch inter-task affinity probing for MTL.

Implements the Fifty et al. (2021) probe: for each task u in a batch, take a
single gradient step on shared parameters using only u's loss, then measure
the relative loss change on every other task v. Aggregating across batches
gives an NxN affinity matrix where (u, v) = "training on u helps/hurts v".

This implementation reuses the main training optimizer (with full save/restore
around each probe step) so the temporary update inherits the accumulated Adam
moments from training. Task-specific parameters are frozen during the probe
by setting their grads to None so Adam skips them entirely.
"""
import copy
import numpy as np
import torch


def compute_inter_task_affinity(model, X_batch, y_batch, task_ids_batch,
                                main_optimizer, shared_params, task_params,
                                loss_fn, num_tasks):
    """
    Adam-adapted TAG affinity computation.

    For each source task u and target task v:
        Z[u, v] = 1 - L_v(after temporary Adam update from u) / L_v(before update)

    The probe reuses the main training optimizer so its update inherits the
    accumulated Adam moments. Model and optimizer states are deep-copied and
    restored around every probe so affinity extraction does not perturb the
    real training trajectory. The model is kept in train() mode throughout
    (with state restoration after each evaluation) to keep BatchNorm behaviour
    consistent with training while preventing running-stat contamination.

    `shared_params` is accepted for API symmetry with the caller but is not
    used inside the function — only `task_params` are touched (their grads
    are set to None so Adam skips them during the probe step).

    Parameters
    ----------
    model : MTLModel
    X_batch, y_batch, task_ids_batch : tensors already on device
    main_optimizer : optim.Optimizer
        The training optimizer. Its state will be saved/restored around each
        probe step so the probe inherits the accumulated Adam moments.
    shared_params : list[nn.Parameter]
        Shared backbone parameters (kept for API symmetry; not used here).
    task_params : list[nn.Parameter]
        Task-specific parameters whose grads are set to None during the probe
        step (TAG rule: only shared params should move).
    loss_fn : nn.Module with reduction='none'.
    num_tasks : int

    Returns
    -------
    np.ndarray (num_tasks, num_tasks), float32. Tasks not present in the batch
    leave their rows/columns as zeros.
    """
    was_training = model.training
    model.train()

    task_ids_cpu = task_ids_batch.detach().cpu().tolist()
    unique_tasks = sorted(set(task_ids_cpu))
    task_to_indices = {t: [] for t in unique_tasks}
    for bidx, tid in enumerate(task_ids_cpu):
        task_to_indices[tid].append(bidx)

    base_model_state     = copy.deepcopy(model.state_dict())
    base_optimizer_state = copy.deepcopy(main_optimizer.state_dict())

    # Old losses at base parameters.
    with torch.no_grad():
        preds_old = model(X_batch, task_ids_batch)
        losses_old_all = loss_fn(preds_old, y_batch).squeeze(-1).detach().cpu().numpy()
    loss_old_per_task = {
        t: float(np.mean([losses_old_all[b] for b in idxs]))
        for t, idxs in task_to_indices.items()
    }

    # Restore — BN running stats can shift even under no_grad in train mode.
    model.load_state_dict(base_model_state)
    main_optimizer.load_state_dict(base_optimizer_state)

    affinity_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    for u in unique_tasks:
        # Start every source-task probe from the exact same base model + Adam state.
        model.load_state_dict(base_model_state)
        main_optimizer.load_state_dict(base_optimizer_state)
        model.train()
        main_optimizer.zero_grad(set_to_none=True)

        idxs_u = task_to_indices[u]
        pred_u = model(X_batch[idxs_u], task_ids_batch[idxs_u])
        loss_u = loss_fn(pred_u, y_batch[idxs_u]).mean()
        loss_u.backward()

        # Freeze task-specific params during the probe.
        # grad=None (not zeros) makes Adam skip them entirely for this step.
        for p in task_params:
            p.grad = None

        main_optimizer.step()

        # New losses after the temporary source-task update.
        with torch.no_grad():
            preds_new = model(X_batch, task_ids_batch)
            losses_new_all = loss_fn(preds_new, y_batch).squeeze(-1).detach().cpu().numpy()
        loss_new_per_task = {
            t: float(np.mean([losses_new_all[b] for b in idxs]))
            for t, idxs in task_to_indices.items()
        }

        for v in unique_tasks:
            old = loss_old_per_task[v]
            new = loss_new_per_task[v]
            affinity_matrix[u, v] = 1.0 - new / old if old != 0.0 else 0.0

    # Restore the real training trajectory exactly.
    model.load_state_dict(base_model_state)
    main_optimizer.load_state_dict(base_optimizer_state)
    model.train(was_training)
    return affinity_matrix


def extract_per_participant_scores(affinity_matrix, participant_ids):
    """
    Per-participant score = mean of column j excluding diagonal:
    "how much does training on each other participant help/hurt j".
    """
    n = len(participant_ids)
    out = {}
    for j in range(n):
        col = [affinity_matrix[i, j] for i in range(n) if i != j]
        out[participant_ids[j]] = float(np.mean(col)) if col else float('nan')
    return out
