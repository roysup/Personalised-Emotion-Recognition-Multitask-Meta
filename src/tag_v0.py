"""
Task Affinity Grouping (TAG) — per-batch inter-task affinity probing for MTL.

Implements the Fifty et al. (2021) probe: for each task u in a batch, take a
single gradient step on shared parameters using only u's loss, then measure
the relative loss change on every other task v. Aggregating across batches
gives an NxN affinity matrix where (u, v) = "training on u helps/hurts v".

Designed to plug into MTLModel + make_mtl_loader directly.
"""
import copy
import numpy as np
import torch


def compute_inter_task_affinity(model, X_batch, y_batch, task_ids_batch,
                                affinity_optimizer, task_params, loss_fn,
                                num_tasks):
    """
    Compute one NxN affinity matrix from a single batch.

    Parameters
    ----------
    model : MTLModel
    X_batch, y_batch, task_ids_batch : tensors already on device
    affinity_optimizer : optim.Optimizer over SHARED params only — separate
        from the main training optimizer so its state can be saved/restored
        without disturbing training.
    task_params : list[nn.Parameter]
        Task-specific parameters whose grads are zeroed before the probe step
        (TAG rule: only shared params should move).
    loss_fn : nn.Module with reduction='none'.
    num_tasks : int

    Returns
    -------
    np.ndarray (num_tasks, num_tasks), float32. Tasks not present in the batch
    leave their rows/columns as zeros.
    """
    was_training = model.training
    model.eval()

    unique_tasks = sorted(set(task_ids_batch.tolist()))
    task_to_idx = {t: [] for t in unique_tasks}
    for b, tid in enumerate(task_ids_batch.tolist()):
        task_to_idx[tid].append(b)

    with torch.no_grad():
        losses_old = loss_fn(model(X_batch, task_ids_batch),
                             y_batch).squeeze(-1).cpu().numpy()
    loss_old_per_task = {t: float(np.mean([losses_old[b] for b in idxs]))
                         for t, idxs in task_to_idx.items()}

    aff = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    for u in unique_tasks:
        saved_model = copy.deepcopy(model.state_dict())
        saved_opt   = copy.deepcopy(affinity_optimizer.state_dict())

        model.train()
        affinity_optimizer.zero_grad()

        idx_u = task_to_idx[u]
        loss_u = loss_fn(model(X_batch[idx_u], task_ids_batch[idx_u]),
                         y_batch[idx_u]).mean()
        loss_u.backward()

        # TAG rule: zero task-specific grads (defensive — affinity_optimizer
        # only holds shared params, but keeps semantics explicit).
        for p in task_params:
            if p.grad is not None:
                p.grad.zero_()

        affinity_optimizer.step()

        model.eval()
        with torch.no_grad():
            losses_new = loss_fn(model(X_batch, task_ids_batch),
                                 y_batch).squeeze(-1).cpu().numpy()

        for v in unique_tasks:
            old_v = loss_old_per_task[v]
            new_v = float(np.mean([losses_new[b] for b in task_to_idx[v]]))
            aff[u, v] = (1.0 - new_v / old_v) if old_v != 0.0 else 0.0

        model.load_state_dict(saved_model)
        affinity_optimizer.load_state_dict(saved_opt)

    model.train(was_training)
    return aff


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
