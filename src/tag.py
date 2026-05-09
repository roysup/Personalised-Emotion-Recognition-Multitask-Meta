"""
Task Affinity Grouping (TAG) — per-batch inter-task affinity probing for MTL.

Implements the Fifty et al. (2021) probe, aligned with the official reference
implementation (google-research/tag/taskonomy/tag.patch — `lookahead` and
`train_batch`). For each source task u in a batch, take a single SGD step on
shared parameters using only u's loss, then measure the relative loss change
on every other task v. Aggregating across batches gives an NxN affinity matrix
where affinity_matrix[u, v] = "training on u helps/hurts v".

Implementation choices, all matching the reference:
  * ONE full-batch forward pass; per-task losses are sliced from it. The
    gradient computation therefore backprops through the same BatchNorm
    statistics as a real training step on this batch, instead of the
    single-sample BN context that a per-task slice forward would produce.
  * Gradients via `torch.autograd.grad(..., retain_graph=True)` against
    shared parameters only. No `.backward()` call, so no leaked grads on
    task-specific parameters.
  * Manual vanilla-SGD step on the shared parameters (the reference's
    momentum branch is dead code: it checks `'momentum_buffer' in opt_state`
    where opt_state is `param_groups[0]`, but momentum_buffer lives in
    `optimizer.state[param_id]`; so the reference probe is effectively
    SGD with weight decay, no momentum, no Adam moments). This isolates
    the per-task gradient direction without mixing in optimizer history,
    matches Eq. (1) of the paper, and removes the entire optimizer-state-
    aliasing bug class (no `optimizer.step()` is called).
  * Restoration: `param.data = init_weight` reassigns the data tensor back
    to the saved reference. Works because the manual SGD step uses
    `param.data = param.data - lr * grad`, which creates a NEW tensor
    rather than mutating in place; the original tensor is preserved.
  * BatchNorm running stats: we temporarily disable `track_running_stats`
    on all BN modules for the duration of the probe. The model stays in
    train() mode, so BN still normalizes with BATCH statistics (matching
    reference lookahead behavior). Only the side-effect — the in-place
    update of `running_mean` / `running_var` — is suppressed. This is
    needed because the in-place BN buffer update during the probe forward
    trips autograd's version check on `autograd.grad(..., retain_graph=True)`
    under deterministic-algorithm mode, and would also pollute the real
    training trajectory's BN stats with probe steps.

Affinity formula:
    Z_{u->v} = (1 - L_v(after) / L_v(before)) / lr
The /lr is in the reference code but NOT in paper Eq. (1). It makes Z
scale-invariant w.r.t. learning rate (since L_v change is proportional to
lr to first order). Without /lr, decreasing the learning rate trivially
shrinks all affinities, breaking cross-experiment comparison and absolute
interpretation. For ranking-based network selection it doesn't matter;
for correlation analysis it changes magnitudes but not signs.

Diagonal: Z_{u->u} (self-affinity) is computed but is biased on training
data (paper Appendix B.4: "TAG acting on the training dataset will never
group a task by itself"). Downstream `extract_per_participant_scores`
excludes the diagonal, matching the reference's network selection logic.
"""
import numpy as np
import torch


_BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
             torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)


def _disable_bn_tracking(model):
    """Temporarily set track_running_stats=False on all BN modules.
    Returns a list of (module, prior_state) tuples for restoration.

    With train mode preserved, BN keeps using batch statistics for
    normalization. Only the running-stat update side-effect is suppressed,
    so nothing in-place modifies `running_mean` / `running_var` during the
    probe forward, which (a) avoids autograd version-counter errors and
    (b) keeps the probe from polluting real training's BN stats."""
    states = []
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            states.append((m, m.track_running_stats))
            m.track_running_stats = False
    return states


def _restore_bn_tracking(states):
    for m, was_tracking in states:
        m.track_running_stats = was_tracking


def compute_inter_task_affinity(model, X_batch, y_batch, task_ids_batch,
                                main_optimizer, shared_params, task_params,
                                loss_fn, num_tasks):
    """
    Reference-aligned TAG affinity computation for one batch.

    For each source task u present in the batch and each target task v:
        affinity_matrix[u, v] = (1 - L_v(after temp SGD step from u) / L_v(before)) / lr

    Parameters
    ----------
    model : nn.Module
        The shared MTL model. Should already be in train() mode for the
        forward pass to use training-style BN/dropout (matching reference).
    X_batch, y_batch : tensors on device
        Full balanced batch (all tasks).
    task_ids_batch : tensor on device
        Task ID for each sample in the batch.
    main_optimizer : optim.Optimizer
        Used ONLY to read the current learning rate from
        `param_groups[0]['lr']`. Not stepped, not state-mutated.
    shared_params : list[nn.Parameter]
        Shared backbone parameters that the probe step modifies.
    task_params : list[nn.Parameter]
        Accepted for API compatibility with the previous version. NOT used
        here — gradients are computed only against `shared_params` via
        autograd.grad, so task heads are untouched by construction.
    loss_fn : nn.Module
        Per-sample loss with reduction='none'.
    num_tasks : int
        Total number of tasks (matrix is num_tasks x num_tasks).

    Returns
    -------
    np.ndarray, shape (num_tasks, num_tasks), dtype float32.
        Rows = source task u, columns = target task v.
        Tasks not present in the batch leave their rows/columns as zeros.
    """
    was_training = model.training
    model.train()  # match reference: probe forward uses training BN/dropout

    # Disable BN running-stat updates for the probe (see module docstring).
    bn_tracking_state = _disable_bn_tracking(model)

    try:
        # Bucket batch indices by task ID.
        task_ids_cpu = task_ids_batch.detach().cpu().tolist()
        unique_tasks = sorted(set(task_ids_cpu))
        task_to_indices = {t: [] for t in unique_tasks}
        for bidx, tid in enumerate(task_ids_cpu):
            task_to_indices[tid].append(bidx)

        lr = main_optimizer.param_groups[0]['lr']

        # ONE full-batch forward with grad enabled. All per-task losses are
        # sliced from this single forward, so the gradient w.r.t. shared
        # params backprops through the same multi-task BN context that a
        # real training step would see.
        preds_full = model(X_batch, task_ids_batch)
        losses_per_sample = loss_fn(preds_full, y_batch).squeeze(-1)

        # Old per-task losses. Detached scalars; no grad needed for these.
        losses_per_sample_np = losses_per_sample.detach().cpu().numpy()
        loss_old_per_task = {
            t: float(np.mean([losses_per_sample_np[b] for b in idxs]))
            for t, idxs in task_to_indices.items()
        }

        affinity_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

        for u in unique_tasks:
            # Save references to current shared param tensors. These are
            # aliases, not copies — but the manual SGD step below reassigns
            # param.data to a NEW tensor (param.data = param.data - lr*grad),
            # leaving the original tensor (referenced here) untouched.
            init_weights = [p.data for p in shared_params]

            # u's loss = mean of per-sample losses for u's slice.
            # autograd.grad only computes grads for shared_params;
            # retain_graph=True so subsequent iterations of this loop can
            # reuse the forward graph.
            idxs_u = task_to_indices[u]
            loss_u = losses_per_sample[idxs_u].mean()
            grads = torch.autograd.grad(loss_u, shared_params,
                                        retain_graph=True)

            # Manual vanilla SGD step on shared params. No optimizer.step(),
            # so no Adam moments are touched and no state aliasing can occur.
            with torch.no_grad():
                for param, grad in zip(shared_params, grads):
                    # wd_term = main_optimizer.param_groups[0].get('weight_decay', 0.0) * param
                    # param.data = param.data - lr * (grad + wd_term)
                    param.data = param.data - lr * grad

                # Forward FULL batch with the updated shared params. no_grad
                # because we only need the loss values for the affinity
                # ratio.
                preds_new = model(X_batch, task_ids_batch)
                losses_new_per_sample = loss_fn(preds_new, y_batch).squeeze(-1)
                losses_new_np = losses_new_per_sample.detach().cpu().numpy()

            loss_new_per_task = {
                t: float(np.mean([losses_new_np[b] for b in idxs]))
                for t, idxs in task_to_indices.items()
            }

            for v in unique_tasks:
                old = loss_old_per_task[v]
                new = loss_new_per_task[v]
                if old != 0.0:
                    affinity_matrix[u, v] = (1.0 - new / old) / lr
                else:
                    affinity_matrix[u, v] = 0.0

            # Restore shared param tensors to their pre-probe values.
            with torch.no_grad():
                for param, init_weight in zip(shared_params, init_weights):
                    param.data = init_weight

    finally:
        # Always re-enable BN tracking and restore the original mode,
        # even if the probe loop raised.
        _restore_bn_tracking(bn_tracking_state)
        model.train(was_training)

    return affinity_matrix


def extract_per_participant_scores(affinity_matrix, participant_ids):
    """
    Per-participant score = mean of column j excluding the diagonal entry:
    "how much does training on each other participant help/hurt j".

    Excludes the diagonal because train-data self-affinity is biased
    (paper Appendix B.4: TAG on training data will never group a task by
    itself). This matches the reference's network selection behavior.
    """
    n = len(participant_ids)
    out = {}
    for j in range(n):
        col = [affinity_matrix[i, j] for i in range(n) if i != j]
        out[participant_ids[j]] = float(np.mean(col)) if col else float('nan')
    return out
