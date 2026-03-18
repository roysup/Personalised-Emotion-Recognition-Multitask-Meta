"""
MTL PCGrad — Projected Conflicting Gradients
Same HPS architecture. Per-task gradients on shared params are projected
to remove conflicting components before being averaged and applied.
Task-head gradients come from the normal backward pass.
Seed is reset before AR training and again before VA training.
"""
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
from config import (SEED, WINDOW_SIZE, STRIDE, EPOCHS, MAX_NORM,
                    MTL_BATCH_SIZE, MTL_SHARED_LR, MTL_TASK_LR,
                    L2_SHARED, L2_TASK, HARDCODED_SPLITS, RESULTS_DIR)
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from data import make_mtl_loader
from dataset_configs.vreed import load_vreed_df, participant_ids
from models import MTLModel
from utils import set_all_seeds, aggregate_results
from training import (save_all_results,
                      evaluate_mtl_all, _pcgrad_project)

NUM_TASKS  = len(participant_ids)

OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_hps_pcgrad_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_all_seeds(SEED)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}\nOutput: {OUTPUT_DIR}")
# =============================
# DATA
# =============================
df = load_vreed_df()

# =============================
# PCGRAD APPLICATION
# =============================
def _apply_pcgrad(model, loss_fn, X_b, y_b, task_ids):
    """
    1. Compute per-task losses on shared params only.
    2. Project gradients via PCGrad (imported from training).
    3. Zero all grads, run standard .backward() for the task heads.
    4. Overwrite shared param grads with projected values.
    5. Return total loss (for scheduler and NaN check).
    """
    shared_params = model.shared_parameters()
    unique_tasks  = torch.unique(task_ids)

    task_grads = []
    for t in unique_tasks:
        mask   = (task_ids == t)
        loss_t = loss_fn(model(X_b, task_ids), y_b)[mask].mean()
        grads  = torch.autograd.grad(
            loss_t, shared_params,
            retain_graph=True, create_graph=False, allow_unused=True)
        flat = []
        for p, g in zip(shared_params, grads):
            flat.append(g.contiguous().view(-1) if g is not None
                        else torch.zeros_like(p).view(-1))
        task_grads.append(torch.cat(flat))

    projected  = _pcgrad_project(task_grads)
    total_loss = loss_fn(model(X_b, task_ids), y_b).mean()
    total      = total_loss + model.compute_l2(L2_SHARED, L2_TASK)
    total.backward()

    offset = 0
    for p in shared_params:
        n = p.numel()
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.copy_(projected[offset: offset + n].view_as(p))
        offset += n

    return total

# =============================
# TRAINING
# =============================
def _train_pcgrad(label_type, lr_shared, lr_task, train_data_dict):
    loader, _, _ = make_mtl_loader(
        train_data_dict, WINDOW_SIZE, STRIDE,
        label_type=label_type, batch_size=MTL_BATCH_SIZE, seed=SEED)

    model = MTLModel(NUM_TASKS).to(device)
    opt   = optim.Adam([
        {'params': model.shared_parameters(),        'lr': lr_shared},
        {'params': model.task_specific_parameters(), 'lr': lr_task},
    ])
    sched     = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn   = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    ckpt_path = os.path.join(OUTPUT_DIR, f'best_model_{label_type}_hps_pcgrad.pt')

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in loader:
            X_b, y_b, task_ids, _ = [b.to(device, non_blocking=True) for b in batch]
            opt.zero_grad(set_to_none=True)
            total = _apply_pcgrad(model, loss_fn, X_b, y_b, task_ids)

            if torch.isnan(total):
                raise ValueError(f"NaN at epoch {epoch+1} [{label_type.upper()}]")

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
            running += total.item()

        avg = running / len(loader)
        sched.step(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label_type.upper()}] Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return model

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    experiment_t0 = time.time()

    train_data, test_data = {}, {}
    for task_idx, pid in enumerate(participant_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        train_data[task_idx] = p_df[p_df['Trial'].isin(HARDCODED_SPLITS[pid]['train'])].reset_index(drop=True)
        test_data[task_idx]  = p_df[p_df['Trial'].isin(HARDCODED_SPLITS[pid]['test'])].reset_index(drop=True)

    print("\n" + "="*60 + "\nTRAINING AR\n" + "="*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_ar = _train_pcgrad('ar', MTL_SHARED_LR, MTL_TASK_LR, train_data)
    print(f"  AR training complete in {time.time() - train_t0:.1f}s")

    print("\n" + "="*60 + "\nTRAINING VA\n" + "="*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    model_va = _train_pcgrad('va', MTL_SHARED_LR, MTL_TASK_LR, train_data)
    print(f"  VA training complete in {time.time() - train_t0:.1f}s")

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    results = evaluate_mtl_all(model_ar, model_va, test_data,
                               participant_ids, device, WINDOW_SIZE, STRIDE)
    agg     = aggregate_results(results)

    results_df, ar_stds, va_stds = save_all_results(
        results, agg, OUTPUT_DIR,
        method_name='MTL-PCGrad',
        misclassification_csv='VREED_hps_pcgrad_misclassification_rates.csv')

    with open(os.path.join(OUTPUT_DIR, 'hps_pcgrad_results.pkl'), 'wb') as f:
        pickle.dump({**agg, 'per_participant': results,
                     'per_participant_table': results_df,
                     **ar_stds, **va_stds}, f)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
