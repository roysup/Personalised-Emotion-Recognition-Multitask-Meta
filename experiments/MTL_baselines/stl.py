"""
Single-Task Learning (STL) — per-participant
One AR model and one VA model trained independently per participant.
Seed is reset before the full AR pass and again before the full VA pass.
"""
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
from config import *
from data import create_sliding_windows, arrays_to_loader
from dataset_configs.vreed import load_vreed_df, participant_ids
from models import SingleTaskModel
from utils import set_all_seeds, create_kfold_splits, aggregate_results
from training import save_all_results, evaluate_stl_all

BATCH_SIZE = STL_BATCH_SIZE
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_stl_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}\nOutput: {OUTPUT_DIR}")

set_all_seeds(SEED)

# =============================
# DATA
# =============================
df = load_vreed_df()

# =============================
# HELPERS
# =============================
def _train_participant(task_idx, label_type, lr, l2_lambda, train_videos, participant_data):
    train_df = participant_data[participant_data['Trial'].isin(train_videos)].reset_index(drop=True)
    if len(train_df) == 0:
        return None
    X, y_ar, y_va, _, _ = create_sliding_windows(train_df, WINDOW_SIZE, STRIDE, task_id=task_idx)
    if len(X) == 0:
        return None

    y      = y_ar if label_type == 'ar' else y_va
    loader = arrays_to_loader(X, y, BATCH_SIZE, shuffle=True, seed=SEED)
    model  = SingleTaskModel().to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_b), y_b)
            if torch.isnan(loss):
                raise ValueError(f"NaN at epoch {epoch+1} [task {task_idx} {label_type.upper()}]")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step()
            running += loss.item()
        sched.step(running / len(loader))
    return model


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, learning_rates, l2_lambdas):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING  [{label_type.upper()}]  STL\n{'='*60}")
    results = []
    for lr in learning_rates:
        for l2 in l2_lambdas:
            fold_f1s = []
            for fold_i in range(N_FOLDS):
                per_participant_val_f1 = []
                for task_idx, pid in enumerate(participant_ids):
                    p_df  = df[df['ID'] == pid].reset_index(drop=True)
                    folds = create_kfold_splits(HARDCODED_SPLITS[pid]['train'], N_FOLDS)
                    tr_v, va_v = folds[fold_i]

                    X_tr, y_ar_tr, y_va_tr, _, _ = create_sliding_windows(
                        p_df[p_df['Trial'].isin(tr_v)].reset_index(drop=True),
                        WINDOW_SIZE, STRIDE, task_id=task_idx)
                    X_va, y_ar_va, y_va_va, _, _ = create_sliding_windows(
                        p_df[p_df['Trial'].isin(va_v)].reset_index(drop=True),
                        WINDOW_SIZE, STRIDE, task_id=task_idx)

                    if len(X_tr) == 0 or len(X_va) == 0:
                        continue

                    y_tr     = y_ar_tr if label_type == 'ar' else y_va_tr
                    y_va_lbl = y_ar_va if label_type == 'ar' else y_va_va

                    #set_all_seeds(SEED)
                    model   = SingleTaskModel().to(device)
                    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
                    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
                    lfn     = nn.BCEWithLogitsLoss()
                    ldr_tr  = arrays_to_loader(X_tr, y_tr,     BATCH_SIZE, shuffle=True,  seed=SEED)
                    ldr_va  = arrays_to_loader(X_va, y_va_lbl, BATCH_SIZE, shuffle=False)

                    best_f1 = 0.0
                    for _ in range(EPOCHS):
                        model.train()
                        run = 0.0
                        for X_b, y_b in ldr_tr:
                            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
                            opt.zero_grad(set_to_none=True)
                            total = lfn(model(X_b), y_b)
                            total.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                            opt.step(); run += total.item()
                        sched.step(run / len(ldr_tr))

                        model.eval()
                        tp = fp = fn = 0
                        with torch.no_grad():
                            for X_v, y_v in ldr_va:
                                out  = model(X_v.to(device, non_blocking=True))
                                pred = (torch.sigmoid(out) > 0.5).float().cpu()
                                tp  += (y_v * pred).sum().item()
                                fp  += ((1-y_v)*pred).sum().item()
                                fn  += (y_v*(1-pred)).sum().item()
                        p = tp/(tp+fp+1e-7); r = tp/(tp+fn+1e-7)
                        best_f1 = max(best_f1, 2*p*r/(p+r+1e-7))

                    per_participant_val_f1.append(best_f1)

                fold_f1s.append(np.mean(per_participant_val_f1) if per_participant_val_f1 else 0.0)
                print(f"  fold {fold_i+1}: avg_f1={fold_f1s[-1]:.4f}  (lr={lr}, l2={l2})")

            avg = np.mean(fold_f1s)
            results.append({'lr': lr, 'l2': l2, 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
            print(f"  avg f1={avg:.4f}")

    best = max(results, key=lambda x: x['avg_f1'])
    print(f"\nBest: lr={best['lr']}, l2={best['l2']}, f1={best['avg_f1']:.4f}")
    with open(os.path.join(OUTPUT_DIR, f'{label_type}_tuning.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr'], best['l2']

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    experiment_t0 = time.time()

    best_lr_ar, best_l2_ar = hyperparameter_tuning('ar', [MTL_SHARED_LR], [MTL_L2_TASK])
    best_lr_va, best_l2_va = hyperparameter_tuning('va', [MTL_SHARED_LR], [MTL_L2_TASK])

    models_ar, models_va = {}, {}
    test_data = {}

    print("\n" + "="*60 + "\nTRAINING AR — ALL PARTICIPANTS\n" + "="*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    for task_idx, pid in enumerate(participant_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        m = _train_participant(task_idx, 'ar', best_lr_ar, best_l2_ar,
                               HARDCODED_SPLITS[pid]['train'], p_df)
        if m is not None:
            models_ar[task_idx] = m
            torch.save(m.state_dict(),
                       os.path.join(OUTPUT_DIR, f'final_model_ar_participant_{pid}_tuned.pt'))
    print(f"  AR training complete in {time.time() - train_t0:.1f}s")

    print("\n" + "="*60 + "\nTRAINING VA — ALL PARTICIPANTS\n" + "="*60)
    set_all_seeds(SEED)
    train_t0 = time.time()
    for task_idx, pid in enumerate(participant_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        m = _train_participant(task_idx, 'va', best_lr_va, best_l2_va,
                               HARDCODED_SPLITS[pid]['train'], p_df)
        if m is not None:
            models_va[task_idx] = m
    print(f"  VA training complete in {time.time() - train_t0:.1f}s")

    # Build per-participant test DataFrames
    for task_idx, pid in enumerate(participant_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        test_data[task_idx] = p_df[p_df['Trial'].isin(
            HARDCODED_SPLITS[pid]['test'])].reset_index(drop=True)

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    results = evaluate_stl_all(models_ar, models_va, test_data,
                               participant_ids, device, WINDOW_SIZE, STRIDE)

    agg = aggregate_results(results)

    results_df, ar_stds, va_stds = save_all_results(
        results, agg, OUTPUT_DIR,
        method_name='STL',
        misclassification_csv='VREED_stl_misclassification_rates.csv')

    with open(os.path.join(OUTPUT_DIR, 'stl_tuned_results.pkl'), 'wb') as f:
        pickle.dump({**agg, 'per_participant': results,
                     'per_participant_table': results_df,
                     **ar_stds, **va_stds}, f)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
