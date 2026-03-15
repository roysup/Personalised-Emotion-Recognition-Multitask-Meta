"""
MTL Hard Parameter Sharing (HPS)
Shared CNN+LSTM backbone, task-specific dense heads per participant.
Separate AR and VA models trained sequentially.
Seed is reset before AR training and again before VA training.
"""
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
from config import *
from data import create_sliding_windows, make_mtl_loader
from dataset_configs.vreed import load_vreed_df, participant_ids
from models import MTLModel
from utils import set_all_seeds, compute_metrics_from_cm, create_kfold_splits
from training import save_all_results, evaluate_mtl_all
from utils import aggregate_results

BATCH_SIZE = MTL_BATCH_SIZE
NUM_TASKS  = len(participant_ids)
SHARED_LR  = MTL_SHARED_LR
TASK_LR    = MTL_TASK_LR
L2_TASK    = MTL_L2_TASK
L2_SHARED  = 0.0

OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_hps_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_all_seeds(SEED)
print(f"Device: {device}\nOutput: {OUTPUT_DIR}")

# =============================
# DATA
# =============================
df = load_vreed_df()

# =============================
# HELPERS
# =============================
def _l2_task_only(model):
    return L2_TASK * sum(p.norm(2)**2 for p in model.task_specific_parameters()
                         if p.requires_grad)

def _train_mtl(label_type, lr_shared, lr_task, l2_fn, train_data_dict):
    loader, _, _ = make_mtl_loader(
        train_data_dict, WINDOW_SIZE, STRIDE,
        label_type=label_type, batch_size=BATCH_SIZE, seed=SEED)

    model = MTLModel(NUM_TASKS).to(device)
    opt   = optim.Adam([
        {'params': model.shared_parameters(),        'lr': lr_shared},
        {'params': model.task_specific_parameters(), 'lr': lr_task},
    ])
    sched     = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    loss_fn   = nn.BCEWithLogitsLoss(reduction='none')
    best_loss = float('inf')
    ckpt_path = os.path.join(OUTPUT_DIR, f'best_model_{label_type}_hps_tuned.pt')

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in loader:
            X_b, y_b, task_ids, _ = [b.to(device) for b in batch]
            opt.zero_grad()
            loss  = loss_fn(model(X_b, task_ids), y_b).mean()
            total = loss + l2_fn(model)
            if torch.isnan(total):
                raise ValueError(f"NaN at epoch {epoch+1} [{label_type.upper()}]")
            total.backward()
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

    model.load_state_dict(torch.load(ckpt_path))
    return model

# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type, shared_lrs, task_lrs, l2_lambdas_task):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING  [{label_type.upper()}]  MTL-HPS\n{'='*60}")
    all_results = []
    for sh_lr in shared_lrs:
        for tk_lr in task_lrs:
            for l2 in l2_lambdas_task:
                fold_f1s = []
                for fold_i in range(N_FOLDS):
                    train_data, val_data = {}, {}
                    for task_idx, pid in enumerate(participant_ids):
                        p_df = df[df['ID'] == pid].reset_index(drop=True)
                        folds = create_kfold_splits(HARDCODED_SPLITS[pid]['train'], N_FOLDS)
                        tr_v, va_v = folds[fold_i]
                        train_data[task_idx] = p_df[p_df['Trial'].isin(tr_v)].reset_index(drop=True)
                        val_data[task_idx]   = p_df[p_df['Trial'].isin(va_v)].reset_index(drop=True)

                    set_all_seeds(SEED)
                    loader, _, _ = make_mtl_loader(
                        train_data, WINDOW_SIZE, STRIDE,
                        label_type=label_type, batch_size=BATCH_SIZE, seed=SEED)

                    model = MTLModel(NUM_TASKS).to(device)
                    opt   = optim.Adam([
                        {'params': model.shared_parameters(),        'lr': sh_lr},
                        {'params': model.task_specific_parameters(), 'lr': tk_lr},
                    ])
                    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
                    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

                    for _ in range(EPOCHS):
                        model.train()
                        run = 0.0
                        for batch in loader:
                            X_b, y_b, tids, _ = [b.to(device) for b in batch]
                            opt.zero_grad()
                            total = (loss_fn(model(X_b, tids), y_b).mean() +
                                     l2 * sum(p.norm(2)**2
                                              for p in model.task_specific_parameters()
                                              if p.requires_grad))
                            total.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                            opt.step(); run += total.item()
                        sched.step(run / len(loader))

                    tp = fp = fn = 0
                    model.eval()
                    with torch.no_grad():
                        for task_idx, val_df in val_data.items():
                            if len(val_df) == 0: continue
                            X_v, y_ar_v, y_va_v, _, _ = create_sliding_windows(
                                val_df, WINDOW_SIZE, STRIDE, task_id=task_idx)
                            if len(X_v) == 0: continue
                            y_v   = torch.tensor(
                                y_ar_v if label_type == 'ar' else y_va_v,
                                dtype=torch.float32).unsqueeze(1)
                            X_vt  = torch.tensor(X_v, dtype=torch.float32).to(device)
                            tids  = torch.full((len(X_v),), task_idx, dtype=torch.long).to(device)
                            pred  = (torch.sigmoid(model(X_vt, tids)) > 0.5).float().cpu()
                            tp   += (y_v * pred).sum().item()
                            fp   += ((1-y_v)*pred).sum().item()
                            fn   += (y_v*(1-pred)).sum().item()

                    p = tp/(tp+fp+1e-7); r = tp/(tp+fn+1e-7)
                    f1 = 2*p*r/(p+r+1e-7)
                    fold_f1s.append(f1)
                    print(f"  fold {fold_i+1}: f1={f1:.4f}  "
                          f"(sh_lr={sh_lr}, tk_lr={tk_lr}, l2={l2})")

                avg = np.mean(fold_f1s)
                all_results.append({'sh_lr': sh_lr, 'tk_lr': tk_lr, 'l2': l2,
                                    'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
                print(f"  avg f1={avg:.4f}")

    best = max(all_results, key=lambda x: x['avg_f1'])
    print(f"\nBest: sh_lr={best['sh_lr']}, tk_lr={best['tk_lr']}, "
          f"l2={best['l2']}, f1={best['avg_f1']:.4f}")
    with open(os.path.join(OUTPUT_DIR, f'{label_type}_tuning.pkl'), 'wb') as f:
        pickle.dump({'all': all_results, 'best': best}, f)
    return best['sh_lr'], best['tk_lr'], best['l2']

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    best_sh_ar, best_tk_ar, best_l2_ar = hyperparameter_tuning('ar', [SHARED_LR], [TASK_LR], [L2_TASK])
    best_sh_va, best_tk_va, best_l2_va = hyperparameter_tuning('va', [SHARED_LR], [TASK_LR], [L2_TASK])

    train_data, test_data = {}, {}
    for task_idx, pid in enumerate(participant_ids):
        p_df = df[df['ID'] == pid].reset_index(drop=True)
        train_data[task_idx] = p_df[p_df['Trial'].isin(HARDCODED_SPLITS[pid]['train'])].reset_index(drop=True)
        test_data[task_idx]  = p_df[p_df['Trial'].isin(HARDCODED_SPLITS[pid]['test'])].reset_index(drop=True)

    print("\n" + "="*60 + "\nTRAINING AR\n" + "="*60)
    set_all_seeds(SEED)
    model_ar = _train_mtl('ar', best_sh_ar, best_tk_ar, _l2_task_only, train_data)

    print("\n" + "="*60 + "\nTRAINING VA\n" + "="*60)
    set_all_seeds(SEED)
    model_va = _train_mtl('va', best_sh_va, best_tk_va, _l2_task_only, train_data)

    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    results = evaluate_mtl_all(model_ar, model_va, test_data,
                               participant_ids, device, WINDOW_SIZE, STRIDE)
    agg     = aggregate_results(results)

    results_df, ar_stds, va_stds = save_all_results(
        results, agg, OUTPUT_DIR,
        method_name='MTL-HPS',
        misclassification_csv='VREED_hps_misclassification_rates.csv')

    with open(os.path.join(OUTPUT_DIR, 'hps_tuned_results.pkl'), 'wb') as f:
        pickle.dump({**agg, 'per_participant': results,
                     'per_participant_table': results_df,
                     **ar_stds, **va_stds}, f)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
