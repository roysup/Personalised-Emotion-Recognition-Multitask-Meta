"""
Pure Meta-Learning — Reptile with single-task episodes, single shared model (no task heads).
"""
import os, sys, time
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
from config import (SEED, WINDOW_SIZE, STRIDE, EPOCHS, MAX_NORM, N_FOLDS,
                    META_STEPS, META_LR, INNER_STEPS, INNER_LR, L2_TASK,
                    HARDCODED_SPLITS, TEST_PARTICIPANTS, RESULTS_DIR)
import copy
import gc
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score
from utils import (set_all_seeds, compute_metrics_from_cm,
                   aggregate_mtml_results, make_kfolds, compute_per_participant_stds,
                   print_determinism_summary)
from data import build_support_query
from models import SingleTaskModel
from training import reptile_outer_update
from dataset_configs.vreed import load_vreed_df

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir       = os.path.join(BASE_OUTPUT_DIR, 'VREED_PureMeta')
os.makedirs(output_dir, exist_ok=True)

meta_steps_grid  = [META_STEPS]
meta_lr_grid     = [META_LR]
inner_steps_grid = [INNER_STEPS]
inner_lr_grid    = [INNER_LR]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_all_seeds(SEED)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df(mode='mtml')

participant_ids   = sorted([p for p in df['ID'].unique() if p in hardcoded_splits])
test_participants  = list(TEST_PARTICIPANTS)
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {len(train_participants)}  Test: {len(test_participants)}")

# =============================
# INNER LOOP
# =============================
def adapt(model, sup_loader, ar_or_va, inner_steps, inner_lr, l2_lambda):
    adapted = copy.deepcopy(model).to(device); adapted.train()
    opt     = optim.Adam(adapted.parameters(), lr=inner_lr, weight_decay=l2_lambda)
    sched   = ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    for step in range(inner_steps):
        ep_loss = 0.0; nb = 0
        for Xb, yb in sup_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(adapted(Xb), yb)
            if torch.isnan(loss):
                raise ValueError(f"NaN in adapt [step {step+1}]")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapted.parameters(), MAX_NORM)
            opt.step()
            ep_loss += loss.item(); nb += 1
        if nb > 0: sched.step(ep_loss / nb)
    return adapted

def make_sup_q_loader(df_user, splits, uid, ar_or_va, seed=SEED):
    return build_support_query(df_user, splits[uid]['train'], splits[uid]['test'],
                               ar_or_va, seed=seed, window_size=WINDOW_SIZE, stride=STRIDE)

# =============================
# META TRAINING
# =============================
def reptile_train(model, train_users, meta_steps, meta_lr, inner_steps, inner_lr, l2, ar_or_va, seed):
    rng  = np.random.default_rng(seed)
    uids = sorted(train_users.keys())
    model.to(device)
    for step in range(meta_steps):
        uid = int(rng.choice(uids))
        sup_loader, _ = make_sup_q_loader(train_users[uid], hardcoded_splits, uid, ar_or_va, seed)
        adapted = adapt(model, sup_loader, ar_or_va, inner_steps, inner_lr, l2)
        reptile_outer_update(model, [adapted], meta_lr)
    return model

# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type='ar'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] PureMeta\n{'='*60}")
    results     = []
    train_folds = make_kfolds(train_participants)
    for ms in meta_steps_grid:
        for mlr in meta_lr_grid:
            for isp in inner_steps_grid:
                for ilr in inner_lr_grid:
                    fold_f1s = []
                    for fold_i in range(N_FOLDS):
                        val_ps   = train_folds[fold_i]
                        tr_ps    = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
                        tr_users = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in tr_ps}
                        val_users= {uid: df[df['ID']==uid].reset_index(drop=True) for uid in val_ps}
                        try:
                            model = SingleTaskModel().to(device)
                            model = reptile_train(model, tr_users, ms, mlr, isp, ilr, L2_TASK, label_type, SEED)
                        except Exception: continue
                        val_f1s = []
                        for uid in val_ps:
                            sup_loader, q_loader = make_sup_q_loader(val_users[uid], hardcoded_splits, uid, label_type)
                            if len(q_loader.dataset) == 0: continue
                            adapted = adapt(model, sup_loader, label_type, isp, ilr, L2_TASK)
                            adapted.eval(); probs, labels = [], []
                            with torch.no_grad():
                                for Xb, yb in q_loader:
                                    probs.extend(torch.sigmoid(adapted(Xb.to(device, non_blocking=True))).cpu().numpy().flatten())
                                    labels.extend(yb.numpy().flatten())
                            if labels:
                                val_f1s.append(f1_score(np.array(labels).astype(int),
                                                        (np.array(probs)>0.5).astype(int),
                                                        average='macro', zero_division=0))
                        if val_f1s:
                            fold_f1s.append(np.mean(val_f1s))
                            print(f"  fold {fold_i+1}: f1={fold_f1s[-1]:.4f}")
                        del model; torch.cuda.empty_cache(); gc.collect()
                    if not fold_f1s: continue
                    avg = np.mean(fold_f1s)
                    results.append({'ms': ms, 'mlr': mlr, 'isp': isp, 'ilr': ilr,
                                    'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
                    print(f"  avg f1={avg:.4f}")
    if not results: return 50, 0.01, 10, 1e-3
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type}_tuning.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['ms'], best['mlr'], best['isp'], best['ilr']

# =============================
# MAIN
# =============================
if __name__ == '__main__':
    experiment_t0 = time.time()

    bms_ar, bmlr_ar, bisp_ar, bilr_ar = hyperparameter_tuning('ar')
    bms_va, bmlr_va, bisp_va, bilr_va = hyperparameter_tuning('va')

    train_users_ar = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in train_participants}
    test_users     = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in test_participants}

    print('\n' + '='*60 + '\nTRAINING FINAL AR\n' + '='*60)
    set_all_seeds(SEED)
    model_ar = SingleTaskModel().to(device)
    train_t0 = time.time()
    model_ar = reptile_train(model_ar, train_users_ar, bms_ar, bmlr_ar, bisp_ar, bilr_ar, L2_TASK, 'ar', SEED)
    print(f"  AR training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_ar.state_dict(), os.path.join(output_dir, 'meta_model_ar_final.pth'))

    print('\n' + '='*60 + '\nTRAINING FINAL VA\n' + '='*60)
    set_all_seeds(SEED)
    train_users_va = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in train_participants}
    model_va = SingleTaskModel().to(device)
    train_t0 = time.time()
    model_va = reptile_train(model_va, train_users_va, bms_va, bmlr_va, bisp_va, bilr_va, L2_TASK, 'va', SEED)
    print(f"  VA training complete in {time.time() - train_t0:.1f}s")
    torch.save(model_va.state_dict(), os.path.join(output_dir, 'meta_model_va_final.pth'))

    results_ar, results_va = [], []
    for model, results_list, label, bms, bisp, bilr in [
        (model_ar, results_ar, 'ar', bms_ar, bisp_ar, bilr_ar),
        (model_va, results_va, 'va', bms_va, bisp_va, bilr_va)]:
        print(f'\n' + '='*60 + f'\nEVALUATION {label.upper()}\n' + '='*60)
        for uid in sorted(test_participants):
            sup_loader, q_loader = make_sup_q_loader(test_users[uid], hardcoded_splits, uid, label)
            if len(q_loader.dataset) == 0: continue
            print(f"  Participant {uid}: adapting")
            adapted = adapt(model, sup_loader, label, bisp, bilr, L2_TASK)
            adapted.eval(); probs, labels_list = [], []
            with torch.no_grad():
                for Xb, yb in q_loader:
                    probs.extend(torch.sigmoid(adapted(Xb.to(device, non_blocking=True))).cpu().numpy().flatten())
                    labels_list.extend(yb.numpy().flatten())
            y_true = np.array(labels_list).astype(int)
            y_prob = np.array(probs)
            y_pred = (y_prob > 0.5).astype(int)
            cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc, prec, rec, f1 = compute_metrics_from_cm(cm)
            p = label
            results_list.append({
                'participant_id':       uid,
                'cm':                   cm,
                f'{p}_acc':             acc,
                f'{p}_precision':       prec,
                f'{p}_recall':          rec,
                f'{p}_f1':              f1,
                f'y_true_{p}':          y_true,
                f'y_pred_{p}':          y_pred,
                f'y_pred_probs_{p}':    y_prob,
            })
            print(f"  Participant {uid}: Acc={acc:.4f} F1={f1:.4f}")

    agg = aggregate_mtml_results(results_ar, results_va)

    global_roc = {
        'AR': {'fpr': agg['fpr_ar'], 'tpr': agg['tpr_ar'], 'auc': agg['ar_auc'],
               'y_true': agg['all_true_ar'], 'y_pred_probs': agg['all_probs_ar']},
        'VA': {'fpr': agg['fpr_va'], 'tpr': agg['tpr_va'], 'auc': agg['va_auc'],
               'y_true': agg['all_true_va'], 'y_pred_probs': agg['all_probs_va']},
    }
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(global_roc, f)

    ar_stds = compute_per_participant_stds(results_ar, 'ar')
    va_stds = compute_per_participant_stds(results_va, 'va')

    final_results = {
        'train_participants': train_participants, 'test_participants': test_participants,
        'best_hyperparameters': {
            'AR': {'meta_steps': bms_ar, 'meta_lr': bmlr_ar, 'inner_steps': bisp_ar,
                   'inner_lr': bilr_ar, 'l2_lambda': L2_TASK},
            'VA': {'meta_steps': bms_va, 'meta_lr': bmlr_va, 'inner_steps': bisp_va,
                   'inner_lr': bilr_va, 'l2_lambda': L2_TASK}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'puremeta_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    print_determinism_summary(
        {f'ar_{k}': final_results[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final_results[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
    print(f"Total experiment time: {time.time() - experiment_t0:.1f}s")
