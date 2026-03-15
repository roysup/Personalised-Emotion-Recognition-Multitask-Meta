"""
Reptile Meta-MTL — Multi-Task Episodes
Selects 5 participants per episode, averages their backbone deltas.
"""
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

import gc
from collections import OrderedDict
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from config import HARDCODED_SPLITS, SEED
from utils import set_all_seeds, compute_metrics_from_cm, safe_roc_auc, make_kfolds
from models import BaseFeatureExtractor, TaskHead
from training import adapt_inner_loop, evaluate_test_user
from data import build_support_query
from dataset_configs.vreed import load_vreed_df_mtml
from paths import RESULTS_DIR

hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir = os.path.join(BASE_OUTPUT_DIR, 'VREED_ReptileMeta_MT_episode')
os.makedirs(output_dir, exist_ok=True)

WINDOW_SIZE = 2560; STRIDE = 1280; N_FOLDS = 5
L2_SHARED = 0.0; L2_TASK = 1e-5; EPISODE_SIZE = 5
meta_steps_grid  = [50];  meta_lr_grid     = [0.01]
inner_steps_grid = [10];  inner_lr_grid    = [1e-3]

set_all_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df_mtml()

participant_ids   = sorted([p for p in df['ID'].unique() if p in hardcoded_splits])
test_participants  = [105,109,112,125,131,132]
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {len(train_participants)}  Test: {len(test_participants)}")


# =============================
# META TRAINING — MULTI TASK EPISODES
# =============================
def reptile_train_mt(train_users, meta_steps, meta_lr, inner_steps, inner_lr,
                     ar_or_va, l2_shared, l2_task, seed):
    base = BaseFeatureExtractor().to(device)
    heads = {uid: TaskHead().to(device) for uid in train_users}
    rng = np.random.default_rng(seed)
    uids = sorted(train_users.keys())
    for step in range(meta_steps):
        selected = list(rng.choice(uids, size=min(EPISODE_SIZE, len(uids)), replace=False))
        deltas = []
        for uid in selected:
            sup_loader, _ = build_support_query(train_users[uid],
                                                 hardcoded_splits[uid]['train'], [],
                                                 ar_or_va, seed=SEED)
            adapted_base, adapted_head = adapt_inner_loop(
                base, heads[uid], sup_loader, ar_or_va,
                inner_steps, inner_lr, device, l2_shared, l2_task)
            heads[uid] = adapted_head
            delta = OrderedDict()
            for (name, p0), (_, pa) in zip(base.named_parameters(), adapted_base.named_parameters()):
                delta[name] = pa.data - p0.data
            deltas.append(delta)
        mean_delta = {name: torch.stack([d[name] for d in deltas]).mean(0) for name in deltas[0]}
        with torch.no_grad():
            for name, p in base.named_parameters():
                p.data.add_(meta_lr * mean_delta[name])
    return base


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type='ar'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type.upper()}] Reptile-MT\n{'='*60}")
    results = []; train_folds = make_kfolds(train_participants)
    for ms in meta_steps_grid:
        for mlr in meta_lr_grid:
            for isp in inner_steps_grid:
                for ilr in inner_lr_grid:
                    fold_f1s = []
                    for fold_i in range(N_FOLDS):
                        val_ps = train_folds[fold_i]
                        tr_ps  = [p for j,f in enumerate(train_folds) if j != fold_i for p in f]
                        tr_users = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in tr_ps}
                        val_users = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in val_ps}
                        try:
                            base = reptile_train_mt(tr_users, ms, mlr, isp, ilr, label_type,
                                                     L2_SHARED, L2_TASK, SEED)
                        except Exception as e:
                            print(f"  fold {fold_i+1}: training failed: {e}"); continue
                        val_f1s = []
                        for uid in val_ps:
                            r = evaluate_test_user(base, TaskHead(), val_users[uid], hardcoded_splits,
                                                    uid, label_type, device, isp, ilr, L2_SHARED, L2_TASK)
                            if r is not None: val_f1s.append(r['f1'])
                        if val_f1s:
                            fold_f1s.append(np.mean(val_f1s))
                            print(f"  fold {fold_i+1}: f1={fold_f1s[-1]:.4f}")
                        del base; torch.cuda.empty_cache(); gc.collect()
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
    bms_ar, bmlr_ar, bisp_ar, bilr_ar = hyperparameter_tuning('ar')
    bms_va, bmlr_va, bisp_va, bilr_va = hyperparameter_tuning('va')

    train_users = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in train_participants}
    test_users  = {uid: df[df['ID']==uid].reset_index(drop=True) for uid in test_participants}

    print('\n' + '='*60 + '\nTRAINING FINAL AR\n' + '='*60)
    set_all_seeds(SEED)
    model_ar = reptile_train_mt(train_users, bms_ar, bmlr_ar, bisp_ar, bilr_ar,
                                 'ar', L2_SHARED, L2_TASK, SEED)
    torch.save(model_ar.state_dict(), os.path.join(output_dir, 'reptile_mt_model_ar_final.pth'))

    print('\n' + '='*60 + '\nTRAINING FINAL VA\n' + '='*60)
    set_all_seeds(SEED)
    model_va = reptile_train_mt(train_users, bms_va, bmlr_va, bisp_va, bilr_va,
                                 'va', L2_SHARED, L2_TASK, SEED)
    torch.save(model_va.state_dict(), os.path.join(output_dir, 'reptile_mt_model_va_final.pth'))

    results_ar, results_va = [], []
    for model, results, label, bisp, bilr in [
        (model_ar, results_ar, 'ar', bisp_ar, bilr_ar),
        (model_va, results_va, 'va', bisp_va, bilr_va)]:
        print(f'\n' + '='*60 + f'\nEVALUATION {label.upper()}\n' + '='*60)
        for uid in sorted(test_participants):
            print(f"  Participant {uid}: adapting")
            r = evaluate_test_user(model, TaskHead(), test_users[uid], hardcoded_splits,
                                    uid, label, device, bisp, bilr, L2_SHARED, L2_TASK)
            if r is not None:
                results.append(r)
                print(f"  Participant {uid}: Acc={r['accuracy']:.4f} F1={r['f1']:.4f}")

    def aggregate(results, label):
        all_true  = np.concatenate([r['y_true'] for r in results])
        all_pred  = np.concatenate([r['y_pred'] for r in results])
        all_probs = np.concatenate([r['y_pred_probs'] for r in results])
        cm = confusion_matrix(all_true, all_pred, labels=[0,1])
        acc, prec, rec, f1 = compute_metrics_from_cm(cm)
        auc_val, fpr, tpr = safe_roc_auc(all_true, all_probs)
        print(f"\n{label}: Acc={acc:.4f} F1={f1:.4f} AUC={auc_val:.4f}")
        return all_true, all_probs, cm, acc, prec, rec, f1, auc_val, fpr, tpr

    all_true_ar, all_probs_ar, cm_AR, ar_acc, ar_prec, ar_rec, ar_f1, ar_auc, ar_fpr, ar_tpr = aggregate(results_ar, 'AR')
    all_true_va, all_probs_va, cm_VA, va_acc, va_prec, va_rec, va_f1, va_auc, va_fpr, va_tpr = aggregate(results_va, 'VA')

    global_roc = {'AR': {'fpr': ar_fpr, 'tpr': ar_tpr, 'auc': ar_auc, 'y_true': all_true_ar, 'y_pred_probs': all_probs_ar},
                  'VA': {'fpr': va_fpr, 'tpr': va_tpr, 'auc': va_auc, 'y_true': all_true_va, 'y_pred_probs': all_probs_va}}
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(global_roc, f)

    final_results = {
        'train_participants': train_participants, 'test_participants': test_participants,
        'best_hyperparameters': {
            'AR': {'meta_steps': bms_ar, 'meta_lr': bmlr_ar, 'inner_steps': bisp_ar, 'inner_lr': bilr_ar},
            'VA': {'meta_steps': bms_va, 'meta_lr': bmlr_va, 'inner_steps': bisp_va, 'inner_lr': bilr_va}},
        'ar_acc': ar_acc, 'ar_precision': ar_prec, 'ar_recall': ar_rec, 'ar_f1': ar_f1, 'ar_auc': ar_auc,
        'va_acc': va_acc, 'va_precision': va_prec, 'va_recall': va_rec, 'va_f1': va_f1, 'va_auc': va_auc,
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': cm_AR, 'cm_va': cm_VA,
    }
    with open(os.path.join(output_dir, 'reptile_mt_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    # =============================
    # DETERMINISM SUMMARY
    # =============================
    from utils import compute_per_participant_stds, print_determinism_summary

    def _prefix(results, prefix):
        return [{f"{prefix}_acc": r["accuracy"], f"{prefix}_precision": r["precision"],
                 f"{prefix}_recall": r["recall"], f"{prefix}_f1": r["f1"],
                 f"y_true_{prefix}": r["y_true"], f"y_pred_probs_{prefix}": r["y_pred_probs"]}
                for r in results]

    ar_stds = compute_per_participant_stds(_prefix(results_ar, "ar"), "ar")
    va_stds = compute_per_participant_stds(_prefix(results_va, "va"), "va")
    print_determinism_summary(
        {f"ar_{k}": final_results[f"ar_{k}"] for k in ["auc", "acc", "precision", "recall", "f1"]},
        {f"va_{k}": final_results[f"va_{k}"] for k in ["auc", "acc", "precision", "recall", "f1"]},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
