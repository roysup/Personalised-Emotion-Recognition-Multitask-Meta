"""
Transfer Learning + Fine-Tuning (TL-FT)
Pre-train on 20 participants, fine-tune per test participant.
"""
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))
from config import *
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (set_all_seeds, compute_metrics_from_cm, safe_roc_auc,
                   aggregate_mtml_results, create_kfold_splits, make_kfolds,
                   compute_per_participant_stds, print_determinism_summary,
                   prefix_results)
from data import create_sliding_windows, arrays_to_loader
from models import SingleTaskModel
from dataset_configs.vreed import load_vreed_df
hardcoded_splits = HARDCODED_SPLITS
BASE_OUTPUT_DIR  = os.path.join(RESULTS_DIR, 'VREED_MTML')
output_dir       = os.path.join(BASE_OUTPUT_DIR, 'VREED_TF')
os.makedirs(output_dir, exist_ok=True)

BATCH_SIZE      = 32
EPOCHS_PRETRAIN = EPOCHS
EPOCHS_FINETUNE = 10
learning_rates_pre = [1e-3]
learning_rates_ft  = [1e-3]
l2_lambdas = [1e-5]

set_all_seeds(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\nOutput: {output_dir}")

# =============================
# DATA
# =============================
df = load_vreed_df()
df['participant_trial_encoded'] = df['ID_video'].astype(str)

existing_ids    = set(df['ID'].unique())
participant_ids = sorted([int(pid) for pid in hardcoded_splits.keys() if pid in existing_ids])

test_participants  = [105, 109, 112, 125, 131, 132]
train_participants = sorted([p for p in participant_ids if p not in test_participants])
print(f"Train: {train_participants}\nTest:  {test_participants}")


# =============================
# HELPERS
# =============================
def _get_windows(sub_df, label_type):
    """
    Extract sliding windows from a sub-DataFrame grouped by participant_trial_encoded.
    Replaces the local get_frames() helper; delegates to the shared create_sliding_windows.
    Returns (X, y) numpy arrays.
    """
    X, y_ar, y_va, _, _ = create_sliding_windows(
        sub_df, WINDOW_SIZE, STRIDE, trial_col='participant_trial_encoded')
    return X, (y_ar if label_type.upper() == 'AR' else y_va)


def pretrain(X, y, lr, l2_lambda, epochs):
    set_all_seeds(SEED)
    model   = SingleTaskModel().to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader  = arrays_to_loader(X, y, BATCH_SIZE, shuffle=True, seed=SEED)
    for ep in range(epochs):
        model.train(); run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss  = loss_fn(model(X_b), y_b)
            l2    = l2_lambda * sum(p.norm(2)**2 for p in model.parameters() if p.requires_grad)
            total = loss + l2
            if torch.isnan(total): return None
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += total.item()
        sched.step(run / len(loader))
    return model


def finetune(base_model, X, y, lr, l2_lambda, epochs, pid):
    model = SingleTaskModel().to(device)
    model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader  = arrays_to_loader(X, y, BATCH_SIZE, shuffle=True, seed=SEED + pid)
    for ep in range(epochs):
        model.train(); run = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss  = loss_fn(model(X_b), y_b)
            l2    = l2_lambda * sum(p.norm(2)**2 for p in model.parameters() if p.requires_grad)
            total = loss + l2
            if torch.isnan(total): return None
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            opt.step(); run += total.item()
        sched.step(run / len(loader))
    return model


def eval_model(model, X, y):
    model.eval()
    loader = arrays_to_loader(X, y, BATCH_SIZE, shuffle=False)
    probs, trues = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            probs.extend(torch.sigmoid(model(X_b.to(device))).cpu().numpy().flatten())
            trues.extend(y_b.numpy().flatten())
    y_true = np.array(trues).astype(int)
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc, prec, rec, f1 = compute_metrics_from_cm(cm)
    return {'y_true': y_true, 'y_pred': y_pred, 'y_pred_probs': y_prob,
            'cm': cm, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# =============================
# HYPERPARAMETER TUNING
# =============================
def hyperparameter_tuning(label_type='AR'):
    print(f"\n{'='*60}\nHYPERPARAMETER TUNING [{label_type}] TL-FT\n{'='*60}")
    results     = []
    train_folds = make_kfolds(train_participants, seed=SEED)  # replaces local create_participant_kfolds
    for lr_pre in learning_rates_pre:
        for lr_ft in learning_rates_ft:
            for l2 in l2_lambdas:
                fold_f1s = []
                for fold_i in range(N_FOLDS):
                    val_ps = train_folds[fold_i]
                    tr_ps  = [p for j, f in enumerate(train_folds) if j != fold_i for p in f]
                    tr_pte = [f"{p}_{v}" for p in tr_ps if p in hardcoded_splits
                              for v in hardcoded_splits[p]['train']]
                    tr_df  = df[df['participant_trial_encoded'].isin(tr_pte)].reset_index(drop=True)
                    Xpre, ypre = _get_windows(tr_df, label_type)
                    if len(Xpre) == 0: continue
                    base = pretrain(Xpre, ypre, lr_pre, l2, EPOCHS_PRETRAIN)
                    if base is None: continue
                    val_f1s = []
                    for pid in val_ps:
                        if pid not in hardcoded_splits: continue
                        tr_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['train']]
                        te_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['test']]
                        u_tr = df[df['participant_trial_encoded'].isin(tr_pte_u)].reset_index(drop=True)
                        u_te = df[df['participant_trial_encoded'].isin(te_pte_u)].reset_index(drop=True)
                        Xft, yft = _get_windows(u_tr, label_type)
                        Xte, yte = _get_windows(u_te, label_type)
                        if len(Xft) == 0 or len(Xte) == 0: continue
                        ft = finetune(base, Xft, yft, lr_ft, l2, EPOCHS_FINETUNE, pid)
                        if ft is None: continue
                        r = eval_model(ft, Xte, yte)
                        val_f1s.append(f1_score(r['y_true'], r['y_pred'],
                                                average='macro', zero_division=0))
                    if val_f1s:
                        fold_f1s.append(np.mean(val_f1s))
                        print(f"  fold {fold_i+1}: f1={fold_f1s[-1]:.4f}")
                if not fold_f1s: continue
                avg = np.mean(fold_f1s)
                results.append({'lr_pre': lr_pre, 'lr_ft': lr_ft, 'l2': l2,
                                 'avg_f1': avg, 'std_f1': np.std(fold_f1s)})
                print(f"  avg f1={avg:.4f}")
    if not results: return 1e-3, 1e-3, 0.0
    best = max(results, key=lambda x: x['avg_f1'])
    with open(os.path.join(output_dir, f'{label_type.lower()}_tuning_results_tlft.pkl'), 'wb') as f:
        pickle.dump({'all': results, 'best': best}, f)
    return best['lr_pre'], best['lr_ft'], best['l2']


# =============================
# MAIN
# =============================
if __name__ == '__main__':
    best_lr_pre_ar, best_lr_ft_ar, best_l2_ar = hyperparameter_tuning('AR')
    best_lr_pre_va, best_lr_ft_va, best_l2_va = hyperparameter_tuning('VA')

    tr_pte      = [f"{p}_{v}" for p in train_participants if p in hardcoded_splits
                   for v in hardcoded_splits[p]['train']]
    pretrain_df = df[df['participant_trial_encoded'].isin(tr_pte)].reset_index(drop=True)

    print('\n' + '='*60 + '\nPRETRAINING AR\n' + '='*60)
    Xpre_ar, ypre_ar = _get_windows(pretrain_df, 'AR')
    set_all_seeds(SEED)
    base_ar = pretrain(Xpre_ar, ypre_ar, best_lr_pre_ar, best_l2_ar, EPOCHS_PRETRAIN)
    torch.save(base_ar.state_dict(), os.path.join(output_dir, 'base_model_ar_final.pth'))

    print('\n' + '='*60 + '\nPRETRAINING VA\n' + '='*60)
    Xpre_va, ypre_va = _get_windows(pretrain_df, 'VA')
    set_all_seeds(SEED)
    base_va = pretrain(Xpre_va, ypre_va, best_lr_pre_va, best_l2_va, EPOCHS_PRETRAIN)
    torch.save(base_va.state_dict(), os.path.join(output_dir, 'base_model_va_final.pth'))

    results_ar, results_va = [], []
    for pid in sorted(test_participants):
        if pid not in hardcoded_splits: continue
        tr_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['train']]
        te_pte_u = [f"{pid}_{v}" for v in hardcoded_splits[pid]['test']]
        u_tr = df[df['participant_trial_encoded'].isin(tr_pte_u)].reset_index(drop=True)
        u_te = df[df['participant_trial_encoded'].isin(te_pte_u)].reset_index(drop=True)

        Xft_ar, yft_ar = _get_windows(u_tr, 'AR')
        Xte_ar, yte_ar = _get_windows(u_te, 'AR')
        Xft_va, yft_va = _get_windows(u_tr, 'VA')
        Xte_va, yte_va = _get_windows(u_te, 'VA')

        print(f"\nParticipant {pid}: fine-tuning")
        ft_ar = finetune(base_ar, Xft_ar, yft_ar, best_lr_ft_ar, best_l2_ar, EPOCHS_FINETUNE, pid)
        ft_va = finetune(base_va, Xft_va, yft_va, best_lr_ft_va, best_l2_va, EPOCHS_FINETUNE, pid)

        r_ar = eval_model(ft_ar, Xte_ar, yte_ar); r_ar['participant_id'] = pid
        r_va = eval_model(ft_va, Xte_va, yte_va); r_va['participant_id'] = pid
        results_ar.append(r_ar); results_va.append(r_va)
        print(f"  AR Acc={r_ar['accuracy']:.4f} F1={r_ar['f1']:.4f} | "
              f"VA Acc={r_va['accuracy']:.4f} F1={r_va['f1']:.4f}")

    agg = aggregate_mtml_results(results_ar, results_va)

    # CM plots
    for cm, label, fname in [(agg['cm_ar'], 'AR', 'ar_cm_tlft.png'),
                              (agg['cm_va'], 'VA', 'va_cm_tlft.png')]:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'{label}=0', f'{label}=1'],
                    yticklabels=[f'{label}=0', f'{label}=1'])
        plt.title(f'CM {label} (TL-FT)'); plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, fname)); plt.close()

    roc_data = {
        'AR': {'true': agg['all_true_ar'], 'probs': agg['all_probs_ar']},
        'VA': {'true': agg['all_true_va'], 'probs': agg['all_probs_va']},
    }
    with open(os.path.join(output_dir, 'global_roc_data.pkl'), 'wb') as f:
        pickle.dump(roc_data, f)

    ar_stds = compute_per_participant_stds(prefix_results(results_ar, 'ar'), 'ar')
    va_stds = compute_per_participant_stds(prefix_results(results_va, 'va'), 'va')

    final_results = {
        'train_participants': train_participants,
        'test_participants':  test_participants,
        'best_hyperparameters': {
            'AR': {'lr_pre': best_lr_pre_ar, 'lr_ft': best_lr_ft_ar, 'l2': best_l2_ar},
            'VA': {'lr_pre': best_lr_pre_va, 'lr_ft': best_lr_ft_va, 'l2': best_l2_va}},
        **{f'ar_{k}': agg[f'ar_{k}'] for k in ['acc','precision','recall','f1','auc']},
        **{f'va_{k}': agg[f'va_{k}'] for k in ['acc','precision','recall','f1','auc']},
        'test_results_per_participant_ar': results_ar,
        'test_results_per_participant_va': results_va,
        'cm_ar': agg['cm_ar'], 'cm_va': agg['cm_va'],
    }
    with open(os.path.join(output_dir, 'tlft_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)

    print_determinism_summary(
        {f'ar_{k}': final_results[f'ar_{k}'] for k in ['auc','acc','precision','recall','f1']},
        {f'va_{k}': final_results[f'va_{k}'] for k in ['auc','acc','precision','recall','f1']},
        ar_stds, va_stds)

    print(f"\n✓ All results saved to: {output_dir}")
