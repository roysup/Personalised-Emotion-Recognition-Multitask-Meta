"""
P-STL embedding visualization (t-SNE / PCA / UMAP) — motivation figure.

Replicates the *process* of the Colab similarity-analysis notebook, but built on
the repository's own components instead of ad-hoc code:

    * P-STL model      -> src/models.SingleTaskModel (the trained population model),
                          loaded from results/<PREFIX>_MTL/<PREFIX>_pstl_results/
                          model_{ar,va}.pth  (falls back to random init if absent).
    * data / windowing -> dataset_configs.loader.load_dataset + data.create_sliding_windows
    * splits           -> src/config participant + per-user trial splits.

For each emotion label (AR / VA) it runs every windowed segment through the
population model and taps two 64-d representations:
    lstm_mean : pooled LSTM output (the backbone representation)
    z2        : activation after the second dense layer (one layer from the output)

These are aggregated to window / trial / participant level and projected to 2-D
with PCA, t-SNE and (optionally) UMAP. Each projection is drawn TWICE for the same
points — colored by participant and colored by emotion label — so the panels answer
"does the population feature space cluster by user, or by emotion?".

By default it embeds the TRAIN trials of ALL users (the exact data P-STL was fit
on), which is the most conservative basis for the personalization argument.

NOTE: t-SNE/UMAP distances and cluster sizes are NOT metric-meaningful; use only
for the qualitative "clusters by user" reading. Report perplexity/seed. This script
does NO similarity-metric computation — visualization only.

Examples
--------
    python analysis/tsne_embeddings.py --dataset vreed
    python analysis/tsne_embeddings.py --dataset vreed --label ar --levels window trial
    python analysis/tsne_embeddings.py --dataset dssn_em --features lstm_mean
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import create_sliding_windows
from dataset_configs.loader import load_dataset
from models import SingleTaskModel

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

_PREFIX = {'vreed': 'VREED', 'dssn_eq': 'DSSN_EQ', 'dssn_em': 'DSSN_EM'}


# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='P-STL t-SNE/PCA/UMAP embedding figures')
    p.add_argument('--dataset', default='vreed', choices=['vreed', 'dssn_eq', 'dssn_em'])
    p.add_argument('--label', default='both', choices=['ar', 'va', 'both'])
    p.add_argument('--features', nargs='+', default=['lstm_mean'],
                   choices=['lstm_mean', 'z2'],
                   help='default: lstm_mean (backbone tap). z2 available but not '
                        'recommended (closer to the classifier head)')
    p.add_argument('--levels', nargs='+', default=['window'],
                   choices=['window', 'trial', 'participant'],
                   help='default: window')
    p.add_argument('--methods', nargs='+', default=['tsne'],
                   choices=['pca', 'tsne', 'umap'],
                   help='default: tsne. Add pca for a linear sanity-check panel')
    p.add_argument('--combined', action='store_true',
                   help='single figure with AR and VA side by side, both colored '
                        'by participant (shared color map). Ignores by-label panels.')
    p.add_argument('--participants', default='all', choices=['all', 'train', 'test'],
                   help='which participants to include')
    p.add_argument('--trials', default='train', choices=['all', 'train', 'test'],
                   help='within each participant, which trials to window '
                        '(default: train = the data P-STL was fit on)')
    p.add_argument('--ckpt-path', default='auto',
                   help='P-STL checkpoint path, "auto" to locate model_{label}.pth, '
                        'or "init" for random init')
    p.add_argument('--max-per-user', type=int, default=0,
                   help='cap windows/user (0 = use all)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', default=None)
    return p.parse_args()


# ------------------------------------------------------------------
# data -> windows  (train trials of the selected users)
# ------------------------------------------------------------------
def build_windows(df, cfg, which, trials, max_per_user, seed):
    rng = np.random.default_rng(seed)
    if which == 'train':
        keep = set(cfg['train_participants'])
    elif which == 'test':
        keep = set(cfg['test_participants'])
    else:
        keep = set(cfg['participant_ids'])

    Xs, yar, yva, pids, tids = [], [], [], [], []
    for uid in cfg['participant_ids']:
        if uid not in keep:
            continue
        sub = df[df['ID'] == uid].reset_index(drop=True)
        if trials in ('train', 'test') and uid in cfg['splits']:
            sub = sub[sub['Trial'].isin(cfg['splits'][uid][trials])].reset_index(drop=True)
        if len(sub) == 0:
            continue
        X, y_ar, y_va, task_ids, trial_ids = create_sliding_windows(
            sub, cfg['window_size'], cfg['stride'],
            task_id=uid, feature_cols=cfg['feature_cols'])
        if len(X) == 0:
            continue
        if max_per_user and len(X) > max_per_user:
            idx = rng.choice(len(X), max_per_user, replace=False)
            X, y_ar, y_va, trial_ids = X[idx], y_ar[idx], y_va[idx], trial_ids[idx]
        Xs.append(X); yar.append(y_ar); yva.append(y_va)
        pids.append(np.full(len(X), uid, dtype=np.int64))
        tids.append(trial_ids.astype(np.int64))

    if not Xs:
        raise RuntimeError('No windows built — check dataset/splits.')
    return (np.concatenate(Xs), np.concatenate(yar).astype(int),
            np.concatenate(yva).astype(int), np.concatenate(pids),
            np.concatenate(tids))


# ------------------------------------------------------------------
# P-STL feature taps  (lstm_mean and z2)
# ------------------------------------------------------------------
def load_pstl(cfg, ckpt_path, dataset, label, device):
    net = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
    if ckpt_path == 'auto':
        prefix = _PREFIX[dataset]
        ckpt_path = os.path.join(_REPO_ROOT, 'results', f'{prefix}_MTL',
                                 f'{prefix}_pstl_results', f'model_{label}.pth')
    if ckpt_path != 'init' and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        net.load_state_dict(state, strict=False)
        print(f'[pstl:{label}] loaded checkpoint: {ckpt_path}')
    else:
        print(f'[pstl:{label}] WARNING: no checkpoint — using RANDOM INIT '
              f'({ckpt_path}). Plot will not reflect a trained model.')
    net.eval()
    return net


def extract_feats(net, X, device, bs=256):
    """Return dict with 'lstm_mean' (N,64) and 'z2' (N,64)."""
    xt = torch.from_numpy(X.astype(np.float32))
    lstm_mean, z2 = [], []
    with torch.no_grad():
        for i in range(0, len(xt), bs):
            x = xt[i:i + bs].to(device)
            x = x.permute(0, 2, 1)
            x = F.relu(net.bn1(net.conv1(x))); x = net.pool1(x)
            x = F.relu(net.bn2(net.conv2(x))); x = net.pool2(x)
            x = x.permute(0, 2, 1)
            x, _ = net.lstm(x)
            h = torch.mean(x, dim=1)                 # lstm_mean (64)
            z = F.relu(net.dense2(F.relu(net.dense1(h))))  # z2 (64)
            lstm_mean.append(h.cpu().numpy())
            z2.append(z.cpu().numpy())
    return {'lstm_mean': np.concatenate(lstm_mean),
            'z2': np.concatenate(z2)}


# ------------------------------------------------------------------
# aggregation: window / trial / participant
# ------------------------------------------------------------------
def aggregate(feat, pid, trial, label, level):
    """Return (F matrix, pid array, label array) at the requested level."""
    if level == 'window':
        return feat, pid, label
    cols = [f'f{i}' for i in range(feat.shape[1])]
    d = pd.DataFrame(feat, columns=cols)
    d['pid'], d['trial'], d['label'] = pid, trial, label.astype(float)
    keys = ['pid', 'trial'] if level == 'trial' else ['pid']
    g = d.groupby(keys, as_index=False).agg({**{c: 'mean' for c in cols},
                                             'label': 'mean'})
    return (g[cols].to_numpy(np.float32),
            g['pid'].to_numpy(np.int64),
            (g['label'].to_numpy() >= 0.5).astype(int))


# ------------------------------------------------------------------
# projections + plotting
# ------------------------------------------------------------------
def safe_perplexity(n):
    if n <= 5:
        return max(2, n - 1)
    return int(max(5, min(30, n // 3)))


def project_all(F_hi, methods, seed):
    Xz = StandardScaler().fit_transform(F_hi)
    out = {}
    if 'pca' in methods and Xz.shape[1] >= 2:
        out['PCA'] = PCA(n_components=2, random_state=seed).fit_transform(Xz)
    if 'tsne' in methods:
        out['t-SNE'] = TSNE(n_components=2, perplexity=safe_perplexity(len(Xz)),
                            init='pca', learning_rate='auto',
                            random_state=seed).fit_transform(Xz)
    if 'umap' in methods and HAS_UMAP:
        out['UMAP'] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                random_state=seed).fit_transform(Xz)
    return out


def plot_panels(embs, color_vals, title, out_png, discrete_labels=None,
                pid_labels=None):
    methods = list(embs.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.8),
                             squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, methods):
        e = embs[m]
        if discrete_labels is None:  # continuous-ish (participant ids) -> tab20
            cmap = plt.get_cmap('tab20', max(len(np.unique(color_vals)), 3))
            for k, v in enumerate(np.unique(color_vals)):
                mk = color_vals == v
                ax.scatter(e[mk, 0], e[mk, 1], s=16, color=cmap(k % cmap.N),
                           alpha=0.75, linewidths=0)
        else:  # binary label legend
            palette = {0: '#3b6fb0', 1: '#c0392b'}
            for c, nm in discrete_labels.items():
                mk = color_vals == c
                ax.scatter(e[mk, 0], e[mk, 1], s=16, color=palette[c],
                           alpha=0.65, linewidths=0, label=nm)
            ax.legend(loc='best', fontsize=8, frameon=True)
        if pid_labels is not None:
            for x, y, pid in zip(e[:, 0], e[:, 1], pid_labels):
                ax.text(x, y, str(pid), fontsize=6, ha='center', va='center')
        ax.set_title(m); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_png}')


def plot_combined(panels, suptitle, out_png):
    """panels: list of (panel_title, emb, ids). One column per panel, colored by
    participant with a color map shared across panels."""
    all_ids = np.unique(np.concatenate([ids for _, _, ids in panels]))
    cmap = plt.get_cmap('tab20', max(len(all_ids), 3))
    id2c = {v: cmap(k % cmap.N) for k, v in enumerate(all_ids)}

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.6),
                             squeeze=False)
    axes = axes[0]
    for ax, (ptitle, emb, ids) in zip(axes, panels):
        for v in np.unique(ids):
            m = ids == v
            ax.scatter(emb[m, 0], emb[m, 1], s=14, color=id2c[v],
                       alpha=0.75, linewidths=0)
        ax.set_title(ptitle); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_png}')


# ------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    labels = ['ar', 'va'] if args.label == 'both' else [args.label]

    outdir = args.outdir or os.path.join(_REPO_ROOT, 'results', 'embedding_plots')
    os.makedirs(outdir, exist_ok=True)

    df, cfg = load_dataset(args.dataset, mode='mtml')
    X, y_ar, y_va, pid, trial = build_windows(
        df, cfg, args.participants, args.trials, args.max_per_user, args.seed)
    print(f'[data] {len(X)} windows | {len(np.unique(pid))} participants | '
          f'participants={args.participants} trials={args.trials} | shape={X.shape}')
    if not HAS_UMAP and 'umap' in args.methods:
        print('[info] umap-learn not installed — UMAP panels skipped.')

    # feature dict per label (each label uses its own P-STL model)
    feats_by_label = {}
    for label in labels:
        net = load_pstl(cfg, args.ckpt_path, args.dataset, label, device)
        feats_by_label[label] = extract_feats(net, X, device)

    method = 't-SNE' if 'tsne' in args.methods else args.methods[0].upper()

    if args.combined:
        # one figure: AR and VA side by side, both colored by participant
        for feat_name in args.features:
            for level in args.levels:
                panels = []
                for label in labels:
                    y = y_ar if label == 'ar' else y_va
                    F_hi, ids, _ = aggregate(feats_by_label[label][feat_name],
                                             pid, trial, y, level)
                    if len(F_hi) < 3:
                        print(f'[skip] {label}/{feat_name}/{level}: too few points')
                        continue
                    emb = project_all(F_hi, args.methods, args.seed)[method]
                    panels.append((f'{label.upper()} (n={len(F_hi)})', emb, ids))
                if not panels:
                    continue
                out = os.path.join(
                    outdir, f'{args.dataset}_{level}_{feat_name}_'
                            f'{"-".join(l.upper() for l in labels)}_by_participant.png')
                plot_combined(
                    panels,
                    f'{args.dataset.upper()} · {level} · {feat_name} · {method} '
                    f'— colored by participant',
                    out)
        print(f'\n[done] figures in {outdir}')
        return

    for label in labels:
        y = y_ar if label == 'ar' else y_va
        feats = feats_by_label[label]

        for feat_name in args.features:
            for level in args.levels:
                F_hi, ids, lab = aggregate(feats[feat_name], pid, trial, y, level)
                if len(F_hi) < 3:
                    print(f'[skip] {label}/{feat_name}/{level}: too few points ({len(F_hi)})')
                    continue
                embs = project_all(F_hi, args.methods, args.seed)
                base = f'{args.dataset}_{label}_{level}_{feat_name}'
                title = (f'{args.dataset.upper()} · {label.upper()} · {level} · '
                         f'{feat_name} (n={len(F_hi)})')
                # participant labels only useful at participant level
                plabels = ids if level == 'participant' else None
                plot_panels(embs, ids, title + ' — by participant',
                            os.path.join(outdir, base + '_by_participant.png'),
                            discrete_labels=None, pid_labels=plabels)
                plot_panels(embs, lab, title + f' — by {label.upper()} label',
                            os.path.join(outdir, base + '_by_label.png'),
                            discrete_labels={0: f'{label.upper()} low',
                                             1: f'{label.upper()} high'})

    print(f'\n[done] figures in {outdir}')


if __name__ == '__main__':
    main()
