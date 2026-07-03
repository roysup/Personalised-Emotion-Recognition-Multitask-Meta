"""
t-SNE / UMAP embedding visualization — motivation figure for personalization.

Projects each participant's windowed physiological segments to 2-D and produces
two side-by-side panels for the same points:

    (left)  colored by PARTICIPANT   — if points cluster by user, that visually
                                        motivates modeling each user as a task.
    (right) colored by EMOTION LABEL  — shows how weakly the raw/shared feature
                                        space separates the target label.

Choice of feature space (--features):
    pstl     — features from the trained POPULATION model (P-STL SingleTaskModel,
               64-d pooled representation). BEST fit for the "why personalize?"
               argument: if the non-personalized population model still organizes
               its feature space by participant rather than by emotion, that
               directly motivates STL/MTL personalization over P-STL.
    backbone — features from the shared MTML CNN-LSTM backbone (BaseFeatureExtractor).
               Supports the secondary point that residual per-user structure
               survives parameter sharing (justifying per-user heads / adaptation).
    raw      — flattened raw windows (model-agnostic). Shows the data itself is
               user-structured, independent of any trained model.

IMPORTANT (reviewer-safe usage): t-SNE/UMAP distances, cluster sizes and gaps are
NOT metric-meaningful. Use these plots only for the qualitative "clusters by user"
point, and always report perplexity/seed. Nothing quantitative should be read off
the axes.

Examples
--------
    python analysis/tsne_embeddings.py --dataset vreed --label ar               # P-STL features (default)
    python analysis/tsne_embeddings.py --dataset dssn_em --label va --method umap
    python analysis/tsne_embeddings.py --dataset vreed --label ar --features raw
    python analysis/tsne_embeddings.py --dataset vreed --label ar --features backbone
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data import create_sliding_windows
from dataset_configs.loader import load_dataset
from models import BaseFeatureExtractor, SingleTaskModel


# ------------------------------------------------------------------
# args
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='t-SNE / UMAP embedding figure')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    p.add_argument('--label', type=str, default='ar', choices=['ar', 'va'],
                   help='which emotion label to color the right panel by')
    p.add_argument('--features', type=str, default='pstl',
                   choices=['pstl', 'backbone', 'raw'],
                   help='"pstl" = population model features (default; best for the '
                        'personalization argument), "backbone" = shared MTML '
                        'CNN-LSTM features, "raw" = flattened windows')
    p.add_argument('--ckpt-path', type=str, default='auto',
                   help='path to a model checkpoint, or "auto" to locate the '
                        'P-STL / MI checkpoint for the chosen --features, or '
                        '"init" for random init')
    p.add_argument('--method', type=str, default='tsne',
                   choices=['tsne', 'umap'])
    p.add_argument('--perplexity', type=float, default=30.0)
    p.add_argument('--max-per-user', type=int, default=60,
                   help='cap windows per participant (balance + tractability)')
    p.add_argument('--participants', type=str, default='all',
                   choices=['all', 'train', 'test'],
                   help='which PARTICIPANTS to include (train = users P-STL/MTML '
                        'were fit on; test = held-out unseen users)')
    p.add_argument('--trials', type=str, default='all',
                   choices=['all', 'train', 'test'],
                   help='within each participant, which TRIALS to window '
                        '(train = support trials the models were fit on; '
                        'test = held-out query trials)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', type=str, default=None)
    return p.parse_args()


# ------------------------------------------------------------------
# data -> windows
# ------------------------------------------------------------------
def build_windows(df, cfg, which='all', trials='all', max_per_user=60, seed=42):
    """Return X (N, win, C), y_ar, y_va, user_ids, aligned per participant.

    `which`  selects participants (all/train/test).
    `trials` selects, within each participant, which trials to window
             (all / train-support / test-query), via cfg['splits'][uid].
    """
    rng = np.random.default_rng(seed)
    all_ids = cfg['participant_ids']
    if which == 'train':
        keep = set(cfg['train_participants'])
    elif which == 'test':
        keep = set(cfg['test_participants'])
    else:
        keep = set(all_ids)

    Xs, yar, yva, uids = [], [], [], []
    for uid in all_ids:
        if uid not in keep:
            continue
        sub = df[df['ID'] == uid].reset_index(drop=True)
        if trials in ('train', 'test') and uid in cfg['splits']:
            keep_trials = cfg['splits'][uid][trials]
            sub = sub[sub['Trial'].isin(keep_trials)].reset_index(drop=True)
        if len(sub) == 0:
            continue
        X, y_ar, y_va, _, _ = create_sliding_windows(
            sub, cfg['window_size'], cfg['stride'],
            task_id=uid, feature_cols=cfg['feature_cols'])
        if len(X) == 0:
            continue
        if len(X) > max_per_user:
            idx = rng.choice(len(X), max_per_user, replace=False)
            X, y_ar, y_va = X[idx], y_ar[idx], y_va[idx]
        Xs.append(X)
        yar.append(y_ar)
        yva.append(y_va)
        uids.append(np.full(len(X), uid, dtype=np.int64))

    if not Xs:
        raise RuntimeError('No windows built — check dataset/splits.')
    return (np.concatenate(Xs), np.concatenate(yar),
            np.concatenate(yva), np.concatenate(uids))


def _binarize(y):
    """Ratings are typically already {0,1}; otherwise median-split."""
    u = np.unique(y)
    if len(u) <= 2:
        return (y > (u.min() if len(u) == 1 else np.mean(u))).astype(int)
    return (y > np.median(y)).astype(int)


# ------------------------------------------------------------------
# features
# ------------------------------------------------------------------
_PREFIX = {'vreed': 'VREED', 'dssn_eq': 'DSSN_EQ', 'dssn_em': 'DSSN_EM'}


def resolve_ckpt_path(arg, features, dataset, label):
    if arg != 'auto':
        return arg
    prefix = _PREFIX[dataset]
    if features == 'pstl':
        cand = os.path.join(_REPO_ROOT, 'results', f'{prefix}_MTL',
                            f'{prefix}_pstl_results', f'model_{label}.pth')
    else:  # backbone (MTML MI)
        cand = os.path.join(_REPO_ROOT, 'results', f'{prefix}_MTML',
                            f'{prefix}_reptile_mi', f'reptile_mi_base_{label}.pth')
    return cand if os.path.exists(cand) else 'init'


def _pstl_features(net, x):
    """Pooled 64-d representation from SingleTaskModel (forward up to torch.mean)."""
    import torch.nn.functional as F
    x = x.permute(0, 2, 1)
    x = F.relu(net.bn1(net.conv1(x))); x = net.pool1(x)
    x = F.relu(net.bn2(net.conv2(x))); x = net.pool2(x)
    x = x.permute(0, 2, 1)
    x, _ = net.lstm(x)
    return torch.mean(x, dim=1)


def extract_features(X, cfg, features, ckpt_path, dataset, label, device):
    if features == 'raw':
        return X.reshape(len(X), -1).astype(np.float32)

    path = resolve_ckpt_path(ckpt_path, features, dataset, label)
    if features == 'pstl':
        net = SingleTaskModel(input_dim=cfg['input_dim']).to(device)
        fwd = _pstl_features
    else:
        net = BaseFeatureExtractor(input_dim=cfg['input_dim']).to(device)
        fwd = lambda n, x: n(x)

    if path != 'init' and os.path.exists(path):
        state = torch.load(path, map_location=device)
        state = state.get('state_dict', state) if isinstance(state, dict) else state
        net.load_state_dict(state, strict=False)
        print(f'[features] loaded {features} checkpoint: {path}')
    else:
        print(f'[features] WARNING: no {features} checkpoint found — using RANDOM '
              'INIT. Pass --features raw or a valid --ckpt-path for a meaningful plot.')
    net.eval()

    feats, bs = [], 256
    xt = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(xt), bs):
            feats.append(fwd(net, xt[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


# ------------------------------------------------------------------
# projection
# ------------------------------------------------------------------
def project(F, method, perplexity, seed):
    # standardize before projection
    mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True) + 1e-8
    F = (F - mu) / sd
    if method == 'umap':
        try:
            import umap
        except ImportError:
            raise SystemExit('umap-learn not installed. pip install umap-learn '
                             '--break-system-packages, or use --method tsne.')
        reducer = umap.UMAP(n_components=2, random_state=seed,
                            n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(F)
    from sklearn.manifold import TSNE
    perp = min(perplexity, max(5, (len(F) - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perp, init='pca',
                learning_rate='auto', random_state=seed)
    return tsne.fit_transform(F)


# ------------------------------------------------------------------
# plot
# ------------------------------------------------------------------
def plot(emb, user_ids, label_bin, args, cfg, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    users = sorted(np.unique(user_ids))
    cmap = plt.get_cmap('tab20', max(len(users), 3))
    u2c = {u: cmap(i % cmap.N) for i, u in enumerate(users)}

    # left: by participant
    for u in users:
        m = user_ids == u
        axes[0].scatter(emb[m, 0], emb[m, 1], s=14, color=u2c[u],
                        alpha=0.7, linewidths=0)
    axes[0].set_title(f'Colored by participant (n={len(users)})')

    # right: by label
    lname = args.label.upper()
    colors = {0: '#3b6fb0', 1: '#c0392b'}
    names = {0: f'{lname} low (0)', 1: f'{lname} high (1)'}
    for c in (0, 1):
        m = label_bin == c
        axes[1].scatter(emb[m, 0], emb[m, 1], s=14, color=colors[c],
                        alpha=0.6, linewidths=0, label=names[c])
    axes[1].set_title(f'Colored by {lname} label')
    axes[1].legend(loc='best', frameon=True, fontsize=9)

    method = args.method.upper()
    for ax in axes:
        ax.set_xlabel(f'{method}-1')
        ax.set_ylabel(f'{method}-2')
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f'{args.dataset.upper()} — {method} of {args.features} features '
        f'(perplexity={args.perplexity:g}, seed={args.seed})',
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'[saved] {out_png}')


# ------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df, cfg = load_dataset(args.dataset, mode='mtml')
    X, y_ar, y_va, user_ids = build_windows(
        df, cfg, which=args.participants, trials=args.trials,
        max_per_user=args.max_per_user, seed=args.seed)
    print(f'[data] {len(X)} windows from {len(np.unique(user_ids))} participants '
          f'(participants={args.participants}, trials={args.trials}), shape={X.shape}')

    y = y_ar if args.label == 'ar' else y_va
    label_bin = _binarize(y)

    F = extract_features(X, cfg, args.features, args.ckpt_path,
                         args.dataset, args.label, device)
    emb = project(F, args.method, args.perplexity, args.seed)

    outdir = args.outdir or os.path.join(_REPO_ROOT, 'results',
                                         'embedding_plots')
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(
        outdir, f'{args.dataset}_{args.method}_{args.features}_{args.label}'
                f'_p-{args.participants}_t-{args.trials}.png')
    plot(emb, user_ids, label_bin, args, cfg, out_png)

    # quick quantitative sanity check (silhouette by user vs by label)
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(user_ids)) > 1:
            s_user = silhouette_score(emb, user_ids)
            s_lab = silhouette_score(emb, label_bin)
            print(f'[silhouette] by user = {s_user:.3f} | by {args.label} label = {s_lab:.3f}')
            print('  (higher by-user than by-label supports the personalization motivation)')
    except Exception as e:
        print(f'[silhouette] skipped: {e}')


if __name__ == '__main__':
    main()
