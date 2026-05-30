"""
TAG affinity analysis - post-hoc consolidation of per-participant affinity
scores against MTL vs STL gains.
"""
import argparse
import os, sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, zscore, ttest_ind
from config import get_dataset_config, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description='TAG affinity analysis')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'])
    p.add_argument('--z_threshold', type=float, default=2.5,
                   help='Z-score threshold for outlier removal (default 2.5)')
    return p.parse_args()


def _plot_heatmaps(mat_ar, mat_va, p_ids, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mat, title in [(axes[0], mat_ar, 'AR Inter-Task Affinity'),
                           (axes[1], mat_va, 'VA Inter-Task Affinity')]:
        mask = np.eye(mat.shape[0], dtype=bool)
        off_diag = mat[~mask & np.isfinite(mat)]
        v = np.percentile(np.abs(off_diag), 99)
        sns.heatmap(mat, annot=False, cmap='RdYlGn', center=0,
                    vmin=-v, vmax=v, mask=mask,
                    xticklabels=p_ids, yticklabels=p_ids, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Participant ID (Target)')
        ax.set_ylabel('Participant ID (Source)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"[OK] {out_path}")


def _plot_distribution(scores_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, color, title in [
        (axes[0], 'ar_affinity_score', 'steelblue', 'AR Affinity Score'),
        (axes[1], 'va_affinity_score', 'seagreen',  'VA Affinity Score')]:
        ax.hist(scores_df[col], bins=15, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(scores_df[col].mean(), color='red', linestyle='--',
                   linewidth=2, label='Mean')
        ax.set_xlabel('Average Affinity Score')
        ax.set_ylabel('Number of Participants')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"[OK] {out_path}")


def _print_summary(scores_df):
    for col, label in [('ar_affinity_score', 'AR'), ('va_affinity_score', 'VA')]:
        s = scores_df[col]
        print(f"\n{label} affinity scores:")
        print(f"  mean={s.mean():.4f}  median={s.median():.4f}  std={s.std():.4f}")
        print(f"  min={s.min():.4f}  max={s.max():.4f}")


def _scatter_with_fit(ax, df, x_col, y_col, color, title, outliers=None):
    ax.scatter(df[x_col], df[y_col], alpha=0.6, s=100, color=color, label='All')
    if outliers is not None and outliers.sum() > 0:
        ax.scatter(df[outliers][x_col], df[outliers][y_col],
                   alpha=0.8, s=150, color='red', marker='x',
                   linewidths=3, label='Outliers')
    if len(df) > 1:
        z = np.polyfit(df[x_col], df[y_col], 1)
        x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), 'r--', alpha=0.6, linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if outliers is not None: ax.legend()


def _correlation_block(merged, z_thr, out_path):
    def _z_outliers(df, score_col, gain_col):
        return (np.abs(zscore(df[score_col])) > z_thr) | \
               (np.abs(zscore(df[gain_col]))  > z_thr)
    ar_out = _z_outliers(merged, 'ar_affinity_score', 'AR_gain_%')
    va_out = _z_outliers(merged, 'va_affinity_score', 'VA_gain_%')
    ar_clean = merged[~ar_out]
    va_clean = merged[~va_out]
    def _corr(df, x, y):
        r,  pp = pearsonr(df[x], df[y])
        rh, ps = spearmanr(df[x], df[y])
        return r, pp, rh, ps
    ar_r,  ar_p,  ar_rho,  ar_ps  = _corr(merged,   'ar_affinity_score', 'AR_gain_%')
    va_r,  va_p,  va_rho,  va_ps  = _corr(merged,   'va_affinity_score', 'VA_gain_%')
    ar_rc, ar_pc, ar_rhoc, ar_psc = _corr(ar_clean, 'ar_affinity_score', 'AR_gain_%')
    va_rc, va_pc, va_rhoc, va_psc = _corr(va_clean, 'va_affinity_score', 'VA_gain_%')
    print("\nAffinity vs MTL gain (Pearson r, p):")
    print(f"  AR full      n={len(merged):2d}  r={ar_r:+.3f}  p={ar_p:.4f}")
    print(f"  AR z<{z_thr:.1f}  n={len(ar_clean):2d}  r={ar_rc:+.3f}  p={ar_pc:.4f}")
    print(f"  VA full      n={len(merged):2d}  r={va_r:+.3f}  p={va_p:.4f}")
    print(f"  VA z<{z_thr:.1f}  n={len(va_clean):2d}  r={va_rc:+.3f}  p={va_pc:.4f}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    _scatter_with_fit(axes[0,0], merged,   'ar_affinity_score', 'AR_gain_%',
                      'steelblue', f'AR full (r={ar_r:+.3f}, p={ar_p:.4f})', outliers=ar_out)
    _scatter_with_fit(axes[0,1], ar_clean, 'ar_affinity_score', 'AR_gain_%',
                      'steelblue', f'AR z<{z_thr} (r={ar_rc:+.3f}, p={ar_pc:.4f})')
    _scatter_with_fit(axes[1,0], merged,   'va_affinity_score', 'VA_gain_%',
                      'seagreen',  f'VA full (r={va_r:+.3f}, p={va_p:.4f})', outliers=va_out)
    _scatter_with_fit(axes[1,1], va_clean, 'va_affinity_score', 'VA_gain_%',
                      'seagreen',  f'VA z<{z_thr} (r={va_rc:+.3f}, p={va_pc:.4f})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"[OK] {out_path}")
    return pd.DataFrame([
        {'group':'ar_full',  'n':len(merged),   'r':ar_r,  'p':ar_p,  'rho':ar_rho,  'p_spear':ar_ps},
        {'group':'ar_clean', 'n':len(ar_clean), 'r':ar_rc, 'p':ar_pc, 'rho':ar_rhoc, 'p_spear':ar_psc},
        {'group':'va_full',  'n':len(merged),   'r':va_r,  'p':va_p,  'rho':va_rho,  'p_spear':va_ps},
        {'group':'va_clean', 'n':len(va_clean), 'r':va_rc, 'p':va_pc, 'rho':va_rhoc, 'p_spear':va_psc},
    ])


def _rescue_effect(merged, out_path):
    print("\nRescue-effect (STL baseline split at 0.5):")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for ax, label, stl_col, gain_col, color in [
        (axes[0], 'AR', 'AR_acc_STL', 'AR_gain_%', 'steelblue'),
        (axes[1], 'VA', 'VA_acc_STL', 'VA_gain_%', 'seagreen'),
    ]:
        low  = merged[merged[stl_col] <  0.5][gain_col].values
        high = merged[merged[stl_col] >= 0.5][gain_col].values
        if len(low) > 0 and len(high) > 0:
            t, p = ttest_ind(low, high)
            print(f"  {label}: low n={len(low)} mean={low.mean():+.2f}%, "
                  f"high n={len(high)} mean={high.mean():+.2f}%, "
                  f"t={t:+.3f}  p={p:.4f}")
            rows.append({'task':label, 'low_n':len(low), 'low_mean':low.mean(),
                         'high_n':len(high), 'high_mean':high.mean(),
                         't':t, 'p':p})
        else:
            print(f"  {label}: insufficient data (low n={len(low)}, high n={len(high)})")
            rows.append({'task':label, 'low_n':len(low), 'high_n':len(high)})
        bp = ax.boxplot([low, high], positions=[1, 2], widths=0.6, patch_artist=True,
                        labels=[f'Low\nn={len(low)}', f'High\nn={len(high)}'])
        for box in bp['boxes']:
            box.set_facecolor(color); box.set_alpha(0.6)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_ylabel(f'{label} gain (%)')
        ax.set_title(f'{label}: gain by STL baseline group', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"[OK] {out_path}")
    return pd.DataFrame(rows)


if __name__ == '__main__':
    args = parse_args()
    cfg    = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']

    tag_dir = os.path.join(RESULTS_DIR, f'{prefix}_TAG', f'{prefix}_tag_results')
    paths = {
        'ar': os.path.join(tag_dir, f'ar_final_affinity_matrix_{prefix}.npy'),
        'va': os.path.join(tag_dir, f'va_final_affinity_matrix_{prefix}.npy'),
        'scores': os.path.join(tag_dir, f'affinity_scores_per_participant_{prefix}.csv'),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"[X] Missing: {path}")
            print( "  Run experiments/TAG_analysis/tag_train.py for this dataset first.")
            sys.exit(1)

    out_dir = os.path.join(tag_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    mat_ar    = np.load(paths['ar'])
    mat_va    = np.load(paths['va'])
    scores_df = pd.read_csv(paths['scores'])
    p_ids     = scores_df['participant_id'].tolist()

    print(f"\nDataset: {prefix}")
    print(f"AR matrix: {mat_ar.shape}, VA matrix: {mat_va.shape}, participants: {len(p_ids)}")

    _plot_heatmaps(mat_ar, mat_va, p_ids,
                   os.path.join(out_dir, 'affinity_heatmaps.png'))
    _plot_distribution(scores_df,
                       os.path.join(out_dir, 'affinity_distributions.png'))
    _print_summary(scores_df)

    gains_csv = os.path.join(RESULTS_DIR, f'{prefix}_MTL_vs_STL_Gains.csv')
    if not os.path.exists(gains_csv):
        print(f"\n(Skipping gain correlation - {gains_csv} not found.")
        print( " Run analysis/mtl_vs_stl_gains.py to enable this section.)")
        sys.exit(0)

    merged = pd.merge(pd.read_csv(gains_csv), scores_df, on='participant_id')
    print(f"\nMerged with gains: {len(merged)} participants")

    corr_df = _correlation_block(
        merged, args.z_threshold,
        os.path.join(out_dir, 'affinity_vs_gain_correlation.png'))
    corr_df.to_csv(os.path.join(out_dir, 'correlation_summary.csv'), index=False)
    print(f"[OK] {os.path.join(out_dir, 'correlation_summary.csv')}")

    rescue_df = _rescue_effect(
        merged, os.path.join(out_dir, 'rescue_effect_by_stl_baseline.png'))
    rescue_df.to_csv(os.path.join(out_dir, 'rescue_effect_summary.csv'), index=False)
    print(f"[OK] {os.path.join(out_dir, 'rescue_effect_summary.csv')}")

    print("\nDone.")
