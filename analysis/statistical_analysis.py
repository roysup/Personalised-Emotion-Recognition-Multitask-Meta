"""
Comprehensive Statistical Analysis
Paired t-tests, Levene variance tests, rescue-effect correlations,
low/high baseline group comparisons, McNemar (if available),
and a 6-panel publication figure.

Usage
-----
    python statistical_analysis.py                  # runs on VREED (default)
    python statistical_analysis.py --dataset dssn_eq
    python statistical_analysis.py --dataset dssn_em
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'datasets'))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_rel, ttest_ind, levene
from config import get_dataset_config, RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description='Comprehensive statistical analysis')
    p.add_argument('--dataset', type=str, default='vreed',
                   choices=['vreed', 'dssn_eq', 'dssn_em'],
                   help='Dataset to analyse (default: vreed)')
    return p.parse_args()


def load_pstl_stats(prefix):
    """
    Load PSTL per-participant results from the pickle and compute
    population-level mean and std for AR and VA accuracy.

    Returns
    -------
    pstl_ar_mean, pstl_ar_std, pstl_va_mean, pstl_va_std
    """
    pstl_pkl = os.path.join(RESULTS_DIR, f'{prefix}_MTL',
                            f'{prefix}_pstl_results', 'pstl_results.pkl')
    if not os.path.exists(pstl_pkl):
        print(f'⚠ PSTL results not found: {pstl_pkl}')
        print('  PSTL panels in the figure will use NaN.')
        return np.nan, np.nan, np.nan, np.nan

    with open(pstl_pkl, 'rb') as f:
        pstl = pickle.load(f)

    per_p = pstl.get('per_participant', [])
    if not per_p:
        return np.nan, np.nan, np.nan, np.nan

    ar_accs = [r['ar_acc'] for r in per_p]
    va_accs = [r['va_acc'] for r in per_p]
    return (np.mean(ar_accs), np.std(ar_accs, ddof=1),
            np.mean(va_accs), np.std(va_accs, ddof=1))


if __name__ == '__main__':
    args = parse_args()
    cfg = get_dataset_config(args.dataset)
    prefix = cfg['results_prefix']

    # ---- Load gains file (produced by mtl_vs_stl_gains.py) ----
    gains_file = os.path.join(RESULTS_DIR, f'{prefix}_MTL_vs_STL_Gains.csv')
    if not os.path.exists(gains_file):
        print(f'✗ Gains file not found: {gains_file}')
        print('  Run mtl_vs_stl_gains.py first.')
        raise SystemExit(1)

    gains_df = pd.read_csv(gains_file)
    n = len(gains_df)

    # ---- Load PSTL population-level stats from results pickle ----
    PSTL_AR_MEAN, PSTL_AR_STD, PSTL_VA_MEAN, PSTL_VA_STD = load_pstl_stats(prefix)

    # =====================================================================
    # PART 1: DESCRIPTIVE STATISTICS
    # =====================================================================
    print('\n' + '=' * 80)
    print(f'PART 1: DESCRIPTIVE STATISTICS  ({prefix})')
    print('=' * 80)
    print(f'\nSample size: n = {n} participants')

    for label, pstl_m, pstl_s, acc_stl, acc_mtl, gain in [
        ('AROUSAL (AR)', PSTL_AR_MEAN, PSTL_AR_STD,
         'AR_acc_STL', 'AR_acc_MTL', 'AR_gain_%'),
        ('VALENCE (VA)', PSTL_VA_MEAN, PSTL_VA_STD,
         'VA_acc_STL', 'VA_acc_MTL', 'VA_gain_%'),
    ]:
        print(f'\n{"-" * 60}\n{label}\n{"-" * 60}')
        print(f'PSTL: {pstl_m:.4f} ± {pstl_s:.4f}')
        print(f'STL:  {gains_df[acc_stl].mean():.4f} ± {gains_df[acc_stl].std():.4f}')
        print(f'MTL:  {gains_df[acc_mtl].mean():.4f} ± {gains_df[acc_mtl].std():.4f}')
        print(f'STL→MTL gain: {gains_df[gain].mean():.2f}% ± {gains_df[gain].std():.2f}%')

    # =====================================================================
    # PART 2: FOUR KEY STATISTICAL TESTS
    # =====================================================================
    print('\n' + '=' * 80)
    print('PART 2: STATISTICAL SIGNIFICANCE TESTS')
    print('=' * 80)

    results = []

    # Test 1: Paired t-test
    print(f'\n{"-" * 60}\nTEST 1: PAIRED T-TEST (STL vs MTL)\n{"-" * 60}')
    ar_t_mean, ar_p_mean = ttest_rel(gains_df['AR_acc_STL'], gains_df['AR_acc_MTL'])
    va_t_mean, va_p_mean = ttest_rel(gains_df['VA_acc_STL'], gains_df['VA_acc_MTL'])
    print(f'AR: t={ar_t_mean:.3f}, p={ar_p_mean:.4f} '
          f'{"✓ SIGNIFICANT" if ar_p_mean < 0.05 else "✗ Not significant"}')
    print(f'VA: t={va_t_mean:.3f}, p={va_p_mean:.4f} '
          f'{"✓ SIGNIFICANT" if va_p_mean < 0.05 else "✗ Not significant"}')
    results.append({
        'Test': 'Mean improvement (STL→MTL)',
        'AR_stat': f't={ar_t_mean:.3f}', 'AR_p': ar_p_mean,
        'AR_sig': '✓' if ar_p_mean < 0.05 else '✗',
        'VA_stat': f't={va_t_mean:.3f}', 'VA_p': va_p_mean,
        'VA_sig': '✓' if va_p_mean < 0.05 else '✗',
    })

    # Test 2: Levene's test
    print(f'\n{"-" * 60}\nTEST 2: LEVENE\'S TEST (Variance)\n{"-" * 60}')
    ar_var_stl = gains_df['AR_acc_STL'].var()
    ar_var_mtl = gains_df['AR_acc_MTL'].var()
    va_var_stl = gains_df['VA_acc_STL'].var()
    va_var_mtl = gains_df['VA_acc_MTL'].var()
    ar_lev, ar_p_var = levene(gains_df['AR_acc_STL'], gains_df['AR_acc_MTL'])
    va_lev, va_p_var = levene(gains_df['VA_acc_STL'], gains_df['VA_acc_MTL'])
    ar_var_change = (ar_var_mtl - ar_var_stl) / ar_var_stl * 100
    va_var_change = (va_var_mtl - va_var_stl) / va_var_stl * 100
    print(f'AR: {ar_var_change:+.1f}% variance change, W={ar_lev:.3f}, p={ar_p_var:.4f} '
          f'{"✓ SIGNIFICANT" if ar_p_var < 0.05 else "✗ Not significant"}')
    print(f'VA: {va_var_change:+.1f}% variance change, W={va_lev:.3f}, p={va_p_var:.4f} '
          f'{"✓ SIGNIFICANT" if va_p_var < 0.05 else "✗ Not significant"}')
    results.append({
        'Test': 'Variance reduction',
        'AR_stat': f'{ar_var_change:+.1f}%', 'AR_p': ar_p_var,
        'AR_sig': '✓' if ar_p_var < 0.05 else '✗',
        'VA_stat': f'{va_var_change:+.1f}%', 'VA_p': va_p_var,
        'VA_sig': '✓' if va_p_var < 0.05 else '✗',
    })

    # Test 3: Rescue effect correlation
    print(f'\n{"-" * 60}\nTEST 3: RESCUE EFFECT CORRELATION\n{"-" * 60}')
    ar_r, ar_p_corr = pearsonr(gains_df['AR_acc_STL'], gains_df['AR_gain_%'])
    va_r, va_p_corr = pearsonr(gains_df['VA_acc_STL'], gains_df['VA_gain_%'])
    ar_rho, ar_p_spear = spearmanr(gains_df['AR_acc_STL'], gains_df['AR_gain_%'])
    va_rho, va_p_spear = spearmanr(gains_df['VA_acc_STL'], gains_df['VA_gain_%'])
    print(f'AR: Pearson r={ar_r:.4f}, p={ar_p_corr:.4f}  |  '
          f'Spearman ρ={ar_rho:.4f}, p={ar_p_spear:.4f}')
    print(f'VA: Pearson r={va_r:.4f}, p={va_p_corr:.4f}  |  '
          f'Spearman ρ={va_rho:.4f}, p={va_p_spear:.4f}')
    if va_p_corr < 0.05:
        print('    ✓✓ SIGNIFICANT RESCUE EFFECT for VA ⭐')
    results.append({
        'Test': 'Baseline-gain correlation (rescue effect)',
        'AR_stat': f'r={ar_r:.3f}', 'AR_p': ar_p_corr,
        'AR_sig': '✓' if ar_p_corr < 0.05 else '✗',
        'VA_stat': f'r={va_r:.3f}', 'VA_p': va_p_corr,
        'VA_sig': '✓' if va_p_corr < 0.05 else '✗',
    })

    # Test 4: Low vs high baseline group comparison
    print(f'\n{"-" * 60}\nTEST 4: LOW vs HIGH BASELINE\n{"-" * 60}')
    va_low  = gains_df[gains_df['VA_acc_STL'] < 0.5]
    va_high = gains_df[gains_df['VA_acc_STL'] >= 0.5]
    ar_low  = gains_df[gains_df['AR_acc_STL'] < 0.5]
    ar_high = gains_df[gains_df['AR_acc_STL'] >= 0.5]

    ar_t_group = ar_p_group = np.nan
    va_t_group = va_p_group = np.nan

    if len(ar_low) > 0 and len(ar_high) > 0:
        ar_t_group, ar_p_group = ttest_ind(ar_low['AR_gain_%'], ar_high['AR_gain_%'])
        print(f'AR: Low (n={len(ar_low)}): {ar_low["AR_gain_%"].mean():.2f}% vs '
              f'High (n={len(ar_high)}): {ar_high["AR_gain_%"].mean():.2f}%')
        print(f'    t={ar_t_group:.3f}, p={ar_p_group:.4f} '
              f'{"✓ SIGNIFICANT" if ar_p_group < 0.05 else "✗ Not significant"}')

    if len(va_low) > 0 and len(va_high) > 0:
        va_t_group, va_p_group = ttest_ind(va_low['VA_gain_%'], va_high['VA_gain_%'])
        print(f'VA: Low (n={len(va_low)}): {va_low["VA_gain_%"].mean():.2f}% vs '
              f'High (n={len(va_high)}): {va_high["VA_gain_%"].mean():.2f}%')
        sig_label = ('✓ SIGNIFICANT' if va_p_group < 0.05
                     else '(Marginal)' if va_p_group < 0.10
                     else '✗ Not significant')
        print(f'    t={va_t_group:.3f}, p={va_p_group:.4f} {sig_label}')

    results.append({
        'Test': 'Low vs High baseline',
        'AR_stat': f't={ar_t_group:.3f}' if not np.isnan(ar_t_group) else 'N/A',
        'AR_p': ar_p_group, 'AR_sig': '✓' if not np.isnan(ar_p_group) and ar_p_group < 0.05 else '✗',
        'VA_stat': f't={va_t_group:.3f}' if not np.isnan(va_t_group) else 'N/A',
        'VA_p': va_p_group, 'VA_sig': '✓' if not np.isnan(va_p_group) and va_p_group < 0.05 else '✗',
    })

    # =====================================================================
    # PART 3: SUMMARY TABLE
    # =====================================================================
    print('\n' + '=' * 80)
    print('PART 3: SUMMARY TABLE')
    print('=' * 80)
    summary_df = pd.DataFrame(results)
    print('\n' + summary_df.to_string(index=False))
    output_csv = os.path.join(RESULTS_DIR, f'{prefix}_comprehensive_statistical_summary.csv')
    summary_df.to_csv(output_csv, index=False)
    print(f'\n✓ Saved: {output_csv}')

    # =====================================================================
    # PART 4: McNEMAR (if available)
    # =====================================================================
    print('\n' + '=' * 80)
    print('PART 4: PARTICIPANT-LEVEL McNEMAR TESTS')
    print('=' * 80)
    mcnemar_file = os.path.join(RESULTS_DIR,
        f'{prefix}_mcnemar_participant_level', 'mcnemar_participant_weighted.csv')
    mcnemar_df = None
    if os.path.exists(mcnemar_file):
        mcnemar_df = pd.read_csv(mcnemar_file)
        print('\n' + mcnemar_df.to_string(index=False))
    else:
        print(f'\n⚠ McNemar file not found: {mcnemar_file}')

    # =====================================================================
    # PART 5: KEY FINDINGS
    # =====================================================================
    print('\n' + '=' * 80)
    print('PART 5: KEY FINDINGS')
    print('=' * 80)
    print(f'\n🎯 AROUSAL (AR):')
    print(f'  Mean improvement: p={ar_p_mean:.4f} '
          f'{"✓" if ar_p_mean < 0.05 else "✗ Not significant"}')
    print(f'  Variance change: {ar_var_change:+.1f}% (p={ar_p_var:.4f})')
    print(f'  Rescue effect: r={ar_r:.3f} (p={ar_p_corr:.4f})'
          f' {"✓" if ar_p_corr < 0.05 else "✗"}')

    print(f'\n🎯 VALENCE (VA):')
    print(f'  Mean improvement: p={va_p_mean:.4f} '
          f'{"✓" if va_p_mean < 0.05 else "✗ Not significant"}')
    print(f'  Variance change: {va_var_change:+.1f}% (p={va_p_var:.4f})')
    print(f'  Rescue effect: r={va_r:.3f} (p={va_p_corr:.4f})'
          f' {"✓✓ SIGNIFICANT ⭐" if va_p_corr < 0.05 else "✗"}')

    # =====================================================================
    # PART 6: PUBLICATION FIGURE
    # =====================================================================
    print('\n' + '=' * 80)
    print('PART 6: GENERATING PUBLICATION FIGURE')
    print('=' * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Comprehensive Statistical Analysis: MTL Rescue Effect ({prefix})',
                 fontsize=16, fontweight='bold')

    methods = ['PSTL', 'STL', 'MTL']
    x = np.arange(len(methods))
    width = 0.35
    ar_means = [PSTL_AR_MEAN, gains_df['AR_acc_STL'].mean(), gains_df['AR_acc_MTL'].mean()]
    va_means = [PSTL_VA_MEAN, gains_df['VA_acc_STL'].mean(), gains_df['VA_acc_MTL'].mean()]

    # Panel 1: Performance progression
    ax = axes[0, 0]
    ax.bar(x - width / 2, ar_means, width, label='Arousal', alpha=0.7, color='steelblue')
    ax.bar(x + width / 2, va_means, width, label='Valence', alpha=0.7, color='seagreen')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Performance Progression', fontsize=12, fontweight='bold')

    # Panel 2: Variance trajectory
    ax = axes[0, 1]
    ar_vars = [PSTL_AR_STD ** 2, ar_var_stl, ar_var_mtl]
    va_vars = [PSTL_VA_STD ** 2, va_var_stl, va_var_mtl]
    ax.plot(methods, ar_vars, 'o-', linewidth=2, markersize=8,
            label='Arousal', color='steelblue')
    ax.plot(methods, va_vars, 's-', linewidth=2, markersize=8,
            label='Valence', color='seagreen')
    ax.set_ylabel('Variance (σ²)', fontsize=11, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(f'Variance: AR {ar_var_change:+.1f}%, VA {va_var_change:+.1f}%',
                 fontsize=12, fontweight='bold')

    # Panel 3: Rescue effect scatter
    ax = axes[0, 2]
    ax.scatter(gains_df['VA_acc_STL'], gains_df['VA_gain_%'],
               alpha=0.6, s=80, color='seagreen', label=f'VA (r={va_r:.3f}*)')
    ax.scatter(gains_df['AR_acc_STL'], gains_df['AR_gain_%'],
               alpha=0.6, s=80, color='steelblue', label=f'AR (r={ar_r:.3f})')
    z = np.polyfit(gains_df['VA_acc_STL'], gains_df['VA_gain_%'], 1)
    x_line = np.linspace(0.1, 1.0, 100)
    ax.plot(x_line, np.poly1d(z)(x_line), '--', color='darkgreen', linewidth=2)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('STL Baseline Accuracy', fontsize=11, fontweight='bold')
    ax.set_ylabel('MTL Gain (%)', fontsize=11, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(f'Rescue Effect (p={va_p_corr:.3f}{"*" if va_p_corr < 0.05 else ""})',
                 fontsize=12, fontweight='bold')

    # Panel 4: Group boxplot
    ax = axes[1, 0]
    if len(va_low) > 0 and len(va_high) > 0:
        data_boxes = [va_low['VA_gain_%'], va_high['VA_gain_%']]
        positions = [1, 2]
        if len(ar_low) > 0: data_boxes.append(ar_low['AR_gain_%']); positions.append(4)
        if len(ar_high) > 0: data_boxes.append(ar_high['AR_gain_%']); positions.append(5)
        bp = ax.boxplot(data_boxes, positions=positions, widths=0.6, patch_artist=True)
        colors_bp = ['seagreen', 'lightgreen', 'steelblue', 'lightblue']
        for patch, c in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(['VA\nLow', 'VA\nHigh', 'AR\nLow', 'AR\nHigh'])
        ax.set_ylabel('MTL Gain (%)', fontsize=11, fontweight='bold')
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_title(
            f'Low vs High Baseline (VA p={va_p_group:.3f})',
            fontsize=12, fontweight='bold')

    # Panel 5: Distribution histogram
    ax = axes[1, 1]
    ax.hist(gains_df['VA_gain_%'], bins=15, alpha=0.6, color='seagreen',
            edgecolor='black', label='Valence')
    ax.hist(gains_df['AR_gain_%'], bins=15, alpha=0.6, color='steelblue',
            edgecolor='black', label='Arousal')
    ax.axvline(0, color='black', linewidth=2)
    ax.axvline(gains_df['VA_gain_%'].mean(), color='seagreen', linestyle='--', linewidth=2)
    ax.axvline(gains_df['AR_gain_%'].mean(), color='steelblue', linestyle='--', linewidth=2)
    ax.set_xlabel('MTL Gain (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Gain Distributions', fontsize=12, fontweight='bold')

    # Panel 6: Text summary
    ax = axes[1, 2]
    ax.axis('off')
    low_va_mean = va_low['VA_gain_%'].mean() if len(va_low) > 0 else float('nan')
    high_va_mean = va_high['VA_gain_%'].mean() if len(va_high) > 0 else float('nan')
    summary_text = (
        f'STATISTICAL SUMMARY ({prefix})\n\n'
        f'VALENCE:\n'
        f'{"✓" if va_p_corr < 0.05 else "○"} Rescue: r={va_r:.3f}, p={va_p_corr:.4f}\n'
        f'○ Mean gain: +{gains_df["VA_gain_%"].mean():.1f}%, p={va_p_mean:.4f}\n'
        f'○ Variance: {va_var_change:.1f}%, p={va_p_var:.4f}\n'
        f'○ Group diff: p={va_p_group:.4f}\n'
        f'  Low baseline: +{low_va_mean:.1f}%\n'
        f'  High baseline: +{high_va_mean:.1f}%\n\n'
        f'AROUSAL:\n'
        f'{"✓" if ar_p_corr < 0.05 else "✗"} Rescue: r={ar_r:.3f}, p={ar_p_corr:.4f}\n'
        f'✗ Improvement: p={ar_p_mean:.4f}\n'
        f'  Variance: {ar_var_change:+.1f}%\n\n'
        f'CONCLUSION:\n'
        f'Valence: Baseline-dependent rescue\n'
        f'Arousal: Population model best'
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f'{prefix}_comprehensive_statistical_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'\n✓ Saved figure: {fig_path}')

    print('\n' + '=' * 80)
    print('ANALYSIS COMPLETE')
    print('=' * 80)
