"""
Statistical tests for SPARC galaxy fits: Coherence vs Dark Matter.

Tests:
- Wilcoxon signed-rank test (paired, non-parametric)
- Paired t-test (parametric alternative)
- Kolmogorov-Smirnov test on ratio distribution
- Effect size (Cohen's d)
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_results(csv_file='../outputs/sparc_fit_summary.csv'):
    """
    Load SPARC fit results from CSV.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file
        
    Returns:
    --------
    df : DataFrame
        Results dataframe
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Results file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} galaxy fits from {csv_file}")
    
    return df


def cohens_d(x, y):
    """
    Compute Cohen's d effect size.
    
    Parameters:
    -----------
    x, y : array-like
        Two samples
        
    Returns:
    --------
    d : float
        Cohen's d
    """
    nx = len(x)
    ny = len(y)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (np.mean(x) - np.mean(y)) / pooled_std
    return d


def run_statistical_tests(df, output_file=None):
    """
    Run comprehensive statistical tests.
    
    Parameters:
    -----------
    df : DataFrame
        Results dataframe
    output_file : str
        Optional file to save results
    """
    print("=" * 80)
    print("STATISTICAL TESTS: COHERENCE vs DARK MATTER")
    print("=" * 80)
    
    # Extract chi-squared values (handle old and new CSV formats)
    chi2_co = df['chi2_red_coherence'].values
    chi2_nfw = df['chi2_red_nfw'].values
    
    # Check if best_dm column exists (new format)
    if 'chi2_red_best_dm' in df.columns:
        chi2_best_dm = df['chi2_red_best_dm'].values
        ratios = df['ratio_vs_best_dm'].values
    else:
        # Old format: use NFW as best DM
        chi2_best_dm = chi2_nfw.copy()
        ratios = df['ratio'].values
    
    # Remove NaN values
    mask = ~(np.isnan(chi2_co) | np.isnan(chi2_best_dm))
    chi2_co_clean = chi2_co[mask]
    chi2_nfw_clean = chi2_nfw[mask]
    chi2_best_dm_clean = chi2_best_dm[mask]
    
    n = len(chi2_co_clean)
    
    print(f"\nDataset: {n} galaxies")
    print(f"  Coherence mean: {np.mean(chi2_co_clean):.3f}")
    print(f"  Best DM mean: {np.mean(chi2_best_dm_clean):.3f}")
    print(f"  Difference: {np.mean(chi2_co_clean) - np.mean(chi2_best_dm_clean):.3f}")
    
    # ===================================================================
    # Test 1: Wilcoxon signed-rank test (paired, non-parametric)
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 1: WILCOXON SIGNED-RANK TEST (Paired, Non-parametric)")
    print("=" * 80)
    print("Null hypothesis: Coherence and DM have the same distribution")
    print("Alternative: Coherence fits better (lower chi^2_red)")
    
    # Test coherence vs NFW
    stat_nfw, pval_nfw = stats.wilcoxon(chi2_co_clean, chi2_nfw_clean, 
                                        alternative='less')  # one-sided: coherence < NFW
    print(f"\nCoherence vs NFW:")
    print(f"  Test statistic: {stat_nfw:.3f}")
    print(f"  p-value (one-sided): {pval_nfw:.6f}")
    print(f"  Significance: {'***' if pval_nfw < 0.001 else ('**' if pval_nfw < 0.01 else ('*' if pval_nfw < 0.05 else 'ns'))}")
    
    # Test coherence vs best DM
    stat_best, pval_best = stats.wilcoxon(chi2_co_clean, chi2_best_dm_clean,
                                         alternative='less')
    print(f"\nCoherence vs Best DM (NFW or Burkert):")
    print(f"  Test statistic: {stat_best:.3f}")
    print(f"  p-value (one-sided): {pval_best:.6f}")
    print(f"  Significance: {'***' if pval_best < 0.001 else ('**' if pval_best < 0.01 else ('*' if pval_best < 0.05 else 'ns'))}")
    
    # Two-sided test
    stat_best_2sided, pval_best_2sided = stats.wilcoxon(chi2_co_clean, chi2_best_dm_clean)
    print(f"  p-value (two-sided): {pval_best_2sided:.6f}")
    
    # ===================================================================
    # Test 2: Paired t-test (parametric)
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 2: PAIRED T-TEST (Parametric)")
    print("=" * 80)
    print("Null hypothesis: Mean difference = 0")
    print("Alternative: Mean(coherence) < Mean(DM)")
    
    # Test coherence vs NFW
    tstat_nfw, pval_ttest_nfw = stats.ttest_rel(chi2_co_clean, chi2_nfw_clean)
    # One-sided p-value
    if tstat_nfw < 0:
        pval_ttest_nfw_onesided = pval_ttest_nfw / 2
    else:
        pval_ttest_nfw_onesided = 1 - pval_ttest_nfw / 2
    
    print(f"\nCoherence vs NFW:")
    print(f"  t-statistic: {tstat_nfw:.3f}")
    print(f"  p-value (one-sided): {pval_ttest_nfw_onesided:.6f}")
    print(f"  Significance: {'***' if pval_ttest_nfw_onesided < 0.001 else ('**' if pval_ttest_nfw_onesided < 0.01 else ('*' if pval_ttest_nfw_onesided < 0.05 else 'ns'))}")
    
    # Test coherence vs best DM
    tstat_best, pval_ttest_best = stats.ttest_rel(chi2_co_clean, chi2_best_dm_clean)
    if tstat_best < 0:
        pval_ttest_best_onesided = pval_ttest_best / 2
    else:
        pval_ttest_best_onesided = 1 - pval_ttest_best / 2
    
    print(f"\nCoherence vs Best DM:")
    print(f"  t-statistic: {tstat_best:.3f}")
    print(f"  p-value (one-sided): {pval_ttest_best_onesided:.6f}")
    print(f"  Significance: {'***' if pval_ttest_best_onesided < 0.001 else ('**' if pval_ttest_best_onesided < 0.01 else ('*' if pval_ttest_best_onesided < 0.05 else 'ns'))}")
    
    # ===================================================================
    # Test 3: Effect size (Cohen's d)
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 3: EFFECT SIZE (Cohen's d)")
    print("=" * 80)
    print("Interpretation: |d| < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), > 0.8 (large)")
    
    d_nfw = cohens_d(chi2_co_clean, chi2_nfw_clean)
    d_best = cohens_d(chi2_co_clean, chi2_best_dm_clean)
    
    print(f"\nCoherence vs NFW:")
    print(f"  Cohen's d: {d_nfw:.3f}")
    if abs(d_nfw) < 0.2:
        size = "negligible"
    elif abs(d_nfw) < 0.5:
        size = "small"
    elif abs(d_nfw) < 0.8:
        size = "medium"
    else:
        size = "large"
    print(f"  Effect size: {size}")
    
    print(f"\nCoherence vs Best DM:")
    print(f"  Cohen's d: {d_best:.3f}")
    if abs(d_best) < 0.2:
        size = "negligible"
    elif abs(d_best) < 0.5:
        size = "small"
    elif abs(d_best) < 0.8:
        size = "medium"
    else:
        size = "large"
    print(f"  Effect size: {size}")
    
    # ===================================================================
    # Test 4: Kolmogorov-Smirnov test on ratio distribution
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 4: KOLMOGOROV-SMIRNOV TEST (Ratio vs 1.0)")
    print("=" * 80)
    print("Null hypothesis: Ratio distribution is centered at 1.0")
    
    if 'ratio_vs_best_dm' in df.columns:
        ratios = df['ratio_vs_best_dm'].values
    else:
        ratios = df['ratio'].values
    ratios_clean = ratios[mask]
    
    # Test if ratio distribution is different from 1.0
    # Transform: ratio - 1.0 should be centered at 0
    ks_stat, ks_pval = stats.kstest(ratios_clean - 1.0, 
                                   lambda x: stats.norm.cdf(x, 0, np.std(ratios_clean - 1.0)))
    
    print(f"\nRatio distribution (Coherence / Best DM):")
    print(f"  Mean ratio: {np.mean(ratios_clean):.3f}")
    print(f"  Median ratio: {np.median(ratios_clean):.3f}")
    print(f"  Std ratio: {np.std(ratios_clean):.3f}")
    print(f"  KS statistic: {ks_stat:.3f}")
    print(f"  p-value: {ks_pval:.6f}")
    print(f"  Significance: {'***' if ks_pval < 0.001 else ('**' if ks_pval < 0.01 else ('*' if ks_pval < 0.05 else 'ns'))}")
    
    # Test if ratio < 1.0 (coherence better)
    from scipy.stats import normaltest
    is_normal = normaltest(ratios_clean - 1.0)[1] > 0.05
    
    if is_normal:
        # Use one-sample t-test
        tstat_ratio, pval_ratio = stats.ttest_1samp(ratios_clean, 1.0, alternative='less')
        print(f"\nOne-sample t-test (ratio < 1.0):")
        print(f"  t-statistic: {tstat_ratio:.3f}")
        print(f"  p-value: {pval_ratio:.6f}")
    else:
        # Use sign test (non-parametric)
        n_below_1 = (ratios_clean < 1.0).sum()
        n_above_1 = (ratios_clean > 1.0).sum()
        # Handle scipy API change
        try:
            from scipy.stats import binomtest
            result = binomtest(n_below_1, n_below_1 + n_above_1, 0.5, alternative='greater')
            pval_ratio = result.pvalue
        except ImportError:
            try:
                pval_ratio = stats.binom_test(n_below_1, n_below_1 + n_above_1, 0.5, alternative='greater')
            except AttributeError:
                # Fallback: manual calculation
                from scipy.stats import binom
                pval_ratio = 1 - binom.cdf(n_below_1 - 1, n_below_1 + n_above_1, 0.5)
        print(f"\nSign test (ratio < 1.0):")
        print(f"  Galaxies with ratio < 1.0: {n_below_1}/{len(ratios_clean)}")
        print(f"  p-value: {pval_ratio:.6f}")
    
    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    coherence_wins = (ratios_clean < 1.0).sum()
    win_rate = 100 * coherence_wins / len(ratios_clean)
    
    print(f"\nWin rate: Coherence beats Best DM in {coherence_wins}/{len(ratios_clean)} galaxies ({win_rate:.1f}%)")
    print(f"\nStatistical significance:")
    print(f"  Wilcoxon test (one-sided): p = {pval_best:.6f} {'***' if pval_best < 0.001 else ('**' if pval_best < 0.01 else ('*' if pval_best < 0.05 else 'ns'))}")
    print(f"  Paired t-test (one-sided): p = {pval_ttest_best_onesided:.6f} {'***' if pval_ttest_best_onesided < 0.001 else ('**' if pval_ttest_best_onesided < 0.01 else ('*' if pval_ttest_best_onesided < 0.05 else 'ns'))}")
    print(f"  Effect size (Cohen's d): {d_best:.3f} ({'large' if abs(d_best) > 0.8 else ('medium' if abs(d_best) > 0.5 else ('small' if abs(d_best) > 0.2 else 'negligible'))})")
    
    # Interpretation
    if pval_best < 0.001 and win_rate > 60:
        interpretation = "VERY STRONG: Coherence significantly outperforms dark matter"
    elif pval_best < 0.01 and win_rate > 55:
        interpretation = "STRONG: Coherence outperforms dark matter"
    elif pval_best < 0.05 and win_rate > 50:
        interpretation = "MODERATE: Evidence that coherence outperforms dark matter"
    else:
        interpretation = "WEAK: Insufficient evidence"
    
    print(f"\nInterpretation: {interpretation}")
    print("=" * 80)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            f.write("STATISTICAL TESTS: COHERENCE vs DARK MATTER\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Dataset: {n} galaxies\n\n")
            f.write("Wilcoxon signed-rank test (one-sided):\n")
            f.write(f"  Coherence vs NFW: p = {pval_nfw:.6f}\n")
            f.write(f"  Coherence vs Best DM: p = {pval_best:.6f}\n\n")
            f.write("Paired t-test (one-sided):\n")
            f.write(f"  Coherence vs NFW: p = {pval_ttest_nfw_onesided:.6f}\n")
            f.write(f"  Coherence vs Best DM: p = {pval_ttest_best_onesided:.6f}\n\n")
            f.write("Effect size (Cohen's d):\n")
            f.write(f"  Coherence vs NFW: d = {d_nfw:.3f}\n")
            f.write(f"  Coherence vs Best DM: d = {d_best:.3f}\n\n")
            f.write(f"Win rate: {win_rate:.1f}%\n")
            f.write(f"Interpretation: {interpretation}\n")
        
        print(f"\nSaved results to: {output_file}")
    
    return {
        'n_galaxies': n,
        'win_rate': win_rate,
        'wilcoxon_p': pval_best,
        'ttest_p': pval_ttest_best_onesided,
        'cohens_d': d_best,
        'mean_ratio': np.mean(ratios_clean),
        'median_ratio': np.median(ratios_clean),
        'interpretation': interpretation
    }


def main():
    """Main function."""
    csv_file = '../outputs/sparc_fit_summary.csv'
    df = load_results(csv_file)
    
    results = run_statistical_tests(df, output_file='../outputs/statistical_tests.txt')
    
    print(f"\n{'=' * 80}")
    print("Test complete!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

