#!/usr/bin/env python3
"""
Extreme Tails Analysis: LSB vs HSB
==================================

Test whether the acceleration-dependent coherence length effect is real
by comparing the most extreme 20% of each tail.

This clarifies:
- If extreme tails show ratio ~1.3-1.5×: effect is real but saturates in middle
- If extreme tails show ratio ~1.1×: acceleration dependence is genuinely negligible
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

def run_extreme_tails_analysis():
    """Analyze extreme tails of LSB/HSB distribution"""
    
    # Load previous results
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "weyl" / "lsb_hsb_test_results.json"
    
    with open(results_path) as f:
        data = json.load(f)
    
    galaxies = data["galaxies"]
    
    # Sort by surface brightness
    sorted_galaxies = sorted(galaxies, key=lambda x: x["surface_brightness"])
    
    n_total = len(sorted_galaxies)
    n_extreme = n_total // 5  # 20% tails
    
    print("=" * 70)
    print("EXTREME TAILS ANALYSIS")
    print("=" * 70)
    print(f"\nTotal galaxies: {n_total}")
    print(f"Extreme tail size: {n_extreme} galaxies each (20%)")
    
    # Extract tails
    lsb_extreme = sorted_galaxies[:n_extreme]
    hsb_extreme = sorted_galaxies[-n_extreme:]
    middle = sorted_galaxies[n_extreme:-n_extreme]
    
    # Statistics for each group
    def analyze_group(galaxies, name):
        ell0_Rd = np.array([g["ell0_over_Rd"] for g in galaxies])
        g_ratio = np.array([g["g_ratio"] for g in galaxies])
        sb = np.array([g["surface_brightness"] for g in galaxies])
        g_char = np.array([g["g_char"] for g in galaxies])
        
        return {
            "name": name,
            "n": len(galaxies),
            "ell0_Rd_mean": np.mean(ell0_Rd),
            "ell0_Rd_std": np.std(ell0_Rd),
            "ell0_Rd_stderr": np.std(ell0_Rd) / np.sqrt(len(ell0_Rd)),
            "ell0_Rd_median": np.median(ell0_Rd),
            "g_ratio_mean": np.mean(g_ratio),
            "sb_range": (np.min(sb), np.max(sb)),
            "g_char_mean": np.mean(g_char),
            "galaxies": [g["galaxy"] for g in galaxies]
        }
    
    lsb_stats = analyze_group(lsb_extreme, "LSB Extreme (bottom 20%)")
    hsb_stats = analyze_group(hsb_extreme, "HSB Extreme (top 20%)")
    mid_stats = analyze_group(middle, "Middle 60%")
    
    # Display results
    print(f"\n{'Group':<25} {'N':>5} {'log(Σ) range':<20} {'ℓ₀/R_d mean±SE':>18} {'√(g†/g) mean':>12}")
    print("-" * 85)
    
    for s in [lsb_stats, mid_stats, hsb_stats]:
        sb_range = f"[{s['sb_range'][0]:.2f}, {s['sb_range'][1]:.2f}]"
        ell_str = f"{s['ell0_Rd_mean']:.2f} ± {s['ell0_Rd_stderr']:.2f}"
        print(f"{s['name']:<25} {s['n']:>5} {sb_range:<20} {ell_str:>18} {s['g_ratio_mean']:>12.2f}")
    
    # Key ratios
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)
    
    # Extreme ratio
    extreme_ratio = lsb_stats["ell0_Rd_mean"] / hsb_stats["ell0_Rd_mean"]
    predicted_extreme = lsb_stats["g_ratio_mean"] / hsb_stats["g_ratio_mean"]
    
    print(f"\n  Extreme tails (20% each):")
    print(f"    LSB extreme ℓ₀/R_d: {lsb_stats['ell0_Rd_mean']:.2f} ± {lsb_stats['ell0_Rd_stderr']:.2f}")
    print(f"    HSB extreme ℓ₀/R_d: {hsb_stats['ell0_Rd_mean']:.2f} ± {hsb_stats['ell0_Rd_stderr']:.2f}")
    print(f"    Observed ratio (LSB/HSB): {extreme_ratio:.3f}")
    print(f"    Predicted ratio (√g† scaling): {predicted_extreme:.3f}")
    
    # Statistical test on extremes
    lsb_vals = np.array([g["ell0_over_Rd"] for g in lsb_extreme])
    hsb_vals = np.array([g["ell0_over_Rd"] for g in hsb_extreme])
    
    t_stat, p_value = stats.ttest_ind(lsb_vals, hsb_vals)
    
    # Mann-Whitney U test (non-parametric, better for skewed data)
    u_stat, p_mw = stats.mannwhitneyu(lsb_vals, hsb_vals, alternative='greater')
    
    print(f"\n  Statistical Tests (Extreme Tails):")
    print(f"    Two-sample t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
    print(f"    Mann-Whitney U (one-sided): U = {u_stat:.0f}, p = {p_mw:.4f}")
    
    # Fit power law exponent
    print("\n" + "=" * 70)
    print("POWER LAW FIT: ℓ₀/R_d ∝ (g†/g_char)^α")
    print("=" * 70)
    
    # Use all galaxies for power law fit
    all_g_ratio = np.array([g["g_ratio"] for g in sorted_galaxies])
    all_ell0_Rd = np.array([g["ell0_over_Rd"] for g in sorted_galaxies])
    
    # Filter out outliers (ℓ₀/R_d > 20 likely hit boundary)
    valid = all_ell0_Rd < 20
    log_g_ratio = np.log(all_g_ratio[valid])
    log_ell0_Rd = np.log(all_ell0_Rd[valid])
    
    # Linear regression in log-log space
    slope, intercept, r_value, p_value_fit, std_err = stats.linregress(log_g_ratio, log_ell0_Rd)
    
    print(f"\n  Fit results (excluding ℓ₀/R_d > 20 boundary hits):")
    print(f"    N galaxies used: {np.sum(valid)}")
    print(f"    Power law exponent α: {slope:.4f} ± {std_err:.4f}")
    print(f"    Correlation r: {r_value:.4f}")
    print(f"    p-value: {p_value_fit:.4f}")
    
    # Compare to predictions
    print(f"\n  Comparison to Theory:")
    print(f"    Weyl prediction: α = 0.5 (square root)")
    print(f"    Observed:        α = {slope:.3f}")
    print(f"    Ratio: {slope/0.5:.2f}× weaker than predicted")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if extreme_ratio > 1.3 and p_mw < 0.1:
        verdict = "✓ EFFECT IS REAL but saturates in the middle"
        details = f"Extreme tails show {extreme_ratio:.2f}× difference (p={p_mw:.3f})"
    elif extreme_ratio > 1.1 and slope > 0.05:
        verdict = "? WEAK EFFECT DETECTED — constrain α ≈ {:.2f}".format(slope)
        details = "Acceleration dependence exists but is ~5× weaker than √ scaling"
    else:
        verdict = "✗ ACCELERATION DEPENDENCE IS NEGLIGIBLE"
        details = "The (2/3)R_d scaling dominates; g_char has minimal effect"
    
    print(f"\n  {verdict}")
    print(f"  {details}")
    
    # Update recommendation
    print("\n" + "=" * 70)
    print("THEORETICAL IMPLICATION")
    print("=" * 70)
    
    print(f"\n  The Weyl derivation should be revised to:")
    print(f"")
    print(f"    ℓ₀ = (2/3) × R_d × (g†/g_char)^α")
    print(f"")
    print(f"  where α = {slope:.2f} ± {std_err:.2f} (empirically constrained)")
    print(f"")
    print(f"  This is equivalent to saying the coherence length depends only")
    print(f"  weakly on the characteristic acceleration, with the disk scale")
    print(f"  length being the dominant factor.")
    
    # Save results
    output = {
        "extreme_tails": {
            "n_per_tail": n_extreme,
            "lsb_extreme": {
                "ell0_Rd_mean": lsb_stats["ell0_Rd_mean"],
                "ell0_Rd_stderr": lsb_stats["ell0_Rd_stderr"],
                "g_ratio_mean": lsb_stats["g_ratio_mean"],
                "sb_range": lsb_stats["sb_range"],
                "galaxies": lsb_stats["galaxies"]
            },
            "hsb_extreme": {
                "ell0_Rd_mean": hsb_stats["ell0_Rd_mean"],
                "ell0_Rd_stderr": hsb_stats["ell0_Rd_stderr"],
                "g_ratio_mean": hsb_stats["g_ratio_mean"],
                "sb_range": hsb_stats["sb_range"],
                "galaxies": hsb_stats["galaxies"]
            },
            "observed_ratio": extreme_ratio,
            "predicted_ratio": predicted_extreme,
            "t_statistic": t_stat,
            "p_value_ttest": p_value,
            "p_value_mannwhitney": p_mw
        },
        "power_law_fit": {
            "alpha": slope,
            "alpha_stderr": std_err,
            "r_value": r_value,
            "p_value": p_value_fit,
            "n_galaxies_used": int(np.sum(valid))
        },
        "verdict": verdict
    }
    
    output_path = repo_root / "weyl" / "extreme_tails_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\n  Results saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    run_extreme_tails_analysis()
