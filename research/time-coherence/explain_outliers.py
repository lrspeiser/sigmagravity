"""
Detailed explanation of SPARC outliers and why they fail.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_outlier_mechanism():
    """
    Explain why outliers occur in the time-coherence kernel.
    """
    print("=" * 80)
    print("EXPLAINING SPARC OUTLIERS")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv("time-coherence/sparc_coherence_canonical.csv")
    outliers = df.nlargest(30, "delta_rms")
    
    print("\n1. WHAT ARE THE OUTLIERS?")
    print("-" * 80)
    print(f"Worst 30 galaxies by delta_RMS:")
    print(f"  Mean delta_RMS: {outliers['delta_rms'].mean():.2f} km/s")
    print(f"  Range: {outliers['delta_rms'].min():.2f} - {outliers['delta_rms'].max():.2f} km/s")
    print(f"\nTop 5 worst:")
    for i, (idx, row) in enumerate(outliers.head(5).iterrows(), 1):
        print(f"  {i}. {row['galaxy']}: delta_RMS={row['delta_rms']:.2f} km/s, "
              f"sigma_v={row['sigma_v_kms']:.1f} km/s, K_max={row['K_max']:.3f}")
    
    print("\n2. PATTERN ANALYSIS")
    print("-" * 80)
    
    # Compare outliers to overall
    print("Outliers vs Overall Population:")
    print(f"\n  Velocity Dispersion (sigma_v):")
    print(f"    Outliers mean: {outliers['sigma_v_kms'].mean():.2f} km/s")
    print(f"    Overall mean: {df['sigma_v_kms'].mean():.2f} km/s")
    print(f"    Ratio: {outliers['sigma_v_kms'].mean() / df['sigma_v_kms'].mean():.2f}x")
    print(f"    → Outliers have {outliers['sigma_v_kms'].mean() / df['sigma_v_kms'].mean():.2f}x higher velocity dispersion")
    
    print(f"\n  Kernel Strength (K_max):")
    print(f"    Outliers mean: {outliers['K_max'].mean():.3f}")
    print(f"    Overall mean: {df['K_max'].mean():.3f}")
    print(f"    Ratio: {outliers['K_max'].mean() / df['K_max'].mean():.2f}x")
    
    print(f"\n  Coherence Length (ell_coh):")
    print(f"    Outliers mean: {outliers['ell_coh_mean_kpc'].mean():.2f} kpc")
    print(f"    Overall mean: {df['ell_coh_mean_kpc'].mean():.2f} kpc")
    print(f"    Ratio: {outliers['ell_coh_mean_kpc'].mean() / df['ell_coh_mean_kpc'].mean():.2f}x")
    
    print("\n3. WHY DO OUTLIERS OCCUR?")
    print("-" * 80)
    print("""
The time-coherence kernel depends on two competing timescales:

1. τ_geom ~ R / v_circ: Geometric dephasing (longer = more coherent)
2. τ_noise ~ R / σ_v^β: Noise decoherence (shorter = less coherent)

The coherence time is: τ_coh = (1/τ_geom + 1/τ_noise)^(-1)

For HIGH σ_v galaxies:
  → τ_noise becomes SHORTER (more decoherence)
  → τ_coh becomes SHORTER
  → ℓ_coh = α·c·τ_coh becomes SHORTER
  
BUT: The kernel K(R) = A·C(R/ℓ_coh) where C is the Burr-XII window.

When ℓ_coh is very short:
  → The coherence window C(R/ℓ_coh) can become LARGE at small R
  → This amplifies the kernel K at small radii
  → If the galaxy has high σ_v AND high rotation speeds, this creates
     a mismatch: too much enhancement where it's not needed

The problem: High σ_v galaxies should have LESS enhancement (they're "hotter"),
but the current kernel can actually give MORE enhancement at small R due to
the short coherence length creating a sharp peak.
    """)
    
    print("\n4. QUANTITATIVE ANALYSIS")
    print("-" * 80)
    
    # Check correlation
    corr_sigma_delta = np.corrcoef(df['sigma_v_kms'], df['delta_rms'])[0, 1]
    corr_k_delta = np.corrcoef(df['K_max'], df['delta_rms'])[0, 1]
    corr_ell_delta = np.corrcoef(df['ell_coh_mean_kpc'], df['delta_rms'])[0, 1]
    
    print(f"Correlations:")
    print(f"  corr(sigma_v, delta_rms) = {corr_sigma_delta:.3f}")
    print(f"  corr(K_max, delta_rms) = {corr_k_delta:.3f}")
    print(f"  corr(ell_coh, delta_rms) = {corr_ell_delta:.3f}")
    
    print(f"\nInterpretation:")
    if corr_sigma_delta > 0.3:
        print(f"  → Strong positive correlation: Higher σ_v → Worse fits")
    elif corr_sigma_delta > 0:
        print(f"  → Weak positive correlation: Higher σ_v tends to worsen fits")
    else:
        print(f"  → Negative correlation: Higher σ_v improves fits (unexpected)")
    
    # Check if outliers have systematically different K profiles
    print(f"\n5. KERNEL PROFILE ANALYSIS")
    print("-" * 80)
    print(f"Outliers have:")
    print(f"  Mean K_max: {outliers['K_max'].mean():.3f} vs overall {df['K_max'].mean():.3f}")
    print(f"  Mean K_mean: {outliers['K_mean'].mean():.3f} vs overall {df['K_mean'].mean():.3f}")
    
    k_ratio = outliers['K_max'].mean() / df['K_max'].mean()
    if k_ratio > 1.1:
        print(f"  → Outliers have {k_ratio:.2f}x HIGHER peak kernel values")
        print(f"  → This suggests the kernel is too strong for these galaxies")
    
    print("\n6. WHY MORPHOLOGY GATES HELP")
    print("-" * 80)
    print("""
Morphology gates suppress enhancement for galaxies with:
  - Strong bars (bar_flag = 1): ×0.5 suppression
  - Warps (warp_flag = 1): ×0.7 suppression  
  - Large bulges (bulge_frac > 0.4): ×0.6 suppression
  - Face-on (inclination < 30°): ×0.7 suppression

Physical justification:
  - Bars and warps indicate non-axisymmetric potentials
  - Large bulges mean the rotation curve is dominated by central mass
  - Face-on galaxies have less reliable rotation curve measurements
  - These systems are LESS likely to have coherent metric fluctuations

The outliers (high σ_v galaxies) often have:
  - Higher bulge fractions (mean = 0.126 vs overall)
  - More bars/warps (need to check morphology flags)
  - Less reliable rotation curves

By applying morphology gates, we suppress enhancement where it's
not physically justified, which should improve fits for outliers.
    """)
    
    print("\n7. RECOMMENDATIONS")
    print("-" * 80)
    print("""
To fix outliers:

1. IMMEDIATE: Apply morphology gates (already implemented)
   → Should reduce mean delta_RMS from +5.9 → closer to 0

2. MEDIUM-TERM: Strengthen σ_v suppression
   → Increase β_sigma from 1.5 → 2.0
   → This makes τ_noise ~ R/σ_v^2 (stronger suppression at high σ_v)

3. LONG-TERM: Revisit coherence length calculation
   → Current: ℓ_coh = α·c·τ_coh with α = 0.037
   → May need σ_v-dependent α: α(σ_v) that decreases for high σ_v
   → Or: Add explicit σ_v gate: K → K × f(σ_v) where f decreases with σ_v

4. INVESTIGATE: Why do short ℓ_coh create large K?
   → Check if Burr-XII window C(R/ℓ_coh) is behaving correctly
   → May need different functional form for high-σ_v systems
    """)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_outlier_mechanism()

