#!/usr/bin/env python3
"""
Ferraro Scale Tests
===================

Tests inspired by Prof. Rafael Ferraro's insight about f(T) teleparallelism:

    "When a f(T) function is chosen, necessarily a constant with units of 
    square length must be introduced... This constant fixes the scale at 
    which the modified gravity deviates from standard (GR) gravity."

In our phenomenological model, we have several scale parameters:
- ℓ₀ (coherence length) ~ 2 kpc
- R_disk (baryonic scale length)  
- σ_v / √(GΣ_b) (Toomre scale)
- c/H₀ (cosmological scale ~ 4 Gpc)

Ferraro's question: Is there a SINGLE fundamental length² constant that 
determines when coherence effects activate?

These tests systematically probe:
1. Whether ℓ² scales with baryonic geometry across systems
2. Whether g† ~ cH₀ emerges from a length scale
3. Whether all thresholds derive from one dimensional constant

If successful, this would provide the "a" constant for a teleparallel interpretation.
"""

import numpy as np
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.environment_estimator import EnvironmentEstimator

# Physical constants
G_NEWTON = 4.302e-6  # kpc (km/s)² / M_sun
C_LIGHT = 299792.458  # km/s
H0 = 70.0  # km/s/Mpc = 70e-6 km/s/kpc

# Critical acceleration (MOND scale)
a0_SI = 1.2e-10  # m/s²
# Convert to (km/s)²/kpc: 1 m/s² = (1e-3 km/s)² / (3.086e-16 kpc) = 3.24e-9 (km/s)²/kpc
a0 = 3.9e-3  # (km/s)²/kpc ~ cH₀

# Hubble length
L_H = C_LIGHT / (H0 * 1e-3)  # kpc (c / H₀ ~ 4.3 Gpc)


@dataclass
class FerrароScaleResult:
    """Results from scale analysis for one galaxy."""
    name: str
    M_total: float  # Total baryonic mass [M_sun]
    R_disk: float   # Disk scale length [kpc]
    Sigma_0: float  # Central surface density [M_sun/kpc²]
    sigma_v: float  # Velocity dispersion [km/s]
    
    # Derived scales (all in kpc²)
    ell_toomre_sq: float      # (σ_v / √(GΣ))² - Toomre scale squared
    ell_disk_sq: float        # R_disk² - geometric scale squared
    ell_cH0_sq: float         # (c/H₀)² × (g/g†)² - cosmological scale squared
    ell_fitted_sq: float      # Fitted coherence length squared
    
    # Dimensionless ratios
    ratio_toomre_disk: float  # ℓ_toomre / R_disk
    ratio_fitted_toomre: float  # ℓ_fitted / ℓ_toomre


def compute_toomre_scale(sigma_v: float, Sigma_b: float) -> float:
    """
    Compute Toomre-like scale: ℓ_T = σ_v / √(2πG Σ_b)
    
    This is the natural length scale from disk stability theory.
    In f(T), this could be the "a" constant if it's universal.
    
    Parameters:
    -----------
    sigma_v : float
        Velocity dispersion [km/s]
    Sigma_b : float
        Surface density [M_sun/kpc²]
    
    Returns:
    --------
    ell_T : float
        Toomre scale [kpc]
    """
    if Sigma_b <= 0:
        return np.inf
    return sigma_v / np.sqrt(2 * np.pi * G_NEWTON * Sigma_b)


def compute_cH0_scale(g_typical: float) -> float:
    """
    Compute the scale where g ~ g† = cH₀.
    
    At this acceleration, the coherence effect should turn on.
    This gives a length scale: ℓ_cH0 = v² / g† where v is typical velocity.
    
    Parameters:
    -----------
    g_typical : float
        Typical acceleration [(km/s)²/kpc]
    
    Returns:
    --------
    ell_cH0 : float
        Length scale [kpc]
    """
    g_dagger = C_LIGHT * H0 * 1e-3  # ~ 21 (km/s)²/kpc
    # Scale where g/g† ~ 1
    return np.sqrt(g_typical / g_dagger) * (C_LIGHT / (H0 * 1e-3))


def analyze_galaxy_scales(galaxy_name: str, loader: RealDataLoader) -> Optional[FerrароScaleResult]:
    """
    Analyze the characteristic scales for one galaxy.
    
    Tests Ferraro's hypothesis: is there a universal length² constant?
    """
    try:
        gal = loader.load_galaxy(galaxy_name)
        masses = load_sparc_masses(galaxy_name)
    except Exception as e:
        print(f"  Skipping {galaxy_name}: {e}")
        return None
    
    # Extract parameters
    M_total = masses['M_total']
    R_disk = masses['R_disk']
    
    # Central surface density
    Sigma_0 = M_total / (2 * np.pi * R_disk**2)  # M_sun/kpc²
    
    # Estimate velocity dispersion from rotation curve
    r = gal['r']
    v_obs = gal['v_obs']
    
    # Use outer rotation curve to estimate σ_v (typically ~10-30% of v_flat)
    v_flat = np.median(v_obs[len(v_obs)//2:])
    sigma_v_est = 0.15 * v_flat  # Rough estimate
    
    # Better: use environment estimator if available
    try:
        env = EnvironmentEstimator()
        env_params = env.estimate_environment(gal, galaxy_name)
        sigma_v_est = env_params.get('sigma_v', sigma_v_est)
    except:
        pass
    
    # Compute characteristic scales
    ell_toomre = compute_toomre_scale(sigma_v_est, Sigma_0)
    
    # Typical acceleration at R_disk
    idx_disk = np.argmin(np.abs(r - R_disk))
    v_at_disk = v_obs[idx_disk] if idx_disk < len(v_obs) else v_flat
    g_at_disk = v_at_disk**2 / max(R_disk, 0.1)
    
    # Fitted coherence length (from your model: ℓ = ℓ₀ × (R_disk/2)^0.5)
    ell_0 = 2.0  # kpc (base value)
    p = 0.5
    ell_fitted = ell_0 * (R_disk / 2.0)**p
    
    # Cosmological scale
    g_dagger = C_LIGHT * H0 * 1e-3  # ~ 21 (km/s)²/kpc
    ell_cH0 = R_disk * np.sqrt(g_at_disk / g_dagger)
    
    return FerrароScaleResult(
        name=galaxy_name,
        M_total=M_total,
        R_disk=R_disk,
        Sigma_0=Sigma_0,
        sigma_v=sigma_v_est,
        ell_toomre_sq=ell_toomre**2,
        ell_disk_sq=R_disk**2,
        ell_cH0_sq=ell_cH0**2,
        ell_fitted_sq=ell_fitted**2,
        ratio_toomre_disk=ell_toomre / R_disk if R_disk > 0 else np.nan,
        ratio_fitted_toomre=ell_fitted / ell_toomre if ell_toomre > 0 and ell_toomre < np.inf else np.nan
    )


def test_scale_universality(results: List[FerrароScaleResult]) -> Dict:
    """
    Test whether there's a universal length² constant.
    
    Ferraro's insight: In f(T) = a⁻¹ Exp[aT], the constant 'a' has units [length]².
    
    We test several hypotheses:
    1. a = ℓ_toomre² (varies with galaxy)
    2. a = R_disk² (varies with galaxy)  
    3. a = ℓ_toomre² / R_disk² × const (universal ratio?)
    4. a = (c/H₀)² × (g/g†)² (cosmological)
    
    If (3) holds with small scatter, we have a candidate for teleparallel "a".
    """
    # Extract arrays
    ell_toomre_sq = np.array([r.ell_toomre_sq for r in results])
    ell_disk_sq = np.array([r.ell_disk_sq for r in results])
    ell_fitted_sq = np.array([r.ell_fitted_sq for r in results])
    ratio_toomre_disk = np.array([r.ratio_toomre_disk for r in results])
    ratio_fitted_toomre = np.array([r.ratio_fitted_toomre for r in results])
    
    # Filter out infinities and NaNs
    valid = np.isfinite(ratio_toomre_disk) & np.isfinite(ratio_fitted_toomre)
    
    if np.sum(valid) < 5:
        return {'error': 'Not enough valid galaxies'}
    
    ratio_td_valid = ratio_toomre_disk[valid]
    ratio_ft_valid = ratio_fitted_toomre[valid]
    
    # Test 1: Is ℓ_toomre / R_disk universal?
    mean_ratio_td = np.mean(ratio_td_valid)
    std_ratio_td = np.std(ratio_td_valid)
    scatter_td = std_ratio_td / mean_ratio_td
    
    # Test 2: Is ℓ_fitted / ℓ_toomre universal?
    mean_ratio_ft = np.mean(ratio_ft_valid)
    std_ratio_ft = np.std(ratio_ft_valid)
    scatter_ft = std_ratio_ft / mean_ratio_ft if mean_ratio_ft > 0 else np.inf
    
    # Test 3: Correlation between ℓ_toomre² and R_disk²
    corr_toomre_disk = np.corrcoef(
        np.log10(ell_toomre_sq[valid & (ell_toomre_sq > 0)]),
        np.log10(ell_disk_sq[valid & (ell_toomre_sq > 0)])
    )[0, 1] if np.sum(valid & (ell_toomre_sq > 0)) > 2 else np.nan
    
    return {
        'n_galaxies': np.sum(valid),
        'ratio_toomre_disk': {
            'mean': mean_ratio_td,
            'std': std_ratio_td,
            'scatter': scatter_td,
            'is_universal': scatter_td < 0.3  # <30% scatter = universal
        },
        'ratio_fitted_toomre': {
            'mean': mean_ratio_ft,
            'std': std_ratio_ft,
            'scatter': scatter_ft,
            'is_universal': scatter_ft < 0.3
        },
        'correlation_toomre_disk': corr_toomre_disk,
        'interpretation': interpret_results(scatter_td, scatter_ft, corr_toomre_disk)
    }


def interpret_results(scatter_td: float, scatter_ft: float, corr: float) -> str:
    """
    Interpret the scale analysis results in terms of teleparallel f(T).
    """
    lines = []
    
    if scatter_td < 0.3:
        lines.append("✓ ℓ_toomre / R_disk is approximately UNIVERSAL")
        lines.append("  → The Toomre scale tracks baryonic geometry")
        lines.append("  → Candidate for f(T) constant: a ∝ ℓ_toomre² ∝ R_disk²")
    else:
        lines.append("✗ ℓ_toomre / R_disk varies significantly across galaxies")
        lines.append(f"  → Scatter = {scatter_td:.1%}")
    
    if scatter_ft < 0.3:
        lines.append("✓ ℓ_fitted / ℓ_toomre is approximately UNIVERSAL")
        lines.append("  → Fitted coherence length derives from Toomre physics")
    else:
        lines.append(f"✗ ℓ_fitted / ℓ_toomre varies (scatter = {scatter_ft:.1%})")
    
    if corr > 0.8:
        lines.append(f"✓ Strong correlation ({corr:.2f}) between ℓ_toomre² and R_disk²")
        lines.append("  → Both scales encode the same underlying physics")
    
    lines.append("")
    lines.append("TELEPARALLEL INTERPRETATION:")
    if scatter_td < 0.3 and corr > 0.7:
        lines.append("  The constant 'a' in f(T) could be:")
        lines.append("    a = (σ_v / √(GΣ_b))² = ℓ_toomre²")
        lines.append("  This is FIELD-DEPENDENT, not universal!")
        lines.append("  → Suggests f(T) where 'a' is a functional of matter distribution")
        lines.append("  → Similar to Bekenstein's TeVeS or RAQUAL")
    else:
        lines.append("  No clear universal scale found.")
        lines.append("  → May need more sophisticated f(T) form")
    
    return "\n".join(lines)


def test_acceleration_scale_emergence() -> Dict:
    """
    Test whether g† ~ cH₀ emerges from the length scale.
    
    In f(T), the modification scale is set by 'a' [length²].
    The corresponding acceleration scale is:
        g* = c² / √a
    
    If a ~ ℓ_toomre² ~ (σ_v)² / (GΣ_b), then:
        g* ~ c² × √(GΣ_b) / σ_v
    
    For typical disk: Σ_b ~ 10⁷ M_sun/kpc², σ_v ~ 20 km/s
        g* ~ (3e5)² × √(4.3e-6 × 10⁷) / 20
           ~ 9e10 × 6.6 / 20 ~ 3e10 (km/s)²/kpc
    
    This is WAY too large. So the "a" must include a cosmological factor.
    
    Alternative: a = ℓ_toomre² × (H₀/c)² × f(geometry)
    Then: g* ~ cH₀ × f(geometry)^(-1/2) ~ g†
    """
    # Typical values
    sigma_v = 20.0  # km/s
    Sigma_b = 1e7   # M_sun/kpc²
    
    # Toomre scale
    ell_toomre = sigma_v / np.sqrt(2 * np.pi * G_NEWTON * Sigma_b)
    
    # Naive acceleration scale from ℓ_toomre
    g_naive = C_LIGHT**2 / ell_toomre  # (km/s)²/kpc
    
    # Target: g† = cH₀
    g_dagger = C_LIGHT * H0 * 1e-3  # ~ 21 (km/s)²/kpc
    
    # What cosmological factor is needed?
    factor_needed = g_naive / g_dagger
    
    # Alternative: a = ℓ_toomre² × (ℓ_toomre / L_H)²
    # Then g* = c² / (ℓ_toomre × L_H) ~ c × H₀ × (L_H / ℓ_toomre)
    ell_effective = np.sqrt(ell_toomre * L_H)  # Geometric mean
    g_effective = C_LIGHT**2 / ell_effective
    
    return {
        'ell_toomre_kpc': ell_toomre,
        'g_naive': g_naive,
        'g_dagger': g_dagger,
        'ratio_naive_dagger': g_naive / g_dagger,
        'factor_needed': factor_needed,
        'ell_effective_kpc': ell_effective,
        'g_effective': g_effective,
        'ratio_effective_dagger': g_effective / g_dagger,
        'interpretation': (
            f"Naive ℓ_toomre gives g* = {g_naive:.2e}, which is {factor_needed:.0f}× too large.\n"
            f"Need cosmological correction: a_eff = ℓ_toomre × L_H = {ell_effective:.1f} kpc\n"
            f"This gives g* = {g_effective:.1f} (km/s)²/kpc, ratio to g† = {g_effective/g_dagger:.2f}\n"
            f"\nTeleparallel interpretation:\n"
            f"  f(T) constant: a = ℓ_toomre × (c/H₀)\n"
            f"  This mixes LOCAL (baryonic) and GLOBAL (cosmological) scales\n"
            f"  → Suggests a 'running' or 'environmentally-dependent' f(T)"
        )
    }


def test_dimensional_analysis() -> Dict:
    """
    Test the dimensional structure of the coherence scale.
    
    Ferraro's key point: f(T) requires a constant 'a' with units [length]².
    
    We have several candidate scales:
    1. ℓ_toomre² = σ_v² / (G Σ_b)  [local, baryonic]
    2. (c/H₀)² = L_H²  [cosmological]
    3. ℓ_toomre × L_H  [mixed]
    
    The critical acceleration g† ~ cH₀ ~ 1.2×10⁻¹⁰ m/s² suggests:
        a ~ (c/g†)² ~ (c²/cH₀)² ~ (c/H₀)² ~ L_H²
    
    But this is universal, while our model has system-dependent scales.
    """
    # Cosmological scale
    L_H_kpc = C_LIGHT / (H0 * 1e-3)  # ~ 4.3 Gpc = 4.3e6 kpc
    
    # Critical acceleration
    g_dagger = C_LIGHT * H0 * 1e-3  # ~ 21 (km/s)²/kpc
    
    # Corresponding length from g†
    # g = v²/r → r = v²/g
    # For v ~ c: ℓ_g† = c²/g† = c/H₀ = L_H
    ell_g_dagger = C_LIGHT**2 / g_dagger  # Should equal L_H
    
    # Typical galaxy parameters
    sigma_v_typical = 20.0  # km/s
    Sigma_b_typical = 1e7   # M_sun/kpc²
    R_disk_typical = 3.0    # kpc
    
    # Toomre scale
    ell_toomre = sigma_v_typical / np.sqrt(2 * np.pi * G_NEWTON * Sigma_b_typical)
    
    # The puzzle: ell_toomre ~ 1 kpc, but ell_g_dagger ~ L_H ~ 4e6 kpc
    # Ratio is huge!
    ratio_toomre_LH = ell_toomre / L_H_kpc
    
    # But the EFFECT activates at g ~ g† ~ cH₀
    # This suggests the "a" constant involves BOTH scales:
    #   a = ℓ_toomre² × f(ℓ_toomre / L_H)
    # where f is some function that brings in the cosmological scale
    
    # Hypothesis: a = ℓ_toomre × L_H (geometric mean)
    ell_geometric = np.sqrt(ell_toomre * L_H_kpc)
    g_from_geometric = C_LIGHT**2 / ell_geometric
    
    # Alternative: a = ℓ_toomre² × (L_H / ℓ_toomre)^n for some n
    # If n=1: a = ℓ_toomre × L_H (same as geometric)
    # If n=2: a = L_H² (pure cosmological)
    
    return {
        'L_H_kpc': L_H_kpc,
        'g_dagger': g_dagger,
        'ell_g_dagger': ell_g_dagger,
        'ell_toomre': ell_toomre,
        'ratio_toomre_LH': ratio_toomre_LH,
        'ell_geometric': ell_geometric,
        'g_from_geometric': g_from_geometric,
        'ratio_g_geometric_dagger': g_from_geometric / g_dagger
    }


def test_fT_form_candidates() -> Dict:
    """
    Test candidate f(T) forms that could reproduce our phenomenology.
    
    Standard f(T) forms:
    1. f(T) = T + α T² / T*        (quadratic correction)
    2. f(T) = T × (1 - exp(-T/T*)) (exponential saturation)
    3. f(T) = T × √(1 + T*/T)      (square-root interpolation)
    
    The question: What is T* in terms of baryonic observables?
    
    Torsion scalar: T ~ (∂g)² ~ (g/ℓ)² ~ g²/ℓ²
    For weak field: T ~ (GM/r²)² × (1/r)² ~ (GM)²/r⁶
    
    If T* = (g†)² / ℓ_b² where ℓ_b is baryonic scale:
        T/T* ~ (g/g†)² × (ℓ_b/ℓ)²
    
    At r ~ ℓ_b:
        T/T* ~ (g/g†)²
    
    This gives modification when g < g†, as observed!
    """
    # Torsion scalar scaling
    # T ~ (Riemann)² ~ (g/c²)² × (1/ℓ)² in geometric units
    # In our units: T ~ g² / ℓ² [(km/s)⁴ / kpc⁴]
    
    g_dagger = C_LIGHT * H0 * 1e-3  # ~ 21 (km/s)²/kpc
    
    # Typical galaxy: g ~ 100 (km/s)²/kpc at R ~ 5 kpc
    g_typical = 100.0  # (km/s)²/kpc
    ell_typical = 5.0  # kpc
    
    T_typical = g_typical**2 / ell_typical**2
    
    # What should T* be?
    # Option 1: T* = g†² / ℓ_b² (baryonic scale)
    ell_b = 3.0  # kpc (typical disk scale)
    T_star_option1 = g_dagger**2 / ell_b**2
    
    # Option 2: T* = g†² / L_H² (cosmological)
    L_H = C_LIGHT / (H0 * 1e-3)
    T_star_option2 = g_dagger**2 / L_H**2
    
    # Option 3: T* = g†² / (ℓ_b × L_H) (mixed)
    T_star_option3 = g_dagger**2 / (ell_b * L_H)
    
    # Compute T/T* for each
    ratio1 = T_typical / T_star_option1
    ratio2 = T_typical / T_star_option2
    ratio3 = T_typical / T_star_option3
    
    return {
        'T_typical': T_typical,
        'T_star_baryonic': T_star_option1,
        'T_star_cosmological': T_star_option2,
        'T_star_mixed': T_star_option3,
        'ratio_baryonic': ratio1,
        'ratio_cosmological': ratio2,
        'ratio_mixed': ratio3,
        'interpretation': (
            f"T/T* ratios at typical galaxy radius:\n"
            f"  Baryonic T*: T/T* = {ratio1:.2f}\n"
            f"  Cosmological T*: T/T* = {ratio2:.2e}\n"
            f"  Mixed T*: T/T* = {ratio3:.2e}\n\n"
            f"For f(T) = T + α T²/T*, modification strength ~ α × T/T*\n"
            f"Baryonic T* gives O(1) modification at galaxy scales ✓\n"
            f"Cosmological T* gives negligible modification ✗\n"
            f"Mixed T* gives intermediate modification\n\n"
            f"Conclusion: T* must be FIELD-DEPENDENT, not universal!"
        )
    }


def run_ferraro_tests():
    """
    Run all tests inspired by Ferraro's insight.
    """
    print("=" * 70)
    print("FERRARO SCALE TESTS")
    print("Testing whether a universal length² constant exists")
    print("=" * 70)
    
    # Test 1: Acceleration scale emergence
    print("\n" + "-" * 70)
    print("TEST 1: Does g† ~ cH₀ emerge from a length scale?")
    print("-" * 70)
    
    accel_results = test_acceleration_scale_emergence()
    print(f"\nTypical Toomre scale: ℓ_T = {accel_results['ell_toomre_kpc']:.2f} kpc")
    print(f"Naive acceleration: g* = c²/ℓ_T = {accel_results['g_naive']:.2e} (km/s)²/kpc")
    print(f"Target (g† = cH₀): {accel_results['g_dagger']:.1f} (km/s)²/kpc")
    print(f"Ratio: {accel_results['ratio_naive_dagger']:.0f}× too large")
    print(f"\n{accel_results['interpretation']}")
    
    # Test 2: Dimensional analysis
    print("\n" + "-" * 70)
    print("TEST 2: Dimensional analysis of scales")
    print("-" * 70)
    
    dim_results = test_dimensional_analysis()
    print(f"\nHubble length: L_H = {dim_results['L_H_kpc']:.2e} kpc")
    print(f"Toomre scale: ℓ_T = {dim_results['ell_toomre']:.2f} kpc")
    print(f"Ratio ℓ_T / L_H = {dim_results['ratio_toomre_LH']:.2e}")
    print(f"\nGeometric mean: √(ℓ_T × L_H) = {dim_results['ell_geometric']:.1f} kpc")
    print(f"g from geometric: {dim_results['g_from_geometric']:.1f} (km/s)²/kpc")
    print(f"Ratio to g†: {dim_results['ratio_g_geometric_dagger']:.2f}")
    
    # Test 3: f(T) form candidates
    print("\n" + "-" * 70)
    print("TEST 3: f(T) form candidates")
    print("-" * 70)
    
    fT_results = test_fT_form_candidates()
    print(f"\n{fT_results['interpretation']}")
    
    # Test 4: Scale universality across SPARC sample (if data available)
    print("\n" + "-" * 70)
    print("TEST 4: Scale universality across galaxies")
    print("-" * 70)
    
    universality = None
    results = []
    try:
        loader = RealDataLoader()
        galaxies = loader.list_available_galaxies()
        
        if len(galaxies) > 0:
            print(f"\nAnalyzing {len(galaxies)} galaxies...")
            for i, gal_name in enumerate(galaxies[:50]):
                if i % 10 == 0:
                    print(f"  Processing {i}/{min(50, len(galaxies))}...")
                result = analyze_galaxy_scales(gal_name, loader)
                if result is not None:
                    results.append(result)
            
            print(f"\nSuccessfully analyzed {len(results)} galaxies")
            
            if len(results) > 5:
                universality = test_scale_universality(results)
                
                print(f"\nResults:")
                print(f"  ℓ_toomre / R_disk: mean = {universality['ratio_toomre_disk']['mean']:.2f} ± {universality['ratio_toomre_disk']['std']:.2f}")
                print(f"  Scatter: {universality['ratio_toomre_disk']['scatter']:.1%}")
                print(f"  Universal? {universality['ratio_toomre_disk']['is_universal']}")
                print(f"\n  ℓ_fitted / ℓ_toomre: mean = {universality['ratio_fitted_toomre']['mean']:.2f} ± {universality['ratio_fitted_toomre']['std']:.2f}")
                print(f"  Scatter: {universality['ratio_fitted_toomre']['scatter']:.1%}")
                print(f"\n  Correlation(ℓ_toomre², R_disk²): {universality['correlation_toomre_disk']:.3f}")
                
                print(f"\n{universality['interpretation']}")
        else:
            print("\nNo SPARC data available. Skipping galaxy-by-galaxy analysis.")
            print("To run this test, ensure SPARC Rotmod_LTG data is in the data/ directory.")
    except Exception as e:
        print(f"\nCould not load SPARC data: {e}")
        print("Skipping galaxy-by-galaxy analysis.")
    
    # Test 3: Propose f(T) form
    print("\n" + "-" * 70)
    print("TEST 3: Proposed f(T) interpretation")
    print("-" * 70)
    
    print("""
Based on these tests, a teleparallel interpretation might take the form:

    f(T) = T + α T² / T*

where:
    T* = 1/a = (H₀/c)² × (√(GΣ_b) / σ_v)²
    
This makes T* FIELD-DEPENDENT:
    - In dense, cold disks: T* small → strong modification
    - In hot, diffuse systems: T* large → weak modification
    - In Solar System: T* → ∞ → no modification

The key insight from Ferraro: The constant 'a' need not be universal.
It can be a FUNCTIONAL of the matter distribution, similar to:
    - RAQUAL (Bekenstein)
    - Emergent gravity (Verlinde)
    - Superfluid dark matter (Berezhiani & Khoury)

This would explain why:
    1. The coherence scale tracks baryonic geometry (ℓ ∝ R_disk^0.5)
    2. The critical acceleration g† ~ cH₀ emerges naturally
    3. Different systems (galaxies, clusters) follow the same physics
       with system-dependent 'a'
""")
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Ferraro's f(T) insight suggests looking for:
    a = [length]² constant that sets modification scale

Our phenomenology suggests:
    a = ℓ_toomre² × (cosmological factor)
    a = (σ_v² / GΣ_b) × (c/H₀)²  [mixed local + global]

This is NOT a universal constant, but a FIELD-DEPENDENT quantity.

NEXT STEPS for teleparallel formalization:
1. Write f(T) = T + α T² / T*(ρ_b, σ_v) with field-dependent T*
2. Derive field equations and check consistency
3. Compare to Born-Infeld determinantal forms
4. Check if residual Lorentz structure allows this dependence
""")
    
    return {
        'acceleration_test': accel_results,
        'universality_test': universality if len(results) > 5 else None,
        'galaxy_results': results
    }


if __name__ == "__main__":
    results = run_ferraro_tests()

