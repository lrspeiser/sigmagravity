"""
CRITICAL TEST: Derived Formula vs Empirical Formula on SPARC
=============================================================

This script tests whether the DERIVED formula from first-principles
actually fits the SPARC data as well as the EMPIRICAL formula.

DERIVED FORMULA (from teleparallel gravity derivation):
    Σ = 1 + √2 × W(r) × h(g)
    where:
        W(r) = 1 - (ξ/(ξ+r))^0.5       [ξ = (2/3)×R_d, galaxy-dependent]
        h(g) = √(g†/g) × g†/(g†+g)     [p_eff = 0.5 at low g]
        g† = cH₀/(2e) = 1.2×10⁻¹⁰ m/s²

EMPIRICAL FORMULA (best-fit to SPARC):
    Σ = 1 + A₀ × C(r) × (g†/g)^p
    where:
        C(r) = 1 - [1 + (r/ℓ₀)^p]^(-n_coh)  [Burr-XII]
        A₀ = 0.591, ℓ₀ = 5 kpc, p = 0.757, n_coh = 0.5

QUESTION: Does the derived formula fit as well as the empirical one?

Author: Sigma Gravity Team
Date: November 2025
"""

import numpy as np
from pathlib import Path
import json

# Physical constants
c = 2.998e8           # m/s
H0 = 2.184e-18        # s^-1 (70 km/s/Mpc)
kpc_to_m = 3.086e19   # m per kpc
e = np.e

# Derived g†
g_dagger = c * H0 / (2 * e)  # = 1.204e-10 m/s²

print("=" * 80)
print("CRITICAL TEST: Derived vs Empirical Formula on SPARC")
print("=" * 80)
print(f"\ng† = cH₀/(2e) = {g_dagger:.4e} m/s²")

# =============================================================================
# FORMULA DEFINITIONS
# =============================================================================

def h_derived(g):
    """
    DERIVED h(g) from teleparallel gravity / geometric mean argument.
    
    h(g) = √(g†/g) × g†/(g†+g)
    
    This gives:
    - At g << g†: h ~ √(g†/g) (MOND deep limit)
    - At g >> g†: h ~ g†/g (GR limit)
    """
    g_safe = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g_safe) * g_dagger / (g_dagger + g_safe)

def W_derived(r, R_d):
    """
    DERIVED coherence window with galaxy-dependent ξ.
    
    W(r) = 1 - (ξ/(ξ+r))^0.5
    
    where ξ = (2/3) × R_d (derived from torsion gradient argument)
    """
    xi = (2/3) * R_d  # Galaxy-dependent coherence length
    return 1 - (xi / (xi + r))**0.5

def Sigma_derived(r, g, R_d, A_max=np.sqrt(2)):
    """
    DERIVED enhancement formula.
    
    Σ = 1 + A_max × W(r) × h(g)
    
    Parameters:
    - A_max = √2 (from graviton polarization / quadrature addition)
    - W(r) = coherence window with galaxy-dependent ξ = (2/3)×R_d
    - h(g) = √(g†/g) × g†/(g†+g) (from geometric mean of torsion)
    """
    W = W_derived(r, R_d)
    h = h_derived(g)
    return 1 + A_max * W * h

def Sigma_empirical(r, g, A0=0.591, ell0=5.0, p=0.757, n_coh=0.5):
    """
    EMPIRICAL best-fit formula (Burr-XII coherence).
    
    Σ = 1 + A₀ × C(r) × (g†/g)^p
    
    where C(r) = 1 - [1 + (r/ℓ₀)^p]^(-n_coh)
    """
    g_safe = np.maximum(g, 1e-15)
    C = 1 - (1 + (r/ell0)**p)**(-n_coh)
    H = (g_dagger / g_safe)**p
    return 1 + A0 * C * H

# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_data(sparc_dir):
    """Load SPARC rotation curves with scale radii."""
    galaxies = {}
    sparc_dir = Path(sparc_dir)
    
    # Also load galaxy properties for R_d
    properties_file = sparc_dir.parent / "SPARC_Lelli2016c.mrt"
    
    # Default R_d values (will be overwritten if properties file exists)
    R_d_dict = {}
    
    # Try to load scale radii from properties file
    if properties_file.exists():
        try:
            with open(properties_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 10:  # Standard SPARC format
                        name = parts[0]
                        try:
                            R_d = float(parts[4])  # Scale length in kpc
                            R_d_dict[name] = R_d
                        except (ValueError, IndexError):
                            pass
        except Exception as e:
            print(f"Warning: Could not load properties file: {e}")
    
    # Load rotation curves
    for rotmod_file in sparc_dir.glob('*_rotmod.dat'):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
        
        with open(rotmod_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
        
        if len(R) < 3:
            continue
        
        R = np.array(R)
        V_obs = np.array(V_obs)
        V_err = np.array(V_err)
        V_gas = np.array(V_gas)
        V_disk = np.array(V_disk)
        V_bulge = np.array(V_bulge)
        
        # Compute V_bar
        V_bar = np.sqrt(
            np.abs(V_gas) * V_gas +  # Preserve sign
            np.abs(V_disk) * V_disk + 
            V_bulge**2
        )
        
        # Get R_d (default to R_max/4 if not available)
        R_d = R_d_dict.get(name, np.max(R) / 4)
        
        galaxies[name] = {
            'R': R,
            'V_obs': V_obs,
            'V_err': V_err,
            'V_bar': V_bar,
            'R_d': R_d
        }
    
    return galaxies

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def predict_velocity(R, V_bar, formula='empirical', R_d=3.0, **kwargs):
    """
    Predict rotation curve using specified formula.
    
    Returns V_pred and Σ values.
    """
    # Convert V_bar to g_bar
    R_safe = np.maximum(R, 0.1)
    g_bar_kpc = V_bar**2 / R_safe  # (km/s)²/kpc
    g_bar_mks = g_bar_kpc * 1e6 / kpc_to_m  # m/s²
    
    if formula == 'derived':
        Sigma = Sigma_derived(R, g_bar_mks, R_d, **kwargs)
    else:
        Sigma = Sigma_empirical(R, g_bar_mks, **kwargs)
    
    V_pred = V_bar * np.sqrt(Sigma)
    return V_pred, Sigma

def compute_scatter(galaxies, formula='empirical', **kwargs):
    """
    Compute RAR scatter for a given formula.
    
    Returns scatter (dex), bias (dex), and per-galaxy results.
    """
    all_residuals = []
    galaxy_results = {}
    
    for name, data in galaxies.items():
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        R_d = data['R_d']
        
        # Quality cuts
        mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5)
        if np.sum(mask) < 3:
            continue
        
        R_cut = R[mask]
        V_obs_cut = V_obs[mask]
        V_bar_cut = V_bar[mask]
        
        # Predict
        V_pred, Sigma = predict_velocity(
            R_cut, V_bar_cut, 
            formula=formula, 
            R_d=R_d,
            **kwargs
        )
        
        # Compute residuals (in dex)
        residuals = np.log10(V_pred / V_obs_cut)
        all_residuals.extend(residuals)
        
        galaxy_results[name] = {
            'scatter': np.std(residuals),
            'bias': np.mean(residuals),
            'n_points': len(residuals)
        }
    
    total_scatter = np.std(all_residuals)
    total_bias = np.mean(all_residuals)
    
    return total_scatter, total_bias, galaxy_results

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    # Load SPARC data
    sparc_dir = Path("C:/Users/henry/dev/sigmagravity/data/Rotmod_LTG")
    
    if not sparc_dir.exists():
        print(f"\nERROR: SPARC data not found at {sparc_dir}")
        return None
    
    print(f"\nLoading SPARC data from {sparc_dir}...")
    galaxies = load_sparc_data(sparc_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # ==========================================================================
    # TEST 1: Compare formulas with their canonical parameters
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 1: CANONICAL FORMULAS")
    print("=" * 80)
    
    # Derived formula with canonical A_max = √2
    scatter_derived, bias_derived, _ = compute_scatter(
        galaxies, 
        formula='derived',
        A_max=np.sqrt(2)
    )
    
    # Empirical formula with best-fit parameters
    scatter_empirical, bias_empirical, _ = compute_scatter(
        galaxies,
        formula='empirical',
        A0=0.591, ell0=5.0, p=0.757, n_coh=0.5
    )
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                     DERIVED vs EMPIRICAL FORMULA COMPARISON                    ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║   DERIVED (first-principles):                                                 ║
    ║     Σ = 1 + √2 × [1-(ξ/(ξ+r))^0.5] × √(g†/g) × g†/(g†+g)                    ║
    ║     where ξ = (2/3)×R_d (galaxy-dependent)                                    ║
    ║                                                                               ║
    ║     RAR Scatter: {scatter_derived:.4f} dex                                              ║
    ║     RAR Bias:    {bias_derived:+.4f} dex                                              ║
    ║                                                                               ║
    ╠───────────────────────────────────────────────────────────────────────────────╣
    ║                                                                               ║
    ║   EMPIRICAL (best-fit):                                                       ║
    ║     Σ = 1 + 0.591 × [1-(1+(r/5)^0.757)^(-0.5)] × (g†/g)^0.757               ║
    ║     (universal ℓ₀ = 5 kpc)                                                    ║
    ║                                                                               ║
    ║     RAR Scatter: {scatter_empirical:.4f} dex                                              ║
    ║     RAR Bias:    {bias_empirical:+.4f} dex                                              ║
    ║                                                                               ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║   COMPARISON:                                                                 ║
    ║     Derived scatter / Empirical scatter = {scatter_derived/scatter_empirical:.2f}                             ║
    ║     Difference: {100*(scatter_derived - scatter_empirical)/scatter_empirical:+.1f}%                                                       ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ==========================================================================
    # TEST 2: Optimize A_max for derived formula
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 2: OPTIMIZING A_max FOR DERIVED FORMULA")
    print("=" * 80)
    
    A_max_values = np.linspace(0.3, 2.0, 50)
    best_A_max = np.sqrt(2)
    best_scatter = scatter_derived
    
    print("\nScanning A_max...")
    for A_max in A_max_values:
        scatter, bias, _ = compute_scatter(galaxies, formula='derived', A_max=A_max)
        if scatter < best_scatter:
            best_scatter = scatter
            best_A_max = A_max
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║   OPTIMIZED A_max FOR DERIVED FORMULA                                         ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║   Theoretical A_max:  √2 = {np.sqrt(2):.4f}                                            ║
    ║   Optimal A_max:      {best_A_max:.4f}                                                  ║
    ║   Ratio:              {best_A_max/np.sqrt(2):.3f}                                                  ║
    ║                                                                               ║
    ║   With optimal A_max:                                                         ║
    ║     RAR Scatter: {best_scatter:.4f} dex                                               ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ==========================================================================
    # TEST 3: Try different ξ scalings
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 3: TESTING ξ SCALING (galaxy-dependent vs universal)")
    print("=" * 80)
    
    # Test various ξ models
    xi_models = {
        'Derived: ξ = (2/3)×R_d': lambda R_d: (2/3) * R_d,
        'Alternative: ξ = R_d': lambda R_d: R_d,
        'Alternative: ξ = 2×R_d': lambda R_d: 2 * R_d,
        'Universal: ξ = 5 kpc': lambda R_d: 5.0,
        'Universal: ξ = 2 kpc': lambda R_d: 2.0,
    }
    
    print("\n  Testing different ξ models with A_max = √2...\n")
    print(f"  {'Model':<35} {'Scatter (dex)':<15} {'Bias (dex)':<15}")
    print("  " + "-" * 65)
    
    xi_results = {}
    for model_name, xi_func in xi_models.items():
        # Temporarily modify W_derived to use different ξ
        def W_custom(r, R_d, xi_func=xi_func):
            xi = xi_func(R_d)
            return 1 - (xi / (xi + r))**0.5
        
        # Custom Sigma with this ξ model
        def Sigma_custom(r, g, R_d, A_max=np.sqrt(2)):
            W = W_custom(r, R_d)
            h = h_derived(g)
            return 1 + A_max * W * h
        
        # Compute scatter
        all_residuals = []
        for name, data in galaxies.items():
            R = data['R']
            V_obs = data['V_obs']
            V_bar = data['V_bar']
            R_d = data['R_d']
            
            mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5)
            if np.sum(mask) < 3:
                continue
            
            R_cut, V_obs_cut, V_bar_cut = R[mask], V_obs[mask], V_bar[mask]
            R_safe = np.maximum(R_cut, 0.1)
            g_bar = (V_bar_cut**2 / R_safe) * 1e6 / kpc_to_m
            
            Sigma = Sigma_custom(R_cut, g_bar, R_d)
            V_pred = V_bar_cut * np.sqrt(Sigma)
            residuals = np.log10(V_pred / V_obs_cut)
            all_residuals.extend(residuals)
        
        scatter = np.std(all_residuals)
        bias = np.mean(all_residuals)
        xi_results[model_name] = (scatter, bias)
        print(f"  {model_name:<35} {scatter:<15.4f} {bias:<+15.4f}")
    
    # ==========================================================================
    # TEST 4: Pure MOND comparison
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("TEST 4: COMPARISON TO PURE MOND")
    print("=" * 80)
    
    def Sigma_MOND(r, g, a0=1.2e-10):
        """Standard MOND interpolating function."""
        g_safe = np.maximum(g, 1e-15)
        y = g_safe / a0
        # Simple interpolating function
        nu = 0.5 + 0.5 * np.sqrt(1 + 4/y)
        return nu
    
    # Compute MOND scatter
    all_mond_residuals = []
    for name, data in galaxies.items():
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        
        mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5)
        if np.sum(mask) < 3:
            continue
        
        R_cut, V_obs_cut, V_bar_cut = R[mask], V_obs[mask], V_bar[mask]
        R_safe = np.maximum(R_cut, 0.1)
        g_bar = (V_bar_cut**2 / R_safe) * 1e6 / kpc_to_m
        
        Sigma_m = Sigma_MOND(R_cut, g_bar)
        V_pred = V_bar_cut * np.sqrt(Sigma_m)
        residuals = np.log10(V_pred / V_obs_cut)
        all_mond_residuals.extend(residuals)
    
    scatter_mond = np.std(all_mond_residuals)
    bias_mond = np.mean(all_mond_residuals)
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║   COMPARISON TO MOND                                                          ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║   Pure MOND (simple ν):    {scatter_mond:.4f} dex scatter, {bias_mond:+.4f} dex bias          ║
    ║   Empirical Σ-Gravity:    {scatter_empirical:.4f} dex scatter, {bias_empirical:+.4f} dex bias          ║
    ║   Derived Σ-Gravity:      {scatter_derived:.4f} dex scatter, {bias_derived:+.4f} dex bias          ║
    ║   Derived (opt A_max):    {best_scatter:.4f} dex scatter                                ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    derived_vs_empirical = scatter_derived / scatter_empirical
    derived_opt_vs_empirical = best_scatter / scatter_empirical
    
    if derived_vs_empirical < 1.1:
        verdict = "SUCCESS: Derived formula performs comparably to empirical"
    elif derived_vs_empirical < 1.3:
        verdict = "PARTIAL: Derived formula within 30% of empirical"
    else:
        verdict = "DIFFERENT: Derived formula significantly worse than empirical"
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║   SUMMARY: DERIVED FORMULA vs EMPIRICAL                                       ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║   Derived (canonical A_max=√2):   {scatter_derived:.4f} dex                             ║
    ║   Derived (optimal A_max={best_A_max:.2f}):   {best_scatter:.4f} dex                             ║
    ║   Empirical (A₀=0.591):           {scatter_empirical:.4f} dex                             ║
    ║                                                                               ║
    ║   Ratio (derived/empirical): {derived_vs_empirical:.2f}×                                        ║
    ║   Ratio (optimized/empirical): {derived_opt_vs_empirical:.2f}×                                      ║
    ║                                                                               ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║   VERDICT: {verdict:<58}║
    ║                                                                               ║
    ║   KEY DIFFERENCES:                                                            ║
    ║   - Amplitude: A_max=√2=1.414 (derived) vs A₀=0.591 (empirical) → 2.4×       ║
    ║   - Exponent: p~0.5 (derived at low g) vs p=0.757 (empirical)                ║
    ║   - Coherence: ξ = (2/3)R_d (derived) vs ℓ₀=5 kpc (empirical)               ║
    ║                                                                               ║
    ║   INTERPRETATION:                                                             ║
    ║   The derived formula captures the STRUCTURE but not exact coefficients.      ║
    ║   This is similar to deriving Newton's law but not knowing G.                 ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Save results
    results = {
        'scatter_derived_canonical': scatter_derived,
        'scatter_derived_optimized': best_scatter,
        'scatter_empirical': scatter_empirical,
        'scatter_mond': scatter_mond,
        'optimal_A_max': best_A_max,
        'theoretical_A_max': np.sqrt(2),
        'ratio_derived_empirical': derived_vs_empirical,
        'ratio_optimized_empirical': derived_opt_vs_empirical,
        'xi_model_results': {k: {'scatter': v[0], 'bias': v[1]} for k, v in xi_results.items()}
    }
    
    output_path = Path(__file__).parent / "derived_vs_empirical_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results

if __name__ == "__main__":
    main()
