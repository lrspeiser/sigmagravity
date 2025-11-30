#!/usr/bin/env python3
"""
Test Derived Unified Formula WITH Gates on SPARC
=================================================

This script tests whether the theoretically-derived formula can match
or beat the empirical formula when gates are properly included.

KEY FINDING FROM ANALYSIS:
- Empirical (no gates): 0.104 dex
- Derived (no gates): 0.093 dex  <-- DERIVED WINS!

- Empirical (with gates): 0.0854 dex (paper result)
- Derived (with gates): ??? (THIS TEST)

If derived + gates achieves ~0.08 dex, it becomes the preferred formula.

Author: Sigma Gravity Team
Date: November 2025
"""

import numpy as np
from pathlib import Path
import sys

# Physical constants
c = 2.998e8
H0_SI = 70 * 1000 / 3.086e22  # H0 in SI units
G = 6.674e-11
kpc_to_m = 3.086e19

g_dagger = c * H0_SI / (2 * np.e)

print("=" * 80)
print("DERIVED FORMULA + GATES TEST ON SPARC")
print("=" * 80)
print(f"\ng† = {g_dagger:.3e} m/s²")

# =============================================================================
# THE FORMULAS TO COMPARE
# =============================================================================

def h_universal(g):
    """
    Universal h(g) from coherence theory.
    
    h(g) = √(g†/g) × g†/(g†+g)
    """
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def h_empirical(g, p=0.757):
    """
    Empirical power-law h(g).
    
    h(g) = (g†/g)^p
    """
    g = np.maximum(g, 1e-15)
    return (g_dagger / g) ** p

# =============================================================================
# GATES (same for both formulas)
# =============================================================================

def G_coherence(R, ell_0=4.993, n_coh=0.5):
    """
    Coherence window gate.
    
    Empirical form: (ℓ₀/(ℓ₀+R))^n_coh
    """
    return (ell_0 / (ell_0 + R)) ** n_coh

def W_derived(r, R_d):
    """
    Derived coherence window.
    
    W(r) = 1 - (ξ/(ξ+r))^0.5
    where ξ = (2/3) × R_d
    """
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5

def G_winding(R, v_c, t_age=10.0, N_crit=150.0, power=2.0):
    """
    Spiral winding gate.
    
    Suppresses coherence after ~N_crit orbits.
    Paper uses N_crit = 150 (less aggressive than theoretical N_crit=10).
    """
    R_safe = np.maximum(R, 0.1)
    v_safe = np.maximum(v_c, 1.0)
    N_orbits = t_age * v_safe / (2.0 * np.pi * R_safe * 0.978)
    return np.exp(-N_orbits / N_crit)  # Exponential form from paper

def G_bulge(bulge_frac, alpha=0.3):
    """
    Bulge suppression gate.
    
    Suppresses coherence in bulge-dominated regions.
    """
    return np.exp(-alpha * bulge_frac)

def G_solar_system(R, R_gate=0.5):
    """
    Solar system safety gate.
    
    Ensures no anomalous effects at small R.
    """
    return 1 - np.exp(-(R / R_gate)**2)

# =============================================================================
# KERNEL FUNCTIONS
# =============================================================================

def kernel_empirical(R, g_bar, A0=0.591, p=0.757, ell_0=4.993, n_coh=0.5, 
                      v_c=None, bulge_frac=None, use_gates=True):
    """
    Empirical kernel from paper.
    
    K = A₀ × (g†/g)^p × (ℓ₀/(ℓ₀+R))^n × gates
    """
    K = A0 * h_empirical(g_bar, p) * G_coherence(R, ell_0, n_coh)
    
    if use_gates:
        K *= G_solar_system(R)
        if v_c is not None:
            K *= G_winding(R, v_c)
        if bulge_frac is not None:
            K *= G_bulge(bulge_frac)
    
    return K

def kernel_derived(R, g_bar, A=None, R_d=3.0, 
                    v_c=None, bulge_frac=None, use_gates=True, 
                    use_W_derived=True):
    """
    Derived kernel from coherence theory.
    
    K = A × W(r) × h(g)
    
    where h(g) = √(g†/g) × g†/(g†+g)
    
    Parameters:
    -----------
    A : float, optional
        Amplitude. If None, use √2 (default) or √3 (test)
    R_d : float
        Disk scale length [kpc]
    use_W_derived : bool
        If True, use W(r) = 1 - (ξ/(ξ+r))^0.5
        If False, use empirical coherence window
    """
    if A is None:
        A = np.sqrt(2)  # Default from theory
    
    h = h_universal(g_bar)
    
    if use_W_derived:
        W = W_derived(R, R_d)
    else:
        # Use empirical coherence window for fair comparison
        W = G_coherence(R, ell_0=4.993, n_coh=0.5)
    
    K = A * W * h
    
    if use_gates:
        K *= G_solar_system(R)
        if v_c is not None:
            K *= G_winding(R, v_c)
        if bulge_frac is not None:
            K *= G_bulge(bulge_frac)
    
    return K

# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_sparc_data(sparc_dir):
    """Load SPARC rotation curve data."""
    galaxies = {}
    sparc_dir = Path(sparc_dir)
    
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
            np.sign(V_gas) * V_gas**2 + 
            np.sign(V_disk) * V_disk**2 + 
            V_bulge**2
        )
        
        # Compute bulge fraction at each radius
        total_squared = V_disk**2 + V_bulge**2 + np.abs(V_gas**2)
        bulge_frac = np.where(total_squared > 0, V_bulge**2 / total_squared, 0)
        
        galaxies[name] = {
            'R': R,
            'V_obs': V_obs,
            'V_err': V_err,
            'V_gas': V_gas,
            'V_disk': V_disk,
            'V_bulge': V_bulge,
            'V_bar': V_bar,
            'bulge_frac': bulge_frac
        }
    
    return galaxies

def load_sparc_master(master_file):
    """Load SPARC master table for R_d values."""
    R_d_values = {}
    
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('-------'):
            data_start = i + 1
            break
    
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('#'):
            continue
        if len(line) < 67:
            continue
        try:
            name = line[0:11].strip()
            Rdisk_str = line[62:67].strip()
            if name and Rdisk_str:
                R_d_values[name] = float(Rdisk_str)
        except:
            continue
    
    return R_d_values

# =============================================================================
# COMPUTE SCATTER
# =============================================================================

def compute_scatter(galaxies, kernel_func, R_d_values=None, use_gates=True, 
                     use_W_derived=True, A=None):
    """
    Compute RAR scatter across all galaxies.
    
    Returns scatter in dex.
    """
    all_log_residuals = []
    
    for name, data in galaxies.items():
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        bulge_frac = data['bulge_frac']
        
        # Quality cuts
        mask = (R > 0.5) & (V_bar > 5) & (V_obs > 5)
        if np.sum(mask) < 3:
            continue
        
        R = R[mask]
        V_obs = V_obs[mask]
        V_bar = V_bar[mask]
        bulge_frac = bulge_frac[mask]
        
        # Get R_d for this galaxy
        R_d = R_d_values.get(name, 3.0) if R_d_values else 3.0
        
        # Compute baryonic acceleration
        g_bar = (V_bar * 1000)**2 / (R * kpc_to_m)  # m/s²
        
        # Get circular velocity for winding gate
        v_c = np.sqrt(V_obs**2 + 10)  # km/s with floor
        
        # Compute kernel
        if 'derived' in kernel_func.__name__ or kernel_func == kernel_derived:
            K = kernel_func(R, g_bar, A=A, R_d=R_d, v_c=v_c, 
                           bulge_frac=bulge_frac, use_gates=use_gates,
                           use_W_derived=use_W_derived)
        else:
            K = kernel_func(R, g_bar, v_c=v_c, bulge_frac=bulge_frac, 
                           use_gates=use_gates)
        
        # Predicted velocity
        V_pred = V_bar * np.sqrt(1 + K)
        
        # Log residual
        mask_good = (V_pred > 0) & (V_obs > 0)
        log_residual = np.log10(V_obs[mask_good] / V_pred[mask_good])
        all_log_residuals.extend(log_residual)
    
    all_log_residuals = np.array(all_log_residuals)
    scatter_dex = np.std(all_log_residuals)
    bias_dex = np.mean(all_log_residuals)
    
    return scatter_dex, bias_dex, len(all_log_residuals)

# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    # Find SPARC data
    sparc_paths = [
        Path(r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"),
        Path(r"C:\Users\henry\dev\sigmagravity\data\SPARC"),
        Path(r"C:\Users\henry\dev\sigmagravity\coherence-field-theory\data\SPARC"),
    ]
    
    sparc_dir = None
    for p in sparc_paths:
        if p.exists():
            sparc_dir = p
            break
    
    if sparc_dir is None:
        print("\nERROR: SPARC data not found!")
        sys.exit(1)
    
    print(f"\nLoading SPARC data from: {sparc_dir}")
    galaxies = load_sparc_data(sparc_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Load R_d values
    master_paths = [
        sparc_dir / 'SPARC_Lelli2016c.mrt',
        sparc_dir.parent / 'SPARC_Lelli2016c.mrt',
    ]
    R_d_values = {}
    for p in master_paths:
        if p.exists():
            R_d_values = load_sparc_master(p)
            print(f"Loaded R_d for {len(R_d_values)} galaxies")
            break
    
    # ==========================================================================
    # TEST 1: Reproduce known results (no gates)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: REPRODUCE KNOWN RESULTS (NO GATES)")
    print("=" * 80)
    
    scatter_emp_no_gates, bias_emp, n_emp = compute_scatter(
        galaxies, kernel_empirical, R_d_values, use_gates=False
    )
    print(f"\nEmpirical (no gates):  {scatter_emp_no_gates:.4f} dex (bias: {bias_emp:+.4f}), {n_emp} points")
    
    scatter_der_no_gates, bias_der, n_der = compute_scatter(
        galaxies, kernel_derived, R_d_values, use_gates=False, use_W_derived=True
    )
    print(f"Derived (no gates):    {scatter_der_no_gates:.4f} dex (bias: {bias_der:+.4f}), {n_der} points")
    
    winner = "DERIVED" if scatter_der_no_gates < scatter_emp_no_gates else "EMPIRICAL"
    improvement = (scatter_emp_no_gates - scatter_der_no_gates) / scatter_emp_no_gates * 100
    print(f"\nWinner (no gates): {winner} by {abs(improvement):.1f}%")
    
    # ==========================================================================
    # TEST 2: Both formulas WITH gates
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: BOTH FORMULAS WITH GATES")
    print("=" * 80)
    
    scatter_emp_gates, bias_emp_g, n_emp_g = compute_scatter(
        galaxies, kernel_empirical, R_d_values, use_gates=True
    )
    print(f"\nEmpirical (with gates): {scatter_emp_gates:.4f} dex (bias: {bias_emp_g:+.4f})")
    
    scatter_der_gates, bias_der_g, n_der_g = compute_scatter(
        galaxies, kernel_derived, R_d_values, use_gates=True, use_W_derived=True
    )
    print(f"Derived (with gates):   {scatter_der_gates:.4f} dex (bias: {bias_der_g:+.4f})")
    
    winner_gates = "DERIVED" if scatter_der_gates < scatter_emp_gates else "EMPIRICAL"
    improvement_gates = (scatter_emp_gates - scatter_der_gates) / scatter_emp_gates * 100
    print(f"\nWinner (with gates): {winner_gates} by {abs(improvement_gates):.1f}%")
    
    # ==========================================================================
    # TEST 3: Amplitude optimization for derived formula
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: OPTIMAL AMPLITUDE FOR DERIVED FORMULA")
    print("=" * 80)
    
    A_values = [1.2, 1.3, 1.41, 1.5, 1.6, 1.73, 1.8, 2.0]
    print(f"\n{'A':<10} {'Name':<15} {'Scatter (dex)':<15} {'Bias':<10}")
    print("-" * 50)
    
    best_A = None
    best_scatter = float('inf')
    
    for A in A_values:
        name = ""
        if abs(A - np.sqrt(2)) < 0.02:
            name = "√2"
        elif abs(A - np.sqrt(3)) < 0.02:
            name = "√3"
        elif abs(A - 1.73) < 0.02:
            name = "~√3"
        
        scatter, bias, _ = compute_scatter(
            galaxies, kernel_derived, R_d_values, use_gates=True, 
            use_W_derived=True, A=A
        )
        print(f"{A:<10.3f} {name:<15} {scatter:<15.4f} {bias:+.4f}")
        
        if scatter < best_scatter:
            best_scatter = scatter
            best_A = A
    
    print(f"\nOptimal A = {best_A:.3f} with scatter = {best_scatter:.4f} dex")
    
    # Check if √3 is special
    sqrt3_scatter, _, _ = compute_scatter(
        galaxies, kernel_derived, R_d_values, use_gates=True, 
        use_W_derived=True, A=np.sqrt(3)
    )
    print(f"A = √3 = {np.sqrt(3):.4f} gives scatter = {sqrt3_scatter:.4f} dex")
    
    # ==========================================================================
    # TEST 4: Hybrid formula (derived W with empirical coherence)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: HYBRID FORMULAS")
    print("=" * 80)
    
    # Derived h(g) with empirical coherence window
    scatter_hybrid1, bias_h1, _ = compute_scatter(
        galaxies, kernel_derived, R_d_values, use_gates=True, 
        use_W_derived=False, A=np.sqrt(2)
    )
    print(f"\nDerived h(g) + empirical W: {scatter_hybrid1:.4f} dex (bias: {bias_h1:+.4f})")
    
    scatter_hybrid2, bias_h2, _ = compute_scatter(
        galaxies, kernel_derived, R_d_values, use_gates=True, 
        use_W_derived=False, A=best_A
    )
    print(f"Derived h(g) + empirical W + optimal A={best_A:.2f}: {scatter_hybrid2:.4f} dex (bias: {bias_h2:+.4f})")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"""
FORMULA COMPARISON:

WITHOUT GATES:
  Empirical: {scatter_emp_no_gates:.4f} dex
  Derived:   {scatter_der_no_gates:.4f} dex
  Winner:    {winner} by {abs(improvement):.1f}%

WITH GATES:
  Empirical: {scatter_emp_gates:.4f} dex
  Derived:   {scatter_der_gates:.4f} dex
  Winner:    {winner_gates} by {abs(improvement_gates):.1f}%

OPTIMIZED DERIVED:
  A = {best_A:.3f}: {best_scatter:.4f} dex
  A = √3:    {sqrt3_scatter:.4f} dex

PAPER RESULT: 0.0854 dex (empirical with all gates)

CONCLUSIONS:
""")

    if scatter_der_gates < scatter_emp_gates:
        print("✓ Derived formula BEATS empirical even with gates!")
        print(f"  Improvement: {abs(improvement_gates):.1f}%")
    else:
        print("○ Empirical formula still wins with gates")
        print(f"  But derived is close: only {abs(improvement_gates):.1f}% worse")
    
    if best_scatter < scatter_emp_gates:
        print(f"\n✓ Optimized derived (A={best_A:.2f}) achieves {best_scatter:.4f} dex")
        if abs(best_A - np.sqrt(3)) < 0.1:
            print("  And A ≈ √3 is potentially derivable!")
    
    print(f"""
RECOMMENDED NEXT STEPS:
1. If A_optimal ≈ √3, derive this from 3D geometry correction
2. Test if optimal A varies with galaxy type
3. Compare to MOND and ΛCDM at matched parameter count
""")
