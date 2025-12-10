"""
Viability Scanner for Symmetron Coherence Gravity
==================================================

Systematically test (μ, λ, M, V₀, β) parameter space against three filters:
1. Cosmology: Ω_m ≈ 0.3, Ω_φ ≈ 0.7 at z=0
2. Galaxy screening: R_c ~ kpc (not >> Mpc, not << kpc)
3. Solar System: PPN safe (φ ≈ 0 due to high density)

Goal: Find ANY parameter set that passes all three simultaneously.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmology.symmetron_potential import (
    SymmetronParams, dark_energy_fraction, critical_density, d2V_dphi2
)


# ==============================================================================
# FILTER FUNCTIONS
# ==============================================================================

def passes_cosmology(params: SymmetronParams, 
                    Omega_m_range: Tuple[float, float] = (0.25, 0.35),
                    Omega_phi_range: Tuple[float, float] = (0.65, 0.75)) -> Tuple[bool, Dict]:
    """
    Test cosmology: Ω_m and Ω_φ at z=0.
    
    Returns:
        pass_flag, diagnostics
    """
    try:
        rho_m0 = 2.5e-27  # kg/m³ (roughly Ω_m ~ 0.3 today in matter-only)
        a = np.array([1.0])  # Today
        
        Omega_m, Omega_phi = dark_energy_fraction(a, rho_m0, params)
        
        Omega_m_val = Omega_m[0]
        Omega_phi_val = Omega_phi[0]
        
        # Check bounds
        pass_Omega_m = Omega_m_range[0] <= Omega_m_val <= Omega_m_range[1]
        pass_Omega_phi = Omega_phi_range[0] <= Omega_phi_val <= Omega_phi_range[1]
        
        passed = pass_Omega_m and pass_Omega_phi
        
        diagnostics = {
            'Omega_m': Omega_m_val,
            'Omega_phi': Omega_phi_val,
            'pass_Omega_m': pass_Omega_m,
            'pass_Omega_phi': pass_Omega_phi
        }
        
        return passed, diagnostics
        
    except Exception as e:
        return False, {'error': str(e), 'Omega_m': np.nan, 'Omega_phi': np.nan}


def passes_galaxy_screening(params: SymmetronParams,
                           rho_crit_range: Tuple[float, float] = (1e-22, 1e-20)) -> Tuple[bool, Dict]:
    """
    Test galaxy screening: ρ_crit should be at galaxy halo densities.
    
    For exponential profile ρ(r) = ρ_c exp(-r/R_d):
    - Typical ρ_c ~ 10^-20 kg/m³
    - Want R_c ~ few kpc where ρ(R_c) = ρ_crit
    - So ρ_crit ~ 10^-21 to 10^-22 kg/m³
    """
    try:
        rho_crit = critical_density(params)
        
        passed = rho_crit_range[0] <= rho_crit <= rho_crit_range[1]
        
        # Estimate screening radius for typical galaxy
        # Exponential: R_c ~ R_d ln(ρ_c/ρ_crit)
        R_d_typical = 3.0  # kpc
        rho_c_typical = 1e-20  # kg/m³
        
        if rho_crit < rho_c_typical:
            R_c_est = R_d_typical * np.log(rho_c_typical / rho_crit)
        else:
            R_c_est = 0.0  # Fully screened
        
        diagnostics = {
            'rho_crit': rho_crit,
            'R_c_estimate_kpc': R_c_est
        }
        
        return passed, diagnostics
        
    except Exception as e:
        return False, {'error': str(e), 'rho_crit': np.nan, 'R_c_estimate_kpc': np.nan}


def passes_ppn(params: SymmetronParams,
              m_eff_min: float = 1e8) -> Tuple[bool, Dict]:
    """
    Test PPN constraints: field must be screened in Solar System.
    
    In Solar System: ρ ~ 10^-15 kg/m³ >> ρ_crit
    → m_eff² = ρ/M² - μ² should be >> μ²
    → φ ≈ 0 (huge effective mass)
    → no fifth force
    
    Rough criterion: m_eff² > 10^8 [normalized units]
    """
    try:
        rho_solar = 1e-15  # kg/m³ (order of magnitude)
        
        # Effective mass² at φ=0 in Solar System
        m_eff_sq = d2V_dphi2(np.array([0.0]), rho_solar, params)[0]
        
        # Check if heavy enough
        passed = m_eff_sq > m_eff_min
        
        # Also check: is Solar System above ρ_crit?
        rho_crit = critical_density(params)
        above_rho_crit = rho_solar > rho_crit
        
        diagnostics = {
            'm_eff_squared': m_eff_sq,
            'rho_solar_over_rho_crit': rho_solar / rho_crit,
            'solar_system_screened': above_rho_crit
        }
        
        return passed and above_rho_crit, diagnostics
        
    except Exception as e:
        return False, {'error': str(e), 'm_eff_squared': np.nan}


# ==============================================================================
# MAIN SCANNER
# ==============================================================================

def run_viability_scan(
    n_mu: int = 20,
    n_lambda: int = 15,
    n_M: int = 20,
    n_V0: int = 10,
    beta_values: list = [0.1, 0.5, 1.0, 2.0],
    output_dir: str = 'coherence-field-theory/outputs/symmetron_viability_scan'
):
    """
    Run full parameter scan.
    
    Parameter ranges motivated by:
    - μ ~ H₀ ~ 10^-33 eV (in natural units, or ~10^-3 eV/c²)
    - λ ~ O(1) (dimensionless self-coupling)
    - M ~ M_Pl scale or suppressed
    - V₀ ~ dark energy scale (tune for Ω_φ)
    - β ~ O(1) coupling
    """
    print("="*80)
    print("SYMMETRON VIABILITY SCAN")
    print("="*80)
    
    # Parameter grids (log-uniform where appropriate)
    mu_range = np.logspace(-4, -2, n_mu)      # [normalized units]
    lambda_range = np.logspace(-2, 1, n_lambda)  # dimensionless
    M_range = np.logspace(-4, -2, n_M)        # [normalized units]
    V0_range = np.linspace(-1e-6, 1e-6, n_V0)  # Small vacuum energy
    
    total_points = n_mu * n_lambda * n_M * n_V0 * len(beta_values)
    
    print(f"\nParameter ranges:")
    print(f"  μ: [{mu_range[0]:.2e}, {mu_range[-1]:.2e}] ({n_mu} points)")
    print(f"  λ: [{lambda_range[0]:.2e}, {lambda_range[-1]:.2e}] ({n_lambda} points)")
    print(f"  M: [{M_range[0]:.2e}, {M_range[-1]:.2e}] ({n_M} points)")
    print(f"  V₀: [{V0_range[0]:.2e}, {V0_range[-1]:.2e}] ({n_V0} points)")
    print(f"  β: {beta_values}")
    print(f"\nTotal combinations: {total_points:,}")
    print(f"Estimated time: ~{total_points/1000:.1f} minutes\n")
    
    # Storage
    results = []
    
    # Progress tracking
    start_time = time.time()
    count = 0
    passed_cosmology = 0
    passed_galaxy = 0
    passed_ppn = 0
    passed_all = 0
    
    # Scan
    for mu in mu_range:
        for lambda_self in lambda_range:
            for M in M_range:
                for V0 in V0_range:
                    for beta in beta_values:
                        count += 1
                        
                        # Progress update every 1000 points
                        if count % 1000 == 0:
                            elapsed = time.time() - start_time
                            rate = count / elapsed
                            eta = (total_points - count) / rate
                            print(f"Progress: {count:,}/{total_points:,} ({100*count/total_points:.1f}%) | "
                                  f"Passed: {passed_all} | ETA: {eta/60:.1f} min")
                        
                        # Create parameter set
                        params = SymmetronParams(
                            mu=mu,
                            lambda_self=lambda_self,
                            M=M,
                            V0=V0,
                            beta=beta
                        )
                        
                        # Test filters
                        pass_cosmo, diag_cosmo = passes_cosmology(params)
                        pass_gal, diag_gal = passes_galaxy_screening(params)
                        pass_ppn_test, diag_ppn = passes_ppn(params)
                        
                        # Count passes
                        if pass_cosmo:
                            passed_cosmology += 1
                        if pass_gal:
                            passed_galaxy += 1
                        if pass_ppn_test:
                            passed_ppn += 1
                        if pass_cosmo and pass_gal and pass_ppn_test:
                            passed_all += 1
                        
                        # Store result
                        result = {
                            'mu': mu,
                            'lambda': lambda_self,
                            'M': M,
                            'V0': V0,
                            'beta': beta,
                            'pass_cosmology': pass_cosmo,
                            'pass_galaxy': pass_gal,
                            'pass_ppn': pass_ppn_test,
                            'pass_all': pass_cosmo and pass_gal and pass_ppn_test,
                            **diag_cosmo,
                            **diag_gal,
                            **diag_ppn
                        }
                        
                        results.append(result)
    
    # Summary
    elapsed_total = time.time() - start_time
    print("\n" + "="*80)
    print("SCAN COMPLETE")
    print("="*80)
    print(f"Time: {elapsed_total/60:.1f} minutes")
    print(f"Rate: {total_points/elapsed_total:.1f} points/sec")
    print(f"\nResults:")
    print(f"  Passed cosmology:  {passed_cosmology:,} / {total_points:,} ({100*passed_cosmology/total_points:.2f}%)")
    print(f"  Passed galaxy:     {passed_galaxy:,} / {total_points:,} ({100*passed_galaxy/total_points:.2f}%)")
    print(f"  Passed PPN:        {passed_ppn:,} / {total_points:,} ({100*passed_ppn/total_points:.2f}%)")
    print(f"  Passed ALL THREE:  {passed_all:,} / {total_points:,} ({100*passed_all/total_points:.2f}%)")
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    
    # Full results
    full_path = Path(output_dir) / 'symmetron_scan_full.csv'
    df.to_csv(full_path, index=False)
    print(f"\n✓ Full results: {full_path}")
    
    # Viable points only
    if passed_all > 0:
        df_viable = df[df['pass_all']]
        viable_path = Path(output_dir) / 'symmetron_scan_viable.csv'
        df_viable.to_csv(viable_path, index=False)
        print(f"✓ Viable points: {viable_path}")
        
        print("\n" + "="*80)
        print("VIABLE PARAMETER SETS")
        print("="*80)
        print(df_viable[['mu', 'lambda', 'M', 'V0', 'beta', 
                        'Omega_m', 'Omega_phi', 'rho_crit', 'R_c_estimate_kpc']].to_string(index=False))
    else:
        print("\n⚠️  NO VIABLE POINTS FOUND")
        print("\nMost common failure modes:")
        
        # Analyze failures
        df_cosmo_pass = df[df['pass_cosmology']]
        df_gal_pass = df[df['pass_galaxy']]
        df_ppn_pass = df[df['pass_ppn']]
        
        print(f"\n  Cosmology only:     {len(df_cosmo_pass):,} points")
        if len(df_cosmo_pass) > 0:
            print(f"    Typical Ω_m: {df_cosmo_pass['Omega_m'].mean():.3f} ± {df_cosmo_pass['Omega_m'].std():.3f}")
            print(f"    Typical Ω_φ: {df_cosmo_pass['Omega_phi'].mean():.3f} ± {df_cosmo_pass['Omega_phi'].std():.3f}")
        
        print(f"\n  Galaxy only:        {len(df_gal_pass):,} points")
        if len(df_gal_pass) > 0:
            print(f"    Typical ρ_crit: {df_gal_pass['rho_crit'].median():.2e} kg/m³")
            print(f"    Typical R_c: {df_gal_pass['R_c_estimate_kpc'].median():.1f} kpc")
        
        print(f"\n  PPN only:           {len(df_ppn_pass):,} points")
        
        # Check overlap
        df_cosmo_gal = df[df['pass_cosmology'] & df['pass_galaxy']]
        df_cosmo_ppn = df[df['pass_cosmology'] & df['pass_ppn']]
        df_gal_ppn = df[df['pass_galaxy'] & df['pass_ppn']]
        
        print(f"\n  Cosmology + Galaxy: {len(df_cosmo_gal):,} points")
        print(f"  Cosmology + PPN:    {len(df_cosmo_ppn):,} points")
        print(f"  Galaxy + PPN:       {len(df_gal_ppn):,} points")
    
    print("\n" + "="*80)
    
    return df


# ==============================================================================
# QUICK TEST MODE
# ==============================================================================

def quick_test():
    """
    Quick test on small grid to verify scanner works.
    """
    print("QUICK TEST MODE (small grid)")
    print("="*80)
    
    df = run_viability_scan(
        n_mu=5,
        n_lambda=5,
        n_M=5,
        n_V0=5,
        beta_values=[0.5, 1.0],
        output_dir='coherence-field-theory/outputs/symmetron_viability_scan_test'
    )
    
    return df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick test mode
        df = quick_test()
    else:
        # Full scan
        df = run_viability_scan()
    
    print("\n✓ Scan complete!")
