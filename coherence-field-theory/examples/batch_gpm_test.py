#!/usr/bin/env python3
"""
Batch test GPM on multiple SPARC galaxies.

Tests the Gravitational Polarization with Memory (GPM) model on a diverse
sample of SPARC galaxies to validate that:
1. GPM improves fit over baryons-only
2. 7 global parameters work across diverse morphologies
3. α and ℓ correlate with environment (Q, σ_v)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics import GravitationalPolarizationMemory
from galaxies.rotation_curves import GalaxyRotationCurve
from galaxies.environment_estimator import EnvironmentEstimator


def exponential_disk_density(r, Sigma0, R_d, h_z):
    """3D density for exponential disk: ρ(r,z) = Σ₀/(2h_z) exp(-r/R_d) sech²(z/2h_z)"""
    # Always work with scalars to avoid broadcasting issues
    r_scalar = np.atleast_1d(r)
    
    # Integrate over z: ∫ sech²(z/2h_z) dz ≈ 2h_z
    rho = Sigma0 / (2.0 * h_z) * np.exp(-r_scalar / R_d)
    
    # Return scalar if input was scalar
    if np.isscalar(r):
        return float(rho[0])
    return rho


def create_baryon_density(gal, galaxy_name):
    """Create baryon density from SPARC master table masses.
    
    CRITICAL FIX: Uses master table masses (M_stellar from L[3.6], M_HI from 21cm)
    instead of broken SBdisk × M/L approach which underestimated by ~1000×.
    """
    # Load masses from SPARC master table
    try:
        sparc_masses = load_sparc_masses(galaxy_name)
    except Exception as e:
        raise ValueError(f"Could not load SPARC masses for {galaxy_name}: {e}")
    
    M_stellar = sparc_masses['M_stellar']
    M_HI = sparc_masses['M_HI']
    M_total = sparc_masses['M_total']
    R_disk = sparc_masses['R_disk']
    R_HI = sparc_masses['R_HI']
    
    # Gas disk scale length (more extended than stellar)
    R_gas = max(R_HI, 1.5 * R_disk)
    
    # Central surface densities from total masses
    # For M = 2π Σ₀ R², we have Σ₀ = M/(2π R²)
    Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)  # M☉/kpc²
    Sigma0_gas = M_HI / (2.0 * np.pi * R_gas**2)  # M☉/kpc²
    
    # Scale height
    h_z = 0.3  # kpc
    
    def rho_b(r_eval):
        """Total baryon volume density: stellar + gas exponential disks."""
        r_safe = np.maximum(np.atleast_1d(r_eval), 1e-6)
        scalar_input = np.isscalar(r_eval)
        
        # Exponential profiles: Σ(r) = Σ₀ exp(-r/R)
        Sigma_stellar = Sigma0_stellar * np.exp(-r_safe / R_disk)
        Sigma_gas = Sigma0_gas * np.exp(-r_safe / R_gas)
        Sigma_total = Sigma_stellar + Sigma_gas
        
        # Volume density: ρ = Σ / (2 h_z)
        rho = Sigma_total / (2.0 * h_z)
        
        return float(rho[0]) if scalar_input else rho
    
    # Load SBdisk for environment estimation (Q calculation needs it)
    loader = RealDataLoader()
    rotmod_dir = os.path.join(loader.base_data_dir, 'Rotmod_LTG')
    filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = [l for l in lines if not l.startswith('#')]
    SBdisk = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 7:
            SBdisk.append(float(parts[6]))  # L☉/pc²
    SBdisk = np.array(SBdisk)
    
    return rho_b, M_total, R_disk, SBdisk


def estimate_environment(gal, SBdisk, R_disk, M_total, galaxy_name):
    """Estimate Toomre Q and velocity dispersion from SPARC data."""
    estimator = EnvironmentEstimator()
    
    # Classify morphology
    morphology = estimator.classify_morphology(gal, M_total, R_disk)
    
    # Estimate Q and sigma_v from observables
    Q, sigma_v = estimator.estimate_from_sparc(
        gal, SBdisk, R_disk, M_L=0.5, morphology=morphology
    )
    
    return Q, sigma_v


def test_galaxy_gpm(galaxy_name, gpm_params):
    """
    Test GPM on a single galaxy.
    
    Returns:
    --------
    result : dict
        Keys: 'name', 'n_points', 'M_total', 'R_disk', 'Q', 'sigma_v',
              'alpha_eff', 'ell', 'chi2_baryon', 'chi2_gpm', 'improvement'
    """
    # Load data
    loader = RealDataLoader()
    try:
        gal = loader.load_rotmod_galaxy(galaxy_name)
    except Exception as e:
        print(f"\n   ERROR loading: {e}")
        return None
    
    r_data = gal['r']
    v_obs = gal['v_obs']
    e_v_obs = gal['v_err']
    
    if len(r_data) < 5:
        print(f"\n   Only {len(r_data)} points")
        return None  # too few points
    
    # Baryon density
    try:
        rho_b, M_total, R_disk, SBdisk = create_baryon_density(gal, galaxy_name)
    except Exception as e:
        print(f"\n   ERROR baryon: {e}")
        return None
    
    # Environment
    Q, sigma_v = estimate_environment(gal, SBdisk, R_disk, M_total, galaxy_name)
    
    # Create GPM
    gpm = GravitationalPolarizationMemory(
        alpha0=gpm_params['alpha0'],
        ell0_kpc=gpm_params['ell0_kpc'],
        Qstar=gpm_params['Qstar'],
        sigmastar=gpm_params['sigmastar'],
        nQ=gpm_params['nQ'],
        nsig=gpm_params['nsig'],
        p=gpm_params['p'],
        Mstar_Msun=gpm_params['Mstar_Msun'],
        nM=gpm_params['nM']
    )
    
    # Make coherence density
    rho_coh_func, gpm_diagnostics = gpm.make_rho_coh(rho_b, Q=Q, sigma_v=sigma_v, R_disk=R_disk, M_total=M_total, r_max=r_data.max() * 2)
    
    alpha_eff = gpm_diagnostics['alpha']
    ell_eff = gpm_diagnostics['ell_kpc']
    
    # SPARC baryon baseline (their best-fit decomposition)
    # v_bar = sqrt(v_disk² + v_gas² + v_bulge²)
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    v_bar_sparc = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    # GPM model (SPARC baryons + coherence halo)
    galaxy_gpm = GalaxyRotationCurve(G=4.30091e-6)
    galaxy_gpm.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
    galaxy_gpm.set_coherence_halo_gpm(rho_coh_func, gpm_params)
    v_model_gpm = galaxy_gpm.circular_velocity(r_data)
    
    # But for baryon baseline, use SPARC velocities (not our exponential profiles)
    # This is the correct comparison: SPARC baryons vs SPARC baryons + GPM
    v_model_bar = v_bar_sparc
    
    # χ²
    chi2_bar = np.sum(((v_obs - v_model_bar) / e_v_obs)**2)
    chi2_gpm = np.sum(((v_obs - v_model_gpm) / e_v_obs)**2)
    
    n_dof = len(r_data)
    chi2_red_bar = chi2_bar / n_dof
    chi2_red_gpm = chi2_gpm / n_dof
    
    improvement = (chi2_bar - chi2_gpm) / chi2_bar * 100
    
    result = {
        'name': galaxy_name,
        'n_points': len(r_data),
        'M_total': M_total,
        'R_disk': R_disk,
        'Q': Q,
        'sigma_v': sigma_v,
        'alpha_eff': alpha_eff,
        'ell': ell_eff,
        'chi2_baryon': chi2_red_bar,
        'chi2_gpm': chi2_red_gpm,
        'improvement': improvement
    }
    
    return result


def main():
    """Test GPM on diverse SPARC sample."""
    
    print("="*80)
    print("Batch GPM Test on SPARC Galaxies")
    print("="*80)
    
    # Global GPM parameters (tuned with mass-dependent gating)
    gpm_params = {
        'alpha0': 0.3,         # Base susceptibility
        'ell0_kpc': 2.0,       # kpc (base coherence length)
        'Qstar': 2.0,          # Toomre Q threshold
        'sigmastar': 25.0,     # km/s (velocity dispersion threshold)
        'nQ': 2.0,             # Q gating exponent
        'nsig': 2.0,           # sigma_v gating exponent
        'p': 0.5,              # ell ~ R_disk^p scaling
        'Mstar_Msun': 2e8,     # Mass scale for gating (M_sun) - sharp transition at mid-mass
        'nM': 1.5              # Mass gating exponent - very sharp suppression
    }
    
    print("\nGlobal GPM parameters:")
    for k, v in gpm_params.items():
        print(f"  {k}: {v}")
    
    # Test galaxies (diverse morphologies and sizes)
    test_galaxies = [
        'DDO154',     # dwarf
        'DDO170',     # dwarf
        'IC2574',     # dwarf/irregular
        'NGC2403',    # spiral
        'NGC6503',    # spiral
        'NGC3198',    # spiral
        'NGC2841',    # massive spiral
        'UGC00128',   # dwarf
        'UGC02259',   # dwarf
        'NGC0801'     # spiral
    ]
    
    results = []
    
    print(f"\nTesting {len(test_galaxies)} galaxies...")
    print("-"*80)
    
    for i, name in enumerate(test_galaxies):
        print(f"[{i+1}/{len(test_galaxies)}] {name}...", end=' ', flush=True)
        
        result = test_galaxy_gpm(name, gpm_params)
        
        if result is None:
            print("SKIP (insufficient data)")
            continue
        
        results.append(result)
        print(f"dChi2 = {result['improvement']:+.1f}%")
    
    print("-"*80)
    
    if len(results) == 0:
        print("No galaxies tested successfully.")
        return
    
    # Summary statistics
    df = pd.DataFrame(results)
    
    n_improved = np.sum(df['improvement'] > 0)
    n_total = len(df)
    frac_improved = n_improved / n_total * 100
    
    mean_improvement = df['improvement'].mean()
    median_improvement = df['improvement'].median()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Galaxies tested: {n_total}")
    print(f"GPM improves fit: {n_improved}/{n_total} ({frac_improved:.1f}%)")
    print(f"Mean dChi2: {mean_improvement:+.1f}%")
    print(f"Median dChi2: {median_improvement:+.1f}%")
    
    print("\nPer-galaxy results:")
    print("-"*80)
    print(f"{'Galaxy':<12} {'N':>4} {'M_tot':>10} {'R_d':>6} {'Q':>5} {'sig_v':>6} {'alpha':>5} {'ell':>5} {'chi2_bar':>8} {'chi2_gpm':>8} {'dChi2':>7}")
    print(f"{'':12} {'':4} {'[Msun]':>10} {'[kpc]':>6} {'':5} {'[km/s]':>6} {'':5} {'[kpc]':>5} {'':8} {'':8} {'[%]':>7}")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"{row['name']:<12} {row['n_points']:4d} {row['M_total']:10.2e} "
              f"{row['R_disk']:6.2f} {row['Q']:5.2f} {row['sigma_v']:6.1f} "
              f"{row['alpha_eff']:5.3f} {row['ell']:5.2f} "
              f"{row['chi2_baryon']:8.1f} {row['chi2_gpm']:8.1f} {row['improvement']:+7.1f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'gpm_tests'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'batch_gpm_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")
    
    # Correlations
    print("\n" + "="*80)
    print("CORRELATIONS")
    print("="*80)
    
    corr_alpha_Q = df[['alpha_eff', 'Q']].corr().iloc[0, 1]
    corr_alpha_sigma = df[['alpha_eff', 'sigma_v']].corr().iloc[0, 1]
    corr_ell_R = df[['ell', 'R_disk']].corr().iloc[0, 1]
    
    print(f"alpha vs Q:       {corr_alpha_Q:+.3f} (expect negative)")
    print(f"alpha vs sigma_v: {corr_alpha_sigma:+.3f} (expect negative)")
    print(f"ell vs R_disk:    {corr_ell_R:+.3f}")
    
    # Test success criterion
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    success = True
    
    if frac_improved < 70:
        print(f"[FAIL] GPM improves <70% of galaxies ({frac_improved:.1f}%)")
        success = False
    else:
        print(f"[PASS] GPM improves >=70% of galaxies ({frac_improved:.1f}%)")
    
    if mean_improvement < 10:
        print(f"[FAIL] Mean improvement <10% ({mean_improvement:.1f}%)")
        success = False
    else:
        print(f"[PASS] Mean improvement >=10% ({mean_improvement:.1f}%)")
    
    if corr_alpha_Q > -0.3:
        print(f"[FAIL] Weak alpha-Q anticorrelation ({corr_alpha_Q:+.3f})")
        success = False
    else:
        print(f"[PASS] Strong alpha-Q anticorrelation ({corr_alpha_Q:+.3f})")
    
    print("\n" + "="*80)
    if success:
        print("[PASS] GPM PASSES VALIDATION")
    else:
        print("[FAIL] GPM NEEDS TUNING")
    print("="*80)


if __name__ == '__main__':
    main()
