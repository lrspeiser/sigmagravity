#!/usr/bin/env python3
"""
Grid search to find optimal GPM parameters.

Searches over (alpha0, Mstar) to maximize:
1. Success rate (fraction of galaxies with improvement > 0)
2. Mean improvement (average Δχ²)

Fixed parameters: ell0, Qstar, sigmastar, nQ, nsig, p, nM
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics import GravitationalPolarizationMemory
from galaxies.rotation_curves import GalaxyRotationCurve
from galaxies.environment_estimator import EnvironmentEstimator


def create_baryon_density(gal, galaxy_name):
    """Create baryon density from SPARC master table masses."""
    # Load masses from SPARC master table
    sparc_masses = load_sparc_masses(galaxy_name)
    
    M_stellar = sparc_masses['M_stellar']
    M_HI = sparc_masses['M_HI']
    M_total = sparc_masses['M_total']
    R_disk = sparc_masses['R_disk']
    R_HI = sparc_masses['R_HI']
    
    # Gas disk scale length (more extended than stellar)
    R_gas = max(R_HI, 1.5 * R_disk)
    
    # Central surface densities from total masses
    Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)
    Sigma0_gas = M_HI / (2.0 * np.pi * R_gas**2)
    
    # Scale height
    h_z = 0.3  # kpc
    
    def rho_b(r_eval):
        """Total baryon volume density: stellar + gas exponential disks."""
        r_safe = np.maximum(np.atleast_1d(r_eval), 1e-6)
        scalar_input = np.isscalar(r_eval)
        
        Sigma_stellar = Sigma0_stellar * np.exp(-r_safe / R_disk)
        Sigma_gas = Sigma0_gas * np.exp(-r_safe / R_gas)
        Sigma_total = Sigma_stellar + Sigma_gas
        
        rho = Sigma_total / (2.0 * h_z)
        
        return float(rho[0]) if scalar_input else rho
    
    # Load SBdisk for environment estimation
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
            SBdisk.append(float(parts[6]))
    SBdisk = np.array(SBdisk)
    
    return rho_b, M_total, R_disk, SBdisk


def estimate_environment(gal, SBdisk, R_disk, M_total, galaxy_name):
    """Estimate Toomre Q and velocity dispersion from SPARC data."""
    estimator = EnvironmentEstimator()
    morphology = estimator.classify_morphology(gal, M_total, R_disk)
    Q, sigma_v = estimator.estimate_from_sparc(
        gal, SBdisk, R_disk, M_L=0.5, morphology=morphology
    )
    return Q, sigma_v


def test_galaxy_with_params(galaxy_name, alpha0, Mstar, fixed_params):
    """
    Test GPM on a single galaxy with given parameters.
    
    Returns:
    --------
    result : dict or None
        Keys: 'name', 'M_total', 'chi2_baryon', 'chi2_gpm', 'improvement'
    """
    # Load data
    loader = RealDataLoader()
    try:
        gal = loader.load_rotmod_galaxy(galaxy_name)
    except Exception:
        return None
    
    r_data = gal['r']
    v_obs = gal['v_obs']
    e_v_obs = gal['v_err']
    
    if len(r_data) < 5:
        return None
    
    # Baryon density
    try:
        rho_b, M_total, R_disk, SBdisk = create_baryon_density(gal, galaxy_name)
    except Exception:
        return None
    
    # Environment
    Q, sigma_v = estimate_environment(gal, SBdisk, R_disk, M_total, galaxy_name)
    
    # Create GPM with test parameters
    gpm = GravitationalPolarizationMemory(
        alpha0=alpha0,
        ell0_kpc=fixed_params['ell0_kpc'],
        Qstar=fixed_params['Qstar'],
        sigmastar=fixed_params['sigmastar'],
        nQ=fixed_params['nQ'],
        nsig=fixed_params['nsig'],
        p=fixed_params['p'],
        Mstar_Msun=Mstar,
        nM=fixed_params['nM']
    )
    
    # Make coherence density
    try:
        rho_coh_func, gpm_diagnostics = gpm.make_rho_coh(
            rho_b, Q=Q, sigma_v=sigma_v, R_disk=R_disk, 
            M_total=M_total, r_max=r_data.max() * 2
        )
    except Exception:
        return None
    
    # SPARC baryon baseline
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    v_bar_sparc = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    # GPM model
    galaxy_gpm = GalaxyRotationCurve(G=4.30091e-6)
    galaxy_gpm.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
    galaxy_gpm.set_coherence_halo_gpm(rho_coh_func, fixed_params)
    v_model_gpm = galaxy_gpm.circular_velocity(r_data)
    
    # Chi-squared
    chi2_bar = np.sum(((v_obs - v_bar_sparc) / e_v_obs)**2)
    chi2_gpm = np.sum(((v_obs - v_model_gpm) / e_v_obs)**2)
    
    improvement = (chi2_bar - chi2_gpm) / chi2_bar * 100
    
    return {
        'name': galaxy_name,
        'M_total': M_total,
        'alpha_eff': gpm_diagnostics['alpha'],
        'chi2_baryon': chi2_bar,
        'chi2_gpm': chi2_gpm,
        'improvement': improvement
    }


def grid_search(galaxies, alpha0_range, Mstar_range, fixed_params):
    """
    Perform grid search over (alpha0, Mstar).
    
    Returns:
    --------
    results_df : DataFrame
        Columns: alpha0, Mstar, n_galaxies, n_improved, success_rate, 
                 mean_improvement, median_improvement
    """
    print(f"Grid search: {len(alpha0_range)} α₀ × {len(Mstar_range)} M* = {len(alpha0_range)*len(Mstar_range)} combinations")
    print(f"Testing on {len(galaxies)} galaxies")
    print("-" * 80)
    
    all_results = []
    
    total_combos = len(alpha0_range) * len(Mstar_range)
    combo_num = 0
    
    for alpha0 in alpha0_range:
        for Mstar in Mstar_range:
            combo_num += 1
            print(f"[{combo_num}/{total_combos}] α₀={alpha0:.3f}, M*={Mstar:.1e} ... ", end='', flush=True)
            
            galaxy_results = []
            
            for gal_name in galaxies:
                result = test_galaxy_with_params(gal_name, alpha0, Mstar, fixed_params)
                if result is not None:
                    galaxy_results.append(result)
            
            if len(galaxy_results) == 0:
                print("NO DATA")
                continue
            
            # Compute statistics
            improvements = [r['improvement'] for r in galaxy_results]
            n_improved = sum(1 for imp in improvements if imp > 0)
            success_rate = n_improved / len(galaxy_results) * 100
            mean_imp = np.mean(improvements)
            median_imp = np.median(improvements)
            
            print(f"✓ {n_improved}/{len(galaxy_results)} ({success_rate:.0f}%), mean={mean_imp:+.1f}%")
            
            all_results.append({
                'alpha0': alpha0,
                'Mstar': Mstar,
                'n_galaxies': len(galaxy_results),
                'n_improved': n_improved,
                'success_rate': success_rate,
                'mean_improvement': mean_imp,
                'median_improvement': median_imp,
                'galaxy_results': galaxy_results  # Store individual results
            })
    
    return pd.DataFrame(all_results)


def main():
    """Run grid search."""
    
    print("=" * 80)
    print("GPM Parameter Grid Search")
    print("=" * 80)
    
    # Test galaxies (diverse sample)
    test_galaxies = [
        'DDO154',     # dwarf, M~3e8
        'DDO170',     # dwarf, M~1e9
        'IC2574',     # irregular, M~2e9
        'NGC2403',    # spiral, M~1.5e10
        'NGC6503',    # spiral, M~1.6e10
        'NGC3198',    # spiral, M~5e10
        'UGC02259',   # dwarf, M~2e9
    ]
    
    # Fixed parameters (not searching over these)
    fixed_params = {
        'ell0_kpc': 1.5,      # Reduced from 2.0
        'Qstar': 2.0,
        'sigmastar': 25.0,
        'nQ': 2.0,
        'nsig': 2.0,
        'p': 0.5,
        'nM': 2.0             # Increased from 1.5 for stronger suppression
    }
    
    print("\nFixed parameters:")
    for k, v in fixed_params.items():
        print(f"  {k}: {v}")
    
    # Grid search ranges (focused on promising region)
    alpha0_range = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    Mstar_range = [3e8, 5e8, 1e9, 2e9, 5e9]
    
    print(f"\nSearching:")
    print(f"  α₀: {alpha0_range}")
    print(f"  M*: {[f'{m:.1e}' for m in Mstar_range]}")
    
    # Run grid search
    print("\n" + "=" * 80)
    start_time = time.time()
    
    results_df = grid_search(test_galaxies, alpha0_range, Mstar_range, fixed_params)
    
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Completed in {elapsed:.1f} seconds")
    
    # Find best parameters
    print("\n" + "=" * 80)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 80)
    
    # Sort by: 1) success_rate, 2) mean_improvement
    results_sorted = results_df.sort_values(
        by=['success_rate', 'mean_improvement'], 
        ascending=[False, False]
    )
    
    print(f"{'Rank':<6} {'α₀':<8} {'M*':<12} {'N':<4} {'Success':<10} {'Mean Δχ²':<12} {'Median Δχ²':<12}")
    print(f"{'':6} {'':8} {'[M☉]':<12} {'':4} {'[%]':<10} {'[%]':<12} {'[%]':<12}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_sorted.head(10).iterrows()):
        print(f"{i+1:<6} {row['alpha0']:<8.3f} {row['Mstar']:<12.2e} "
              f"{row['n_galaxies']:<4d} {row['success_rate']:<10.1f} "
              f"{row['mean_improvement']:<+12.1f} {row['median_improvement']:<+12.1f}")
    
    # Best parameters
    best = results_sorted.iloc[0]
    
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)
    print(f"α₀ = {best['alpha0']:.3f}")
    print(f"M* = {best['Mstar']:.2e} M☉")
    print(f"Success rate: {best['n_improved']:.0f}/{best['n_galaxies']:.0f} ({best['success_rate']:.1f}%)")
    print(f"Mean improvement: {best['mean_improvement']:+.1f}%")
    print(f"Median improvement: {best['median_improvement']:+.1f}%")
    
    # Show individual galaxy results for best parameters
    print("\nPer-galaxy results (best parameters):")
    print("-" * 80)
    print(f"{'Galaxy':<12} {'M_total':<12} {'χ²_bar':<10} {'χ²_gpm':<10} {'Δχ²':<8}")
    print(f"{'':12} {'[M☉]':<12} {'':10} {'':10} {'[%]':<8}")
    print("-" * 80)
    
    for gal_result in best['galaxy_results']:
        print(f"{gal_result['name']:<12} {gal_result['M_total']:<12.2e} "
              f"{gal_result['chi2_baryon']:<10.1f} {gal_result['chi2_gpm']:<10.1f} "
              f"{gal_result['improvement']:<+8.1f}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs' / 'gpm_tests'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full grid results
    results_save = results_df.drop(columns=['galaxy_results'])  # Drop nested data
    csv_path = output_dir / 'grid_search_results.csv'
    results_save.to_csv(csv_path, index=False)
    print(f"\nGrid results saved: {csv_path}")
    
    # Save best parameters
    best_params_path = output_dir / 'best_gpm_parameters.txt'
    with open(best_params_path, 'w') as f:
        f.write("# Best GPM Parameters (from grid search)\n")
        f.write(f"# Success rate: {best['success_rate']:.1f}%\n")
        f.write(f"# Mean improvement: {best['mean_improvement']:+.1f}%\n\n")
        f.write(f"alpha0 = {best['alpha0']:.3f}\n")
        f.write(f"Mstar_Msun = {best['Mstar']:.2e}\n")
        for k, v in fixed_params.items():
            f.write(f"{k} = {v}\n")
    
    print(f"Best parameters saved: {best_params_path}")
    
    print("\n" + "=" * 80)
    if best['success_rate'] >= 70 and best['mean_improvement'] >= 10:
        print("[SUCCESS] Found parameters meeting criteria!")
    elif best['success_rate'] >= 70:
        print("[PARTIAL] High success rate but low mean improvement")
    elif best['mean_improvement'] >= 10:
        print("[PARTIAL] Good mean improvement but low success rate")
    else:
        print("[FAIL] Did not meet criteria - need broader search or different approach")
    print("=" * 80)


if __name__ == '__main__':
    main()
