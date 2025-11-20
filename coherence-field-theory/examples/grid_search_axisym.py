"""
Parameter Grid Search with Axisymmetric Convolution

Re-optimize GPM parameters with proper disk geometry.
Focus on mass gating (M*, n_M) to suppress massive spiral failures.
"""

import numpy as np
import pandas as pd
import sys
import os
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics import GravitationalPolarizationMemory
from galaxies.rotation_curves import GalaxyRotationCurve
from galaxies.environment_estimator import EnvironmentEstimator
from scipy.interpolate import PchipInterpolator


def create_baryon_density(gal, galaxy_name):
    """Create baryon density from SPARC data."""
    sparc_masses = load_sparc_masses(galaxy_name)
    M_stellar = sparc_masses['M_stellar']
    M_HI = sparc_masses['M_HI']
    M_total = sparc_masses['M_total']
    R_disk = sparc_masses['R_disk']
    R_HI = sparc_masses['R_HI']
    
    R_gas = max(R_HI, 1.5 * R_disk)
    Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)
    Sigma0_gas = M_HI / (2.0 * np.pi * R_gas**2)
    h_z = 0.3  # kpc
    
    def rho_b(r_eval):
        r_safe = np.maximum(np.atleast_1d(r_eval), 1e-6)
        scalar_input = np.isscalar(r_eval)
        
        Sigma_stellar = Sigma0_stellar * np.exp(-r_safe / R_disk)
        Sigma_gas = Sigma0_gas * np.exp(-r_safe / R_gas)
        Sigma_total = Sigma_stellar + Sigma_gas
        rho = Sigma_total / (2.0 * h_z)
        
        return float(rho[0]) if scalar_input else rho
    
    # Load SBdisk for Q estimation
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


def test_galaxy_with_params(galaxy_name, alpha0, ell0, Mstar, nM):
    """Test one galaxy with given GPM parameters."""
    
    try:
        # Load data
        loader = RealDataLoader()
        gal = loader.load_rotmod_galaxy(galaxy_name)
        
        r_data = gal['r']
        v_obs = gal['v_obs']
        v_err = gal['v_err']
        v_disk = gal['v_disk']
        v_gas = gal['v_gas']
        v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
        v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
        
        if len(r_data) < 5:
            return None
        
        # Create baryon density
        rho_b, M_total, R_disk, SBdisk = create_baryon_density(gal, galaxy_name)
        
        # Environment estimation
        estimator = EnvironmentEstimator()
        morphology = estimator.classify_morphology(gal, M_total, R_disk)
        Q, sigma_v = estimator.estimate_from_sparc(gal, SBdisk, R_disk, M_L=0.5, morphology=morphology)
        
        # GPM model with axisymmetric convolution
        gpm = GravitationalPolarizationMemory(
            alpha0=alpha0, ell0_kpc=ell0,
            Qstar=2.0, sigmastar=25.0,
            nQ=2.0, nsig=2.0, p=0.5,
            Mstar_Msun=Mstar, nM=nM
        )
        
        h_z = 0.3  # kpc
        rho_coh, gpm_params = gpm.make_rho_coh(
            rho_b, Q=Q, sigma_v=sigma_v, R_disk=R_disk, M_total=M_total,
            use_axisymmetric=True, h_z=h_z, r_max=r_data.max() * 2
        )
        
        # Compute rotation curve
        galaxy_gpm = GalaxyRotationCurve(G=4.30091e-6)
        galaxy_gpm.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
        galaxy_gpm.set_coherence_halo_gpm(rho_coh, gpm_params)
        v_model_gpm = galaxy_gpm.circular_velocity(r_data)
        
        # χ²
        chi2_bar = np.sum(((v_obs - v_bar) / v_err)**2)
        chi2_gpm = np.sum(((v_obs - v_model_gpm) / v_err)**2)
        
        dof = len(r_data)
        chi2_red_bar = chi2_bar / dof
        chi2_red_gpm = chi2_gpm / dof
        
        improvement = (chi2_bar - chi2_gpm) / chi2_bar * 100
        
        return {
            'galaxy': galaxy_name,
            'morphology': morphology,
            'M_total': M_total,
            'alpha_eff': gpm_params['alpha'],
            'chi2_red_bar': chi2_red_bar,
            'chi2_red_gpm': chi2_red_gpm,
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"      ERROR {galaxy_name}: {e}")
        return None


def test_parameter_set(params_tuple, test_galaxies):
    """Test all galaxies for a single parameter set (used by parallel workers)."""
    alpha0, ell0, Mstar, nM = params_tuple
    
    results_galaxies = []
    for gal_name in test_galaxies:
        result = test_galaxy_with_params(gal_name, alpha0, ell0, Mstar, nM)
        if result is not None:
            result['alpha0'] = alpha0
            result['ell0'] = ell0
            result['Mstar'] = Mstar
            result['nM'] = nM
            results_galaxies.append(result)
    
    return results_galaxies


def grid_search():
    """Run parameter grid search with axisymmetric convolution (parallelized)."""
    
    print("="*80)
    print("PARAMETER GRID SEARCH: Axisymmetric Convolution (PARALLEL)")
    print("="*80)
    print()
    
    # Detect CPU cores
    n_cores = cpu_count()
    print(f"Detected {n_cores} CPU cores - using {n_cores} parallel workers")
    print()
    
    # Test galaxies (diverse sample)
    test_galaxies = [
        'DDO154',     # Dwarf (3e8 Msun)
        'DDO170',     # Dwarf
        'IC2574',     # Dwarf/irregular
        'NGC2403',    # Normal spiral (8e9 Msun)
        'NGC6503',    # Normal spiral (8e9 Msun)
        'NGC3198',    # Normal spiral (3e10 Msun)
        'NGC2841',    # Massive spiral (1e11 Msun) - previous failure
        'UGC00128',   # Dwarf
        'NGC0801',    # Spiral - previous catastrophic failure
    ]
    
    # Parameter grid
    # Focus on mass gating (M*, n_M) to suppress massive spiral failures
    alpha0_values = [0.20, 0.25, 0.30]  # Slightly wider range
    ell0_values = [0.8, 1.0, 1.2]       # Test shorter coherence lengths
    Mstar_values = [5e9, 1e10, 2e10, 5e10]  # CRITICAL: test stronger mass suppression
    nM_values = [2.5, 3.0, 3.5, 4.0]    # CRITICAL: stronger mass exponents
    
    print("Parameter grid:")
    print(f"  alpha0: {alpha0_values}")
    print(f"  ell0 [kpc]: {ell0_values}")
    print(f"  Mstar [Msun]: {[f'{m:.1e}' for m in Mstar_values]}")
    print(f"  nM: {nM_values}")
    print(f"  Fixed: Qstar=2.0, sigmastar=25.0, nQ=2.0, nsig=2.0, p=0.5")
    print()
    
    # Create all parameter combinations
    param_combinations = list(product(alpha0_values, ell0_values, Mstar_values, nM_values))
    total_combinations = len(param_combinations)
    
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Test galaxies: {len(test_galaxies)}")
    print(f"Total tests: {total_combinations * len(test_galaxies)}")
    print()
    
    print("Running parallel grid search...")
    print("-"*80)
    print()
    
    # Parallel execution
    results_all = []
    best_params = None
    best_score = -np.inf
    
    # Create worker function with test_galaxies bound
    worker_func = partial(test_parameter_set, test_galaxies=test_galaxies)
    
    # Run parallel grid search
    with Pool(processes=n_cores) as pool:
        # Process parameter sets in parallel
        for param_idx, results_galaxies in enumerate(pool.imap(worker_func, param_combinations), 1):
            alpha0, ell0, Mstar, nM = param_combinations[param_idx - 1]
            
            print(f"[{param_idx}/{total_combinations}] α₀={alpha0:.2f}, ℓ₀={ell0:.1f} kpc, M*={Mstar:.1e} M☉, n_M={nM:.1f}")
            
            # Add to global results
            results_all.extend(results_galaxies)
            
            if len(results_galaxies) == 0:
                print("  SKIP: No valid results\n")
                continue
            
            # Compute metrics for this parameter set
            improvements = [r['improvement'] for r in results_galaxies]
            n_improved = sum(1 for imp in improvements if imp > 0)
            n_total = len(improvements)
            success_rate = n_improved / n_total * 100
            mean_improvement = np.mean(improvements)
            median_improvement = np.median(improvements)
            
            # Penalize catastrophic failures (< -100%)
            n_catastrophic = sum(1 for imp in improvements if imp < -100)
            
            # Score: prioritize success rate, then median improvement, penalize catastrophic failures
            score = success_rate + 0.3 * median_improvement - 50 * n_catastrophic
            
            print(f"  Success: {n_improved}/{n_total} ({success_rate:.1f}%)")
            print(f"  Mean: {mean_improvement:+.1f}%, Median: {median_improvement:+.1f}%")
            print(f"  Catastrophic failures: {n_catastrophic}")
            print(f"  Score: {score:.1f}")
            
            if score > best_score:
                best_score = score
                best_params = {
                    'alpha0': alpha0,
                    'ell0': ell0,
                    'Mstar': Mstar,
                    'nM': nM,
                    'success_rate': success_rate,
                    'mean_improvement': mean_improvement,
                    'median_improvement': median_improvement,
                    'n_catastrophic': n_catastrophic,
                    'score': score
                }
                print(f"  *** NEW BEST (score={score:.1f}) ***")
            print()
    
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    print()
    
    if best_params is None:
        print("ERROR: No valid parameter sets found")
        return
    
    print("BEST PARAMETERS:")
    print(f"  alpha0: {best_params['alpha0']:.2f}")
    print(f"  ell0: {best_params['ell0']:.2f} kpc")
    print(f"  Mstar: {best_params['Mstar']:.2e} Msun")
    print(f"  nM: {best_params['nM']:.1f}")
    print()
    print("PERFORMANCE:")
    print(f"  Success rate: {best_params['success_rate']:.1f}%")
    print(f"  Mean improvement: {best_params['mean_improvement']:+.1f}%")
    print(f"  Median improvement: {best_params['median_improvement']:+.1f}%")
    print(f"  Catastrophic failures: {best_params['n_catastrophic']}")
    print(f"  Score: {best_params['score']:.1f}")
    print()
    
    # Save all results
    df = pd.DataFrame(results_all)
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'gpm_tests')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'grid_search_axisym_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved results: {output_file}")
    print()
    
    # Summary by parameter
    print("="*80)
    print("PARAMETER SENSITIVITY")
    print("="*80)
    print()
    
    # Group by each parameter and show mean success rate
    for param in ['alpha0', 'ell0', 'Mstar', 'nM']:
        print(f"\n{param.upper()}:")
        grouped = df.groupby(param)['improvement'].agg(['mean', 'median', lambda x: (x > 0).mean() * 100])
        grouped.columns = ['Mean Δχ²', 'Median Δχ²', 'Success %']
        print(grouped.to_string())
    
    print()
    print("="*80)


if __name__ == '__main__':
    grid_search()
