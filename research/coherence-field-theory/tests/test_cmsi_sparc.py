"""
SPARC Test Harness for CMSI Kernel
==================================

This script tests the CMSI kernel against SPARC rotation curve data.
Uses the project's existing data integration utilities.

Usage:
    python tests/test_cmsi_sparc.py

See: derivations/CMSI_DERIVATION_SUMMARY.md for physics explanation.
"""

import numpy as np
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import csv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CMSI kernel from galaxies folder
from galaxies.cmsi_kernel import (
    CMSIParams,
    cmsi_enhancement,
    compute_v_circ_enhanced,
    compute_rms,
    sigma_exponential_disk,
    sigma_constant,
    check_solar_system_safety
)

# Import project data loaders
from data_integration.load_data import DataLoader


# =============================================================================
# SPARC Data Loader (using project utilities)
# =============================================================================

def load_sparc_galaxy_from_rotmod(filepath: str) -> Dict:
    """
    Load a SPARC galaxy rotation curve from Rotmod_LTG format.
    
    Expected format (space-separated):
        R[kpc]  Vobs[km/s]  Verr[km/s]  Vgas[km/s]  Vdisk[km/s]  Vbul[km/s]  SBdisk  SBbul
    
    Parameters
    ----------
    filepath : str
        Path to .dat file
        
    Returns
    -------
    data : dict
        Galaxy data dictionary
    """
    data = {
        'name': Path(filepath).stem,
        'R': [],
        'v_obs': [],
        'v_err': [],
        'v_gas': [],
        'v_disk': [],
        'v_bul': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    data['R'].append(float(parts[0]))
                    data['v_obs'].append(float(parts[1]))
                    data['v_err'].append(float(parts[2]) if len(parts) > 2 else 5.0)
                    data['v_gas'].append(float(parts[3]) if len(parts) > 3 else 0.0)
                    data['v_disk'].append(float(parts[4]) if len(parts) > 4 else 0.0)
                    data['v_bul'].append(float(parts[5]) if len(parts) > 5 else 0.0)
                except ValueError:
                    continue
    
    for key in data:
        if key != 'name':
            data[key] = np.array(data[key])
    
    # Compute baryonic velocity
    if len(data['R']) > 0:
        data['v_bary'] = np.sqrt(
            np.abs(data['v_gas'])**2 + 
            np.abs(data['v_disk'])**2 + 
            np.abs(data['v_bul'])**2
        )
    else:
        data['v_bary'] = np.array([])
    
    return data


def compute_sigma_v_profile(
    R_kpc: np.ndarray,
    v_circ: np.ndarray,
    method: str = 'exponential',
    **kwargs
) -> np.ndarray:
    """
    Estimate velocity dispersion profile.
    
    Methods:
        'exponential': σ_v = σ_0 exp(-R/R_σ) + σ_floor
        'constant': σ_v = constant
        'scaled': σ_v = fraction × v_circ
    """
    if method == 'exponential':
        sigma_0 = kwargs.get('sigma_0', 30.0)
        R_sigma = kwargs.get('R_sigma', 4.0)
        sigma_floor = kwargs.get('sigma_floor', 8.0)
        return sigma_exponential_disk(R_kpc, sigma_0, R_sigma, sigma_floor)
    
    elif method == 'constant':
        sigma_v = kwargs.get('sigma_v', 20.0)
        return sigma_constant(R_kpc, sigma_v)
    
    elif method == 'scaled':
        fraction = kwargs.get('fraction', 0.1)
        sigma_floor = kwargs.get('sigma_floor', 5.0)
        return np.maximum(fraction * v_circ, sigma_floor)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_surface_density(
    R_kpc: np.ndarray,
    v_disk: np.ndarray,
    v_gas: np.ndarray,
    R_d: float = 3.0,
    h: float = 0.3
) -> np.ndarray:
    """
    Estimate surface density from disk/gas velocities.
    
    Uses: Σ ~ v²/(2πGR) for the disk contribution.
    """
    G = 4.302e-6  # kpc (km/s)^2 / M_sun
    
    # Disk contribution (Freeman disk approximation)
    v_disk_safe = np.maximum(np.abs(v_disk), 1.0)
    R_safe = np.maximum(R_kpc, 0.1)
    
    # Simple approximation: Σ_disk ~ v²/(2πGR)
    Sigma_disk = v_disk_safe**2 / (2 * np.pi * G * R_safe * 1e6)  # M_sun/pc²
    
    # Gas contribution
    v_gas_safe = np.maximum(np.abs(v_gas), 1.0)
    Sigma_gas = v_gas_safe**2 / (2 * np.pi * G * R_safe * 1e6)
    
    # Total (with floor)
    Sigma_total = np.maximum(Sigma_disk + Sigma_gas, 1.0)
    
    return Sigma_total


# =============================================================================
# CMSI Testing Functions
# =============================================================================

@dataclass
class GalaxyResult:
    """Results for a single galaxy."""
    name: str
    n_points: int
    rms_bary: float
    rms_cmsi: float
    delta_rms: float
    improved: bool
    mean_F_CMSI: float
    max_F_CMSI: float
    mean_N_coh: float
    mean_sigma_v: float
    mean_Sigma: float


def test_galaxy(
    galaxy_data: Dict,
    params: CMSIParams,
    sigma_method: str = 'exponential',
    sigma_kwargs: Optional[Dict] = None
) -> GalaxyResult:
    """
    Test CMSI on a single galaxy.
    
    Parameters
    ----------
    galaxy_data : dict
        Galaxy data from load_sparc_galaxy
    params : CMSIParams
        CMSI parameters
    sigma_method : str
        Method for estimating σ_v
    sigma_kwargs : dict
        Additional kwargs for sigma estimation
    
    Returns
    -------
    GalaxyResult
        Results dataclass
    """
    if sigma_kwargs is None:
        sigma_kwargs = {}
    
    R = galaxy_data['R']
    v_obs = galaxy_data['v_obs']
    v_err = galaxy_data['v_err']
    v_bary = galaxy_data['v_bary']
    
    if len(R) == 0:
        raise ValueError(f"No data points for galaxy {galaxy_data['name']}")
    
    # Estimate velocity dispersion
    sigma_v = compute_sigma_v_profile(R, v_obs, method=sigma_method, **sigma_kwargs)
    
    # Estimate surface density
    Sigma = estimate_surface_density(
        R, galaxy_data['v_disk'], galaxy_data['v_gas']
    )
    
    # Compute CMSI enhancement
    v_cmsi, diag = compute_v_circ_enhanced(
        R, v_bary, sigma_v, params, Sigma, use_iterative=True
    )
    
    # Compute RMS values
    rms_bary = compute_rms(v_bary, v_obs, v_err)
    rms_cmsi = compute_rms(v_cmsi, v_obs, v_err)
    
    return GalaxyResult(
        name=galaxy_data['name'],
        n_points=len(R),
        rms_bary=rms_bary,
        rms_cmsi=rms_cmsi,
        delta_rms=rms_cmsi - rms_bary,
        improved=(rms_cmsi < rms_bary),
        mean_F_CMSI=np.mean(diag['F_CMSI']),
        max_F_CMSI=np.max(diag['F_CMSI']),
        mean_N_coh=np.mean(diag['N_coh']),
        mean_sigma_v=np.mean(sigma_v),
        mean_Sigma=np.mean(Sigma)
    )


def run_sparc_batch(
    data_dir: str,
    params: CMSIParams,
    sigma_method: str = 'exponential',
    sigma_kwargs: Optional[Dict] = None,
    file_pattern: str = '*.dat',
    max_galaxies: int = 0
) -> List[GalaxyResult]:
    """
    Run CMSI test on all galaxies in a directory.
    """
    import glob
    
    results = []
    files = glob.glob(os.path.join(data_dir, file_pattern))
    
    if max_galaxies > 0:
        files = files[:max_galaxies]
    
    print(f"Found {len(files)} galaxy files")
    
    for i, filepath in enumerate(files):
        try:
            galaxy_data = load_sparc_galaxy_from_rotmod(filepath)
            if len(galaxy_data['R']) < 3:
                print(f"  [{i+1:3d}/{len(files)}] SKIP {Path(filepath).stem}: too few points")
                continue
                
            result = test_galaxy(galaxy_data, params, sigma_method, sigma_kwargs)
            results.append(result)
            
            status = "✓" if result.improved else "✗"
            print(f"  [{i+1:3d}/{len(files)}] {status} {result.name}: "
                  f"RMS {result.rms_bary:.1f} → {result.rms_cmsi:.1f} km/s "
                  f"(F={result.mean_F_CMSI:.2f})")
        except Exception as e:
            print(f"  [{i+1:3d}/{len(files)}] ERROR {Path(filepath).stem}: {e}")
    
    return results


def summarize_results(results: List[GalaxyResult]) -> Dict:
    """Summarize batch results."""
    n_total = len(results)
    if n_total == 0:
        return {'n_total': 0, 'n_improved': 0, 'pct_improved': 0}
    
    n_improved = sum(1 for r in results if r.improved)
    
    delta_rms_values = [r.delta_rms for r in results]
    
    summary = {
        'n_total': n_total,
        'n_improved': n_improved,
        'pct_improved': 100.0 * n_improved / n_total if n_total > 0 else 0,
        'mean_delta_rms': np.mean(delta_rms_values),
        'median_delta_rms': np.median(delta_rms_values),
        'std_delta_rms': np.std(delta_rms_values),
        'mean_rms_bary': np.mean([r.rms_bary for r in results]),
        'mean_rms_cmsi': np.mean([r.rms_cmsi for r in results]),
        'mean_F_CMSI': np.mean([r.mean_F_CMSI for r in results]),
    }
    
    return summary


def save_results(
    results: List[GalaxyResult],
    output_path: str,
    params: CMSIParams
):
    """Save results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header with parameters
        writer.writerow(['# CMSI SPARC Results'])
        writer.writerow([f'# chi_0={params.chi_0}, gamma_phase={params.gamma_phase}, '
                        f'alpha_Ncoh={params.alpha_Ncoh}, ell_0={params.ell_0_kpc}'])
        writer.writerow([])
        
        # Column headers
        headers = ['name', 'n_points', 'rms_bary', 'rms_cmsi', 'delta_rms', 
                  'improved', 'mean_F_CMSI', 'max_F_CMSI', 'mean_N_coh',
                  'mean_sigma_v', 'mean_Sigma']
        writer.writerow(headers)
        
        # Data rows
        for r in results:
            writer.writerow([
                r.name, r.n_points, f'{r.rms_bary:.2f}', f'{r.rms_cmsi:.2f}',
                f'{r.delta_rms:.2f}', r.improved, f'{r.mean_F_CMSI:.3f}',
                f'{r.max_F_CMSI:.3f}', f'{r.mean_N_coh:.1f}',
                f'{r.mean_sigma_v:.1f}', f'{r.mean_Sigma:.1f}'
            ])
    
    print(f"Results saved to {output_path}")


# =============================================================================
# Demo with Synthetic Data (when SPARC files aren't available)
# =============================================================================

def create_demo_galaxy(
    name: str = "Demo_Galaxy",
    v_flat: float = 150.0,
    R_d: float = 3.0,
    n_points: int = 30
) -> Dict:
    """Create a synthetic galaxy for testing when SPARC data isn't available."""
    R = np.linspace(0.5, 15.0, n_points)
    
    # Simplified disk velocity profile (rising then flat)
    x = R / R_d
    v_disk = v_flat * 0.6 * np.sqrt(x / (1 + x))
    
    # Gas contribution (extended)
    v_gas = v_flat * 0.2 * (1 - np.exp(-R/5.0))
    
    # No bulge for simplicity
    v_bul = np.zeros_like(R)
    
    # Total baryonic
    v_bary = np.sqrt(v_disk**2 + v_gas**2 + v_bul**2)
    
    # "Observed" velocity (needs dark matter or CMSI boost)
    # This is the flat rotation curve we're trying to explain
    v_obs = v_flat * np.sqrt(R / (R + R_d))
    v_obs = np.maximum(v_obs, 50.0)  # Floor
    
    # Add some observational "scatter"
    np.random.seed(42)
    v_err = 5.0 + 0.02 * v_obs
    v_obs_noisy = v_obs + np.random.normal(0, 0.3 * v_err, size=len(R))
    
    return {
        'name': name,
        'R': R,
        'v_obs': v_obs_noisy,
        'v_err': v_err,
        'v_gas': v_gas,
        'v_disk': v_disk,
        'v_bul': v_bul,
        'v_bary': v_bary
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CMSI Kernel - SPARC Test Harness")
    print("=" * 70)
    
    # Default parameters (from calibration)
    params = CMSIParams(
        chi_0=800.0,
        gamma_phase=1.5,
        alpha_Ncoh=0.55,
        ell_0_kpc=2.2,
        n_profile=2.0,
        Sigma_ref=50.0,
        epsilon_Sigma=0.5,
        include_K_rough=True
    )
    
    print("\nCMSI Parameters:")
    print(f"  χ_0 = {params.chi_0}")
    print(f"  γ_phase = {params.gamma_phase}")
    print(f"  α_Ncoh = {params.alpha_Ncoh}")
    print(f"  ℓ_0 = {params.ell_0_kpc} kpc")
    print(f"  Σ_ref = {params.Sigma_ref} M_sun/pc²")
    print(f"  ε_Σ = {params.epsilon_Sigma}")
    
    # Check Solar System first
    print("\n" + "-" * 50)
    print("Solar System Safety Check:")
    ss = check_solar_system_safety(params)
    print(f"  δg/g = {ss['delta_g_over_g']:.2e} (limit: {ss['cassini_limit']:.2e})")
    print(f"  PASSES: {ss['passes_cassini']}")
    
    if not ss['passes_cassini']:
        print("  WARNING: Parameters fail Solar System constraint!")
        return
    
    # Find SPARC data directory
    # Try several possible paths relative to project root
    project_root = Path(__file__).parent.parent
    possible_paths = [
        project_root / "data" / "Rotmod_LTG",
        project_root.parent / "data" / "Rotmod_LTG",
        Path("../../data/Rotmod_LTG"),
    ]
    
    sparc_dir = None
    for path in possible_paths:
        if path.exists():
            sparc_dir = str(path)
            break
    
    if sparc_dir and os.path.exists(sparc_dir):
        print(f"\n" + "-" * 50)
        print(f"Running SPARC batch test...")
        print(f"Data directory: {sparc_dir}")
        
        results = run_sparc_batch(
            sparc_dir, params,
            sigma_method='exponential',
            sigma_kwargs={'sigma_0': 30.0, 'R_sigma': 4.0, 'sigma_floor': 8.0},
            max_galaxies=20  # Limit for quick test
        )
        
        if results:
            summary = summarize_results(results)
            
            print(f"\n" + "=" * 50)
            print("BATCH SUMMARY")
            print("=" * 50)
            print(f"  Total galaxies: {summary['n_total']}")
            print(f"  Improved: {summary['n_improved']} ({summary['pct_improved']:.1f}%)")
            print(f"  Mean ΔRMS: {summary['mean_delta_rms']:.2f} km/s")
            print(f"  Median ΔRMS: {summary['median_delta_rms']:.2f} km/s")
            print(f"  Mean F_CMSI: {summary['mean_F_CMSI']:.3f}")
            
            # Save results
            output_path = project_root / "outputs" / "cmsi_sparc_results.csv"
            output_path.parent.mkdir(exist_ok=True)
            save_results(results, str(output_path), params)
        
    else:
        print(f"\n" + "-" * 50)
        print("SPARC data not found. Running demo with synthetic galaxy...")
        
        # Create demo galaxy
        demo_data = create_demo_galaxy("Synthetic_Spiral", v_flat=180.0)
        
        result = test_galaxy(
            demo_data, params,
            sigma_method='exponential',
            sigma_kwargs={'sigma_0': 35.0, 'R_sigma': 4.0, 'sigma_floor': 10.0}
        )
        
        print(f"\nDemo Galaxy Results ({result.name}):")
        print(f"  N points: {result.n_points}")
        print(f"  RMS (baryonic): {result.rms_bary:.2f} km/s")
        print(f"  RMS (CMSI): {result.rms_cmsi:.2f} km/s")
        print(f"  ΔRMS: {result.delta_rms:.2f} km/s")
        print(f"  Improved: {result.improved}")
        print(f"  Mean F_CMSI: {result.mean_F_CMSI:.3f}")
        print(f"  Mean N_coh: {result.mean_N_coh:.1f}")
        
        # Show radial profile
        print(f"\nRadial profile:")
        R = demo_data['R']
        sigma_v = compute_sigma_v_profile(R, demo_data['v_obs'], 'exponential',
                                          sigma_0=35.0, R_sigma=4.0, sigma_floor=10.0)
        Sigma = estimate_surface_density(R, demo_data['v_disk'], demo_data['v_gas'])
        v_cmsi, diag = compute_v_circ_enhanced(
            R, demo_data['v_bary'], sigma_v, params, Sigma
        )
        
        print(f"  {'R[kpc]':>8} {'v_bary':>8} {'v_CMSI':>8} {'v_obs':>8} {'F_CMSI':>8}")
        for i in range(0, len(R), max(1, len(R)//10)):
            print(f"  {R[i]:8.2f} {demo_data['v_bary'][i]:8.1f} {v_cmsi[i]:8.1f} "
                  f"{demo_data['v_obs'][i]:8.1f} {diag['F_CMSI'][i]:8.3f}")
    
    print("\n" + "=" * 70)
    print("Test complete.")


if __name__ == "__main__":
    main()
