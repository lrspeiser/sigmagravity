"""
CMSI Parameter Optimization on Full SPARC Sample
=================================================

Sweeps over (χ_0, α_Ncoh, ℓ_0) to find optimal parameters.

Usage:
    python fitting/cmsi_parameter_sweep.py

Output:
    - outputs/cmsi_sweep_results.csv - all parameter combinations
    - outputs/cmsi_best_params.json - optimal parameters
"""

import numpy as np
import os
import sys
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from itertools import product
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from galaxies.cmsi_kernel import (
    CMSIParams,
    cmsi_enhancement,
    compute_v_circ_enhanced,
    compute_rms,
    sigma_exponential_disk,
    check_solar_system_safety
)


# =============================================================================
# Data Loading
# =============================================================================

def load_sparc_galaxy(filepath: str) -> Optional[Dict]:
    """Load SPARC galaxy rotation curve from Rotmod_LTG format."""
    data = {
        'name': Path(filepath).stem.replace('_rotmod', ''),
        'R': [],
        'v_obs': [],
        'v_err': [],
        'v_gas': [],
        'v_disk': [],
        'v_bul': []
    }
    
    try:
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
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
    
    for key in data:
        if key != 'name':
            data[key] = np.array(data[key])
    
    if len(data['R']) < 5:
        return None
    
    # Compute baryonic velocity
    data['v_bary'] = np.sqrt(
        np.abs(data['v_gas'])**2 + 
        np.abs(data['v_disk'])**2 + 
        np.abs(data['v_bul'])**2
    )
    
    return data


def load_all_sparc(data_dir: str) -> List[Dict]:
    """Load all SPARC galaxies."""
    import glob
    
    files = glob.glob(os.path.join(data_dir, '*.dat'))
    galaxies = []
    
    for filepath in files:
        data = load_sparc_galaxy(filepath)
        if data is not None:
            galaxies.append(data)
    
    return galaxies


# =============================================================================
# Evaluation Functions
# =============================================================================

def estimate_surface_density(R_kpc, v_disk, v_gas):
    """Estimate surface density from velocity components."""
    G = 4.302e-6
    v_disk_safe = np.maximum(np.abs(v_disk), 1.0)
    v_gas_safe = np.maximum(np.abs(v_gas), 1.0)
    R_safe = np.maximum(R_kpc, 0.1)
    
    Sigma_disk = v_disk_safe**2 / (2 * np.pi * G * R_safe * 1e6)
    Sigma_gas = v_gas_safe**2 / (2 * np.pi * G * R_safe * 1e6)
    
    return np.maximum(Sigma_disk + Sigma_gas, 1.0)


def evaluate_galaxy(galaxy: Dict, params: CMSIParams) -> Dict:
    """Evaluate CMSI on a single galaxy."""
    R = galaxy['R']
    v_obs = galaxy['v_obs']
    v_err = galaxy['v_err']
    v_bary = galaxy['v_bary']
    
    # Estimate sigma_v (exponential profile)
    sigma_v = sigma_exponential_disk(R, sigma_0=30.0, R_sigma=4.0, sigma_floor=8.0)
    
    # Estimate surface density
    Sigma = estimate_surface_density(R, galaxy['v_disk'], galaxy['v_gas'])
    
    # Compute CMSI enhanced velocity
    v_cmsi, diag = compute_v_circ_enhanced(
        R, v_bary, sigma_v, params, Sigma, use_iterative=True, n_iter=3
    )
    
    # Compute RMS
    rms_bary = compute_rms(v_bary, v_obs, v_err)
    rms_cmsi = compute_rms(v_cmsi, v_obs, v_err)
    
    return {
        'name': galaxy['name'],
        'n_points': len(R),
        'rms_bary': rms_bary,
        'rms_cmsi': rms_cmsi,
        'delta_rms': rms_cmsi - rms_bary,
        'improved': rms_cmsi < rms_bary,
        'mean_F': np.mean(diag['F_CMSI']),
        'max_F': np.max(diag['F_CMSI']),
        'v_flat': np.median(v_obs[-5:]) if len(v_obs) >= 5 else v_obs[-1]
    }


def evaluate_batch(galaxies: List[Dict], params: CMSIParams) -> Dict:
    """Evaluate CMSI on all galaxies and return summary statistics."""
    results = []
    
    for galaxy in galaxies:
        try:
            result = evaluate_galaxy(galaxy, params)
            results.append(result)
        except Exception as e:
            pass  # Skip failed galaxies silently
    
    if len(results) == 0:
        return {'n_total': 0}
    
    n_improved = sum(1 for r in results if r['improved'])
    delta_rms = [r['delta_rms'] for r in results]
    
    return {
        'n_total': len(results),
        'n_improved': n_improved,
        'pct_improved': 100.0 * n_improved / len(results),
        'mean_delta_rms': np.mean(delta_rms),
        'median_delta_rms': np.median(delta_rms),
        'mean_rms_bary': np.mean([r['rms_bary'] for r in results]),
        'mean_rms_cmsi': np.mean([r['rms_cmsi'] for r in results]),
        'mean_F': np.mean([r['mean_F'] for r in results]),
        'results': results
    }


# =============================================================================
# Parameter Sweep
# =============================================================================

@dataclass
class SweepResult:
    chi_0: float
    alpha_Ncoh: float
    ell_0: float
    n_total: int
    n_improved: int
    pct_improved: float
    mean_delta_rms: float
    median_delta_rms: float
    mean_F: float
    passes_cassini: bool


def run_parameter_sweep(
    galaxies: List[Dict],
    chi_0_values: List[float],
    alpha_Ncoh_values: List[float],
    ell_0_values: List[float],
    verbose: bool = True
) -> List[SweepResult]:
    """
    Sweep over parameter space.
    """
    results = []
    total_combos = len(chi_0_values) * len(alpha_Ncoh_values) * len(ell_0_values)
    
    print(f"\nRunning parameter sweep: {total_combos} combinations on {len(galaxies)} galaxies")
    print("=" * 70)
    
    start_time = time.time()
    
    for i, (chi_0, alpha_Ncoh, ell_0) in enumerate(product(chi_0_values, alpha_Ncoh_values, ell_0_values)):
        params = CMSIParams(
            chi_0=chi_0,
            gamma_phase=1.5,
            alpha_Ncoh=alpha_Ncoh,
            ell_0_kpc=ell_0,
            n_profile=2.0,
            Sigma_ref=50.0,
            epsilon_Sigma=0.5,
            include_K_rough=True
        )
        
        # Check Solar System safety first
        ss = check_solar_system_safety(params)
        if not ss['passes_cassini']:
            if verbose:
                print(f"  [{i+1:3d}/{total_combos}] χ_0={chi_0:5.0f} α={alpha_Ncoh:.2f} ℓ_0={ell_0:.1f} - FAILS CASSINI")
            continue
        
        # Evaluate on all galaxies
        eval_result = evaluate_batch(galaxies, params)
        
        if eval_result['n_total'] == 0:
            continue
        
        result = SweepResult(
            chi_0=chi_0,
            alpha_Ncoh=alpha_Ncoh,
            ell_0=ell_0,
            n_total=eval_result['n_total'],
            n_improved=eval_result['n_improved'],
            pct_improved=eval_result['pct_improved'],
            mean_delta_rms=eval_result['mean_delta_rms'],
            median_delta_rms=eval_result['median_delta_rms'],
            mean_F=eval_result['mean_F'],
            passes_cassini=True
        )
        results.append(result)
        
        if verbose:
            status = "✓" if result.pct_improved > 50 else "○"
            print(f"  [{i+1:3d}/{total_combos}] {status} χ_0={chi_0:5.0f} α={alpha_Ncoh:.2f} ℓ_0={ell_0:.1f} | "
                  f"{result.pct_improved:5.1f}% improved | ΔRMS={result.mean_delta_rms:+6.2f} km/s | F={result.mean_F:.2f}")
    
    elapsed = time.time() - start_time
    print(f"\nSweep completed in {elapsed:.1f}s")
    
    return results


def find_best_params(results: List[SweepResult]) -> SweepResult:
    """Find best parameters by pct_improved, then median_delta_rms."""
    if not results:
        return None
    
    # Primary: maximize % improved
    # Secondary: minimize median_delta_rms
    sorted_results = sorted(results, key=lambda r: (-r.pct_improved, r.median_delta_rms))
    return sorted_results[0]


# =============================================================================
# Output
# =============================================================================

def save_sweep_results(results: List[SweepResult], output_path: str):
    """Save all sweep results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['chi_0', 'alpha_Ncoh', 'ell_0', 'n_total', 'n_improved', 
                   'pct_improved', 'mean_delta_rms', 'median_delta_rms', 'mean_F']
        writer.writerow(headers)
        
        for r in results:
            writer.writerow([
                r.chi_0, r.alpha_Ncoh, r.ell_0, r.n_total, r.n_improved,
                f'{r.pct_improved:.1f}', f'{r.mean_delta_rms:.2f}', 
                f'{r.median_delta_rms:.2f}', f'{r.mean_F:.3f}'
            ])
    
    print(f"Sweep results saved to {output_path}")


def save_best_params(best: SweepResult, galaxies: List[Dict], output_path: str):
    """Save best parameters and detailed results."""
    # Re-run with best params to get per-galaxy results
    params = CMSIParams(
        chi_0=best.chi_0,
        gamma_phase=1.5,
        alpha_Ncoh=best.alpha_Ncoh,
        ell_0_kpc=best.ell_0,
        n_profile=2.0,
        Sigma_ref=50.0,
        epsilon_Sigma=0.5,
        include_K_rough=True
    )
    
    eval_result = evaluate_batch(galaxies, params)
    
    # Convert numpy types to native Python for JSON serialization
    per_galaxy = []
    for r in eval_result['results']:
        per_galaxy.append({
            'name': r['name'],
            'n_points': int(r['n_points']),
            'rms_bary': float(r['rms_bary']),
            'rms_cmsi': float(r['rms_cmsi']),
            'delta_rms': float(r['delta_rms']),
            'improved': bool(r['improved']),
            'mean_F': float(r['mean_F']),
            'max_F': float(r['max_F']),
            'v_flat': float(r['v_flat'])
        })
    
    output = {
        'best_params': {
            'chi_0': float(best.chi_0),
            'gamma_phase': 1.5,
            'alpha_Ncoh': float(best.alpha_Ncoh),
            'ell_0_kpc': float(best.ell_0),
            'n_profile': 2.0,
            'Sigma_ref': 50.0,
            'epsilon_Sigma': 0.5
        },
        'summary': {
            'n_total': int(best.n_total),
            'n_improved': int(best.n_improved),
            'pct_improved': float(best.pct_improved),
            'mean_delta_rms': float(best.mean_delta_rms),
            'median_delta_rms': float(best.median_delta_rms),
            'mean_F': float(best.mean_F)
        },
        'per_galaxy': per_galaxy
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Best parameters saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CMSI Parameter Optimization - Full SPARC Batch")
    print("=" * 70)
    
    # Find data directory
    project_root = Path(__file__).parent.parent
    possible_paths = [
        project_root.parent / "data" / "Rotmod_LTG",
        project_root / "data" / "Rotmod_LTG",
    ]
    
    data_dir = None
    for path in possible_paths:
        if path.exists():
            data_dir = str(path)
            break
    
    if data_dir is None:
        print("ERROR: SPARC data directory not found!")
        return
    
    print(f"\nData directory: {data_dir}")
    
    # Load all galaxies
    print("\nLoading SPARC galaxies...")
    galaxies = load_all_sparc(data_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    if len(galaxies) == 0:
        print("ERROR: No galaxies loaded!")
        return
    
    # Parameter ranges to sweep
    chi_0_values = [500, 800, 1200, 2000, 3000, 5000]
    alpha_Ncoh_values = [0.45, 0.55, 0.65, 0.75, 0.85]
    ell_0_values = [1.5, 2.2, 3.0, 4.0, 6.0]
    
    print(f"\nParameter ranges:")
    print(f"  χ_0: {chi_0_values}")
    print(f"  α_Ncoh: {alpha_Ncoh_values}")
    print(f"  ℓ_0: {ell_0_values}")
    
    # Run sweep
    results = run_parameter_sweep(
        galaxies, chi_0_values, alpha_Ncoh_values, ell_0_values, verbose=True
    )
    
    if not results:
        print("\nNo valid parameter combinations found!")
        return
    
    # Find best
    best = find_best_params(results)
    
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"  χ_0 = {best.chi_0}")
    print(f"  α_Ncoh = {best.alpha_Ncoh}")
    print(f"  ℓ_0 = {best.ell_0} kpc")
    print(f"\nPerformance:")
    print(f"  Galaxies improved: {best.n_improved}/{best.n_total} ({best.pct_improved:.1f}%)")
    print(f"  Mean ΔRMS: {best.mean_delta_rms:+.2f} km/s")
    print(f"  Median ΔRMS: {best.median_delta_rms:+.2f} km/s")
    print(f"  Mean F_CMSI: {best.mean_F:.2f}")
    
    # Top 5 parameter sets
    sorted_results = sorted(results, key=lambda r: (-r.pct_improved, r.median_delta_rms))[:5]
    print("\nTop 5 parameter sets:")
    for i, r in enumerate(sorted_results):
        print(f"  {i+1}. χ_0={r.chi_0:5.0f} α={r.alpha_Ncoh:.2f} ℓ_0={r.ell_0:.1f} | "
              f"{r.pct_improved:.1f}% | ΔRMS={r.mean_delta_rms:+.2f} | F={r.mean_F:.2f}")
    
    # Save results
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    save_sweep_results(results, str(output_dir / "cmsi_sweep_results.csv"))
    save_best_params(best, galaxies, str(output_dir / "cmsi_best_params.json"))
    
    print("\n" + "=" * 70)
    print("Optimization complete.")


if __name__ == "__main__":
    main()
