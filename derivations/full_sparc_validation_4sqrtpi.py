#!/usr/bin/env python3
"""
Full SPARC Validation: g† = cH₀/(4√π) vs g† = cH₀/(2e)
=======================================================

This script performs a comprehensive comparison on the FULL SPARC dataset
(~175 galaxies) to validate the new critical acceleration formula.

Uses the actual SPARC rotation curve data files.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8              # Speed of light [m/s]
H0_SI = 2.27e-18         # Hubble constant [1/s] (70 km/s/Mpc)
cH0 = c * H0_SI          # c × H₀ [m/s²]
e = math.e               # Euler's number
kpc_to_m = 3.086e19      # meters per kpc

# Two formulas for critical acceleration
g_dagger_old = cH0 / (2 * e)              # Old: cH₀/(2e) ≈ 1.25×10⁻¹⁰ m/s²
g_dagger_new = cH0 / (4 * math.sqrt(math.pi))  # New: cH₀/(4√π) ≈ 0.96×10⁻¹⁰ m/s²

# MOND for comparison
a0_mond = 1.2e-10

print("=" * 80)
print("FULL SPARC VALIDATION: g† = cH₀/(4√π) vs g† = cH₀/(2e)")
print("=" * 80)
print(f"\nOld formula: g† = cH₀/(2e)   = {g_dagger_old:.4e} m/s²")
print(f"New formula: g† = cH₀/(4√π)  = {g_dagger_new:.4e} m/s²")
print(f"MOND a₀:                     = {a0_mond:.4e} m/s²")
print(f"Ratio (new/old): {g_dagger_new/g_dagger_old:.4f}")

# =============================================================================
# FIND SPARC DATA
# =============================================================================

def find_sparc_data() -> Optional[Path]:
    """Find the SPARC data directory."""
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/GravityCalculator/data/Rotmod_LTG"),
        Path(__file__).parent.parent / "data" / "Rotmod_LTG",
        Path(__file__).parent.parent / "many_path_model" / "paper_release" / "data" / "Rotmod_LTG",
    ]
    
    for p in possible_paths:
        if p.exists():
            return p
    return None


def find_master_sheet() -> Optional[Path]:
    """Find the SPARC master sheet with R_d values."""
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        return None
    
    # Try different possible names
    for name in ['MasterSheet_SPARC.mrt', 'SPARC_Lelli2016c.mrt', 'Table1.mrt']:
        p = sparc_dir / name
        if p.exists():
            return p
        p = sparc_dir.parent / name
        if p.exists():
            return p
    return None


# =============================================================================
# LOAD SPARC DATA
# =============================================================================

def load_master_sheet(master_file: Path) -> Dict[str, float]:
    """Load R_d (disk scale length) values from master sheet."""
    R_d_values = {}
    
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.split()
        if len(parts) < 15:
            continue
        
        name = parts[0]
        # Skip header lines
        if name.startswith('-') or name.startswith('=') or name.startswith('Note'):
            continue
        if name.startswith('Byte') or name.startswith('Title') or name.startswith('Table'):
            continue
        
        try:
            # Rdisk is typically the 12th field (index 11)
            R_d = float(parts[11])
            R_d_values[name] = R_d
        except (ValueError, IndexError):
            continue
    
    return R_d_values


def load_galaxy_rotmod(rotmod_file: Path) -> Optional[Dict]:
    """Load a single galaxy rotation curve from rotmod file."""
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    
    with open(rotmod_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))  # kpc
                    V_obs.append(float(parts[1]))  # km/s
                    V_err.append(float(parts[2]))  # km/s
                    V_gas.append(float(parts[3]))  # km/s
                    V_disk.append(float(parts[4]))  # km/s
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
                except ValueError:
                    continue
    
    if len(R) < 3:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    
    # Compute V_bar (handle negative values for counter-rotation)
    V_bar = np.sqrt(
        np.sign(V_gas) * V_gas**2 + 
        np.sign(V_disk) * V_disk**2 + 
        V_bulge**2
    )
    
    return {
        'R': R,
        'V_obs': V_obs,
        'V_err': V_err,
        'V_gas': V_gas,
        'V_disk': V_disk,
        'V_bulge': V_bulge,
        'V_bar': V_bar
    }


def load_all_sparc_galaxies(sparc_dir: Path, R_d_values: Dict[str, float]) -> Dict[str, Dict]:
    """Load all SPARC galaxies with valid data."""
    galaxies = {}
    
    for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        data = load_galaxy_rotmod(rotmod_file)
        if data is None:
            continue
        
        # Get R_d if available, otherwise estimate from data
        if name in R_d_values:
            data['R_d'] = R_d_values[name]
        else:
            # Estimate R_d as 1/3 of maximum radius
            data['R_d'] = data['R'].max() / 3.0
        
        galaxies[name] = data
    
    return galaxies


# =============================================================================
# Σ-GRAVITY KERNEL FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray, g_dagger: float) -> np.ndarray:
    """Universal enhancement function: h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r: np.ndarray, R_d: float) -> np.ndarray:
    """Coherence window: W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d"""
    xi = (2/3) * R_d
    return 1 - (xi / (xi + r)) ** 0.5


def predict_velocity_sigma(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float, 
                           g_dagger: float, A: float = np.sqrt(3)) -> np.ndarray:
    """Predict rotation curve using Σ-Gravity formula."""
    # Convert V_bar to g_bar
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    # Compute enhancement
    h = h_function(g_bar, g_dagger)
    W = W_coherence(R_kpc, R_d)
    Sigma = 1 + A * W * h
    
    # Predict velocity
    V_pred = V_bar * np.sqrt(Sigma)
    return V_pred


def predict_velocity_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """Predict rotation curve using MOND simple interpolation."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    # MOND interpolation: g_obs = g_bar × ν(g_bar/a0)
    # Simple form: ν(x) = 1/(1 - exp(-√x))
    x = g_bar / a0_mond
    x = np.maximum(x, 1e-10)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    
    g_obs = g_bar * nu
    V_pred = np.sqrt(g_obs * R_m) / 1000  # back to km/s
    return V_pred


# =============================================================================
# METRICS
# =============================================================================

def compute_rms_error(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """Compute RMS velocity error in km/s."""
    return np.sqrt(np.mean((V_obs - V_pred)**2))


def compute_rar_scatter(R_kpc: np.ndarray, V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """Compute RAR scatter in dex."""
    R_m = R_kpc * kpc_to_m
    g_obs = (V_obs * 1000)**2 / R_m
    g_pred = (V_pred * 1000)**2 / R_m
    
    mask = (g_obs > 0) & (g_pred > 0)
    if mask.sum() < 2:
        return np.nan
    
    log_residual = np.log10(g_obs[mask] / g_pred[mask])
    return np.std(log_residual)


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_full_validation():
    """Run full SPARC validation comparing old vs new formula."""
    
    # Find data
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        print("\nERROR: SPARC data not found!")
        print("Please download from http://astroweb.cwru.edu/SPARC/")
        print("and place in data/Rotmod_LTG/")
        return None
    
    print(f"\nSPARC data found: {sparc_dir}")
    
    # Load master sheet for R_d values
    master_file = find_master_sheet()
    R_d_values = {}
    if master_file:
        R_d_values = load_master_sheet(master_file)
        print(f"Master sheet loaded: {len(R_d_values)} R_d values")
    else:
        print("Master sheet not found, will estimate R_d from data")
    
    # Load all galaxies
    galaxies = load_all_sparc_galaxies(sparc_dir, R_d_values)
    print(f"Loaded {len(galaxies)} galaxies")
    
    if len(galaxies) == 0:
        print("ERROR: No galaxies loaded!")
        return None
    
    # Run comparison
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON")
    print("=" * 80)
    
    results = []
    A = np.sqrt(3)  # Amplitude for disks
    
    for name, data in galaxies.items():
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        R_d = data['R_d']
        
        # Skip if V_bar has issues
        if np.any(np.isnan(V_bar)) or np.any(V_bar <= 0):
            continue
        
        # Old formula
        V_pred_old = predict_velocity_sigma(R, V_bar, R_d, g_dagger_old, A)
        rms_old = compute_rms_error(V_obs, V_pred_old)
        rar_old = compute_rar_scatter(R, V_obs, V_pred_old)
        
        # New formula
        V_pred_new = predict_velocity_sigma(R, V_bar, R_d, g_dagger_new, A)
        rms_new = compute_rms_error(V_obs, V_pred_new)
        rar_new = compute_rar_scatter(R, V_obs, V_pred_new)
        
        # MOND for comparison
        V_pred_mond = predict_velocity_mond(R, V_bar)
        rms_mond = compute_rms_error(V_obs, V_pred_mond)
        rar_mond = compute_rar_scatter(R, V_obs, V_pred_mond)
        
        results.append({
            'name': name,
            'n_points': len(R),
            'rms_old': rms_old,
            'rms_new': rms_new,
            'rms_mond': rms_mond,
            'rar_old': rar_old,
            'rar_new': rar_new,
            'rar_mond': rar_mond,
        })
    
    # Summary statistics
    n_galaxies = len(results)
    
    rms_old_list = [r['rms_old'] for r in results]
    rms_new_list = [r['rms_new'] for r in results]
    rms_mond_list = [r['rms_mond'] for r in results]
    
    rar_old_list = [r['rar_old'] for r in results if not np.isnan(r['rar_old'])]
    rar_new_list = [r['rar_new'] for r in results if not np.isnan(r['rar_new'])]
    rar_mond_list = [r['rar_mond'] for r in results if not np.isnan(r['rar_mond'])]
    
    mean_rms_old = np.mean(rms_old_list)
    mean_rms_new = np.mean(rms_new_list)
    mean_rms_mond = np.mean(rms_mond_list)
    
    median_rms_old = np.median(rms_old_list)
    median_rms_new = np.median(rms_new_list)
    median_rms_mond = np.median(rms_mond_list)
    
    mean_rar_old = np.mean(rar_old_list)
    mean_rar_new = np.mean(rar_new_list)
    mean_rar_mond = np.mean(rar_mond_list)
    
    median_rar_old = np.median(rar_old_list)
    median_rar_new = np.median(rar_new_list)
    median_rar_mond = np.median(rar_mond_list)
    
    # Count wins (by RMS)
    wins_old_vs_new = sum(1 for r in results if r['rms_old'] < r['rms_new'])
    wins_new_vs_old = sum(1 for r in results if r['rms_new'] < r['rms_old'])
    wins_new_vs_mond = sum(1 for r in results if r['rms_new'] < r['rms_mond'])
    wins_mond_vs_new = sum(1 for r in results if r['rms_mond'] < r['rms_new'])
    
    # Count wins (by RAR)
    wins_old_rar = sum(1 for r in results if not np.isnan(r['rar_old']) and not np.isnan(r['rar_new']) and r['rar_old'] < r['rar_new'])
    wins_new_rar = sum(1 for r in results if not np.isnan(r['rar_old']) and not np.isnan(r['rar_new']) and r['rar_new'] < r['rar_old'])
    
    # Print results
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal galaxies analyzed: {n_galaxies}")
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              RMS VELOCITY ERROR (km/s)                               ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ Formula                │ Mean RMS │ Median RMS │ Galaxies Won (vs other Σ)           ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ OLD: g† = cH₀/(2e)     │ {mean_rms_old:>8.2f} │ {median_rms_old:>10.2f} │ {wins_old_vs_new:>36}  ║
║ NEW: g† = cH₀/(4√π)    │ {mean_rms_new:>8.2f} │ {median_rms_new:>10.2f} │ {wins_new_vs_old:>36}  ║
║ MOND (simple)          │ {mean_rms_mond:>8.2f} │ {median_rms_mond:>10.2f} │ (comparison reference)             ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════════════╗
║                              RAR SCATTER (dex)                                       ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ Formula                │ Mean RAR │ Median RAR │ Galaxies Won (vs other Σ)           ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║ OLD: g† = cH₀/(2e)     │ {mean_rar_old:>8.4f} │ {median_rar_old:>10.4f} │ {wins_old_rar:>36}  ║
║ NEW: g† = cH₀/(4√π)    │ {mean_rar_new:>8.4f} │ {median_rar_new:>10.4f} │ {wins_new_rar:>36}  ║
║ MOND (simple)          │ {mean_rar_mond:>8.4f} │ {median_rar_mond:>10.4f} │ (comparison reference)             ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Improvement calculation
    rms_improvement = 100 * (mean_rms_old - mean_rms_new) / mean_rms_old
    rar_improvement = 100 * (mean_rar_old - mean_rar_new) / mean_rar_old
    
    print(f"""
IMPROVEMENT WITH NEW FORMULA:
  RMS: {rms_improvement:+.1f}% {'(BETTER)' if rms_improvement > 0 else '(WORSE)'}
  RAR: {rar_improvement:+.1f}% {'(BETTER)' if rar_improvement > 0 else '(WORSE)'}

HEAD-TO-HEAD (by RMS):
  Old wins: {wins_old_vs_new}
  New wins: {wins_new_vs_old}

HEAD-TO-HEAD (by RAR):
  Old wins: {wins_old_rar}
  New wins: {wins_new_rar}

NEW vs MOND (by RMS):
  New Σ wins: {wins_new_vs_mond}
  MOND wins: {wins_mond_vs_new}
""")
    
    # Top 10 improvements and regressions
    improvements = [(r['name'], r['rms_old'] - r['rms_new']) for r in results]
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTOP 10 IMPROVEMENTS (Old RMS - New RMS):")
    for name, diff in improvements[:10]:
        print(f"  {name}: {diff:+.2f} km/s")
    
    print("\nTOP 10 REGRESSIONS (New worse than Old):")
    for name, diff in improvements[-10:]:
        print(f"  {name}: {diff:+.2f} km/s")
    
    return {
        'n_galaxies': n_galaxies,
        'mean_rms_old': mean_rms_old,
        'mean_rms_new': mean_rms_new,
        'mean_rms_mond': mean_rms_mond,
        'mean_rar_old': mean_rar_old,
        'mean_rar_new': mean_rar_new,
        'mean_rar_mond': mean_rar_mond,
        'rms_improvement_pct': rms_improvement,
        'rar_improvement_pct': rar_improvement,
        'wins_old': wins_old_vs_new,
        'wins_new': wins_new_vs_old,
        'results': results,
    }


if __name__ == "__main__":
    results = run_full_validation()
    
    if results is not None:
        print("\n" + "=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)
        
        if results['rms_improvement_pct'] > 0 and results['rar_improvement_pct'] > 0:
            print("""
✓ The NEW formula g† = cH₀/(4√π) is BETTER than the old formula g† = cH₀/(2e)

RECOMMENDATION: Update all Σ-Gravity code to use g† = cH₀/(4√π)

This change:
1. Improves rotation curve fits
2. Eliminates the arbitrary constant 'e'
3. Uses only geometric constants (√π from solid angle)
4. Has clear physical interpretation (acceleration at 2×R_coh)
""")
        elif results['rms_improvement_pct'] >= -5:
            print("""
≈ The NEW formula performs COMPARABLY to the old formula

RECOMMENDATION: Consider updating for theoretical elegance
""")
        else:
            print("""
✗ The NEW formula performs WORSE than the old formula

RECOMMENDATION: Keep the old formula g† = cH₀/(2e)
""")

