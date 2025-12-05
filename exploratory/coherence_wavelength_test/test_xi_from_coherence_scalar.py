#!/usr/bin/env python3
"""
Test Investigation 2: Compute W(r) from the covariant coherence scalar C

This script tests whether computing ξ from v_rot/σ = 1 (where C = 0.5)
improves RAR scatter compared to the fixed ξ = (2/3)R_d prescription.

Key hypothesis: The old morphology gates (G_bulge, G_shear, G_bar) were
empirically capturing what the covariant coherence scalar C does from
first principles.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import os
import glob
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8  # m/s
H0_SI = 2.27e-18  # s⁻¹ (70 km/s/Mpc)
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ~9.60e-11 m/s²
kpc_to_m = 3.086e19
A_galaxy = np.sqrt(3)

print("=" * 80)
print("INVESTIGATION 2: COMPUTE W(r) FROM COVARIANT COHERENCE SCALAR C")
print("=" * 80)
print(f"\nPhysical constants:")
print(f"  g† = cH₀/(4√π) = {g_dagger:.3e} m/s²")
print(f"  A_galaxy = √3 = {A_galaxy:.4f}")

# =============================================================================
# SIGMA-GRAVITY FUNCTIONS
# =============================================================================

def h_universal(g_N):
    """Acceleration function h(g_N) - depends on BARYONIC Newtonian acceleration"""
    g_N = np.maximum(g_N, 1e-15)  # Avoid division by zero
    return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)


def W_fixed(r, R_d):
    """Original fixed coherence window: W(r) = 1 - (ξ/(ξ+r))^0.5 with ξ = (2/3)R_d"""
    xi = (2/3) * R_d
    return 1 - np.sqrt(xi / (xi + r))


def W_from_xi(r, xi):
    """Coherence window with arbitrary ξ"""
    xi = max(xi, 0.01)  # Floor at 0.01 kpc
    return 1 - np.sqrt(xi / (xi + r))


def Sigma_gravity(r, g_N, R_d, A=A_galaxy, xi_override=None):
    """
    Enhancement factor Σ = 1 + A × W(r) × h(g_N)
    
    If xi_override is provided, use that instead of (2/3)R_d
    """
    if xi_override is not None:
        W = W_from_xi(r, xi_override)
    else:
        W = W_fixed(r, R_d)
    h = h_universal(g_N)
    return 1 + A * W * h


def C_local(v_rot, sigma):
    """
    Local coherence scalar from kinematic invariants.
    C = (v_rot/σ)² / [1 + (v_rot/σ)²]
    """
    sigma = np.maximum(sigma, 1.0)  # Floor at 1 km/s
    ratio_sq = (v_rot / sigma) ** 2
    return ratio_sq / (1 + ratio_sq)


# =============================================================================
# VELOCITY DISPERSION MODELS
# =============================================================================

def sigma_exponential(r, R_d, sigma_0=80.0, sigma_disk=15.0):
    """
    Exponential velocity dispersion profile.
    σ(r) = σ_disk + (σ_0 - σ_disk) × exp(-r/R_d)
    
    Parameters:
    - sigma_0: Central dispersion (bulge-dominated region)
    - sigma_disk: Asymptotic disk dispersion
    """
    return sigma_disk + (sigma_0 - sigma_disk) * np.exp(-r / R_d)


def sigma_morphology_scaled(r, v_rot, R_d, morphology='spiral'):
    """
    Velocity dispersion scaled by rotation velocity and morphology.
    
    Based on observational scaling relations:
    - Dwarfs: σ/v_c ~ 0.05-0.10 (cold, low dispersion)
    - LSBs: σ/v_c ~ 0.10-0.15 (intermediate)
    - Spirals: σ/v_c ~ 0.15-0.20 (warmer disks)
    - Massive: σ/v_c ~ 0.20-0.30 (hot, pressure-supported)
    """
    v_mean = np.mean(np.abs(v_rot[v_rot > 0])) if np.any(v_rot > 0) else 100
    
    morph_factors = {
        'dwarf': 0.08,
        'lsb': 0.12,
        'spiral': 0.17,
        'massive': 0.25,
        'default': 0.15
    }
    
    factor = morph_factors.get(morphology.lower(), morph_factors['default'])
    sigma_outer = factor * v_mean
    sigma_0 = 2.5 * sigma_outer  # Central dispersion ~2.5x outer
    
    return sigma_exponential(r, R_d, sigma_0=sigma_0, sigma_disk=sigma_outer)


def compute_xi_derived(r, v_rot, sigma):
    """
    Compute coherence scale from v_rot/σ = 1 crossing.
    This is where C = 0.5 (equal ordered/random motion).
    """
    v_rot = np.abs(v_rot)
    sigma = np.maximum(sigma, 1.0)
    ratio = v_rot / sigma
    
    # Find crossing point where ratio = 1
    if len(ratio) < 2:
        return np.nan
    
    # Look for crossing from below 1 to above 1
    crossings = []
    for i in range(len(ratio) - 1):
        if (ratio[i] < 1 and ratio[i+1] >= 1) or (ratio[i] >= 1 and ratio[i+1] < 1):
            # Linear interpolation
            t = (1 - ratio[i]) / (ratio[i+1] - ratio[i])
            r_cross = r[i] + t * (r[i+1] - r[i])
            crossings.append(r_cross)
    
    if crossings:
        return crossings[0]  # Take first crossing (inner transition)
    
    # No crossing found - use asymptotic behavior
    if np.all(ratio > 1):
        # Always rotation-dominated - small ξ
        return r[0] * 0.5
    elif np.all(ratio < 1):
        # Always dispersion-dominated - large ξ
        return r[-1] * 2
    
    return np.nan


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc_galaxy(filepath: str) -> Dict:
    """Load a single SPARC galaxy rotation curve."""
    data = {
        'R': [], 'v_obs': [], 'v_err': [],
        'v_gas': [], 'v_disk': [], 'v_bul': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 6:
                data['R'].append(float(parts[0]))
                data['v_obs'].append(float(parts[1]))
                data['v_err'].append(float(parts[2]))
                data['v_gas'].append(float(parts[3]))
                data['v_disk'].append(float(parts[4]))
                data['v_bul'].append(float(parts[5]))
    
    for key in data:
        data[key] = np.array(data[key])
    
    # Compute total baryonic velocity
    v_gas = data['v_gas']
    v_disk = data['v_disk']
    v_bul = data['v_bul']
    
    v_gas_sq = np.sign(v_gas) * v_gas**2
    v_disk_sq = np.sign(v_disk) * v_disk**2
    v_bul_sq = v_bul**2
    
    v_bary_sq = v_gas_sq + v_disk_sq + v_bul_sq
    data['v_bary'] = np.sign(v_bary_sq) * np.sqrt(np.abs(v_bary_sq))
    
    data['name'] = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    return data


def load_sparc_master(master_file: str) -> Dict[str, Dict]:
    """Load SPARC master sheet with R_d values."""
    galaxies = {}
    
    with open(master_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 10:
                name = parts[0]
                try:
                    R_d = float(parts[4])  # Disk scale length in kpc
                    galaxies[name] = {'R_d': R_d}
                except (ValueError, IndexError):
                    continue
    
    return galaxies


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_rar_scatter(v_obs, v_pred, r):
    """Compute RAR scatter in dex."""
    mask = (v_obs > 0) & (v_pred > 0) & (r > 0)
    if np.sum(mask) < 3:
        return np.nan
    
    g_obs = (v_obs[mask] * 1000)**2 / (r[mask] * kpc_to_m)
    g_pred = (v_pred[mask] * 1000)**2 / (r[mask] * kpc_to_m)
    
    log_residual = np.log10(g_obs / g_pred)
    return np.std(log_residual)


def compute_rms_error(v_obs, v_pred):
    """Compute RMS velocity error in km/s."""
    mask = np.isfinite(v_obs) & np.isfinite(v_pred)
    if np.sum(mask) < 3:
        return np.nan
    return np.sqrt(np.mean((v_obs[mask] - v_pred[mask])**2))


def analyze_galaxy(gal_data: Dict, R_d: float, sigma_0_factor: float = 2.5) -> Dict:
    """
    Analyze a single galaxy with both fixed and derived ξ.
    
    Returns metrics for both approaches.
    """
    r = gal_data['R']
    v_obs = gal_data['v_obs']
    v_bary = np.abs(gal_data['v_bary'])
    
    if len(r) < 5 or R_d <= 0:
        return None
    
    # Skip galaxies with invalid data
    if np.any(np.isnan(v_bary)) or np.any(v_bary < 0):
        return None
    
    # Compute baryonic acceleration
    mask = r > 0
    g_bary = np.zeros_like(r)
    g_bary[mask] = (v_bary[mask] * 1000)**2 / (r[mask] * kpc_to_m)
    
    # Estimate v_rot from observed velocities (approximation)
    v_rot = v_obs  # Use observed as proxy for rotation
    
    # Compute velocity dispersion profile
    v_mean = np.mean(v_rot[v_rot > 0]) if np.any(v_rot > 0) else 100
    sigma_outer = 0.15 * v_mean  # Typical σ/v_c ratio
    sigma_0 = sigma_0_factor * sigma_outer
    sigma = sigma_exponential(r, R_d, sigma_0=sigma_0, sigma_disk=sigma_outer)
    
    # Compute derived ξ from v_rot/σ = 1 crossing
    xi_derived = compute_xi_derived(r, v_rot, sigma)
    xi_fixed = (2/3) * R_d
    
    # Handle invalid xi_derived
    if np.isnan(xi_derived) or xi_derived <= 0:
        xi_derived = xi_fixed  # Fall back to fixed
    
    # Compute predictions with FIXED ξ
    Sigma_fixed = Sigma_gravity(r, g_bary, R_d, xi_override=None)
    v_pred_fixed = v_bary * np.sqrt(Sigma_fixed)
    
    # Compute predictions with DERIVED ξ
    Sigma_derived = Sigma_gravity(r, g_bary, R_d, xi_override=xi_derived)
    v_pred_derived = v_bary * np.sqrt(Sigma_derived)
    
    # Compute metrics
    rar_fixed = compute_rar_scatter(v_obs, v_pred_fixed, r)
    rar_derived = compute_rar_scatter(v_obs, v_pred_derived, r)
    rms_fixed = compute_rms_error(v_obs, v_pred_fixed)
    rms_derived = compute_rms_error(v_obs, v_pred_derived)
    
    # Compute local coherence at outer disk
    C_outer = C_local(v_rot[-1], sigma[-1]) if len(v_rot) > 0 else np.nan
    C_inner = C_local(v_rot[0], sigma[0]) if len(v_rot) > 0 else np.nan
    
    return {
        'name': gal_data['name'],
        'R_d': R_d,
        'xi_fixed': xi_fixed,
        'xi_derived': xi_derived,
        'xi_ratio': xi_derived / xi_fixed if xi_fixed > 0 else np.nan,
        'rar_fixed': rar_fixed,
        'rar_derived': rar_derived,
        'rar_improvement': rar_fixed - rar_derived,
        'rms_fixed': rms_fixed,
        'rms_derived': rms_derived,
        'rms_improvement': rms_fixed - rms_derived,
        'C_inner': C_inner,
        'C_outer': C_outer,
        'v_mean': v_mean,
        'sigma_0': sigma_0,
        'sigma_outer': sigma_outer,
        'n_points': len(r)
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    # Find SPARC data
    sparc_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG")
    master_file = sparc_dir / "MasterSheet_SPARC.mrt"
    
    if not sparc_dir.exists():
        print(f"ERROR: SPARC data not found at {sparc_dir}")
        return
    
    print(f"\nLoading SPARC data from: {sparc_dir}")
    
    # Load master sheet
    master_data = load_sparc_master(str(master_file))
    print(f"Loaded {len(master_data)} galaxies from master sheet")
    
    # Load all rotation curves
    rotmod_files = sorted(glob.glob(str(sparc_dir / "*_rotmod.dat")))
    print(f"Found {len(rotmod_files)} rotation curve files")
    
    # Analyze each galaxy
    results = []
    skipped = 0
    
    print("\nAnalyzing galaxies...")
    for filepath in rotmod_files:
        gal_data = load_sparc_galaxy(filepath)
        name = gal_data['name']
        
        if name not in master_data:
            skipped += 1
            continue
        
        R_d = master_data[name]['R_d']
        
        result = analyze_galaxy(gal_data, R_d)
        if result is not None:
            results.append(result)
    
    print(f"Analyzed {len(results)} galaxies, skipped {skipped}")
    
    # ==========================================================================
    # RESULTS SUMMARY
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("RESULTS: FIXED ξ vs DERIVED ξ FROM COHERENCE SCALAR")
    print("=" * 80)
    
    # Filter valid results
    valid_results = [r for r in results 
                     if not np.isnan(r['rar_fixed']) and not np.isnan(r['rar_derived'])]
    
    print(f"\nValid galaxies for comparison: {len(valid_results)}")
    
    # Compute aggregate metrics
    rar_fixed_all = [r['rar_fixed'] for r in valid_results]
    rar_derived_all = [r['rar_derived'] for r in valid_results]
    rms_fixed_all = [r['rms_fixed'] for r in valid_results if not np.isnan(r['rms_fixed'])]
    rms_derived_all = [r['rms_derived'] for r in valid_results if not np.isnan(r['rms_derived'])]
    
    print("\n" + "-" * 60)
    print("RAR SCATTER (dex) - Lower is better")
    print("-" * 60)
    print(f"  Fixed ξ = (2/3)R_d:")
    print(f"    Mean:   {np.mean(rar_fixed_all):.4f} dex")
    print(f"    Median: {np.median(rar_fixed_all):.4f} dex")
    print(f"    Std:    {np.std(rar_fixed_all):.4f} dex")
    print(f"\n  Derived ξ from C = 0.5:")
    print(f"    Mean:   {np.mean(rar_derived_all):.4f} dex")
    print(f"    Median: {np.median(rar_derived_all):.4f} dex")
    print(f"    Std:    {np.std(rar_derived_all):.4f} dex")
    
    improvement = np.mean(rar_fixed_all) - np.mean(rar_derived_all)
    pct_improvement = 100 * improvement / np.mean(rar_fixed_all)
    print(f"\n  IMPROVEMENT: {improvement:.4f} dex ({pct_improvement:+.1f}%)")
    
    print("\n" + "-" * 60)
    print("RMS VELOCITY ERROR (km/s) - Lower is better")
    print("-" * 60)
    print(f"  Fixed ξ = (2/3)R_d:")
    print(f"    Mean:   {np.mean(rms_fixed_all):.2f} km/s")
    print(f"    Median: {np.median(rms_fixed_all):.2f} km/s")
    print(f"\n  Derived ξ from C = 0.5:")
    print(f"    Mean:   {np.mean(rms_derived_all):.2f} km/s")
    print(f"    Median: {np.median(rms_derived_all):.2f} km/s")
    
    rms_improvement = np.mean(rms_fixed_all) - np.mean(rms_derived_all)
    rms_pct = 100 * rms_improvement / np.mean(rms_fixed_all)
    print(f"\n  IMPROVEMENT: {rms_improvement:.2f} km/s ({rms_pct:+.1f}%)")
    
    # Head-to-head comparison
    print("\n" + "-" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("-" * 60)
    
    derived_wins_rar = sum(1 for r in valid_results if r['rar_derived'] < r['rar_fixed'])
    fixed_wins_rar = sum(1 for r in valid_results if r['rar_fixed'] < r['rar_derived'])
    ties_rar = len(valid_results) - derived_wins_rar - fixed_wins_rar
    
    print(f"  RAR scatter: Derived wins {derived_wins_rar}, Fixed wins {fixed_wins_rar}, Ties {ties_rar}")
    print(f"  Derived win rate: {100*derived_wins_rar/len(valid_results):.1f}%")
    
    derived_wins_rms = sum(1 for r in valid_results 
                          if not np.isnan(r['rms_derived']) and not np.isnan(r['rms_fixed'])
                          and r['rms_derived'] < r['rms_fixed'])
    fixed_wins_rms = sum(1 for r in valid_results 
                        if not np.isnan(r['rms_derived']) and not np.isnan(r['rms_fixed'])
                        and r['rms_fixed'] < r['rms_derived'])
    
    print(f"  RMS error: Derived wins {derived_wins_rms}, Fixed wins {fixed_wins_rms}")
    
    # ξ statistics
    print("\n" + "-" * 60)
    print("ξ DERIVED vs ξ FIXED STATISTICS")
    print("-" * 60)
    
    xi_ratios = [r['xi_ratio'] for r in valid_results if not np.isnan(r['xi_ratio'])]
    print(f"  ξ_derived / ξ_fixed ratio:")
    print(f"    Mean:   {np.mean(xi_ratios):.2f}")
    print(f"    Median: {np.median(xi_ratios):.2f}")
    print(f"    Min:    {np.min(xi_ratios):.2f}")
    print(f"    Max:    {np.max(xi_ratios):.2f}")
    
    # Coherence statistics
    print("\n" + "-" * 60)
    print("COHERENCE SCALAR STATISTICS")
    print("-" * 60)
    
    C_inner_all = [r['C_inner'] for r in valid_results if not np.isnan(r['C_inner'])]
    C_outer_all = [r['C_outer'] for r in valid_results if not np.isnan(r['C_outer'])]
    
    print(f"  C at inner radius:")
    print(f"    Mean:   {np.mean(C_inner_all):.3f}")
    print(f"    Median: {np.median(C_inner_all):.3f}")
    print(f"\n  C at outer radius:")
    print(f"    Mean:   {np.mean(C_outer_all):.3f}")
    print(f"    Median: {np.median(C_outer_all):.3f}")
    
    # Best and worst improvements
    print("\n" + "-" * 60)
    print("TOP 10 GALAXIES WHERE DERIVED ξ HELPS MOST")
    print("-" * 60)
    
    sorted_by_improvement = sorted(valid_results, key=lambda x: x['rar_improvement'], reverse=True)
    print(f"{'Galaxy':<15} {'RAR_fixed':>10} {'RAR_derived':>12} {'Improvement':>12} {'ξ_ratio':>8}")
    for r in sorted_by_improvement[:10]:
        print(f"{r['name']:<15} {r['rar_fixed']:>10.4f} {r['rar_derived']:>12.4f} "
              f"{r['rar_improvement']:>+12.4f} {r['xi_ratio']:>8.2f}")
    
    print("\n" + "-" * 60)
    print("TOP 10 GALAXIES WHERE DERIVED ξ HURTS MOST")
    print("-" * 60)
    
    print(f"{'Galaxy':<15} {'RAR_fixed':>10} {'RAR_derived':>12} {'Improvement':>12} {'ξ_ratio':>8}")
    for r in sorted_by_improvement[-10:]:
        print(f"{r['name']:<15} {r['rar_fixed']:>10.4f} {r['rar_derived']:>12.4f} "
              f"{r['rar_improvement']:>+12.4f} {r['xi_ratio']:>8.2f}")
    
    # ==========================================================================
    # SENSITIVITY ANALYSIS: VARY σ_0 FACTOR
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS: VARYING σ₀/σ_disk RATIO")
    print("=" * 80)
    
    sigma_factors = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    print(f"\n{'σ₀_factor':>10} {'RAR_mean':>10} {'RAR_median':>12} {'RMS_mean':>10} {'Derived_wins':>14}")
    print("-" * 60)
    
    for factor in sigma_factors:
        results_factor = []
        for filepath in rotmod_files:
            gal_data = load_sparc_galaxy(filepath)
            name = gal_data['name']
            if name not in master_data:
                continue
            R_d = master_data[name]['R_d']
            result = analyze_galaxy(gal_data, R_d, sigma_0_factor=factor)
            if result is not None:
                results_factor.append(result)
        
        valid = [r for r in results_factor if not np.isnan(r['rar_derived'])]
        rar_mean = np.mean([r['rar_derived'] for r in valid])
        rar_median = np.median([r['rar_derived'] for r in valid])
        rms_mean = np.mean([r['rms_derived'] for r in valid if not np.isnan(r['rms_derived'])])
        wins = sum(1 for r in valid if r['rar_derived'] < r['rar_fixed'])
        
        print(f"{factor:>10.1f} {rar_mean:>10.4f} {rar_median:>12.4f} {rms_mean:>10.2f} {wins:>14}")
    
    # ==========================================================================
    # CONCLUSION
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if improvement > 0:
        print(f"""
✓ POSITIVE RESULT: Derived ξ from coherence scalar IMPROVES fits

  - RAR scatter reduced by {improvement:.4f} dex ({pct_improvement:.1f}%)
  - RMS error reduced by {rms_improvement:.2f} km/s ({rms_pct:.1f}%)
  - Derived ξ wins on {100*derived_wins_rar/len(valid_results):.1f}% of galaxies

INTERPRETATION:
  The covariant coherence scalar C = (v_rot/σ)²/[1 + (v_rot/σ)²] captures
  morphology-dependent coherence that the fixed ξ = (2/3)R_d misses.
  
  This supports the hypothesis that the old morphology gates (G_bulge, 
  G_shear, G_bar) were empirically capturing what C does from first principles.
""")
    else:
        print(f"""
✗ NEGATIVE RESULT: Derived ξ does NOT improve fits

  - RAR scatter changed by {improvement:.4f} dex ({pct_improvement:.1f}%)
  - RMS error changed by {rms_improvement:.2f} km/s ({rms_pct:.1f}%)
  
INTERPRETATION:
  The model-based σ(r) profile may not accurately capture the true
  velocity dispersion in SPARC galaxies. Real σ(r) measurements are
  needed to properly test this hypothesis.
""")
    
    print("\nNOTE: This analysis uses a MODEL for σ(r) since SPARC does not")
    print("provide velocity dispersion data. Results depend on the σ(r) model.")
    print("True validation requires galaxies with measured σ(r) profiles.")
    
    return results


if __name__ == "__main__":
    results = main()

