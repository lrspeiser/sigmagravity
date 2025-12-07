#!/usr/bin/env python3
"""
Test Wake Coherence Model on SPARC Galaxies
============================================

This script tests the wake-based coherence model on SPARC galaxies
with bulge+disk decomposition data.

Key questions:
1. Does wake coherence improve fits in bulge-dominated inner regions?
2. Do bulge-heavy galaxies show systematically different behavior?
3. Can we predict the bulge→disk transition from wake physics?

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wake_coherence_model import (
    WakeParams, C_wake_discrete, C_wake_continuum,
    predict_velocity_baseline, predict_velocity_wake,
    exponential_disk_profile, sersic_bulge_profile,
    estimate_bulge_dispersion, compute_rms, compare_models,
    A_GALAXY, XI_SCALE, g_dagger, kpc_to_m
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc_galaxy(filepath: str) -> Dict:
    """
    Load a single SPARC galaxy rotation curve with component decomposition.
    
    Returns dict with:
    - R: radii (kpc)
    - V_obs: observed velocity (km/s)
    - V_gas, V_disk, V_bulge: component velocities
    - SBdisk, SBbul: surface brightness profiles
    - V_bar: total baryonic velocity
    """
    data = {
        'R': [], 'V_obs': [], 'V_err': [],
        'V_gas': [], 'V_disk': [], 'V_bulge': [],
        'SBdisk': [], 'SBbul': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    data['R'].append(float(parts[0]))
                    data['V_obs'].append(float(parts[1]))
                    data['V_err'].append(float(parts[2]))
                    data['V_gas'].append(float(parts[3]))
                    data['V_disk'].append(float(parts[4]))
                    data['V_bulge'].append(float(parts[5]))
                    if len(parts) >= 8:
                        data['SBdisk'].append(float(parts[6]))
                        data['SBbul'].append(float(parts[7]))
                except ValueError:
                    continue
    
    for key in data:
        data[key] = np.array(data[key])
    
    # Apply M/L corrections
    ML_DISK = 0.5
    ML_BULGE = 0.7
    
    V_disk_scaled = data['V_disk'] * np.sqrt(ML_DISK)
    V_bulge_scaled = data['V_bulge'] * np.sqrt(ML_BULGE)
    
    # Compute V_bar (handle signed gas velocities)
    V_gas_sq = np.sign(data['V_gas']) * data['V_gas']**2
    V_bar_sq = V_gas_sq + V_disk_scaled**2 + V_bulge_scaled**2
    data['V_bar'] = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
    
    # Store scaled components
    data['V_disk_scaled'] = V_disk_scaled
    data['V_bulge_scaled'] = V_bulge_scaled
    
    # Galaxy name
    data['name'] = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    return data


def load_sparc_master(master_file: str) -> pd.DataFrame:
    """
    Load SPARC master table with galaxy properties.
    
    Returns DataFrame with:
    - Galaxy name, morphological type, distance
    - L[3.6], R_disk, R_eff, M_HI
    - V_flat, quality flag
    """
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    # Find header and data start
    data_rows = []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 16:
                try:
                    row = {
                        'name': parts[0],
                        'T': float(parts[1]) if parts[1] != '---' else np.nan,
                        'D': float(parts[2]) if parts[2] != '---' else np.nan,
                        'L36': float(parts[7]) if parts[7] != '---' else np.nan,
                        'Reff': float(parts[9]) if parts[9] != '---' else np.nan,
                        'Rdisk': float(parts[11]) if parts[11] != '---' else np.nan,
                        'MHI': float(parts[13]) if parts[13] != '---' else np.nan,
                        'Vflat': float(parts[15]) if parts[15] != '---' else np.nan,
                    }
                    data_rows.append(row)
                except (ValueError, IndexError):
                    continue
    
    return pd.DataFrame(data_rows)


def estimate_bulge_fraction(gal: Dict) -> float:
    """
    Estimate bulge fraction from velocity components.
    
    f_bulge = V_bulge² / (V_disk² + V_bulge²) at R ~ R_eff
    """
    R = gal['R']
    V_disk = gal['V_disk_scaled']
    V_bulge = gal['V_bulge_scaled']
    
    # Use inner region (first 1/3 of points)
    n_inner = max(len(R) // 3, 3)
    
    V_disk_inner = np.mean(V_disk[:n_inner]**2)
    V_bulge_inner = np.mean(V_bulge[:n_inner]**2)
    
    total = V_disk_inner + V_bulge_inner
    if total > 0:
        return V_bulge_inner / total
    return 0.0


def estimate_surface_density_from_velocity(
    R: np.ndarray,
    V_disk: np.ndarray,
    V_bulge: np.ndarray,
    R_d: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate surface density profiles from velocity contributions.
    
    For an exponential disk: V² ∝ R × Σ(R) at small R
    We invert this approximately to get relative Σ profiles.
    """
    # Disk: exponential profile
    Sigma_d = np.maximum(V_disk**2, 1.0) * np.exp(-R / R_d) / np.maximum(R, 0.1)
    
    # Bulge: Sérsic-like (concentrated)
    R_e = R_d / 3  # Typical bulge effective radius
    Sigma_b = np.maximum(V_bulge**2, 0.1) * np.exp(-2 * R / R_e) / np.maximum(R, 0.1)
    
    # Normalize
    Sigma_d = Sigma_d / np.max(Sigma_d) if np.max(Sigma_d) > 0 else Sigma_d
    Sigma_b = Sigma_b / np.max(Sigma_b) if np.max(Sigma_b) > 0 else Sigma_b
    
    return Sigma_d * 100, Sigma_b * 100  # Arbitrary units


# =============================================================================
# WAKE ANALYSIS
# =============================================================================

@dataclass
class GalaxyResult:
    """Results for a single galaxy."""
    name: str
    n_points: int
    bulge_frac: float
    R_d: float
    rms_baseline: float
    rms_wake: float
    improvement: float
    C_wake_inner: float
    C_wake_outer: float


def analyze_galaxy(
    gal: Dict,
    R_d: float,
    params: WakeParams = WakeParams()
) -> GalaxyResult:
    """
    Analyze a single galaxy with wake coherence model.
    """
    R = gal['R']
    V_obs = gal['V_obs']
    V_bar = gal['V_bar']
    V_disk = gal['V_disk_scaled']
    V_bulge = gal['V_bulge_scaled']
    
    # Filter valid data
    valid = (V_bar > 0) & (V_obs > 0) & (R > 0)
    if valid.sum() < 5:
        return None
    
    R = R[valid]
    V_obs = V_obs[valid]
    V_bar = V_bar[valid]
    V_disk = V_disk[valid]
    V_bulge = V_bulge[valid]
    
    # Estimate bulge fraction
    bulge_frac = estimate_bulge_fraction(gal)
    
    # Estimate surface density profiles
    Sigma_d, Sigma_b = estimate_surface_density_from_velocity(R, V_disk, V_bulge, R_d)
    
    # Estimate bulge dispersion
    V_flat = np.median(V_obs[-3:]) if len(V_obs) >= 3 else np.max(V_obs)
    sigma_b = estimate_bulge_dispersion(V_flat, bulge_frac)
    
    # Compute wake coherence
    C_wake = C_wake_discrete(R, Sigma_d, Sigma_b, V_obs, sigma_b, params)
    
    # Compute velocities
    V_baseline = predict_velocity_baseline(R, V_bar, R_d)
    V_wake = predict_velocity_wake(R, V_bar, R_d, C_wake, beta=params.beta)
    
    # Metrics
    rms_baseline = compute_rms(V_obs, V_baseline)
    rms_wake = compute_rms(V_obs, V_wake)
    improvement = 1 - rms_wake / rms_baseline if rms_baseline > 0 else 0
    
    # Wake coherence at inner/outer regions
    n_third = len(R) // 3
    C_wake_inner = np.mean(C_wake[:max(n_third, 1)])
    C_wake_outer = np.mean(C_wake[-max(n_third, 1):])
    
    return GalaxyResult(
        name=gal['name'],
        n_points=len(R),
        bulge_frac=bulge_frac,
        R_d=R_d,
        rms_baseline=rms_baseline,
        rms_wake=rms_wake,
        improvement=improvement,
        C_wake_inner=C_wake_inner,
        C_wake_outer=C_wake_outer
    )


def run_sparc_analysis(
    data_dir: Path,
    params: WakeParams = WakeParams(),
    verbose: bool = True
) -> List[GalaxyResult]:
    """
    Run wake coherence analysis on all SPARC galaxies.
    """
    sparc_dir = data_dir / "Rotmod_LTG"
    master_file = sparc_dir / "MasterSheet_SPARC.mrt"
    
    if not sparc_dir.exists():
        print(f"Error: SPARC data not found at {sparc_dir}")
        return []
    
    # Load master table for R_d values
    master_df = None
    if master_file.exists():
        master_df = load_sparc_master(str(master_file))
        master_df = master_df.set_index('name')
    
    results = []
    galaxies_processed = 0
    
    for gf in sorted(sparc_dir.glob("*_rotmod.dat")):
        try:
            gal = load_sparc_galaxy(str(gf))
            
            # Get R_d from master table or estimate
            if master_df is not None and gal['name'] in master_df.index:
                R_d = master_df.loc[gal['name'], 'Rdisk']
                if np.isnan(R_d) or R_d <= 0:
                    R_d = gal['R'][len(gal['R']) // 3]  # Fallback
            else:
                R_d = gal['R'][len(gal['R']) // 3]  # Estimate
            
            result = analyze_galaxy(gal, R_d, params)
            if result is not None:
                results.append(result)
                galaxies_processed += 1
                
                if verbose and galaxies_processed <= 10:
                    print(f"  {result.name}: RMS {result.rms_baseline:.1f} → {result.rms_wake:.1f} "
                          f"({result.improvement*100:+.1f}%), f_bulge={result.bulge_frac:.2f}")
        
        except Exception as e:
            if verbose:
                print(f"  Skipped {gf.stem}: {e}")
    
    return results


# =============================================================================
# ANALYSIS SUMMARY
# =============================================================================

def summarize_results(results: List[GalaxyResult]) -> Dict:
    """
    Summarize wake coherence analysis results.
    """
    if not results:
        return {}
    
    # Convert to arrays
    rms_baseline = np.array([r.rms_baseline for r in results])
    rms_wake = np.array([r.rms_wake for r in results])
    improvement = np.array([r.improvement for r in results])
    bulge_frac = np.array([r.bulge_frac for r in results])
    C_wake_inner = np.array([r.C_wake_inner for r in results])
    C_wake_outer = np.array([r.C_wake_outer for r in results])
    
    # Overall metrics
    mean_rms_baseline = np.mean(rms_baseline)
    mean_rms_wake = np.mean(rms_wake)
    mean_improvement = np.mean(improvement)
    
    # Wins
    wins = np.sum(rms_wake < rms_baseline)
    win_rate = wins / len(results)
    
    # Bulge-heavy vs disk-dominated
    bulge_heavy = bulge_frac > 0.2
    disk_dominated = bulge_frac < 0.1
    
    improvement_bulge_heavy = np.mean(improvement[bulge_heavy]) if bulge_heavy.sum() > 0 else 0
    improvement_disk_dom = np.mean(improvement[disk_dominated]) if disk_dominated.sum() > 0 else 0
    
    # Correlation with C_wake difference
    C_wake_diff = C_wake_outer - C_wake_inner
    corr_improvement_Cdiff = np.corrcoef(improvement, C_wake_diff)[0, 1] if len(improvement) > 2 else 0
    
    return {
        'n_galaxies': len(results),
        'mean_rms_baseline': mean_rms_baseline,
        'mean_rms_wake': mean_rms_wake,
        'mean_improvement': mean_improvement,
        'win_rate': win_rate,
        'n_bulge_heavy': bulge_heavy.sum(),
        'n_disk_dominated': disk_dominated.sum(),
        'improvement_bulge_heavy': improvement_bulge_heavy,
        'improvement_disk_dominated': improvement_disk_dom,
        'corr_improvement_Cdiff': corr_improvement_Cdiff,
    }


def print_summary(summary: Dict, params: WakeParams):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print("WAKE COHERENCE MODEL - SPARC ANALYSIS RESULTS")
    print("=" * 70)
    
    print(f"\nParameters:")
    print(f"  α (velocity weight): {params.alpha}")
    print(f"  v₀ (reference vel): {params.v0} km/s")
    print(f"  β (sharpness): {params.beta}")
    
    print(f"\nOverall Performance ({summary['n_galaxies']} galaxies):")
    print(f"  Mean RMS (baseline): {summary['mean_rms_baseline']:.2f} km/s")
    print(f"  Mean RMS (wake):     {summary['mean_rms_wake']:.2f} km/s")
    print(f"  Mean improvement:    {summary['mean_improvement']*100:+.2f}%")
    print(f"  Win rate:            {summary['win_rate']*100:.1f}%")
    
    print(f"\nBulge vs Disk Analysis:")
    print(f"  Bulge-heavy (f_b > 0.2): {summary['n_bulge_heavy']} galaxies, "
          f"improvement = {summary['improvement_bulge_heavy']*100:+.2f}%")
    print(f"  Disk-dominated (f_b < 0.1): {summary['n_disk_dominated']} galaxies, "
          f"improvement = {summary['improvement_disk_dominated']*100:+.2f}%")
    
    print(f"\nCoherence Correlation:")
    print(f"  Corr(improvement, C_outer - C_inner) = {summary['corr_improvement_Cdiff']:.3f}")
    
    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if summary['improvement_bulge_heavy'] > summary['improvement_disk_dominated']:
        print("  ✓ Wake model helps MORE in bulge-heavy galaxies (as predicted)")
    else:
        print("  ✗ Wake model doesn't preferentially help bulge-heavy galaxies")
    
    if summary['corr_improvement_Cdiff'] > 0.1:
        print("  ✓ Galaxies with larger C_wake gradient show more improvement")
    elif summary['corr_improvement_Cdiff'] < -0.1:
        print("  ✗ Negative correlation with C_wake gradient (unexpected)")
    else:
        print("  ~ Weak correlation with C_wake gradient")
    
    print("=" * 70)


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def sweep_parameters(
    data_dir: Path,
    alpha_values: List[float] = [1.0, 1.5, 2.0],
    beta_values: List[float] = [0.5, 1.0, 1.5, 2.0],
    verbose: bool = False
) -> pd.DataFrame:
    """
    Sweep over wake model parameters to find optimal values.
    """
    results_list = []
    
    for alpha in alpha_values:
        for beta in beta_values:
            params = WakeParams(alpha=alpha, beta=beta)
            results = run_sparc_analysis(data_dir, params, verbose=False)
            
            if results:
                summary = summarize_results(results)
                summary['alpha'] = alpha
                summary['beta'] = beta
                results_list.append(summary)
                
                if verbose:
                    print(f"α={alpha}, β={beta}: RMS {summary['mean_rms_wake']:.2f} km/s, "
                          f"improvement {summary['mean_improvement']*100:+.1f}%")
    
    return pd.DataFrame(results_list)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run wake coherence analysis on SPARC galaxies."""
    data_dir = Path(__file__).parent.parent / "data"
    
    print("Wake Coherence Model - SPARC Galaxy Analysis")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print()
    
    # Default parameters
    params = WakeParams(alpha=1.5, beta=1.0, v0=200.0)
    
    print("Running analysis with default parameters...")
    print("(First 10 galaxies shown)")
    print()
    
    results = run_sparc_analysis(data_dir, params, verbose=True)
    
    if not results:
        print("\nNo results. Check data directory.")
        return
    
    # Summary
    summary = summarize_results(results)
    print_summary(summary, params)
    
    # Parameter sweep
    print("\n\nPARAMETER SWEEP")
    print("-" * 70)
    sweep_df = sweep_parameters(data_dir, verbose=True)
    
    if len(sweep_df) > 0:
        best_idx = sweep_df['mean_improvement'].idxmax()
        best = sweep_df.loc[best_idx]
        print(f"\nBest parameters: α={best['alpha']}, β={best['beta']}")
        print(f"  Improvement: {best['mean_improvement']*100:+.2f}%")
        print(f"  Win rate: {best['win_rate']*100:.1f}%")
    
    # Save detailed results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame([
        {
            'name': r.name,
            'n_points': r.n_points,
            'bulge_frac': r.bulge_frac,
            'R_d': r.R_d,
            'rms_baseline': r.rms_baseline,
            'rms_wake': r.rms_wake,
            'improvement': r.improvement,
            'C_wake_inner': r.C_wake_inner,
            'C_wake_outer': r.C_wake_outer,
        }
        for r in results
    ])
    
    results_df.to_csv(output_dir / "sparc_wake_results.csv", index=False)
    sweep_df.to_csv(output_dir / "parameter_sweep.csv", index=False)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Promising directions
    print("\n" + "=" * 70)
    print("PROMISING DIRECTIONS")
    print("=" * 70)
    
    # Find galaxies where wake model helps most
    results_df_sorted = results_df.sort_values('improvement', ascending=False)
    print("\nTop 5 galaxies where wake correction helps most:")
    for _, row in results_df_sorted.head(5).iterrows():
        print(f"  {row['name']}: {row['improvement']*100:+.1f}% improvement, "
              f"f_bulge={row['bulge_frac']:.2f}, "
              f"C_wake: {row['C_wake_inner']:.2f}→{row['C_wake_outer']:.2f}")
    
    # Find galaxies where it hurts
    print("\nTop 5 galaxies where wake correction hurts:")
    for _, row in results_df_sorted.tail(5).iterrows():
        print(f"  {row['name']}: {row['improvement']*100:+.1f}% improvement, "
              f"f_bulge={row['bulge_frac']:.2f}")


if __name__ == "__main__":
    main()

