#!/usr/bin/env python3
"""
Test 2A: Phase-Dependent Coherence in Gaia DR3
================================================

This script tests whether gravitational coherence depends on orbital phase
alignment by comparing velocity correlations in:

1. THIN DISK: Cold, co-rotating → should show FULL coherence
2. THICK DISK: Warmer, co-rotating → should show REDUCED coherence  
3. HALO/RETROGRADE: Hot, some counter-rotating → should show NO coherence

If phase matters: C_thin(r) >> C_thick(r) >> C_halo(r) ≈ 0
If phase doesn't matter: C_thin(r) ≈ C_thick(r) ≈ C_halo(r)

This is a CRITICAL discriminator for Σ-Gravity mechanism.

Author: Leonard Speiser
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU acceleration
try:
    import cupy as cp
    HAS_GPU = True
    print("GPU acceleration available (CuPy)")
except ImportError:
    HAS_GPU = False
    print("Running on CPU only")


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Kinematic classification thresholds
# Based on Gaia DR3 kinematic studies (e.g., Belokurov et al.)
THIN_DISK_CRITERIA = {
    'v_phi_min': 180,       # km/s - strong prograde rotation
    'v_phi_max': 280,       # km/s
    'v_perp_max': 50,       # km/s - cold kinematics
    'z_max': 0.5,           # kpc - close to plane
    'description': 'Cold, co-rotating thin disk'
}

THICK_DISK_CRITERIA = {
    'v_phi_min': 100,       # km/s - still prograde
    'v_phi_max': 200,       # km/s
    'v_perp_min': 30,       # km/s - warmer
    'v_perp_max': 100,      # km/s
    'z_max': 2.0,           # kpc - extends further from plane
    'description': 'Warm, co-rotating thick disk'
}

HALO_RETROGRADE_CRITERIA = {
    'v_phi_max': 100,       # km/s - slow or counter-rotating
    'v_perp_min': 50,       # km/s - hot kinematics
    'description': 'Hot halo + retrograde stars'
}

# Separation bins for correlation analysis
R_BINS = np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0])


# =============================================================================
# DATA LOADING
# =============================================================================

def load_gaia_data(data_path=None):
    """
    Load Gaia data with full 3D kinematics.
    
    Required columns:
    - R_kpc (or R_cyl, R): Galactocentric radius
    - z_kpc (or z): Height above plane
    - v_phi (or v_tangential): Azimuthal velocity
    - v_R (optional): Radial velocity
    - v_z (optional): Vertical velocity
    """
    # Try multiple possible paths
    if data_path is not None:
        paths = [Path(data_path)]
    else:
        paths = [
            Path("/mnt/user-data/uploads/gaia_processed_corrected.csv"),
            Path("/mnt/user-data/uploads/gaia_processed.csv"),
            Path("data/gaia/gaia_processed_corrected.csv"),
            Path("data/gaia/gaia_processed.csv"),
        ]
    
    df = None
    for path in paths:
        if path.exists():
            print(f"Loading Gaia data from: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        print("ERROR: No Gaia data found!")
        print("Searched paths:", [str(p) for p in paths])
        return None
    
    print(f"Loaded {len(df):,} stars")
    
    # PRIORITY: Use signed v_phi if available (for phase coherence test)
    if 'v_phi_signed' in df.columns:
        df['v_phi'] = df['v_phi_signed']
        print("  Using signed v_phi for phase coherence analysis")
    
    # Standardize column names
    column_map = {
        'R_cyl': 'R_kpc', 'R': 'R_kpc',
        'z': 'z_kpc',
        'v_tangential': 'v_phi',
        'v_r': 'v_R', 'vR': 'v_R',
        'vz': 'v_z', 'v_vertical': 'v_z'
    }
    
    for old, new in column_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Check required columns
    required = ['R_kpc', 'z_kpc', 'v_phi']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Compute derived quantities
    if 'v_R' not in df.columns:
        df['v_R'] = 0.0  # Assume if not available
        print("  Warning: v_R not available, setting to 0")
    
    if 'v_z' not in df.columns:
        df['v_z'] = 0.0
        print("  Warning: v_z not available, setting to 0")
    
    # Total perpendicular velocity (non-rotational)
    df['v_perp'] = np.sqrt(df['v_R']**2 + df['v_z']**2)
    
    # Angular momentum proxy
    df['L_z'] = df['R_kpc'] * df['v_phi']  # kpc × km/s
    
    print(f"  R range: {df['R_kpc'].min():.1f} - {df['R_kpc'].max():.1f} kpc")
    print(f"  v_phi range: {df['v_phi'].min():.0f} - {df['v_phi'].max():.0f} km/s")
    print(f"  v_perp range: {df['v_perp'].min():.0f} - {df['v_perp'].max():.0f} km/s")
    
    return df


# =============================================================================
# POPULATION CLASSIFICATION
# =============================================================================

def classify_populations(df):
    """
    Classify stars into kinematic populations.
    
    Returns dict of boolean masks for each population.
    """
    print("\nClassifying kinematic populations...")
    
    populations = {}
    
    # THIN DISK: Cold, co-rotating
    thin_mask = (
        (df['v_phi'] >= THIN_DISK_CRITERIA['v_phi_min']) &
        (df['v_phi'] <= THIN_DISK_CRITERIA['v_phi_max']) &
        (df['v_perp'] <= THIN_DISK_CRITERIA['v_perp_max']) &
        (np.abs(df['z_kpc']) <= THIN_DISK_CRITERIA['z_max'])
    )
    populations['thin_disk'] = thin_mask
    
    # THICK DISK: Warmer, still prograde
    thick_mask = (
        (df['v_phi'] >= THICK_DISK_CRITERIA['v_phi_min']) &
        (df['v_phi'] <= THICK_DISK_CRITERIA['v_phi_max']) &
        (df['v_perp'] >= THICK_DISK_CRITERIA['v_perp_min']) &
        (df['v_perp'] <= THICK_DISK_CRITERIA['v_perp_max']) &
        (np.abs(df['z_kpc']) <= THICK_DISK_CRITERIA['z_max']) &
        (~thin_mask)  # Exclude thin disk
    )
    populations['thick_disk'] = thick_mask
    
    # HALO + RETROGRADE: Hot and/or counter-rotating
    halo_mask = (
        (df['v_phi'] <= HALO_RETROGRADE_CRITERIA['v_phi_max']) &
        (df['v_perp'] >= HALO_RETROGRADE_CRITERIA['v_perp_min'])
    )
    populations['halo_retrograde'] = halo_mask
    
    # Specifically counter-rotating (v_phi < 0)
    retrograde_mask = df['v_phi'] < 0
    populations['retrograde_only'] = retrograde_mask
    
    # Print statistics
    print("\nPopulation statistics:")
    print("-" * 60)
    for name, mask in populations.items():
        n = mask.sum()
        pct = 100 * n / len(df)
        mean_vphi = df.loc[mask, 'v_phi'].mean() if n > 0 else np.nan
        std_vphi = df.loc[mask, 'v_phi'].std() if n > 0 else np.nan
        print(f"  {name:20s}: {n:8,} ({pct:5.1f}%)  "
              f"<v_phi> = {mean_vphi:6.1f} ± {std_vphi:.1f} km/s")
    
    return populations


# =============================================================================
# VELOCITY CORRELATION ANALYSIS
# =============================================================================

def compute_velocity_correlations(df, r_bins=R_BINS, n_sample=20000, seed=42):
    """
    Compute velocity correlation function C(r) for a population.
    
    C(r) = <δv_i × δv_j> / σ_v² for pairs with separation r
    
    where δv = v_phi - <v_phi>(R) is the residual after subtracting
    the mean rotation curve.
    """
    np.random.seed(seed)
    
    if len(df) < 500:
        print(f"    Warning: Only {len(df)} stars, need at least 500")
        return None, None, None, None
    
    # Subsample if too large
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
        print(f"    Subsampled to {n_sample:,} stars")
    
    # Extract coordinates
    R = df['R_kpc'].values
    z = df['z_kpc'].values
    v_phi = df['v_phi'].values
    
    # Compute velocity residuals
    # Method: Subtract local mean in radial bins
    R_bins_baseline = np.linspace(R.min(), R.max(), 20)
    v_baseline = np.zeros_like(v_phi)
    
    for i in range(len(R_bins_baseline) - 1):
        mask = (R >= R_bins_baseline[i]) & (R < R_bins_baseline[i+1])
        if mask.sum() > 10:
            v_baseline[mask] = np.median(v_phi[mask])
        else:
            v_baseline[mask] = np.median(v_phi)
    
    delta_v = v_phi - v_baseline
    sigma_v = np.std(delta_v)
    
    if sigma_v < 1e-6:
        print("    Warning: Near-zero velocity dispersion")
        return None, None, None, None
    
    delta_v_norm = delta_v / sigma_v
    
    # Build KD-tree for efficient pair finding
    coords = np.column_stack([R, z])
    tree = cKDTree(coords)
    
    # Find all pairs within max separation
    r_max = r_bins[-1] * 1.1
    pairs = tree.query_pairs(r=r_max, output_type='ndarray')
    
    if len(pairs) < 1000:
        print(f"    Warning: Only {len(pairs)} pairs found")
        return None, None, None, None
    
    print(f"    Found {len(pairs):,} pairs")
    
    # Compute separations and velocity products
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    dR = R[i_idx] - R[j_idx]
    dz = z[i_idx] - z[j_idx]
    separations = np.sqrt(dR**2 + dz**2)
    products = delta_v_norm[i_idx] * delta_v_norm[j_idx]
    
    # Bin and compute correlations
    r_centers = []
    correlations = []
    corr_errors = []
    n_pairs_list = []
    
    for i in range(len(r_bins) - 1):
        mask = (separations >= r_bins[i]) & (separations < r_bins[i+1])
        n_pairs = np.sum(mask)
        
        if n_pairs < 100:
            continue
        
        corr = np.mean(products[mask])
        err = np.std(products[mask]) / np.sqrt(n_pairs)
        
        r_centers.append(np.sqrt(r_bins[i] * r_bins[i+1]))
        correlations.append(corr)
        corr_errors.append(err)
        n_pairs_list.append(n_pairs)
    
    return (np.array(r_centers), np.array(correlations), 
            np.array(corr_errors), np.array(n_pairs_list))


def fit_coherence_model(r, C, C_err):
    """
    Fit exponential and power-law coherence models.
    """
    results = {}
    
    # Only fit positive correlations
    mask = C > 0
    if mask.sum() < 3:
        return {'status': 'insufficient_positive_correlations'}
    
    r_fit, C_fit, err_fit = r[mask], C[mask], C_err[mask]
    
    # Model 1: Exponential decay
    def exp_model(r, A, ell):
        return A * np.exp(-r / ell)
    
    try:
        popt, pcov = curve_fit(exp_model, r_fit, C_fit, p0=[0.1, 1.0],
                               sigma=err_fit, bounds=([0, 0.1], [1, 20]))
        A_exp, ell_exp = popt
        y_pred = exp_model(r, A_exp, ell_exp)
        chi2_exp = np.sum(((C - y_pred) / C_err)**2)
        
        results['exponential'] = {
            'A': float(A_exp),
            'ell': float(ell_exp),
            'chi2': float(chi2_exp),
            'chi2_dof': float(chi2_exp / (len(r) - 2))
        }
    except Exception as e:
        results['exponential'] = {'error': str(e)}
    
    # Model 2: Power-law (Σ-Gravity form)
    def powerlaw_model(r, A, ell, n):
        return A * (ell / (ell + r))**n
    
    try:
        popt, pcov = curve_fit(powerlaw_model, r_fit, C_fit, p0=[0.1, 2.0, 0.5],
                               sigma=err_fit, bounds=([0, 0.1, 0.1], [1, 20, 3]))
        A_pl, ell_pl, n_pl = popt
        y_pred = powerlaw_model(r, A_pl, ell_pl, n_pl)
        chi2_pl = np.sum(((C - y_pred) / C_err)**2)
        
        results['powerlaw'] = {
            'A': float(A_pl),
            'ell': float(ell_pl),
            'n': float(n_pl),
            'chi2': float(chi2_pl),
            'chi2_dof': float(chi2_pl / (len(r) - 3))
        }
    except Exception as e:
        results['powerlaw'] = {'error': str(e)}
    
    return results


# =============================================================================
# ANGULAR MOMENTUM BINNING TEST
# =============================================================================

def test_angular_momentum_binning(df, n_sample=15000, seed=42):
    """
    Test if stars with similar L_z show stronger correlations.
    
    Bin pairs by both spatial separation AND |ΔL_z|.
    Prediction: Same-L_z pairs should show stronger C(r).
    """
    print("\n" + "="*60)
    print("ANGULAR MOMENTUM BINNING TEST")
    print("="*60)
    
    np.random.seed(seed)
    
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=seed)
    
    R = df['R_kpc'].values
    z = df['z_kpc'].values
    v_phi = df['v_phi'].values
    L_z = df['L_z'].values
    
    # Velocity residuals
    delta_v = v_phi - np.median(v_phi)
    sigma_v = np.std(delta_v)
    delta_v_norm = delta_v / sigma_v
    
    # Find pairs
    coords = np.column_stack([R, z])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=5.0, output_type='ndarray')
    
    print(f"Analyzing {len(pairs):,} pairs...")
    
    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    
    # Separations
    dR = R[i_idx] - R[j_idx]
    dz = z[i_idx] - z[j_idx]
    separations = np.sqrt(dR**2 + dz**2)
    
    # Angular momentum differences
    delta_Lz = np.abs(L_z[i_idx] - L_z[j_idx])
    
    # Velocity products
    products = delta_v_norm[i_idx] * delta_v_norm[j_idx]
    
    # L_z bins (kpc × km/s)
    Lz_bins = [
        (0, 200, 'Same L_z (Δ<200)'),
        (200, 500, 'Similar L_z (200<Δ<500)'),
        (500, 2000, 'Different L_z (Δ>500)')
    ]
    
    r_bins = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
    
    results = {}
    
    for Lz_lo, Lz_hi, label in Lz_bins:
        Lz_mask = (delta_Lz >= Lz_lo) & (delta_Lz < Lz_hi)
        
        r_centers = []
        correlations = []
        errors = []
        
        for i in range(len(r_bins) - 1):
            r_mask = (separations >= r_bins[i]) & (separations < r_bins[i+1])
            combined_mask = Lz_mask & r_mask
            n = combined_mask.sum()
            
            if n < 50:
                continue
            
            corr = np.mean(products[combined_mask])
            err = np.std(products[combined_mask]) / np.sqrt(n)
            
            r_centers.append(np.sqrt(r_bins[i] * r_bins[i+1]))
            correlations.append(corr)
            errors.append(err)
        
        results[label] = {
            'r': r_centers,
            'C': correlations,
            'err': errors,
            'n_pairs': int(Lz_mask.sum())
        }
        
        print(f"  {label}: {Lz_mask.sum():,} pairs, max C = {max(correlations) if correlations else 0:.4f}")
    
    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_phase_coherence_test(data_path=None):
    """
    Main analysis: Test phase-dependent coherence.
    """
    print("="*70)
    print("PHASE-DEPENDENT COHERENCE TEST")
    print("="*70)
    print("\nHypothesis: If coherence depends on orbital phase alignment,")
    print("prograde (co-rotating) stars should show stronger velocity")
    print("correlations than retrograde or halo stars.")
    
    # Load data
    df = load_gaia_data(data_path)
    if df is None:
        return None
    
    # Classify populations
    populations = classify_populations(df)
    
    # Analyze each population
    results = {
        'test': 'phase_dependent_coherence',
        'n_total_stars': len(df),
        'populations': {}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {
        'thin_disk': 'blue',
        'thick_disk': 'orange', 
        'halo_retrograde': 'red',
        'retrograde_only': 'darkred'
    }
    
    for idx, (pop_name, mask) in enumerate(populations.items()):
        if idx >= 4:
            break
            
        pop_df = df[mask].copy()
        n_stars = len(pop_df)
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {pop_name} ({n_stars:,} stars)")
        print("="*60)
        
        if n_stars < 500:
            print(f"  Skipping: insufficient stars")
            continue
        
        # Compute correlations
        r, C, C_err, n_pairs = compute_velocity_correlations(pop_df)
        
        if r is None:
            print(f"  Skipping: correlation computation failed")
            continue
        
        # Fit models
        fits = fit_coherence_model(r, C, C_err)
        
        # Store results
        results['populations'][pop_name] = {
            'n_stars': n_stars,
            'mean_v_phi': float(pop_df['v_phi'].mean()),
            'std_v_phi': float(pop_df['v_phi'].std()),
            'mean_v_perp': float(pop_df['v_perp'].mean()),
            'r_centers': r.tolist(),
            'correlations': C.tolist(),
            'errors': C_err.tolist(),
            'n_pairs': n_pairs.tolist(),
            'fits': fits
        }
        
        # Plot
        ax = axes.flatten()[idx]
        ax.errorbar(r, C, yerr=C_err, fmt='o-', color=colors[pop_name],
                   capsize=3, markersize=8, label='Data')
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        
        # Add fit line if available
        if 'exponential' in fits and 'ell' in fits['exponential']:
            r_fit = np.linspace(0.1, 8, 100)
            A, ell = fits['exponential']['A'], fits['exponential']['ell']
            ax.plot(r_fit, A * np.exp(-r_fit / ell), 'g--', lw=2,
                   label=f'Exp fit: ℓ={ell:.2f} kpc')
        
        ax.set_xlabel('Separation [kpc]', fontsize=11)
        ax.set_ylabel('Velocity Correlation C(r)', fontsize=11)
        ax.set_title(f'{pop_name.replace("_", " ").title()}\n'
                    f'N={n_stars:,}, ⟨v_φ⟩={pop_df["v_phi"].mean():.0f} km/s',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 8)
        
        # Print summary
        C_max = C.max() if len(C) > 0 else 0
        print(f"  Max correlation: {C_max:.4f}")
        if 'exponential' in fits and 'ell' in fits['exponential']:
            print(f"  Fitted ℓ: {fits['exponential']['ell']:.2f} kpc")
    
    fig.suptitle('Phase-Dependent Coherence Test\n'
                 'Prediction: Thin disk >> Thick disk >> Halo/Retrograde',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase_coherence_populations.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Angular momentum test
    print("\n" + "="*70)
    print("ANGULAR MOMENTUM BINNING TEST")
    print("="*70)
    
    # Use thin disk for cleanest signal
    if 'thin_disk' in populations:
        thin_df = df[populations['thin_disk']].copy()
        if len(thin_df) > 1000:
            Lz_results = test_angular_momentum_binning(thin_df)
            results['angular_momentum_test'] = Lz_results
            
            # Plot L_z binning results
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_Lz = ['blue', 'orange', 'red']
            
            for i, (label, data) in enumerate(Lz_results.items()):
                if data['r']:
                    ax.errorbar(data['r'], data['C'], yerr=data['err'],
                               fmt='o-', color=colors_Lz[i], capsize=3,
                               markersize=8, label=f"{label} ({data['n_pairs']:,} pairs)")
            
            ax.axhline(0, color='gray', ls='--', alpha=0.5)
            ax.set_xlabel('Spatial Separation [kpc]', fontsize=12)
            ax.set_ylabel('Velocity Correlation C(r)', fontsize=12)
            ax.set_title('Angular Momentum Binning Test\n'
                        'If phase matters: Same L_z >> Different L_z',
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'angular_momentum_binning.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("PHASE COHERENCE TEST SUMMARY")
    print("="*70)
    
    summary_table = []
    for pop_name, pop_data in results['populations'].items():
        C_max = max(pop_data['correlations']) if pop_data['correlations'] else 0
        ell = pop_data['fits'].get('exponential', {}).get('ell', np.nan)
        summary_table.append({
            'Population': pop_name,
            'N_stars': pop_data['n_stars'],
            'Mean_v_phi': pop_data['mean_v_phi'],
            'Max_C': C_max,
            'ell_kpc': ell
        })
    
    print("\n| Population          | N stars  | <v_phi> | Max C(r) | ℓ [kpc] |")
    print("|---------------------|----------|---------|----------|---------|")
    for row in summary_table:
        print(f"| {row['Population']:19s} | {row['N_stars']:8,} | {row['Mean_v_phi']:7.0f} | "
              f"{row['Max_C']:8.4f} | {row['ell_kpc']:7.2f} |")
    
    print("\nINTERPRETATION:")
    if len(summary_table) >= 2:
        thin_C = [r['Max_C'] for r in summary_table if 'thin' in r['Population']]
        halo_C = [r['Max_C'] for r in summary_table if 'halo' in r['Population'] or 'retro' in r['Population']]
        
        if thin_C and halo_C:
            ratio = thin_C[0] / halo_C[0] if halo_C[0] != 0 else np.inf
            print(f"  Thin disk / Halo correlation ratio: {ratio:.1f}")
            
            if ratio > 3:
                print("  → SUPPORTS phase-dependent coherence")
                print("  → Prograde rotation enhances gravitational coherence")
            elif ratio > 1.5:
                print("  → WEAK SUPPORT for phase-dependent coherence")
                print("  → Some dependence on rotation, but not dramatic")
            else:
                print("  → NO SUPPORT for phase-dependent coherence")
                print("  → Correlations similar regardless of rotation")
    
    # Save results
    results_path = OUTPUT_DIR / 'phase_coherence_results.json'
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    results = run_phase_coherence_test(data_path)
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
