"""
Calculate GR Baseline Predictions for Gaia Stars

Purpose:
--------
Calculate what General Relativity predicts for each star's velocity
using ONLY observed baryonic matter (no dark matter, no enhancement).

This establishes the baseline that Σ-Gravity must improve upon.

Workflow:
---------
1. Load Gaia data with observed v_phi
2. Calculate v_phi_GR for each star using observed baryonic mass
3. Calculate gap: v_observed - v_GR
4. Save results for testing λ_gw enhancement

Key Point:
----------
We use OBSERVED baryonic masses:
  - Stellar disk: ~3e10 M☉
  - Gas: ~1e10 M☉  
  - Total disk: ~4e10 M☉
  - Bulge: ~1.5e10 M☉
  
NOT fitted masses designed to match observations!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Gravitational constant
G_KPC = 4.30091e-6  # kpc³ M☉⁻¹ (km/s)²

# ============================================================================
# OBSERVED BARYONIC PARAMETERS (from literature)
# ============================================================================

# Milky Way baryonic masses (observational constraints)
M_DISK_STELLAR = 3.0e10  # M☉ (stellar mass from star counts)
M_DISK_GAS = 1.0e10      # M☉ (HI + H2 gas)
M_DISK_TOTAL = 4.0e10    # M☉ (total baryonic disk)
M_BULGE = 1.5e10         # M☉ (central bulge)

# Disk scale parameters (observational)
A_DISK = 3.5   # kpc (scale length)
B_DISK = 0.25  # kpc (scale height)

# Bulge scale
A_BULGE = 0.7  # kpc (Hernquist scale)

print("="*80)
print("GR BASELINE CALCULATION")
print("="*80)
print("\nUsing OBSERVED baryonic masses:")
print(f"  Disk (stellar): {M_DISK_STELLAR:.2e} Msun")
print(f"  Disk (gas):     {M_DISK_GAS:.2e} Msun")
print(f"  Disk (total):   {M_DISK_TOTAL:.2e} Msun")
print(f"  Bulge:          {M_BULGE:.2e} Msun")
print(f"  Halo:           0 Msun (NO DARK MATTER)")
print("\nThese are the REAL baryonic masses, not fitted!")

# ============================================================================
# GR PREDICTIONS (BARYONS ONLY)
# ============================================================================

def miyamoto_nagai_disk(R, z=0, M_disk=M_DISK_TOTAL, a=A_DISK, b=B_DISK):
    """
    Miyamoto-Nagai disk circular velocity.
    
    Uses OBSERVED parameters, not fitted ones!
    
    Returns v_circ in km/s
    """
    z = np.abs(z)
    denom = np.sqrt(R**2 + (a + np.sqrt(z**2 + b**2))**2)
    v_squared = G_KPC * M_disk * R**2 / denom**3
    return np.sqrt(v_squared)


def hernquist_bulge(R, M_bulge=M_BULGE, a_bulge=A_BULGE):
    """
    Hernquist bulge circular velocity.
    
    Returns v_circ in km/s
    """
    v_squared = G_KPC * M_bulge * R / (R + a_bulge)**2
    return np.sqrt(v_squared)


def gr_prediction_baryons_only(R, z=0):
    """
    What GR predicts with ONLY observed baryonic matter.
    
    No dark matter! No enhancement! Pure GR + baryons.
    
    Parameters:
    -----------
    R : array
        Cylindrical radius (kpc)
    z : array
        Height above plane (kpc)
    
    Returns:
    --------
    v_GR : array
        Predicted circular velocity (km/s)
    components : dict
        Breakdown by component
    """
    v_disk = miyamoto_nagai_disk(R, z, M_disk=M_DISK_TOTAL)
    v_bulge = hernquist_bulge(R, M_bulge=M_BULGE)
    
    # Quadrature sum (standard GR combination)
    v_total = np.sqrt(v_disk**2 + v_bulge**2)
    
    return v_total, {
        'disk': v_disk,
        'bulge': v_bulge,
        'total': v_total
    }


# ============================================================================
# PROCESS GAIA DATA
# ============================================================================

def calculate_gr_baseline(
    input_path='gravitywavebaseline/gaia_with_periods.parquet',
    output_path='gravitywavebaseline/gaia_with_gr_baseline.parquet'
):
    """
    Calculate GR baseline predictions for all Gaia stars.
    
    Adds columns:
    - v_phi_GR: GR prediction with baryons only
    - v_phi_gap: Observed - GR (the gap λ_gw must fill)
    - needs_explanation: True if |gap| > 20 km/s
    """
    
    print(f"\n{'='*80}")
    print("LOADING GAIA DATA")
    print("="*80)
    
    gaia = pd.read_parquet(input_path)
    print(f"Loaded {len(gaia):,} stars")
    
    # Check for required columns
    if 'v_phi' not in gaia.columns:
        raise ValueError("Gaia data missing v_phi column!")
    
    print(f"\nObservations:")
    print(f"  R range: {gaia['R'].min():.2f} - {gaia['R'].max():.2f} kpc")
    print(f"  v_phi range: {gaia['v_phi'].min():.1f} - {gaia['v_phi'].max():.1f} km/s")
    
    # ========================================================================
    # Calculate GR predictions
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("CALCULATING GR BASELINE")
    print("="*80)
    
    R = gaia['R'].values
    z = gaia['z'].values if 'z' in gaia.columns else np.zeros_like(R)
    
    v_GR, components = gr_prediction_baryons_only(R, z)
    
    # Store in dataframe
    gaia['v_phi_GR'] = v_GR.astype(np.float32)
    gaia['v_disk_GR'] = components['disk'].astype(np.float32)
    gaia['v_bulge_GR'] = components['bulge'].astype(np.float32)
    
    # ========================================================================
    # Calculate gap (what needs explaining)
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("CALCULATING GAP")
    print("="*80)
    
    # Only calculate gap for valid observations
    valid_mask = np.isfinite(gaia['v_phi']) & (gaia['v_phi'] > 0)
    
    gaia['v_phi_gap'] = np.nan
    gaia.loc[valid_mask, 'v_phi_gap'] = (
        gaia.loc[valid_mask, 'v_phi'] - gaia.loc[valid_mask, 'v_phi_GR']
    )
    
    # Flag stars that need significant explanation
    gaia['needs_explanation'] = False
    gaia.loc[valid_mask, 'needs_explanation'] = (
        np.abs(gaia.loc[valid_mask, 'v_phi_gap']) > 20.0
    )
    
    # ========================================================================
    # Statistics by radius
    # ========================================================================
    
    print("\nStatistics by radial bin:")
    print(f"{'R (kpc)':<12} {'N stars':<10} {'v_obs':<12} {'v_GR':<12} {'Gap':<12} {'Needs Fix'}")
    print("-"*80)
    
    # Bin by radius
    R_bins = np.arange(0, 20, 2)  # 0-2, 2-4, ..., 18-20 kpc
    
    for i in range(len(R_bins)-1):
        R_min, R_max = R_bins[i], R_bins[i+1]
        mask = valid_mask & (R >= R_min) & (R < R_max)
        
        if np.sum(mask) == 0:
            continue
        
        n_stars = np.sum(mask)
        v_obs_mean = gaia.loc[mask, 'v_phi'].mean()
        v_GR_mean = gaia.loc[mask, 'v_phi_GR'].mean()
        gap_mean = gaia.loc[mask, 'v_phi_gap'].mean()
        n_needs_fix = np.sum(gaia.loc[mask, 'needs_explanation'])
        
        print(f"{R_min:2.0f}-{R_max:2.0f} kpc   "
              f"{n_stars:<10,} "
              f"{v_obs_mean:>6.1f} km/s  "
              f"{v_GR_mean:>6.1f} km/s  "
              f"{gap_mean:>+6.1f} km/s  "
              f"{n_needs_fix:>6,} ({100*n_needs_fix/n_stars:.1f}%)")
    
    # ========================================================================
    # Overall statistics
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print("="*80)
    
    valid_data = gaia[valid_mask]
    
    v_obs = valid_data['v_phi'].values
    v_GR = valid_data['v_phi_GR'].values
    gap = valid_data['v_phi_gap'].values
    
    rms_gap = np.sqrt(np.mean(gap**2))
    
    print(f"\nAll stars with valid v_phi:")
    print(f"  N = {len(valid_data):,}")
    print(f"  <v_observed> = {v_obs.mean():.1f} +/- {v_obs.std():.1f} km/s")
    print(f"  <v_GR> = {v_GR.mean():.1f} +/- {v_GR.std():.1f} km/s")
    print(f"  <gap> = {gap.mean():.1f} +/- {gap.std():.1f} km/s")
    print(f"  RMS(gap) = {rms_gap:.1f} km/s")
    print(f"  Stars needing explanation: {np.sum(valid_data['needs_explanation']):,} "
          f"({100*np.sum(valid_data['needs_explanation'])/len(valid_data):.1f}%)")
    
    # By radius range
    print(f"\nOuter disk (R > 10 kpc):")
    outer_mask = valid_mask & (R > 10)
    if np.sum(outer_mask) > 0:
        outer_data = gaia[outer_mask]
        v_obs_outer = outer_data['v_phi'].values
        v_GR_outer = outer_data['v_phi_GR'].values
        gap_outer = outer_data['v_phi_gap'].values
        rms_gap_outer = np.sqrt(np.mean(gap_outer**2))
        
        print(f"  N = {len(outer_data):,}")
        print(f"  <v_observed> = {v_obs_outer.mean():.1f} km/s")
        print(f"  <v_GR> = {v_GR_outer.mean():.1f} km/s")
        print(f"  <gap> = {gap_outer.mean():.1f} km/s")
        print(f"  RMS(gap) = {rms_gap_outer:.1f} km/s")
        print("  ^ THIS IS WHAT lambda_gw MUST EXPLAIN!")
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    
    output_path = Path(output_path)
    gaia.to_parquet(output_path)
    
    print(f"\nSaved to: {output_path}")
    print(f"Added columns:")
    print(f"  - v_phi_GR: GR prediction with baryons only")
    print(f"  - v_disk_GR: Disk contribution")
    print(f"  - v_bulge_GR: Bulge contribution")
    print(f"  - v_phi_gap: Observed - GR (what needs explaining)")
    print(f"  - needs_explanation: True if |gap| > 20 km/s")
    
    return gaia


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_gr_baseline(gaia, output_path='gravitywavebaseline/gr_baseline_plot.png'):
    """Create diagnostic plots of GR baseline vs observations."""
    
    valid_mask = np.isfinite(gaia['v_phi']) & (gaia['v_phi'] > 0)
    data = gaia[valid_mask].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GR Baseline vs Observations (Baryons Only, No Dark Matter)', 
                 fontsize=14, fontweight='bold')
    
    # ========================================================================
    # Plot 1: Rotation curves
    # ========================================================================
    ax = axes[0, 0]
    
    # Bin data
    R_bins = np.linspace(0, 20, 40)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    
    v_obs_binned = []
    v_GR_binned = []
    v_obs_std = []
    
    for i in range(len(R_bins)-1):
        mask = (data['R'] >= R_bins[i]) & (data['R'] < R_bins[i+1])
        if np.sum(mask) > 10:
            v_obs_binned.append(data.loc[mask, 'v_phi'].mean())
            v_GR_binned.append(data.loc[mask, 'v_phi_GR'].mean())
            v_obs_std.append(data.loc[mask, 'v_phi'].std())
        else:
            v_obs_binned.append(np.nan)
            v_GR_binned.append(np.nan)
            v_obs_std.append(np.nan)
    
    v_obs_binned = np.array(v_obs_binned)
    v_GR_binned = np.array(v_GR_binned)
    v_obs_std = np.array(v_obs_std)
    
    # Plot observations
    ax.errorbar(R_centers, v_obs_binned, yerr=v_obs_std, 
                fmt='o', color='black', alpha=0.7, markersize=4,
                label='Gaia observations', capsize=2)
    
    # Plot GR prediction
    ax.plot(R_centers, v_GR_binned, 'r-', linewidth=2.5, 
            label='GR (baryons only, NO dark matter)')
    
    # Highlight the gap
    ax.fill_between(R_centers, v_GR_binned, v_obs_binned, 
                     alpha=0.3, color='orange',
                     label='Gap (needs Σ-Gravity!)')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_circ [km/s]', fontsize=12)
    ax.set_title('Rotation Curves')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 300)
    
    # ========================================================================
    # Plot 2: Gap vs radius
    # ========================================================================
    ax = axes[0, 1]
    
    # 2D histogram of gap
    h = ax.hexbin(data['R'], data['v_phi_gap'], gridsize=40, 
                  cmap='RdYlBu_r', vmin=-100, vmax=100)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.5, label='Zero gap')
    ax.axhline(20, color='r', linestyle=':', alpha=0.7, label='Needs explanation')
    ax.axhline(-20, color='r', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('Gap = v_obs - v_GR [km/s]', fontsize=12)
    ax.set_title('Where GR Fails')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 20)
    ax.set_ylim(-150, 150)
    
    plt.colorbar(h, ax=ax, label='Number of stars')
    
    # ========================================================================
    # Plot 3: Gap distribution
    # ========================================================================
    ax = axes[1, 0]
    
    # Overall
    ax.hist(data['v_phi_gap'], bins=50, alpha=0.5, 
            label=f'All (RMS={np.sqrt(np.mean(data["v_phi_gap"]**2)):.1f} km/s)',
            color='blue', edgecolor='black')
    
    # Outer disk
    outer_mask = data['R'] > 10
    if np.sum(outer_mask) > 0:
        gap_outer = data.loc[outer_mask, 'v_phi_gap']
        ax.hist(gap_outer, bins=50, alpha=0.5,
                label=f'Outer (R>10 kpc, RMS={np.sqrt(np.mean(gap_outer**2)):.1f} km/s)',
                color='red', edgecolor='black')
    
    ax.axvline(0, color='k', linestyle='--', linewidth=2, label='Perfect GR')
    ax.axvline(20, color='r', linestyle=':', alpha=0.7)
    ax.axvline(-20, color='r', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Gap = v_obs - v_GR [km/s]', fontsize=12)
    ax.set_ylabel('Number of stars', fontsize=12)
    ax.set_title('Gap Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 4: Needs explanation map
    # ========================================================================
    ax = axes[1, 1]
    
    # Stars that need explanation
    needs_fix = data[data['needs_explanation']]
    ok = data[~data['needs_explanation']]
    
    ax.scatter(ok['R'], ok['v_phi'], s=1, alpha=0.1, color='blue', 
               label=f'GR OK (|gap|<20 km/s): {len(ok):,}')
    ax.scatter(needs_fix['R'], needs_fix['v_phi'], s=1, alpha=0.3, color='red',
               label=f'Needs Σ-Gravity (|gap|>20): {len(needs_fix):,}')
    
    # GR prediction curve
    R_smooth = np.linspace(0, 20, 100)
    v_GR_smooth, _ = gr_prediction_baryons_only(R_smooth)
    ax.plot(R_smooth, v_GR_smooth, 'k-', linewidth=2, label='GR prediction')
    
    ax.set_xlabel('R [kpc]', fontsize=12)
    ax.set_ylabel('v_phi [km/s]', fontsize=12)
    ax.set_title('Which Stars Need Alternative Gravity?')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 350)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved plot: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Calculate GR baseline
    gaia = calculate_gr_baseline()
    
    # Create plots
    plot_gr_baseline(gaia)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print("\nGR baseline has been calculated using OBSERVED baryonic masses.")
    print("The gap between GR predictions and observations is now quantified.")
    print("\nNext step: Test if lambda_gw enhancement can close this gap!")
    print("\nKey files:")
    print("  - gaia_with_gr_baseline.parquet: Data with GR predictions")
    print("  - gr_baseline_plot.png: Diagnostic plots")

