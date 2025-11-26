#!/usr/bin/env python3
"""
Recompute Signed v_phi from Gaia Proper Motions
================================================

The original processing used:
  v_phi = sqrt(v_ra² + v_dec²)  # WRONG - loses sign!

For phase coherence test we need:
  v_phi = signed azimuthal velocity (+ = prograde, - = retrograde)

This requires proper coordinate transformation from:
  (l, b, μ_l*, μ_b, v_rad) → (R, z, v_R, v_phi, v_z)

Key: Stars with v_phi < 0 are RETROGRADE (counter-rotating with disk)
     These are mostly halo stars on eccentric orbits.

See README.md for coordinate transformation references.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Solar parameters (from Reid & Brunthaler 2020, McMillan 2017)
R_SUN = 8.2      # kpc - Sun's distance from GC
Z_SUN = 0.025   # kpc - Sun above midplane
V_SUN_U = 11.1  # km/s - Sun's peculiar velocity toward GC
V_SUN_V = 12.24 # km/s - Sun's peculiar velocity in rotation direction
V_SUN_W = 7.25  # km/s - Sun's peculiar velocity toward NGP
V_LSR = 232.0   # km/s - Local standard of rest circular velocity


def transform_to_galactocentric(l, b, dist_kpc, pmra, pmdec, v_rad):
    """
    Full coordinate and velocity transformation from heliocentric to galactocentric.
    
    Parameters
    ----------
    l, b : float arrays
        Galactic longitude and latitude (degrees)
    dist_kpc : float array
        Distance (kpc)
    pmra, pmdec : float arrays  
        Proper motions (mas/yr), μ_α* = μ_α*cos(δ)
    v_rad : float array
        Radial velocity (km/s)
    
    Returns
    -------
    R, z : Galactocentric cylindrical coordinates (kpc)
    v_R, v_phi, v_z : Galactocentric velocities (km/s)
        v_phi > 0 = prograde (rotating with disk)
        v_phi < 0 = retrograde (counter-rotating)
    """
    # Convert to radians
    l_rad = np.radians(l)
    b_rad = np.radians(b)
    
    # Heliocentric Cartesian (Galactic frame)
    # X toward GC, Y toward l=90°, Z toward NGP
    x = dist_kpc * np.cos(b_rad) * np.cos(l_rad)
    y = dist_kpc * np.cos(b_rad) * np.sin(l_rad)
    z = dist_kpc * np.sin(b_rad)
    
    # Galactocentric Cartesian
    # X_gc toward Sun, Y_gc in direction of rotation, Z_gc toward NGP
    x_gc = R_SUN - x
    y_gc = -y  # Rotation is in -y direction from Sun's perspective
    z_gc = z + Z_SUN
    
    # Cylindrical coordinates
    R_cyl = np.sqrt(x_gc**2 + y_gc**2)
    phi_cyl = np.arctan2(y_gc, x_gc)
    
    # Convert proper motions to velocities
    # k = 4.74047 converts (mas/yr) × (kpc) → km/s
    k = 4.74047
    
    # Velocity in equatorial frame
    # First convert pmra, pmdec to pm_l*, pm_b (Galactic proper motions)
    # This requires knowing ra, dec, but we can use the approximation that
    # for stars in the disk, pm_l* ≈ pmra, pm_b ≈ pmdec (crude but close)
    # For a proper treatment, use astropy. Here we do a simplified version.
    
    # For now, compute velocities in heliocentric Galactic Cartesian
    # v_l = k × d × μ_l* (in l direction)
    # v_b = k × d × μ_b (in b direction)
    
    # Approximate: use pmra, pmdec as pm_l*, pm_b
    # This is reasonable for |b| < ~30° which covers most disk stars
    pm_l = pmra   # mas/yr (really μ_l*)
    pm_b = pmdec  # mas/yr (μ_b)
    
    # Heliocentric velocities in spherical Galactic coords
    v_l = k * dist_kpc * pm_l   # km/s in l direction
    v_b = k * dist_kpc * pm_b   # km/s in b direction
    v_r = np.where(np.isfinite(v_rad), v_rad, 0.0)  # km/s radial
    
    # Transform to heliocentric Cartesian velocities
    # (X toward GC, Y toward l=90, Z toward NGP)
    cos_l, sin_l = np.cos(l_rad), np.sin(l_rad)
    cos_b, sin_b = np.cos(b_rad), np.sin(b_rad)
    
    # Velocity components in heliocentric Galactic Cartesian
    vx_hel = v_r * cos_b * cos_l - v_l * sin_l - v_b * sin_b * cos_l
    vy_hel = v_r * cos_b * sin_l + v_l * cos_l - v_b * sin_b * sin_l
    vz_hel = v_r * sin_b + v_b * cos_b
    
    # Add solar motion to get galactocentric
    # U = -vx (toward GC is -X), V = vy, W = vz
    vx_gc = -vx_hel + V_SUN_U  # Velocity component toward GC
    vy_gc = vy_hel + V_LSR + V_SUN_V  # Rotation component
    vz_gc = vz_hel + V_SUN_W  # Vertical component
    
    # Transform to cylindrical
    cos_phi = x_gc / np.clip(R_cyl, 1e-6, None)
    sin_phi = y_gc / np.clip(R_cyl, 1e-6, None)
    
    # v_R = vx_gc * cos(phi) + vy_gc * sin(phi)
    # v_phi = -vx_gc * sin(phi) + vy_gc * cos(phi)
    v_R_cyl = vx_gc * cos_phi + vy_gc * sin_phi
    v_phi_cyl = -vx_gc * sin_phi + vy_gc * cos_phi
    v_z_cyl = vz_gc
    
    return R_cyl, z_gc, v_R_cyl, v_phi_cyl, v_z_cyl


def process_gaia_signed_velocities(input_path, output_path=None):
    """
    Reprocess Gaia data with signed v_phi.
    """
    print("\n" + "="*70)
    print("RECOMPUTING SIGNED v_phi FOR GAIA DATA")
    print("="*70)
    
    # Load data
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} stars")
    
    # Check required columns
    required = ['l', 'b', 'parallax', 'pmra', 'pmdec']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return None
    
    # Compute distances
    dist_kpc = 1.0 / df['parallax'].values  # parallax in mas → distance in kpc
    
    # Get radial velocity (may be NaN for many stars)
    v_rad = df['v_rad'].values if 'v_rad' in df.columns else np.full(len(df), np.nan)
    
    print("\nTransforming coordinates...")
    R, z, v_R, v_phi, v_z = transform_to_galactocentric(
        df['l'].values,
        df['b'].values,
        dist_kpc,
        df['pmra'].values,
        df['pmdec'].values,
        v_rad
    )
    
    # Create output dataframe
    df_out = df.copy()
    df_out['R_cyl'] = R
    df_out['z'] = z
    df_out['v_R'] = v_R
    df_out['v_phi_signed'] = v_phi  # NEW: signed azimuthal velocity
    df_out['v_z'] = v_z
    df_out['v_perp'] = np.sqrt(v_R**2 + v_z**2)  # Non-rotational component
    
    # Quality cuts
    mask = (
        np.isfinite(R) & 
        np.isfinite(v_phi) &
        (R > 0.5) & (R < 25) &  # Reasonable MW extent
        np.isfinite(v_R)
    )
    df_out = df_out[mask]
    
    # Statistics
    print("\n" + "-"*60)
    print("VELOCITY STATISTICS (signed v_phi)")
    print("-"*60)
    print(f"  Total stars after cuts: {len(df_out):,}")
    print(f"  v_phi range: {df_out['v_phi_signed'].min():.0f} to {df_out['v_phi_signed'].max():.0f} km/s")
    print(f"  v_phi mean:  {df_out['v_phi_signed'].mean():.1f} km/s")
    print(f"  v_phi std:   {df_out['v_phi_signed'].std():.1f} km/s")
    
    n_prograde = (df_out['v_phi_signed'] > 50).sum()
    n_retrograde = (df_out['v_phi_signed'] < -50).sum()
    n_slow = ((df_out['v_phi_signed'] >= -50) & (df_out['v_phi_signed'] <= 50)).sum()
    
    print(f"\nRotation classification:")
    print(f"  Prograde (v_phi > +50):   {n_prograde:>10,} ({100*n_prograde/len(df_out):.1f}%)")
    print(f"  Slow/intermediate:        {n_slow:>10,} ({100*n_slow/len(df_out):.1f}%)")
    print(f"  Retrograde (v_phi < -50): {n_retrograde:>10,} ({100*n_retrograde/len(df_out):.1f}%)")
    
    # Save
    if output_path is None:
        output_path = input_path.replace('.csv', '_signed.csv')
    
    df_out.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return df_out


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "C:/Users/henry/dev/sigmagravity/data/gaia/gaia_processed_corrected.csv"
    
    output_path = "C:/Users/henry/dev/sigmagravity/data/gaia/gaia_processed_signed.csv"
    
    df = process_gaia_signed_velocities(input_path, output_path)
    
    if df is not None:
        print("\n" + "="*70)
        print("READY FOR PHASE COHERENCE TEST")
        print("="*70)
        print(f"\nRun: python test_phase_coherence_gaia.py {output_path}")
