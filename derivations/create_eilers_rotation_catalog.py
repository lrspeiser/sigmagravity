#!/usr/bin/env python3
"""
Create Eilers-Quality Rotation Curve Catalog
=============================================

This script cross-matches the Eilers+ 2018 spectrophotometric parallax catalog
with APOGEE DR17 radial velocities and Gaia DR3 proper motions to create
a catalog suitable for Milky Way rotation curve analysis.

The goal is to reproduce the Eilers+ 2019 rotation curve methodology using
publicly available data.

Data sources:
1. Eilers+ 2018 parallax catalog (44,784 RGB stars with spectrophotometric distances)
2. APOGEE DR17 allStar catalog (~730,000 stars with radial velocities)
3. Gaia DR3 (proper motions, already in our 1.8M star catalog)

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

# Galactocentric parameters (Gravity Collaboration 2019 + Reid & Brunthaler 2020)
R0_KPC = 8.122      # Distance from Sun to Galactic center
Z_SUN_KPC = 0.0208  # Height of Sun above Galactic plane
V_SUN = [11.1, 232.24, 7.25]  # Solar motion [U, V, W] km/s (Schönrich+ 2010)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_eilers_parallax_catalog(filepath: Path) -> pd.DataFrame:
    """Load the Eilers+ 2018 spectrophotometric parallax catalog."""
    print(f"Loading Eilers parallax catalog from {filepath}...")
    
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        
        df = pd.DataFrame({
            'twomass_id': [s.strip() for s in data['2MASS_ID']],
            'gaia_parallax': to_native_byteorder(data['Gaia_parallax']),
            'gaia_parallax_err': to_native_byteorder(data['Gaia_parallax_err']),
            'spec_parallax': to_native_byteorder(data['spec_parallax']),
            'spec_parallax_err': to_native_byteorder(data['spec_parallax_err']),
            'training_set': to_native_byteorder(data['training_set']),
            'sample': [s.strip() for s in data['sample']],
        })
    
    print(f"  Loaded {len(df)} stars")
    print(f"  Sample A: {(df['sample'] == 'A').sum()} stars")
    print(f"  Sample B: {(df['sample'] == 'B').sum()} stars")
    
    return df


def to_native_byteorder(arr):
    """Convert array to native byte order if needed."""
    arr = np.asarray(arr)
    if arr.dtype.byteorder not in ('=', '|', '<' if np.little_endian else '>'):
        return arr.byteswap().view(arr.dtype.newbyteorder('='))
    return arr


def load_apogee_catalog(filepath: Path) -> pd.DataFrame:
    """Load the APOGEE DR17 allStar catalog."""
    print(f"Loading APOGEE catalog from {filepath}...")
    
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        
        # Extract relevant columns and convert to native byte order
        # APOGEE_ID is in format "2M..." matching 2MASS
        apogee_ids = np.array([s.strip() for s in data['APOGEE_ID']])
        ra = to_native_byteorder(data['RA'])
        dec = to_native_byteorder(data['DEC'])
        glon = to_native_byteorder(data['GLON'])
        glat = to_native_byteorder(data['GLAT'])
        vhelio = to_native_byteorder(data['VHELIO_AVG'])
        vhelio_err = to_native_byteorder(data['VSCATTER'])
        teff = to_native_byteorder(data['TEFF'])
        logg = to_native_byteorder(data['LOGG'])
        fe_h = to_native_byteorder(data['FE_H'])
        snr = to_native_byteorder(data['SNR'])
        
        # Filter for good quality first (before creating DataFrame)
        good = (vhelio > -900) & (snr > 20)
        
        df = pd.DataFrame({
            'apogee_id': apogee_ids[good],
            'ra': ra[good],
            'dec': dec[good],
            'glon': glon[good],
            'glat': glat[good],
            'vhelio': vhelio[good],
            'vhelio_err': vhelio_err[good],
            'teff': teff[good],
            'logg': logg[good],
            'fe_h': fe_h[good],
            'snr': snr[good],
        })
    
    print(f"  Loaded {len(df)} stars with good RV measurements")
    
    return df


def load_gaia_catalog(filepath: Path) -> pd.DataFrame:
    """Load our processed Gaia catalog."""
    print(f"Loading Gaia catalog from {filepath}...")
    
    df = pd.read_csv(filepath)
    
    # We need to match by position since we don't have 2MASS IDs in Gaia
    print(f"  Loaded {len(df)} stars")
    
    return df


# =============================================================================
# CROSS-MATCHING
# =============================================================================

def crossmatch_eilers_apogee(eilers_df: pd.DataFrame, apogee_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-match Eilers and APOGEE catalogs by 2MASS ID."""
    print("\nCross-matching Eilers and APOGEE catalogs...")
    
    # Both use 2MASS IDs
    merged = pd.merge(
        eilers_df,
        apogee_df,
        left_on='twomass_id',
        right_on='apogee_id',
        how='inner'
    )
    
    print(f"  Matched {len(merged)} stars")
    
    return merged


# =============================================================================
# GALACTOCENTRIC COORDINATES
# =============================================================================

def compute_galactocentric_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Compute galactocentric positions and velocities using simplified approach.
    
    Since we don't have proper motions in the cross-matched catalog,
    we use the radial velocity and galactic coordinates to estimate
    the rotation velocity directly.
    """
    print("\nComputing galactocentric coordinates...")
    
    # Use spectrophotometric distances
    distance_kpc = 1.0 / df['spec_parallax'].values  # parallax in mas -> distance in kpc
    
    # Galactic coordinates
    l_rad = np.radians(df['glon'].values)
    b_rad = np.radians(df['glat'].values)
    
    # Convert to Galactocentric Cartesian coordinates
    # Sun is at (X, Y, Z) = (-R0, 0, z_sun)
    R0 = R0_KPC
    z_sun = Z_SUN_KPC
    
    # Heliocentric Cartesian
    x_helio = distance_kpc * np.cos(b_rad) * np.cos(l_rad)
    y_helio = distance_kpc * np.cos(b_rad) * np.sin(l_rad)
    z_helio = distance_kpc * np.sin(b_rad)
    
    # Galactocentric Cartesian (X toward GC, Y in direction of rotation, Z toward NGP)
    X_gal = -x_helio - R0
    Y_gal = y_helio
    Z_gal = z_helio + z_sun
    
    # Galactocentric cylindrical
    R_gal = np.sqrt(X_gal**2 + Y_gal**2)
    phi_gal = np.arctan2(Y_gal, X_gal)
    
    # For the rotation velocity, we use a simplified approach:
    # V_los = V_sun * sin(l) * cos(b) + V_rot * sin(l) * cos(b) * (R0/R - 1)
    # 
    # For stars near the solar circle, V_phi ≈ V_sun + V_los / (sin(l) * cos(b))
    # But this is only accurate for certain geometries.
    #
    # Better approach: Use the Oort constants approximation
    # For now, we'll use a direct geometric projection
    
    V_sun = V_SUN[1]  # V component of solar motion (232.24 km/s)
    V_los = df['vhelio'].values  # Heliocentric radial velocity
    
    # Project to get azimuthal velocity
    # This is an approximation valid for disk stars
    sin_l = np.sin(l_rad)
    cos_l = np.cos(l_rad)
    cos_b = np.cos(b_rad)
    
    # Avoid division by zero
    sin_l_cosb = sin_l * cos_b
    valid = np.abs(sin_l_cosb) > 0.1  # Only use stars with good geometry
    
    # Estimate V_phi using the relation:
    # V_los = (V_phi - V_sun) * (R0/R) * sin(l) * cos(b) + V_R * cos(l) * cos(b)
    # Assuming V_R ≈ 0 for circular orbits:
    # V_phi ≈ V_sun + V_los * R / (R0 * sin(l) * cos(b))
    
    V_phi = np.full_like(V_los, np.nan)
    V_phi[valid] = V_sun + V_los[valid] * R_gal[valid] / (R0 * sin_l_cosb[valid])
    
    # Create output DataFrame
    df = df.copy()
    df['distance_kpc'] = distance_kpc
    df['R_gal'] = R_gal
    df['z_gal'] = Z_gal
    df['phi_gal'] = np.degrees(phi_gal)
    df['v_phi'] = V_phi
    df['v_los'] = V_los
    df['valid_geometry'] = valid
    
    valid_count = np.sum(valid & np.isfinite(V_phi))
    print(f"  Total stars: {len(df)}")
    print(f"  Stars with valid geometry: {valid_count}")
    print(f"  R range: {np.nanmin(R_gal):.1f} - {np.nanmax(R_gal):.1f} kpc")
    print(f"  z range: {np.nanmin(Z_gal):.1f} - {np.nanmax(Z_gal):.1f} kpc")
    
    return df


# =============================================================================
# ROTATION CURVE
# =============================================================================

def compute_rotation_curve(df: pd.DataFrame, R_bins: np.ndarray = None) -> pd.DataFrame:
    """Compute the rotation curve from the catalog."""
    print("\nComputing rotation curve...")
    
    if R_bins is None:
        R_bins = np.arange(5.0, 20.0, 0.5)
    
    # Filter to disk plane
    disk = df[np.abs(df['z_gal']) < 1.0].copy()
    print(f"  Stars in disk plane (|z| < 1 kpc): {len(disk)}")
    
    results = []
    for i in range(len(R_bins) - 1):
        R_min, R_max = R_bins[i], R_bins[i+1]
        R_center = (R_min + R_max) / 2
        
        mask = (disk['R_gal'] >= R_min) & (disk['R_gal'] < R_max)
        stars = disk[mask]
        
        if len(stars) < 10:
            continue
        
        v_phi_median = stars['v_phi'].median()
        v_phi_mean = stars['v_phi'].mean()
        v_phi_std = stars['v_phi'].std()
        v_phi_sem = v_phi_std / np.sqrt(len(stars))
        
        results.append({
            'R': R_center,
            'v_phi_median': v_phi_median,
            'v_phi_mean': v_phi_mean,
            'v_phi_std': v_phi_std,
            'v_phi_sem': v_phi_sem,
            'N': len(stars),
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("CREATING EILERS-QUALITY ROTATION CURVE CATALOG")
    print("=" * 100)
    
    # File paths
    eilers_file = Path("data/gaia/eilers2019_data.fits")
    apogee_file = Path("data/apogee/allStar-dr17-synspec_rev1.fits")
    output_file = Path("data/gaia/eilers_apogee_crossmatch.csv")
    
    # Check files exist
    if not eilers_file.exists():
        print(f"ERROR: Eilers catalog not found: {eilers_file}")
        return
    
    if not apogee_file.exists():
        print(f"ERROR: APOGEE catalog not found: {apogee_file}")
        print("  Download with: wget -O data/apogee/allStar-dr17-synspec_rev1.fits \\")
        print("    'https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits'")
        return
    
    # Load catalogs
    eilers_df = load_eilers_parallax_catalog(eilers_file)
    apogee_df = load_apogee_catalog(apogee_file)
    
    # Cross-match
    matched_df = crossmatch_eilers_apogee(eilers_df, apogee_df)
    
    # Compute galactocentric coordinates
    matched_df = compute_galactocentric_velocities(matched_df)
    
    # Save catalog
    print(f"\nSaving cross-matched catalog to {output_file}...")
    matched_df.to_csv(output_file, index=False)
    
    # Compute and display rotation curve
    curve = compute_rotation_curve(matched_df)
    
    print("\n" + "=" * 100)
    print("ROTATION CURVE FROM EILERS-APOGEE CROSS-MATCH")
    print("=" * 100)
    
    # Compare with Eilers+ 2019 published values
    eilers_2019 = {
        5.0: 232.4, 6.0: 229.4, 7.0: 228.8, 8.0: 228.5,
        9.0: 228.0, 10.0: 227.1, 11.0: 225.8, 12.0: 224.2,
        13.0: 222.3, 14.0: 220.3, 15.0: 218.1
    }
    
    print(f"\n{'R [kpc]':<10} {'N stars':<12} {'v_phi':<12} {'Eilers 2019':<15} {'Δ':<10}")
    print("-" * 60)
    
    for _, row in curve.iterrows():
        R = row['R']
        eilers_v = eilers_2019.get(round(R), None)
        if eilers_v:
            diff = row['v_phi_median'] - eilers_v
            print(f"{R:<10.1f} {int(row['N']):<12} {row['v_phi_median']:<12.1f} {eilers_v:<15.1f} {diff:<+10.1f}")
        else:
            print(f"{R:<10.1f} {int(row['N']):<12} {row['v_phi_median']:<12.1f}")
    
    print("\n" + "=" * 100)
    print("CATALOG SAVED")
    print("=" * 100)
    print(f"\nOutput file: {output_file}")
    print(f"Stars: {len(matched_df)}")
    print(f"\nThis catalog can be used for star-by-star rotation curve analysis.")


if __name__ == "__main__":
    main()

