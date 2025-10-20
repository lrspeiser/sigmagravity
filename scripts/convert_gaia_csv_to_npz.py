#!/usr/bin/env python3
"""
Convert raw Gaia DR3 CSV files to NPZ format with derived galactocentric columns.

Input: CSV with raw Gaia catalog fields (ra, dec, parallax, pmra, pmdec, radial_velocity)
Output: NPZ with derived columns (R_kpc, z_kpc, v_obs_kms, v_err_kms, gN_kms2_per_kpc, Sigma_loc_Msun_pc2)

Based on conversion logic from DensityDependentMetricModel/analysis/analyze_gaia_distribution.py
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def convert_gaia_csv_to_npz(csv_path, output_npz, baryon_model='mw_multi'):
    """
    Convert raw Gaia CSV to NPZ with derived galactocentric columns.
    
    Args:
        csv_path: Path to input CSV with Gaia DR3 columns
        output_npz: Path to output NPZ file
        baryon_model: Baryonic model for gN calculation (default: mw_multi)
    """
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord, Galactocentric, CylindricalDifferential
    except ImportError:
        raise SystemExit("ERROR: astropy is required. Install with: pip install astropy")
    
    print(f"Loading raw Gaia data from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw):,} stars")
    
    # Check required columns
    required_cols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    
    # Quality filters
    print("Applying quality filters...")
    n_initial = len(df_raw)
    
    # Filter out invalid parallax (must be positive for distance calculation)
    df_raw = df_raw[df_raw['parallax'] > 0]
    n_after_plx = len(df_raw)
    print(f"  Positive parallax: {n_after_plx:,} / {n_initial:,} ({100*n_after_plx/n_initial:.1f}%)")
    
    # Filter out rows with NaN radial velocity (required for 3D kinematics)
    df_raw = df_raw[df_raw['radial_velocity'].notna()]
    n_after_rv = len(df_raw)
    print(f"  Valid RV: {n_after_rv:,} / {n_initial:,} ({100*n_after_rv/n_initial:.1f}%)")
    
    # Filter out very large parallax errors (parallax_error/parallax > 0.2)
    if 'parallax_error' in df_raw.columns:
        df_raw = df_raw[(df_raw['parallax_error'] / df_raw['parallax']) < 0.2]
        n_after_plx_err = len(df_raw)
        print(f"  Good parallax precision: {n_after_plx_err:,} / {n_initial:,} ({100*n_after_plx_err/n_initial:.1f}%)")
    
    if len(df_raw) == 0:
        raise SystemExit("ERROR: No stars passed quality filters!")
    
    print(f"Final sample: {len(df_raw):,} stars")
    
    # Galactocentric frame parameters (standard Astropy defaults + DDMM)
    R0_KPC = 8.122  # Distance from Sun to Galactic center (Bennett & Bovy 2019)
    ZSUN_KPC = 0.0208  # Height of Sun above Galactic plane
    VSUN_KMS = [11.1, 232.24, 7.25]  # Solar motion [U, V, W] (Schönrich+ 2010)
    
    gc_frame = Galactocentric(
        galcen_distance=R0_KPC * u.kpc,
        z_sun=ZSUN_KPC * u.kpc,
        galcen_v_sun=VSUN_KMS * u.km/u.s
    )
    
    print("Transforming to Galactocentric coordinates...")
    
    # Create SkyCoord from raw Gaia catalog data
    coords = SkyCoord(
        ra=df_raw['ra'].values * u.deg,
        dec=df_raw['dec'].values * u.deg,
        distance=(1000 / df_raw['parallax'].values) * u.pc,  # parallax in mas -> distance in pc
        pm_ra_cosdec=df_raw['pmra'].values * u.mas/u.yr,
        pm_dec=df_raw['pmdec'].values * u.mas/u.yr,
        radial_velocity=df_raw['radial_velocity'].values * u.km/u.s,
        frame='icrs'
    )
    
    # Transform to Galactocentric frame
    galcen_coords = coords.transform_to(gc_frame)
    cylindrical_velocities = galcen_coords.velocity.represent_as(
        CylindricalDifferential, galcen_coords.data
    )
    
    # Calculate tangential velocity (v_phi) and take absolute value for rotation speed
    v_phi_kms = (galcen_coords.cylindrical.rho * cylindrical_velocities.d_phi).to(
        u.km/u.s, equivalencies=u.dimensionless_angles()
    ).value
    v_obs_kms = np.abs(v_phi_kms)
    
    # Cylindrical radius and height
    R_kpc = galcen_coords.cylindrical.rho.to(u.kpc).value
    z_kpc = galcen_coords.z.to(u.kpc).value
    
    # Error propagation: combine PM and RV uncertainties
    dist_kpc = coords.distance.to(u.kpc).value
    
    # PM error in km/s (factor 4.74047 = 1 mas/yr * 1 kpc in km/s)
    if 'pmra_error' in df_raw.columns and 'pmdec_error' in df_raw.columns:
        pm_error_kms = np.sqrt(
            df_raw['pmra_error'].fillna(0)**2 + df_raw['pmdec_error'].fillna(0)**2
        ) * dist_kpc * 4.74047
    else:
        # Use typical PM uncertainty if error columns missing
        pm_error_kms = np.full_like(dist_kpc, 0.5) * dist_kpc * 4.74047  # ~0.5 mas/yr
    
    # Total velocity error
    if 'radial_velocity_error' in df_raw.columns:
        rv_error = df_raw['radial_velocity_error'].fillna(5.0)  # km/s
    else:
        rv_error = np.full_like(dist_kpc, 5.0)  # typical RV uncertainty km/s
    
    sigma_v = np.sqrt(rv_error**2 + pm_error_kms**2)
    
    # Clip to reasonable bounds
    MIN_VELOCITY_ERROR_KMS = 1.0
    MAX_VELOCITY_ERROR_KMS = 100.0
    v_err_kms = np.clip(sigma_v, MIN_VELOCITY_ERROR_KMS, MAX_VELOCITY_ERROR_KMS)
    
    # Compute Newtonian acceleration g_N from baryonic model
    print(f"Computing baryonic acceleration (model: {baryon_model})...")
    gN_kms2_per_kpc = compute_baryonic_acceleration(R_kpc, baryon_model)
    
    # Compute local surface density (for future vertical analysis)
    # Placeholder: set to typical MW thin-disk value
    Sigma_loc_Msun_pc2 = np.full_like(R_kpc, 40.0)  # Msun/pc^2
    
    # Save to NPZ
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        R_kpc=R_kpc,
        z_kpc=z_kpc,
        v_obs_kms=v_obs_kms,
        v_err_kms=v_err_kms,
        gN_kms2_per_kpc=gN_kms2_per_kpc,
        Sigma_loc_Msun_pc2=Sigma_loc_Msun_pc2
    )
    
    print(f"\n✓ Wrote {output_path} ({len(R_kpc):,} stars)")
    print(f"  R range: {R_kpc.min():.2f} – {R_kpc.max():.2f} kpc")
    print(f"  v_obs range: {v_obs_kms.min():.1f} – {v_obs_kms.max():.1f} km/s")
    print(f"  Mean v_obs: {v_obs_kms.mean():.1f} ± {v_obs_kms.std():.1f} km/s")


def compute_baryonic_acceleration(R_kpc, model='mw_multi'):
    """
    Compute Newtonian circular-orbit acceleration g_N = v_bar^2 / R from baryonic model.
    
    Args:
        R_kpc: Galactocentric cylindrical radius (kpc)
        model: Baryonic model ('mw_multi' = multi-component MW disk)
    
    Returns:
        gN: Newtonian acceleration in (km/s)^2 / kpc
    """
    # Multi-component MW disk model (thin+thick+gas+bulge)
    # From maxdepth_gaia default parameters
    G_KPC_MSUN_KMS2 = 4.302e-6  # Gravitational constant in kpc (km/s)^2 Msun^-1
    
    # Bulge (Hernquist)
    M_b = 5e9  # Msun
    a_b = 0.6  # kpc
    v2_bulge = G_KPC_MSUN_KMS2 * M_b / (R_kpc + a_b)
    
    # Thin disk (Miyamoto-Nagai)
    M_thin = 4.5e10  # Msun
    a_thin = 3.0  # kpc
    b_thin = 0.3  # kpc
    sqrt_term_thin = np.sqrt(R_kpc**2 + (a_thin + b_thin)**2)
    v2_thin = G_KPC_MSUN_KMS2 * M_thin * R_kpc**2 / sqrt_term_thin**3
    
    # Thick disk (Miyamoto-Nagai)
    M_thick = 1.0e10  # Msun
    a_thick = 2.5  # kpc
    b_thick = 0.9  # kpc
    sqrt_term_thick = np.sqrt(R_kpc**2 + (a_thick + b_thick)**2)
    v2_thick = G_KPC_MSUN_KMS2 * M_thick * R_kpc**2 / sqrt_term_thick**3
    
    # H I gas (Miyamoto-Nagai)
    M_HI = 1.1e10  # Msun
    a_HI = 7.0  # kpc
    b_HI = 0.1  # kpc
    sqrt_term_HI = np.sqrt(R_kpc**2 + (a_HI + b_HI)**2)
    v2_HI = G_KPC_MSUN_KMS2 * M_HI * R_kpc**2 / sqrt_term_HI**3
    
    # H2 molecular gas (Miyamoto-Nagai)
    M_H2 = 1.2e9  # Msun
    a_H2 = 1.5  # kpc
    b_H2 = 0.05  # kpc
    sqrt_term_H2 = np.sqrt(R_kpc**2 + (a_H2 + b_H2)**2)
    v2_H2 = G_KPC_MSUN_KMS2 * M_H2 * R_kpc**2 / sqrt_term_H2**3
    
    # Total baryonic circular speed squared
    v2_bar = v2_bulge + v2_thin + v2_thick + v2_HI + v2_H2
    
    # g_N = v_bar^2 / R in (km/s)^2 / kpc
    gN = v2_bar / np.maximum(R_kpc, 1e-6)
    
    return gN


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='Input CSV path with raw Gaia DR3 columns')
    parser.add_argument('--output', required=True, help='Output NPZ path')
    parser.add_argument('--baryon-model', default='mw_multi', help='Baryonic model for gN calculation')
    args = parser.parse_args()
    
    convert_gaia_csv_to_npz(args.input, args.output, args.baryon_model)


if __name__ == '__main__':
    main()
