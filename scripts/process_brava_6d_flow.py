#!/usr/bin/env python3
"""
Process BRAVA 6D Gaia Data: Transform to Galactocentric and Compute Flow Topology

This script:
1. Loads BRAVA-filtered Gaia 6D catalog
2. Transforms to Galactocentric coordinates (R, phi, z, vR, vphi, vz)
3. Bins stars in (R,z) with minimum 50 stars per bin
4. Computes ω² and θ² from velocity gradients
5. Computes C_cov and Σ using existing code
6. Outputs binned data for regression testing

Usage:
    python scripts/process_brava_6d_flow.py --input data/gaia/6d_brava_full.fits --output data/gaia/6d_brava_galcen.parquet
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactocentric, CylindricalDifferential
import astropy.units as u
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19  # m
H0_SI = 2.27e-18  # 1/s
c = 2.998e8  # m/s

# Model parameters
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
L_0 = 0.40  # kpc
N_EXP = 0.27

# g_dagger for h_function
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60×10⁻¹¹ m/s²


def h_function(g: np.ndarray) -> np.ndarray:
    """h(g) = sqrt(g_dagger/g) * g_dagger / (g_dagger + g)"""
    g_safe = np.maximum(g, 1e-30)
    return np.sqrt(g_dagger / g_safe) * g_dagger / (g_dagger + g_safe)


def C_covariant_coherence(
    omega2: np.ndarray,
    rho_kg_m3: np.ndarray,
    theta2: np.ndarray,
) -> np.ndarray:
    """Covariant coherence scalar: C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)"""
    om2 = np.asarray(omega2, dtype=float)
    rho = np.asarray(rho_kg_m3, dtype=float)
    th2 = np.asarray(theta2, dtype=float)
    
    # 4πGρ in (km/s/kpc)^2
    omega_unit_si = 1000.0 / kpc_to_m  # (km/s/kpc) as 1/s
    four_pi_G_rho = 4.0 * np.pi * G * rho  # 1/s²
    four_pi_G_rho_kms_kpc2 = four_pi_G_rho / (omega_unit_si**2)
    
    # H0 in km/s/kpc (≈ 0.07)
    H0_kms_per_kpc = H0_SI * (kpc_to_m / 1000.0)
    H0_sq = H0_kms_per_kpc**2
    
    denom = om2 + four_pi_G_rho_kms_kpc2 + th2 + H0_sq
    denom = np.maximum(denom, 1e-30)
    C = om2 / denom
    C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(C, 0.0, 1.0)

# Galactocentric frame parameters (standard MW values)
R0_KPC = 8.122  # Distance from Sun to Galactic center (Bennett & Bovy 2019)
ZSUN_KPC = 0.0208  # Height of Sun above Galactic plane
VSUN_KMS = [11.1, 232.24, 7.25]  # Solar motion [U, V, W] (Schönrich+ 2010)


def transform_to_galactocentric(
    ra: np.ndarray,
    dec: np.ndarray,
    parallax: np.ndarray,
    pmra: np.ndarray,
    pmdec: np.ndarray,
    radial_velocity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform Gaia observables to Galactocentric cylindrical coordinates.
    
    Returns:
        R_kpc, phi_rad, z_kpc, vR_kms, vphi_kms, vz_kms
    """
    # Create SkyCoord
    coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        distance=(1000.0 / parallax) * u.pc,  # parallax in mas -> distance in pc
        pm_ra_cosdec=pmra * u.mas/u.yr,
        pm_dec=pmdec * u.mas/u.yr,
        radial_velocity=radial_velocity * u.km/u.s,
        frame='icrs'
    )
    
    # Galactocentric frame
    gc_frame = Galactocentric(
        galcen_distance=R0_KPC * u.kpc,
        z_sun=ZSUN_KPC * u.kpc,
        galcen_v_sun=VSUN_KMS * u.km/u.s
    )
    
    # Transform to Galactocentric
    galcen_coords = coords.transform_to(gc_frame)
    
    # Get cylindrical coordinates
    cyl = galcen_coords.cylindrical
    
    R_kpc = cyl.rho.to(u.kpc).value
    phi_rad = cyl.phi.to(u.rad).value
    z_kpc = galcen_coords.z.to(u.kpc).value
    
    # Get cylindrical velocities
    cyl_diff = galcen_coords.velocity.represent_as(
        CylindricalDifferential, galcen_coords.data
    )
    
    vR_kms = cyl_diff.d_rho.to(u.km/u.s).value
    # v_phi = d_phi/dt * R (use equivalencies for angular velocity)
    vphi_kms = (cyl.rho * cyl_diff.d_phi).to(
        u.km/u.s, equivalencies=u.dimensionless_angles()
    ).value
    vz_kms = cyl_diff.d_z.to(u.km/u.s).value
    
    return R_kpc, phi_rad, z_kpc, vR_kms, vphi_kms, vz_kms


def bin_stars(
    df: pd.DataFrame,
    R_bins: Optional[np.ndarray] = None,
    z_bins: Optional[np.ndarray] = None,
    min_stars_per_bin: int = 50,
) -> pd.DataFrame:
    """
    Bin stars in (R,z) space.
    
    Returns binned DataFrame with mean velocities and dispersions.
    If omega2 and theta2 columns exist in df, they will be averaged per bin.
    """
    if R_bins is None:
        R_min, R_max = df['R_kpc'].min(), df['R_kpc'].max()
        R_bins = np.linspace(R_min, R_max, 20)
    
    if z_bins is None:
        z_min, z_max = df['z_kpc'].min(), df['z_kpc'].max()
        z_bins = np.linspace(z_min, z_max, 15)
    
    binned_data = []
    
    for i in range(len(R_bins) - 1):
        for j in range(len(z_bins) - 1):
            R_min, R_max = R_bins[i], R_bins[i+1]
            z_min, z_max = z_bins[j], z_bins[j+1]
            
            # Select stars in this bin
            mask = (
                (df['R_kpc'] >= R_min) & (df['R_kpc'] < R_max) &
                (df['z_kpc'] >= z_min) & (df['z_kpc'] < z_max)
            )
            stars_in_bin = df[mask]
            
            if len(stars_in_bin) < min_stars_per_bin:
                continue
            
            # Compute mean and dispersion
            R_center = (R_min + R_max) / 2
            z_center = (z_min + z_max) / 2
            
            bin_dict = {
                'R_kpc': R_center,
                'z_kpc': z_center,
                'R_min': R_min,
                'R_max': R_max,
                'z_min': z_min,
                'z_max': z_max,
                'n_stars': len(stars_in_bin),
                'vR_mean': stars_in_bin['vR_kms'].mean(),
                'vR_std': stars_in_bin['vR_kms'].std(),
                'vphi_mean': stars_in_bin['vphi_kms'].mean(),
                'vphi_std': stars_in_bin['vphi_kms'].std(),
                'vz_mean': stars_in_bin['vz_kms'].mean(),
                'vz_std': stars_in_bin['vz_kms'].std(),
            }
            
            # If omega2 and theta2 exist, average them
            if 'omega2' in stars_in_bin.columns:
                omega2_valid = stars_in_bin['omega2'].dropna()
                if len(omega2_valid) > 0:
                    bin_dict['omega2'] = omega2_valid.mean()
                else:
                    bin_dict['omega2'] = np.nan
            
            if 'theta2' in stars_in_bin.columns:
                theta2_valid = stars_in_bin['theta2'].dropna()
                if len(theta2_valid) > 0:
                    bin_dict['theta2'] = theta2_valid.mean()
                else:
                    bin_dict['theta2'] = np.nan
            
            binned_data.append(bin_dict)
    
    return pd.DataFrame(binned_data)


def compute_flow_invariants_axisymmetric(
    binned_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ω² and θ² from binned velocity field (axisymmetric approximation).
    
    For axisymmetric rotation:
    - ω ≈ v_phi / R
    - θ ≈ 0 (steady state)
    
    Returns:
        omega2, theta2 in (km/s/kpc)^2
    """
    R = binned_df['R_kpc'].values
    v_phi = binned_df['vphi_mean'].values
    
    # Vorticity: ω ≈ v_phi/R
    R_safe = np.maximum(R, 0.1)  # Avoid division by zero
    omega = v_phi / R_safe  # km/s/kpc
    omega2 = omega**2
    
    # Expansion: θ ≈ 0 for steady-state bulge
    theta2 = np.zeros_like(omega2)
    
    return omega2, theta2


def compute_flow_invariants_3d_gradients(
    df: pd.DataFrame,
    n_neighbors: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ω² and θ² from 3D velocity gradients using KNN.
    
    For each star, finds n_neighbors nearest neighbors and fits a local
    linear velocity field: v(x) ≈ v0 + ∇v · (x - x0)
    
    From ∇v, computes:
    - ω² = |∇×v|² (vorticity magnitude squared)
    - θ² = (∇·v)² (expansion squared)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: R_kpc, phi_rad, z_kpc, vR_kms, vphi_kms, vz_kms
    n_neighbors : int
        Number of nearest neighbors to use for gradient estimation
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (omega2, theta2) in (km/s/kpc)^2, one value per star
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Convert cylindrical to Cartesian coordinates
    x = df['R_kpc'].values * np.cos(df['phi_rad'].values)
    y = df['R_kpc'].values * np.sin(df['phi_rad'].values)
    z = df['z_kpc'].values
    coords = np.column_stack([x, y, z])  # (N, 3)
    
    # Velocities in Cartesian frame
    # v_x = vR * cos(phi) - vphi * sin(phi)
    # v_y = vR * sin(phi) + vphi * cos(phi)
    # v_z = vz
    phi = df['phi_rad'].values
    vR = df['vR_kms'].values
    vphi = df['vphi_kms'].values
    vz = df['vz_kms'].values
    
    vx = vR * np.cos(phi) - vphi * np.sin(phi)
    vy = vR * np.sin(phi) + vphi * np.cos(phi)
    vels = np.column_stack([vx, vy, vz])  # (N, 3)
    
    # Find nearest neighbors
    print(f"    Finding {n_neighbors} nearest neighbors for each star...")
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to exclude self
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    
    # Exclude self (first neighbor is always self)
    indices = indices[:, 1:]  # (N, n_neighbors)
    
    N = len(df)
    omega2 = np.zeros(N)
    theta2 = np.zeros(N)
    
    print(f"    Computing velocity gradients for {N} stars...")
    I = np.eye(3)
    
    for i in range(N):
        if i % 500 == 0:
            print(f"      Progress: {i}/{N} ({i/N*100:.1f}%)")
        
        idx = indices[i]
        # Relative positions and velocities
        dx = coords[idx] - coords[i]  # (k, 3)
        dv = vels[idx] - vels[i]  # (k, 3)
        
        # Skip if degenerate
        if not (np.isfinite(dx).all() and np.isfinite(dv).all()):
            omega2[i] = np.nan
            theta2[i] = np.nan
            continue
        
        # Solve dv ≈ J · dx, where J = ∇v (3x3)
        J = np.zeros((3, 3))
        for comp in range(3):
            try:
                grad_comp, *_ = np.linalg.lstsq(dx, dv[:, comp], rcond=None)
                J[comp, :] = grad_comp
            except:
                omega2[i] = np.nan
                theta2[i] = np.nan
                continue
        
        # Divergence: θ = ∇·v = trace(J)
        theta_i = float(np.trace(J))
        theta2[i] = theta_i**2
        
        # Vorticity vector: ω = ∇×v (from antisymmetric part)
        omega_vec = np.array([
            J[2, 1] - J[1, 2],  # ∂v_z/∂y - ∂v_y/∂z
            J[0, 2] - J[2, 0],  # ∂v_x/∂z - ∂v_z/∂x
            J[1, 0] - J[0, 1],  # ∂v_y/∂x - ∂v_x/∂y
        ])
        omega2[i] = float(np.dot(omega_vec, omega_vec))
    
    # Convert to (km/s/kpc)^2 (already in correct units since coords are in kpc and vels in km/s)
    return omega2, theta2


def compute_baryonic_density_mw(
    R_kpc: np.ndarray,
    z_kpc: np.ndarray,
) -> np.ndarray:
    """
    Compute Milky Way baryonic density at (R,z) positions.
    
    Uses a simplified MW model:
    - Bulge: Hernquist profile
    - Disk: Exponential profile
    - Gas: Thin disk
    
    Returns density in kg/m³
    """
    # Bulge parameters
    M_bulge = 1.0e10 * M_sun  # kg
    a_bulge = 0.5 * kpc_to_m  # m (scale radius)
    
    # Disk parameters
    M_disk = 4.6e10 * M_sun  # kg
    R_d = 2.5 * kpc_to_m  # m (scale length)
    z_d = 0.3 * kpc_to_m  # m (scale height)
    
    # Gas disk
    M_gas = 1.0e10 * M_sun  # kg
    R_gas = 5.0 * kpc_to_m  # m
    z_gas = 0.1 * kpc_to_m  # m
    
    # Convert to meters
    R_m = R_kpc * kpc_to_m
    z_m = np.abs(z_kpc * kpc_to_m)
    
    # Bulge density (Hernquist)
    r_3d = np.sqrt(R_m**2 + z_m**2)
    r_a = r_3d / a_bulge
    rho_bulge = (M_bulge / (2 * np.pi * a_bulge**3)) / (r_a * (1 + r_a)**3)
    
    # Disk density (exponential)
    rho_disk = (M_disk / (4 * np.pi * R_d**2 * z_d)) * np.exp(-R_m / R_d) * np.exp(-z_m / z_d)
    
    # Gas disk
    rho_gas = (M_gas / (4 * np.pi * R_gas**2 * z_gas)) * np.exp(-R_m / R_gas) * np.exp(-z_m / z_gas)
    
    # Total density
    rho_total = rho_bulge + rho_disk + rho_gas
    
    return rho_total  # kg/m³


def main():
    parser = argparse.ArgumentParser(
        description="Process BRAVA 6D Gaia data: transform to Galactocentric and compute flow topology"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/gaia/6d_brava_full.fits"),
        help="Input FITS file with BRAVA-filtered Gaia 6D data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gaia/6d_brava_galcen.parquet"),
        help="Output parquet file with Galactocentric coordinates and binned data",
    )
    parser.add_argument(
        "--min-stars-per-bin",
        type=int,
        default=50,
        help="Minimum stars per bin (default: 50)",
    )
    parser.add_argument(
        "--use-3d-gradients",
        action="store_true",
        help="Use 3D velocity gradients (KNN) instead of axisymmetric approximation",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=50,
        help="Number of neighbors for 3D gradient computation (default: 50)",
    )
    args = parser.parse_args()
    
    print("="*70)
    print("PROCESSING BRAVA 6D GAIA DATA")
    print("="*70)
    
    # Load FITS file
    print(f"\nLoading: {args.input}")
    t = Table.read(str(args.input))
    print(f"  Loaded {len(t):,} stars")
    
    # Filter for valid parallax and reasonable distances
    # Require parallax > 0.1 mas (distance < 10 kpc) for bulge stars
    valid_mask = (
        np.isfinite(t['parallax']) & 
        (t['parallax'] > 0.1) &  # Distance < 10 kpc
        (t['parallax'] < 10.0) &  # Distance > 0.1 kpc (avoid negative parallax issues)
        np.isfinite(t['pmra']) &
        np.isfinite(t['pmdec']) &
        np.isfinite(t['radial_velocity'])
    )
    t_valid = t[valid_mask]
    print(f"  Valid 6D data: {len(t_valid):,} stars ({len(t_valid)/len(t)*100:.1f}%)")
    
    if len(t_valid) == 0:
        print("ERROR: No valid stars found")
        return
    
    # Convert to DataFrame
    df = t_valid.to_pandas()
    
    # Transform to Galactocentric
    print("\nTransforming to Galactocentric coordinates...")
    R_kpc, phi_rad, z_kpc, vR_kms, vphi_kms, vz_kms = transform_to_galactocentric(
        df['ra'].values,
        df['dec'].values,
        df['parallax'].values,
        df['pmra'].values,
        df['pmdec'].values,
        df['radial_velocity'].values,
    )
    
    # Add to DataFrame
    df['R_kpc'] = R_kpc
    df['phi_rad'] = phi_rad
    df['z_kpc'] = z_kpc
    df['vR_kms'] = vR_kms
    df['vphi_kms'] = vphi_kms
    df['vz_kms'] = vz_kms
    
    print(f"  R range: {R_kpc.min():.2f} - {R_kpc.max():.2f} kpc")
    print(f"  z range: {z_kpc.min():.2f} - {z_kpc.max():.2f} kpc")
    print(f"  v_phi range: {vphi_kms.min():.1f} - {vphi_kms.max():.1f} km/s")
    
    # Compute flow invariants
    if args.use_3d_gradients:
        print(f"\nComputing 3D velocity gradients (KNN, n_neighbors={args.n_neighbors})...")
        omega2_star, theta2_star = compute_flow_invariants_3d_gradients(df, n_neighbors=args.n_neighbors)
        df['omega2'] = omega2_star
        df['theta2'] = theta2_star
        print(f"  Computed gradients for {len(df)} stars")
        print(f"  omega^2 range: {np.nanmin(omega2_star):.2f} - {np.nanmax(omega2_star):.2f} (km/s/kpc)^2")
        print(f"  theta^2 range: {np.nanmin(theta2_star):.2f} - {np.nanmax(theta2_star):.2f} (km/s/kpc)^2")
    
    # Bin stars
    print(f"\nBinning stars (min {args.min_stars_per_bin} per bin)...")
    binned_df = bin_stars(df, min_stars_per_bin=args.min_stars_per_bin)
    print(f"  Created {len(binned_df)} bins")
    
    if len(binned_df) == 0:
        print("ERROR: No bins with sufficient stars")
        return
    
    # Compute flow invariants (if not already computed from 3D gradients)
    if not args.use_3d_gradients:
        print("\nComputing flow invariants (axisymmetric approximation)...")
        omega2, theta2 = compute_flow_invariants_axisymmetric(binned_df)
        binned_df['omega2'] = omega2
        binned_df['theta2'] = theta2
    else:
        # Flow invariants already computed per-star and averaged in binning
        omega2 = binned_df['omega2'].values
        theta2 = binned_df['theta2'].values
        print(f"\nUsing 3D gradient-based flow invariants (averaged per bin)")
        print(f"  omega^2 range: {np.nanmin(omega2):.2f} - {np.nanmax(omega2):.2f} (km/s/kpc)^2")
        print(f"  theta^2 range: {np.nanmin(theta2):.2f} - {np.nanmax(theta2):.2f} (km/s/kpc)^2")
    
    # Compute baryonic density
    print("Computing baryonic density...")
    rho_kg_m3 = compute_baryonic_density_mw(
        binned_df['R_kpc'].values,
        binned_df['z_kpc'].values,
    )
    binned_df['rho_kg_m3'] = rho_kg_m3
    
    # Compute covariant coherence
    print("Computing covariant coherence C_cov...")
    C_cov = C_covariant_coherence(omega2, rho_kg_m3, theta2)
    binned_df['C_cov'] = C_cov
    
    # Compute Sigma (enhancement factor)
    print("Computing Sigma (enhancement factor)...")
    R_kpc_arr = binned_df['R_kpc'].values
    R_m = R_kpc_arr * kpc_to_m
    
    # Proper MW baryonic acceleration: bulge + disk
    M_bulge = 1.0e10 * M_sun  # kg
    M_disk = 4.6e10 * M_sun  # kg
    R_d = 2.5 * kpc_to_m  # disk scale length in m
    
    # Enclosed mass: bulge (point mass approx) + disk (exponential)
    M_enc = M_bulge + M_disk * (1.0 - np.exp(-R_m / R_d))
    g_bar = G * M_enc / np.maximum(R_m**2, 1e-9)  # m/s²
    binned_df['g_bar'] = g_bar  # Save for Jeans test
    h = h_function(g_bar)
    
    # Enhancement: Σ = 1 + A₀ × C_cov × h
    A_base = A_0
    Sigma = 1.0 + A_base * C_cov * h
    binned_df['Sigma'] = Sigma
    
    # Save output
    print(f"\nSaving to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binned_df.to_parquet(args.output, index=False)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"  Input stars: {len(t_valid):,}")
    print(f"  Bins created: {len(binned_df)}")
    print(f"  Output file: {args.output}")
    print(f"\nBinned data columns:")
    for col in binned_df.columns:
        print(f"    - {col}")
    print("="*70)


if __name__ == "__main__":
    main()

