#!/usr/bin/env python3
"""
Compare Sigma-Gravity, MOND, and GR predictions vs. observations
for BRAVA bulge and Gaia disk regions.

This script:
1. Loads BRAVA 6D bulge data (binned)
2. Loads/processes Gaia disk data (binned)
3. Computes predictions for Sigma-Gravity, MOND, and GR
4. Compares all three to actual observations
5. Shows results separately for bulge vs. disk
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G = 6.674e-11  # m³/kg/s²
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19  # m
H0_SI = 2.27e-18  # 1/s
c = 2.998e8  # m/s
a0_mond = 1.2e-10  # m/s² (MOND acceleration scale)

# Model parameters
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60×10⁻¹¹ m/s²


def h_function(g: np.ndarray) -> np.ndarray:
    """h(g) = sqrt(g_dagger/g) * g_dagger / (g_dagger + g)"""
    g_safe = np.maximum(g, 1e-30)
    return np.sqrt(g_dagger / g_safe) * g_dagger / (g_dagger + g_safe)


def compute_baryonic_density_mw(
    R_kpc: np.ndarray,
    z_kpc: np.ndarray,
) -> np.ndarray:
    """Compute Milky Way baryonic density at (R,z) positions."""
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


def compute_baryonic_acceleration_mw(R_kpc: np.ndarray) -> np.ndarray:
    """Compute baryonic acceleration from simplified MW model."""
    M_bulge = 1.0e10 * M_sun  # kg
    M_disk = 4.6e10 * M_sun  # kg
    R_d = 2.5 * kpc_to_m  # m
    
    R_m = R_kpc * kpc_to_m
    M_enc = M_bulge + M_disk * (1.0 - np.exp(-R_m / R_d))
    g_bar = G * M_enc / np.maximum(R_m**2, 1e-9)
    return g_bar  # m/s²


def predict_sigma_gravity(
    R_kpc: np.ndarray,
    C_cov: np.ndarray,
    g_bar: np.ndarray,
) -> np.ndarray:
    """Sigma-Gravity prediction for velocity dispersion."""
    h = h_function(g_bar)
    Sigma = 1.0 + A_0 * C_cov * h
    V_circ = np.sqrt(g_bar * R_kpc * kpc_to_m) / 1000.0  # km/s
    calibration_factor = 0.51  # From BRAVA data
    sigma_pred = V_circ * np.sqrt(Sigma) * calibration_factor
    return sigma_pred


def predict_mond(
    R_kpc: np.ndarray,
    g_bar: np.ndarray,
) -> np.ndarray:
    """MOND prediction for velocity dispersion."""
    # MOND interpolation: g_obs = g_bar × ν(g_bar/a0)
    x = g_bar / a0_mond
    x = np.maximum(x, 1e-10)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    g_mond = g_bar * nu
    
    V_circ_mond = np.sqrt(g_mond * R_kpc * kpc_to_m) / 1000.0  # km/s
    calibration_factor = 0.51  # Same calibration for fair comparison
    sigma_pred = V_circ_mond * calibration_factor
    return sigma_pred


def predict_gr_newtonian(
    R_kpc: np.ndarray,
    g_bar: np.ndarray,
) -> np.ndarray:
    """GR/Newtonian prediction (no enhancement)."""
    V_circ = np.sqrt(g_bar * R_kpc * kpc_to_m) / 1000.0  # km/s
    calibration_factor = 0.51  # Same calibration for fair comparison
    sigma_pred = V_circ * calibration_factor
    return sigma_pred


def load_brava_bulge_data(
    binned_path: Optional[Path] = None,
    use_3d_gradients: bool = False,
) -> pd.DataFrame:
    """Load BRAVA bulge binned data."""
    if binned_path is None:
        if use_3d_gradients:
            binned_path = Path("data/gaia/6d_brava_galcen_3d.parquet")
        else:
            binned_path = Path("data/gaia/6d_brava_galcen.parquet")
    
    if not binned_path.exists():
        raise FileNotFoundError(f"BRAVA binned data not found: {binned_path}")
    
    df = pd.read_parquet(binned_path)
    df['region'] = 'bulge'
    return df


def load_gaia_disk_data() -> Optional[pd.DataFrame]:
    """Load Gaia disk data (if available)."""
    # Try to load processed Gaia disk data
    disk_paths = [
        Path("data/gaia/gaia_processed_corrected.csv"),
        Path("data/gaia/mw/gaia_mw_real.csv"),
    ]
    
    for path in disk_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                # Filter for disk region (R > 4 kpc, |z| < 1 kpc)
                if 'R_kpc' in df.columns or 'R_gal' in df.columns:
                    R_col = 'R_kpc' if 'R_kpc' in df.columns else 'R_gal'
                    z_col = 'z_kpc' if 'z_kpc' in df.columns else 'z_gal'
                    
                    mask = (df[R_col] > 4.0) & (df[R_col] < 12.0) & (np.abs(df[z_col]) < 1.0)
                    df = df[mask].copy()
                    
                    if len(df) > 100:
                        # Bin the disk data similar to bulge
                        return bin_disk_data(df, R_col, z_col)
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")
                continue
    
    return None


def bin_disk_data(
    df: pd.DataFrame,
    R_col: str,
    z_col: str,
    min_stars_per_bin: int = 50,
) -> pd.DataFrame:
    """Bin disk data similar to bulge processing."""
    R = df[R_col].values
    z = df[z_col].values
    
    # Create bins
    R_bins = np.linspace(R.min(), R.max(), 15)
    z_bins = np.linspace(z.min(), z.max(), 5)
    
    binned_data = []
    
    for i in range(len(R_bins) - 1):
        for j in range(len(z_bins) - 1):
            R_min, R_max = R_bins[i], R_bins[i+1]
            z_min, z_max = z_bins[j], z_bins[j+1]
            
            mask = (
                (R >= R_min) & (R < R_max) &
                (z >= z_min) & (z < z_max)
            )
            stars_in_bin = df[mask]
            
            if len(stars_in_bin) < min_stars_per_bin:
                continue
            
            # Compute statistics
            R_center = (R_min + R_max) / 2
            z_center = (z_min + z_max) / 2
            
            # Get velocity columns (try different names)
            vR_col = None
            vphi_col = None
            vz_col = None
            
            for col in df.columns:
                if 'vR' in col.lower() or 'vr' in col.lower():
                    vR_col = col
                if 'vphi' in col.lower() or 'v_phi' in col.lower():
                    vphi_col = col
                if 'vz' in col.lower() or 'v_z' in col.lower():
                    vz_col = col
            
            bin_dict = {
                'R_kpc': R_center,
                'z_kpc': z_center,
                'n_stars': len(stars_in_bin),
            }
            
            if vR_col:
                bin_dict['vR_mean'] = stars_in_bin[vR_col].mean()
                bin_dict['vR_std'] = stars_in_bin[vR_col].std()
            if vphi_col:
                bin_dict['vphi_mean'] = stars_in_bin[vphi_col].mean()
                bin_dict['vphi_std'] = stars_in_bin[vphi_col].std()
            if vz_col:
                bin_dict['vz_mean'] = stars_in_bin[vz_col].mean()
                bin_dict['vz_std'] = stars_in_bin[vz_col].std()
            
            # Compute flow invariants (axisymmetric approximation)
            if vphi_col:
                v_phi = bin_dict['vphi_mean']
                R_safe = max(R_center, 0.1)
                omega = v_phi / R_safe
                bin_dict['omega2'] = omega**2
                bin_dict['theta2'] = 0.0
            
            # Compute density and C_cov (simplified)
            # For disk, use simpler model
            rho = compute_baryonic_density_mw(np.array([R_center]), np.array([z_center]))[0]
            bin_dict['rho_kg_m3'] = rho
            
            if 'omega2' in bin_dict:
                omega2 = bin_dict['omega2']
                theta2 = bin_dict['theta2']
                C_cov = C_covariant_coherence(
                    np.array([omega2]),
                    np.array([rho]),
                    np.array([theta2])
                )[0]
                bin_dict['C_cov'] = C_cov
            
            binned_data.append(bin_dict)
    
    result_df = pd.DataFrame(binned_data)
    result_df['region'] = 'disk'
    return result_df


def compute_predictions_and_compare(df: pd.DataFrame) -> dict:
    """Compute predictions for all theories and compare to observations."""
    R_kpc = df['R_kpc'].values
    sigma_tot_obs = np.sqrt(
        df['vR_std'].values**2 +
        df['vphi_std'].values**2 +
        df['vz_std'].values**2
    )
    
    # Compute baryonic acceleration
    g_bar = compute_baryonic_acceleration_mw(R_kpc)
    
    # Get C_cov if available (for Sigma-Gravity)
    if 'C_cov' in df.columns:
        C_cov = df['C_cov'].values
    else:
        # Fallback: use axisymmetric approximation
        v_phi = df['vphi_mean'].values
        R_safe = np.maximum(R_kpc, 0.1)
        omega = v_phi / R_safe
        omega2 = omega**2
        theta2 = np.zeros_like(omega2)
        
        if 'rho_kg_m3' in df.columns:
            rho = df['rho_kg_m3'].values
        else:
            rho = compute_baryonic_density_mw(R_kpc, df['z_kpc'].values)
        
        C_cov = C_covariant_coherence(omega2, rho, theta2)
    
    # Compute predictions
    sigma_sigma = predict_sigma_gravity(R_kpc, C_cov, g_bar)
    sigma_mond = predict_mond(R_kpc, g_bar)
    sigma_gr = predict_gr_newtonian(R_kpc, g_bar)
    
    # Compute residuals and RMS
    resid_sigma = sigma_tot_obs - sigma_sigma
    resid_mond = sigma_tot_obs - sigma_mond
    resid_gr = sigma_tot_obs - sigma_gr
    
    rms_sigma = np.sqrt((resid_sigma**2).mean())
    rms_mond = np.sqrt((resid_mond**2).mean())
    rms_gr = np.sqrt((resid_gr**2).mean())
    
    return {
        'sigma_obs': sigma_tot_obs,
        'sigma_sigma': sigma_sigma,
        'sigma_mond': sigma_mond,
        'sigma_gr': sigma_gr,
        'resid_sigma': resid_sigma,
        'resid_mond': resid_mond,
        'resid_gr': resid_gr,
        'rms_sigma': rms_sigma,
        'rms_mond': rms_mond,
        'rms_gr': rms_gr,
        'C_cov': C_cov,
    }


def print_comparison_table(results: dict, region: str):
    """Print comparison table for a region."""
    print(f"\n{'='*80}")
    print(f"{region.upper()} REGION COMPARISON")
    print(f"{'='*80}")
    
    rms_sigma = results['rms_sigma']
    rms_mond = results['rms_mond']
    rms_gr = results['rms_gr']
    
    print(f"\n{'Theory':<20} {'RMS (km/s)':<15} {'vs GR':<15} {'vs MOND':<15}")
    print("-" * 80)
    print(f"{'GR/Newtonian':<20} {rms_gr:<15.2f} {'(baseline)':<15} {rms_gr - rms_mond:<15.2f}")
    print(f"{'MOND':<20} {rms_mond:<15.2f} {rms_mond - rms_gr:<15.2f} {'(baseline)':<15}")
    print(f"{'Sigma-Gravity':<20} {rms_sigma:<15.2f} {rms_sigma - rms_gr:<15.2f} {rms_sigma - rms_mond:<15.2f}")
    
    # Find best theory
    rms_values = {'GR': rms_gr, 'MOND': rms_mond, 'Sigma-Gravity': rms_sigma}
    best = min(rms_values, key=rms_values.get)
    
    print(f"\nBest fit: {best} (RMS = {rms_values[best]:.2f} km/s)")
    
    # Statistics
    sigma_obs = results['sigma_obs']
    print(f"\nObserved sigma_tot: mean={sigma_obs.mean():.1f}, std={sigma_obs.std():.1f} km/s")
    print(f"  Range: {sigma_obs.min():.1f} - {sigma_obs.max():.1f} km/s")
    
    if 'C_cov' in results:
        C_cov = results['C_cov']
        print(f"\nCovariant coherence (Sigma-Gravity): mean={C_cov.mean():.3f}, std={C_cov.std():.3f}")


def main():
    import sys
    
    use_3d = "--3d" in sys.argv or "--use-3d" in sys.argv
    
    print("="*80)
    print("THEORY COMPARISON: Sigma-Gravity vs MOND vs GR")
    print("BRAVA Bulge vs Gaia Disk")
    print("="*80)
    
    # Load BRAVA bulge data
    print("\nLoading BRAVA bulge data...")
    try:
        bulge_df = load_brava_bulge_data(use_3d_gradients=use_3d)
        print(f"  Loaded {len(bulge_df)} bulge bins")
        bulge_results = compute_predictions_and_compare(bulge_df)
        print_comparison_table(bulge_results, "BULGE")
    except Exception as e:
        print(f"  ERROR loading bulge data: {e}")
        bulge_results = None
    
    # Load Gaia disk data
    print("\nLoading Gaia disk data...")
    try:
        disk_df = load_gaia_disk_data()
        if disk_df is not None and len(disk_df) > 0:
            print(f"  Loaded {len(disk_df)} disk bins")
            disk_results = compute_predictions_and_compare(disk_df)
            print_comparison_table(disk_results, "DISK")
        else:
            print("  Disk data not available (skipping)")
            disk_results = None
    except Exception as e:
        print(f"  ERROR loading disk data: {e}")
        disk_results = None
    
    # Summary comparison
    if bulge_results and disk_results:
        print(f"\n{'='*80}")
        print("SUMMARY: BULGE vs DISK")
        print(f"{'='*80}")
        
        print(f"\n{'Region':<15} {'Sigma-Gravity':<15} {'MOND':<15} {'GR':<15} {'Best':<15}")
        print("-" * 80)
        
        bulge_best = min(['GR', 'MOND', 'Sigma-Gravity'], 
                        key=lambda x: {'GR': bulge_results['rms_gr'], 
                                      'MOND': bulge_results['rms_mond'],
                                      'Sigma-Gravity': bulge_results['rms_sigma']}[x])
        disk_best = min(['GR', 'MOND', 'Sigma-Gravity'],
                       key=lambda x: {'GR': disk_results['rms_gr'],
                                     'MOND': disk_results['rms_mond'],
                                     'Sigma-Gravity': disk_results['rms_sigma']}[x])
        
        print(f"{'Bulge':<15} {bulge_results['rms_sigma']:<15.2f} "
              f"{bulge_results['rms_mond']:<15.2f} {bulge_results['rms_gr']:<15.2f} "
              f"{bulge_best:<15}")
        print(f"{'Disk':<15} {disk_results['rms_sigma']:<15.2f} "
              f"{disk_results['rms_mond']:<15.2f} {disk_results['rms_gr']:<15.2f} "
              f"{disk_best:<15}")
        
        print(f"\n{'='*80}")
        print("KEY FINDINGS:")
        print(f"{'='*80}")
        print(f"1. Bulge best fit: {bulge_best}")
        print(f"2. Disk best fit: {disk_best}")
        print(f"3. Sigma-Gravity improvement over GR (bulge): "
              f"{bulge_results['rms_gr'] - bulge_results['rms_sigma']:.2f} km/s")
        print(f"4. Sigma-Gravity improvement over GR (disk): "
              f"{disk_results['rms_gr'] - disk_results['rms_sigma']:.2f} km/s")
        print(f"5. Sigma-Gravity improvement over MOND (bulge): "
              f"{bulge_results['rms_mond'] - bulge_results['rms_sigma']:.2f} km/s")
        print(f"6. Sigma-Gravity improvement over MOND (disk): "
              f"{disk_results['rms_mond'] - disk_results['rms_sigma']:.2f} km/s")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

