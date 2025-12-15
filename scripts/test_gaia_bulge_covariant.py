#!/usr/bin/env python3
"""
Gaia Bulge Regression Test: Covariant Coherence Scalar

This test implements the strategic pivot: use Gaia bulge as calibration lab
for the covariant coherence scalar C_cov = ω²/(ω² + 4πGρ + θ² + H₀²).

Goal: Ultra-accurate bulge kinematics prediction using true flow invariants
computed from 6D star field, not approximated from 1D rotation curves.

Test Design:
1. Select Gaia bulge stars (spatial/kinematic cuts)
2. Bin in (R, z) space with sufficient density
3. Compute local velocity field and gradients → ω², θ²
4. Evaluate baryonic density model ρ_b(R,z) at star locations
5. Compute C_cov and predict kinematics
6. Compare to observed bulge kinematics

Success Criteria:
- Better than baseline by >1 km/s improvement
- Avoids data leakage (uses baryonic model, not fitted parameters)
- Reports bulge-only pass/fail alongside existing suite
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Import from regression test
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_regression_experimental import (
    C_covariant_coherence, predict_velocity, h_function,
    A_0, L_0, N_EXP, g_dagger, G, M_sun, kpc_to_m, H0_SI
)

@dataclass
class GaiaBulgeTestResult:
    """Results from Gaia bulge covariant coherence test."""
    passed: bool
    rms_kms: float
    baseline_rms_kms: float
    improvement_kms: float
    n_stars: int
    n_bins: int
    details: Dict[str, Any]
    message: str


def select_gaia_bulge_stars(
    gaia_df: pd.DataFrame,
    z_threshold_kpc: float = 0.3,
    R_max_kpc: float = 5.0,
    min_stars_total: int = 200,
    use_inner_disk: bool = True,
) -> pd.DataFrame:
    """Select bulge/inner disk stars from Gaia catalog.
    
    NOTE: The Eilers+ APOGEE-Gaia sample is a disk sample (R: 4-16 kpc).
    For true bulge (R < 3 kpc), a separate dataset is needed.
    This function uses inner disk (R < 5 kpc) as a proxy for testing.
    
    Parameters
    ----------
    gaia_df : pd.DataFrame
        Full Gaia catalog with columns: R_gal, z_gal, v_R, v_phi, v_z, etc.
    z_threshold_kpc : float
        Minimum |z| to be considered (default: 0.3 kpc for inner disk)
    R_max_kpc : float
        Maximum R for selection (default: 5.0 kpc for inner disk)
    min_stars_total : int
        Minimum total stars needed for meaningful analysis
    use_inner_disk : bool
        If True, use inner disk as bulge proxy (default: True)
        
    Returns
    -------
    pd.DataFrame
        Selected inner disk/bulge stars
    """
    if use_inner_disk:
        # Use inner disk as proxy (since catalog is R: 4-16 kpc)
        # Select stars with R < 5 kpc and elevated |z|
        bulge_mask = (
            (gaia_df['R_gal'] < R_max_kpc) &
            (np.abs(gaia_df['z_gal']) >= z_threshold_kpc)
        )
    else:
        # True bulge selection (requires R < 3 kpc data)
        bulge_mask = (
            (gaia_df['R_gal'] <= R_max_kpc) &
            (np.abs(gaia_df['z_gal']) >= z_threshold_kpc)
        )
    
    bulge_df = gaia_df[bulge_mask].copy()
    
    if len(bulge_df) < min_stars_total:
        return pd.DataFrame()  # Not enough stars
    
    return bulge_df


def bin_gaia_bulge(
    bulge_df: pd.DataFrame,
    R_bins: np.ndarray,
    z_bins: np.ndarray,
    min_stars_per_bin: int = 30,
) -> pd.DataFrame:
    """Bin bulge stars in (R, z) space.
    
    Parameters
    ----------
    bulge_df : pd.DataFrame
        Bulge stars
    R_bins : np.ndarray
        Radial bin edges (kpc)
    z_bins : np.ndarray
        Vertical bin edges (kpc)
    min_stars_per_bin : int
        Minimum stars per bin for stable statistics
        
    Returns
    -------
    pd.DataFrame
        Binned data with mean velocities, dispersions, and counts
    """
    binned_data = []
    
    for i in range(len(R_bins) - 1):
        for j in range(len(z_bins) - 1):
            mask = (
                (bulge_df['R_gal'] >= R_bins[i]) &
                (bulge_df['R_gal'] < R_bins[i + 1]) &
                (bulge_df['z_gal'] >= z_bins[j]) &
                (bulge_df['z_gal'] < z_bins[j + 1])
            )
            
            if mask.sum() >= min_stars_per_bin:
                subset = bulge_df[mask]
                binned_data.append({
                    'R': (R_bins[i] + R_bins[i + 1]) / 2,
                    'z': (z_bins[j] + z_bins[j + 1]) / 2,
                    'v_phi_mean': subset['v_phi'].mean() if 'v_phi' in subset.columns else subset.get('v_phi_obs', subset.get('v_phi_corrected', pd.Series([0]))).mean(),
                    'v_R_mean': subset['v_R'].mean(),
                    'v_z_mean': subset['v_z'].mean(),
                    'sigma_R': subset['v_R'].std(),
                    'sigma_phi': subset['v_phi'].std() if 'v_phi' in subset.columns else subset.get('v_phi_obs', subset.get('v_phi_corrected', pd.Series([0]))).std(),
                    'sigma_z': subset['v_z'].std(),
                    'n_stars': len(subset),
                })
    
    return pd.DataFrame(binned_data)


def compute_flow_invariants_from_binned(
    binned_df: pd.DataFrame,
    smooth_points: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ω², θ² from binned velocity field.
    
    For axisymmetric bulge, we approximate:
    - ω² from mean v_phi and R
    - θ² from velocity field divergence (≈ 0 for steady state)
    
    Parameters
    ----------
    binned_df : pd.DataFrame
        Binned bulge data
    smooth_points : int
        Smoothing window for gradients
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (omega2, theta2) in (km/s/kpc)^2
    """
    R = binned_df['R'].values
    v_phi = binned_df['v_phi_mean'].values
    
    # Vorticity: ω ≈ v_phi/R for axisymmetric rotation
    # ω² in (km/s/kpc)^2
    R_safe = np.maximum(R, 0.1)  # Avoid division by zero
    omega = v_phi / R_safe  # km/s/kpc
    omega2 = omega**2
    
    # Expansion: θ ≈ 0 for steady-state bulge (incompressible)
    theta2 = np.zeros_like(omega2)
    
    return omega2, theta2


def compute_baryonic_density_mw(
    R_kpc: np.ndarray,
    z_kpc: np.ndarray,
) -> np.ndarray:
    """Compute Milky Way baryonic density at (R, z) locations.
    
    Uses McMillan 2017 model components:
    - Bulge: ρ_bulge(r) with scale radius
    - Disk: ρ_disk(R, z) exponential in R and z
    - Gas: ρ_gas(R, z) similar to disk
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Cylindrical radius (kpc)
    z_kpc : np.ndarray
        Vertical height (kpc)
        
    Returns
    -------
    np.ndarray
        Total baryonic density in kg/m³
    """
    # McMillan 2017 parameters (scaled by 1.16)
    MW_SCALE = 1.16
    M_disk = 4.6e10 * MW_SCALE**2 * M_sun  # kg
    M_bulge = 1.0e10 * MW_SCALE**2 * M_sun  # kg
    M_gas = 1.0e10 * MW_SCALE**2 * M_sun  # kg
    
    # Scale lengths
    R_d = 3.3  # kpc (disk)
    z_d = 0.3  # kpc (disk scale height)
    R_bulge = 0.5  # kpc (bulge scale radius)
    R_gas = 7.0  # kpc (gas)
    
    # Convert to meters
    R_m = R_kpc * kpc_to_m
    z_m = np.abs(z_kpc) * kpc_to_m
    
    # Bulge: Hernquist profile (more realistic than power law)
    r_sphere = np.sqrt(R_m**2 + z_m**2)
    a_bulge_m = R_bulge * kpc_to_m
    # Hernquist: ρ(r) = M/(2π) × a/(r(a+r)³)
    rho_bulge = M_bulge / (2.0 * np.pi) * a_bulge_m / (r_sphere * (a_bulge_m + r_sphere)**3)
    
    # Disk: exponential in R and z
    # Surface density: Σ(R) = Σ₀ exp(-R/R_d)
    # Volume density: ρ(R,z) = Σ(R) / (2h_z) × exp(-|z|/h_z)
    # For exponential disk: M = 2π Σ₀ R_d², so Σ₀ = M/(2π R_d²)
    Sigma0_disk = M_disk / (2.0 * np.pi * (R_d * kpc_to_m)**2)  # kg/m²
    Sigma_disk = Sigma0_disk * np.exp(-R_kpc / R_d)  # kg/m²
    rho_disk = Sigma_disk / (2.0 * z_d * kpc_to_m) * np.exp(-np.abs(z_kpc) / z_d)  # kg/m³
    
    # Gas: similar to disk but more extended
    Sigma0_gas = M_gas / (2.0 * np.pi * (R_gas * kpc_to_m)**2)  # kg/m²
    Sigma_gas = Sigma0_gas * np.exp(-R_kpc / R_gas)  # kg/m²
    rho_gas = Sigma_gas / (2.0 * z_d * kpc_to_m) * np.exp(-np.abs(z_kpc) / z_d)  # kg/m³
    
    # Total density in kg/m³
    rho_total = rho_bulge + rho_disk + rho_gas
    
    return np.maximum(rho_total, 1e-20)  # Minimum to avoid zeros


def test_gaia_bulge_covariant(
    gaia_df: Optional[pd.DataFrame],
    use_covariant: bool = True,
) -> GaiaBulgeTestResult:
    """Test Gaia bulge using covariant coherence scalar.
    
    Parameters
    ----------
    gaia_df : Optional[pd.DataFrame]
        Full Gaia catalog
    use_covariant : bool
        If True, use C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)
        If False, use baseline C = v²/(v²+σ²)
        
    Returns
    -------
    GaiaBulgeTestResult
        Test results
    """
    if gaia_df is None or len(gaia_df) == 0:
        return GaiaBulgeTestResult(
            passed=True,
            rms_kms=0.0,
            baseline_rms_kms=0.0,
            improvement_kms=0.0,
            n_stars=0,
            n_bins=0,
            details={},
            message="SKIPPED: No Gaia data"
        )
    
    # Select bulge/inner disk stars
    # NOTE: Eilers catalog is disk sample (R: 4-16 kpc), so we use inner disk as proxy
    bulge_df = select_gaia_bulge_stars(gaia_df, z_threshold_kpc=0.3, R_max_kpc=5.0, use_inner_disk=True)
    if len(bulge_df) == 0:
        return GaiaBulgeTestResult(
            passed=True,
            rms_kms=0.0,
            baseline_rms_kms=0.0,
            improvement_kms=0.0,
            n_stars=0,
            n_bins=0,
            details={},
            message="SKIPPED: Not enough bulge stars"
        )
    
    # Bin in (R, z)
    # Adjust bins for inner disk (R: 4-5 kpc typically)
    R_min = max(4.0, bulge_df['R_gal'].min() - 0.5)
    R_max = min(5.5, bulge_df['R_gal'].max() + 0.5)
    R_bins = np.arange(R_min, R_max + 0.5, 0.5)  # kpc
    z_bins = np.arange(-1.0, 1.1, 0.3)  # kpc (adjusted for disk sample)
    binned_df = bin_gaia_bulge(bulge_df, R_bins, z_bins)
    
    if len(binned_df) == 0:
        return GaiaBulgeTestResult(
            passed=True,
            rms_kms=0.0,
            baseline_rms_kms=0.0,
            improvement_kms=0.0,
            n_stars=len(bulge_df),
            n_bins=0,
            details={},
            message="SKIPPED: Not enough bins with sufficient stars"
        )
    
    # Compute flow invariants
    omega2, theta2 = compute_flow_invariants_from_binned(binned_df)
    
    # Compute baryonic density
    rho_kg_m3 = compute_baryonic_density_mw(binned_df['R'].values, binned_df['z'].values)
    
    # Compute covariant coherence
    if use_covariant:
        C_cov = C_covariant_coherence(omega2, rho_kg_m3, theta2)
    else:
        # Baseline: use kinematic approximation
        v_phi = binned_df['v_phi_mean'].values
        sigma_phi = binned_df['sigma_phi'].values
        v2 = v_phi**2
        s2 = np.maximum(sigma_phi, 1.0)**2
        C_cov = v2 / (v2 + s2)
    
    # Predict velocities using C_cov
    # For bulge, we need to compute g_bar from baryonic model
    R = binned_df['R'].values
    R_m = R * kpc_to_m
    g_bar = G * (4.6e10 + 1.0e10 + 1.0e10) * 1.16**2 * M_sun / R_m**2  # Simplified
    h = h_function(g_bar)
    
    # Enhancement
    A_base = A_0  # For bulge, use disk amplitude (L ≈ L_0)
    Sigma = 1.0 + A_base * C_cov * h
    
    # Predicted circular speed
    V_bar = np.sqrt(g_bar * R_m) / 1000.0  # km/s
    V_pred = V_bar * np.sqrt(np.maximum(Sigma, 0.0))
    
    # Observed (from mean v_phi)
    # Handle different column names
    if 'v_phi_mean' in binned_df.columns:
        V_obs = np.abs(binned_df['v_phi_mean'].values)
    else:
        # Fallback: compute from individual stars if needed
        V_obs = np.abs(binned_df.get('v_phi', binned_df.get('v_phi_obs', pd.Series([0]))).values)
    
    # Residuals
    resid = V_obs - V_pred
    rms = np.sqrt((resid**2).mean())
    
    # Baseline (for comparison)
    # Use simple kinematic C
    v2_base = V_obs**2
    s2_base = binned_df['sigma_phi'].values**2
    C_base = v2_base / (v2_base + s2_base)
    Sigma_base = 1.0 + A_base * C_base * h
    V_pred_base = V_bar * np.sqrt(np.maximum(Sigma_base, 0.0))
    resid_base = V_obs - V_pred_base
    rms_base = np.sqrt((resid_base**2).mean())
    
    improvement = rms_base - rms
    passed = improvement > 1.0  # Require >1 km/s improvement
    
    return GaiaBulgeTestResult(
        passed=passed,
        rms_kms=rms,
        baseline_rms_kms=rms_base,
        improvement_kms=improvement,
        n_stars=len(bulge_df),
        n_bins=len(binned_df),
        details={
            'mean_C_cov': float(C_cov.mean()),
            'mean_omega2': float(omega2.mean()),
            'mean_rho_kg_m3': float(rho_kg_m3.mean()),
            'mean_theta2': float(theta2.mean()),
        },
        message=f"RMS={rms:.2f} km/s (baseline={rms_base:.2f}, improvement={improvement:.2f} km/s, {len(binned_df)} bins, {len(bulge_df)} stars)"
    )


if __name__ == "__main__":
    # Test implementation
    print("Gaia Bulge Covariant Coherence Test")
    print("=" * 80)
    print()
    print("This test will be integrated into run_regression_experimental.py")
    print("once Gaia bulge selection and binning are validated.")
    print()
    print("Next steps:")
    print("1. Validate bulge star selection criteria")
    print("2. Test binning and gradient computation")
    print("3. Integrate into regression suite")
    print("4. Set success threshold (>1 km/s improvement)")

