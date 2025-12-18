#!/usr/bin/env python3
"""
Test BRAVA 6D Bulge Data with Covariant Coherence

This test uses the processed BRAVA 6D Gaia data (binned in R,z) to validate
the covariant coherence model on true bulge stars.

Usage:
    python scripts/test_brava_6d_covariant.py
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

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


@dataclass
class BRAVA6DTestResult:
    """Results from BRAVA 6D bulge test."""
    passed: bool
    rms_kms: float
    baseline_rms_kms: float
    improvement_kms: float
    n_bins: int
    n_stars: int
    details: dict
    message: str


def h_function(g: np.ndarray) -> np.ndarray:
    """h(g) = sqrt(g_dagger/g) * g_dagger / (g_dagger + g)"""
    g_safe = np.maximum(g, 1e-30)
    return np.sqrt(g_dagger / g_safe) * g_dagger / (g_dagger + g_safe)


def compute_baryonic_acceleration_mw(R_kpc: np.ndarray) -> np.ndarray:
    """
    Compute baryonic acceleration from simplified MW model.
    
    Uses enclosed mass approximation:
    M_enclosed ≈ M_bulge + M_disk * (1 - exp(-R/R_d))
    
    Returns g_bar in m/s²
    """
    M_bulge = 1.0e10 * M_sun  # kg
    M_disk = 4.6e10 * M_sun  # kg
    R_d = 2.5 * kpc_to_m  # m
    
    R_m = R_kpc * kpc_to_m
    
    # Enclosed mass
    M_enc = M_bulge + M_disk * (1.0 - np.exp(-R_m / R_d))
    
    # Acceleration
    g_bar = G * M_enc / np.maximum(R_m**2, 1e-9)
    
    return g_bar  # m/s²


def test_brava_6d_covariant(
    binned_data_path: Optional[Path] = None,
) -> BRAVA6DTestResult:
    """
    Test BRAVA 6D bulge data using covariant coherence.
    
    Parameters
    ----------
    binned_data_path : Optional[Path]
        Path to binned BRAVA data (parquet file).
        Default: data/gaia/6d_brava_galcen.parquet
        
    Returns
    -------
    BRAVA6DTestResult
        Test results
    """
    if binned_data_path is None:
        binned_data_path = Path("data/gaia/6d_brava_galcen.parquet")
    
    if not binned_data_path.exists():
        return BRAVA6DTestResult(
            passed=False,
            rms_kms=0.0,
            baseline_rms_kms=0.0,
            improvement_kms=0.0,
            n_bins=0,
            n_stars=0,
            details={},
            message=f"SKIPPED: Binned data not found: {binned_data_path}"
        )
    
    # Load binned data
    print(f"Loading binned BRAVA data: {binned_data_path}")
    binned_df = pd.read_parquet(binned_data_path)
    
    if len(binned_df) == 0:
        return BRAVA6DTestResult(
            passed=False,
            rms_kms=0.0,
            baseline_rms_kms=0.0,
            improvement_kms=0.0,
            n_bins=0,
            n_stars=0,
            details={},
            message="SKIPPED: No binned data"
        )
    
    print(f"  Loaded {len(binned_df)} bins")
    
    # Get flow invariants (already computed)
    omega2 = binned_df['omega2'].values
    theta2 = binned_df['theta2'].values
    rho_kg_m3 = binned_df['rho_kg_m3'].values
    C_cov = binned_df['C_cov'].values
    Sigma = binned_df['Sigma'].values
    
    # Compute baryonic acceleration
    R_kpc = binned_df['R_kpc'].values
    g_bar = compute_baryonic_acceleration_mw(R_kpc)
    h = h_function(g_bar)
    
    # Predicted circular speed with enhancement
    R_m = R_kpc * kpc_to_m
    V_bar = np.sqrt(g_bar * R_m) / 1000.0  # km/s
    V_pred = V_bar * np.sqrt(np.maximum(Sigma, 0.0))
    
    # Observed mean streaming velocity (v_phi)
    V_obs = np.abs(binned_df['vphi_mean'].values)
    
    # Residuals
    resid = V_obs - V_pred
    rms = np.sqrt((resid**2).mean())
    
    # Baseline: no enhancement (Sigma = 1)
    V_pred_base = V_bar
    resid_base = V_obs - V_pred_base
    rms_base = np.sqrt((resid_base**2).mean())
    
    improvement = rms_base - rms
    
    # Test passes if improvement > 0 (or set threshold)
    passed = improvement > 0.0
    
    # Total stars (sum of n_stars in bins)
    n_stars_total = binned_df['n_stars'].sum()
    
    return BRAVA6DTestResult(
        passed=passed,
        rms_kms=rms,
        baseline_rms_kms=rms_base,
        improvement_kms=improvement,
        n_bins=len(binned_df),
        n_stars=n_stars_total,
        details={
            'mean_C_cov': float(C_cov.mean()),
            'mean_omega2': float(omega2.mean()),
            'mean_rho_kg_m3': float(rho_kg_m3.mean()),
            'mean_theta2': float(theta2.mean()),
            'mean_Sigma': float(Sigma.mean()),
            'R_range_kpc': [float(R_kpc.min()), float(R_kpc.max())],
            'z_range_kpc': [float(binned_df['z_kpc'].min()), float(binned_df['z_kpc'].max())],
        },
        message=f"RMS={rms:.2f} km/s (baseline={rms_base:.2f}, improvement={improvement:.2f} km/s, {len(binned_df)} bins, {n_stars_total} stars)"
    )


if __name__ == "__main__":
    print("="*70)
    print("BRAVA 6D BULGE COVARIANT COHERENCE TEST")
    print("="*70)
    
    result = test_brava_6d_covariant()
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Status: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  {result.message}")
    print(f"\n  Details:")
    for key, value in result.details.items():
        if isinstance(value, list):
            print(f"    {key}: [{value[0]:.3f}, {value[1]:.3f}]")
        else:
            print(f"    {key}: {value:.6e}" if value < 0.01 else f"    {key}: {value:.3f}")
    print("="*70)

