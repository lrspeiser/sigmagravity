#!/usr/bin/env python3
"""
Test BRAVA bulge data against galpy's calibrated MW potential models.

galpy provides well-calibrated Milky Way potentials from the literature:
- MWPotential2014 (Bovy 2015): Bulge + disk + halo, fitted to observational constraints
- McMillan17 (McMillan 2017): Updated with Gaia DR1 constraints

This test compares observed velocity dispersions to predictions from these
established potentials, providing a proper baseline for Sigma-Gravity.

Usage:
    python scripts/test_brava_galpy_potential.py
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# galpy imports
from galpy.potential import MWPotential2014, vcirc, evaluatezforces, evaluateRforces
from galpy.util import conversion

# Physical constants
G = 6.674e-11  # m^3/kg/s^2
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19  # m
H0_SI = 2.27e-18  # 1/s
c = 2.998e8  # m/s

# Model parameters for Sigma-Gravity
A_0 = np.exp(1 / (2 * np.pi))  # ~ 1.173
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ~ 9.60e-11 m/s^2

# galpy natural units: R0 = 8 kpc, V0 = 220 km/s
R0 = 8.0  # kpc
V0 = 220.0  # km/s


@dataclass
class GalpyTestResult:
    """Results from galpy potential comparison."""
    rms_vcirc_obs_vs_galpy: float  # km/s
    rms_sigma_obs_vs_pred_galpy: float  # km/s
    rms_sigma_obs_vs_pred_sigma_grav: float  # km/s
    improvement_over_galpy: float  # positive = Sigma-Gravity helps
    mean_sigma_enhancement: float
    n_bins: int
    n_stars: int
    details: dict
    message: str


def h_function(g: np.ndarray) -> np.ndarray:
    """h(g) = sqrt(g_dagger/g) * g_dagger / (g_dagger + g)"""
    g_safe = np.maximum(g, 1e-30)
    return np.sqrt(g_dagger / g_safe) * g_dagger / (g_dagger + g_safe)


def C_covariant_coherence(omega2: np.ndarray, rho_kg_m3: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Covariant coherence: C_cov = omega^2/(omega^2 + 4*pi*G*rho + theta^2 + H0^2)"""
    om2 = np.asarray(omega2, dtype=float)
    rho = np.asarray(rho_kg_m3, dtype=float)
    th2 = np.asarray(theta2, dtype=float)
    
    omega_unit_si = 1000.0 / kpc_to_m
    four_pi_G_rho = 4.0 * np.pi * G * rho
    four_pi_G_rho_kms_kpc2 = four_pi_G_rho / (omega_unit_si**2)
    
    H0_kms_per_kpc = H0_SI * (kpc_to_m / 1000.0)
    H0_sq = H0_kms_per_kpc**2
    
    denom = om2 + four_pi_G_rho_kms_kpc2 + th2 + H0_sq
    C = om2 / np.maximum(denom, 1e-30)
    return np.clip(C, 0.0, 1.0)


def get_galpy_vcirc(R_kpc: np.ndarray) -> np.ndarray:
    """Get circular velocity from galpy MWPotential2014."""
    # Convert to galpy natural units (R/R0)
    R_galpy = R_kpc / R0
    
    # Get circular velocity in natural units, then convert to km/s
    v_circ = np.array([vcirc(MWPotential2014, r) * V0 for r in R_galpy])
    
    return v_circ  # km/s


def get_galpy_radial_force(R_kpc: np.ndarray, z_kpc: np.ndarray) -> np.ndarray:
    """Get radial gravitational acceleration from galpy MWPotential2014."""
    R_galpy = R_kpc / R0
    z_galpy = z_kpc / R0
    
    # evaluateRforces returns -dPhi/dR in natural units (V0^2/R0)
    # Convert to m/s^2
    force_natural = np.array([
        evaluateRforces(MWPotential2014, r, z) 
        for r, z in zip(R_galpy, z_galpy)
    ])
    
    # Convert: natural units -> km/s^2/kpc -> m/s^2
    # 1 natural unit = V0^2/R0 = (220 km/s)^2 / (8 kpc)
    # = (220e3 m/s)^2 / (8 * 3.086e19 m) = 1.96e-10 m/s^2
    force_ms2 = force_natural * (V0 * 1000.0)**2 / (R0 * kpc_to_m)
    
    return -force_ms2  # Return positive for inward force


def test_brava_with_galpy(
    binned_data_path: Optional[Path] = None,
) -> GalpyTestResult:
    """
    Test BRAVA bulge data against galpy's MWPotential2014.
    
    This compares:
    1. Observed V_circ (from v_phi) vs galpy prediction
    2. Observed sigma_tot vs prediction from galpy V_circ
    3. Observed sigma_tot vs Sigma-Gravity enhanced prediction
    """
    if binned_data_path is None:
        binned_data_path = Path("data/gaia/6d_brava_galcen.parquet")
    
    if not binned_data_path.exists():
        return GalpyTestResult(
            rms_vcirc_obs_vs_galpy=0.0,
            rms_sigma_obs_vs_pred_galpy=0.0,
            rms_sigma_obs_vs_pred_sigma_grav=0.0,
            improvement_over_galpy=0.0,
            mean_sigma_enhancement=1.0,
            n_bins=0,
            n_stars=0,
            details={},
            message=f"SKIPPED: Binned data not found: {binned_data_path}"
        )
    
    print("=" * 70)
    print("BRAVA BULGE TEST WITH GALPY MWPotential2014")
    print("=" * 70)
    print(f"\nUsing galpy's MWPotential2014 (Bovy 2015)")
    print(f"  R0 = {R0} kpc, V0 = {V0} km/s")
    
    df = pd.read_parquet(binned_data_path)
    print(f"\nLoaded {len(df)} bins from {binned_data_path}")
    
    # Get observed data
    R_kpc = df['R_kpc'].values
    z_kpc = df['z_kpc'].values
    v_phi_obs = np.abs(df['vphi_mean'].values)  # km/s
    
    sigma_R_obs = df['vR_std'].values
    sigma_phi_obs = df['vphi_std'].values
    sigma_z_obs = df['vz_std'].values
    sigma_tot_obs = np.sqrt(sigma_R_obs**2 + sigma_phi_obs**2 + sigma_z_obs**2)
    
    n_stars_total = df['n_stars'].sum()
    
    # Get galpy predictions
    print("\nComputing galpy predictions...")
    v_circ_galpy = get_galpy_vcirc(R_kpc)
    g_bar_galpy = get_galpy_radial_force(R_kpc, z_kpc)
    
    # Sigma-Gravity enhancement
    omega2 = df['omega2'].values
    theta2 = df['theta2'].values
    rho_kg_m3 = df['rho_kg_m3'].values
    C_cov = C_covariant_coherence(omega2, rho_kg_m3, theta2)
    h = h_function(g_bar_galpy)
    Sigma = 1.0 + A_0 * C_cov * h
    
    # Predictions for sigma_tot
    # Calibration factor derived from data: sigma_tot / V_circ
    calib_factor = (sigma_tot_obs / v_circ_galpy).mean()
    print(f"  Calibration factor (sigma_tot / V_circ): {calib_factor:.3f}")
    
    # galpy baseline prediction
    sigma_pred_galpy = v_circ_galpy * calib_factor
    
    # Sigma-Gravity prediction
    v_circ_sigma = v_circ_galpy * np.sqrt(Sigma)
    sigma_pred_sigma_grav = v_circ_sigma * calib_factor
    
    # Compute residuals
    resid_v = v_phi_obs - v_circ_galpy
    resid_sigma_galpy = sigma_tot_obs - sigma_pred_galpy
    resid_sigma_sigma_grav = sigma_tot_obs - sigma_pred_sigma_grav
    
    rms_v = np.sqrt((resid_v**2).mean())
    rms_sigma_galpy = np.sqrt((resid_sigma_galpy**2).mean())
    rms_sigma_sigma_grav = np.sqrt((resid_sigma_sigma_grav**2).mean())
    
    improvement = rms_sigma_galpy - rms_sigma_sigma_grav
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nCircular velocity comparison:")
    print(f"  V_circ (galpy):    mean={v_circ_galpy.mean():.1f} km/s, range=[{v_circ_galpy.min():.1f}, {v_circ_galpy.max():.1f}]")
    print(f"  v_phi (observed):  mean={v_phi_obs.mean():.1f} km/s, range=[{v_phi_obs.min():.1f}, {v_phi_obs.max():.1f}]")
    print(f"  RMS(v_phi - V_circ): {rms_v:.2f} km/s")
    
    print(f"\nVelocity dispersion comparison:")
    print(f"  sigma_tot (observed): mean={sigma_tot_obs.mean():.1f} km/s")
    print(f"  sigma_pred (galpy):   mean={sigma_pred_galpy.mean():.1f} km/s")
    print(f"  sigma_pred (Sigma-G): mean={sigma_pred_sigma_grav.mean():.1f} km/s")
    print(f"  RMS (galpy baseline): {rms_sigma_galpy:.2f} km/s")
    print(f"  RMS (Sigma-Gravity):  {rms_sigma_sigma_grav:.2f} km/s")
    print(f"  Improvement:          {improvement:+.2f} km/s {'(BETTER)' if improvement > 0 else '(WORSE)' if improvement < 0 else ''}")
    
    print(f"\nSigma enhancement:")
    print(f"  mean(Sigma): {Sigma.mean():.6f}")
    print(f"  max(Sigma):  {Sigma.max():.6f}")
    print(f"  mean(C_cov): {C_cov.mean():.4f}")
    
    # Radial profile
    print(f"\nRadial profile (R_kpc | V_circ_galpy | v_phi_obs | sigma_obs | sigma_pred_galpy | Sigma):")
    for i in range(min(len(df), 10)):
        print(f"  {R_kpc[i]:.2f} | {v_circ_galpy[i]:.1f} | {v_phi_obs[i]:.1f} | {sigma_tot_obs[i]:.1f} | {sigma_pred_galpy[i]:.1f} | {Sigma[i]:.4f}")
    if len(df) > 10:
        print(f"  ... ({len(df) - 10} more bins)")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    v_ratio = v_phi_obs.mean() / v_circ_galpy.mean()
    print(f"\n  v_phi_obs / V_circ_galpy = {v_ratio:.3f}")
    
    if v_ratio < 0.8:
        print("  -> Observed rotation is SLOWER than circular (pressure support?)")
    elif v_ratio > 1.1:
        print("  -> Observed rotation is FASTER than circular (unlikely, check data)")
    else:
        print("  -> Observed rotation is consistent with circular velocity")
    
    if improvement > 1.0:
        print(f"\n  Sigma-Gravity improves predictions by {improvement:.1f} km/s")
    elif improvement < -1.0:
        print(f"\n  Sigma-Gravity makes predictions worse by {-improvement:.1f} km/s")
        print("  This is expected for bulges where enhancement is small and not needed")
    else:
        print(f"\n  Sigma-Gravity has minimal effect ({improvement:+.2f} km/s)")
        print("  This is expected: C_cov correctly suppresses enhancement in dense regions")
    
    details = {
        'R_kpc': R_kpc.tolist(),
        'v_circ_galpy': v_circ_galpy.tolist(),
        'v_phi_obs': v_phi_obs.tolist(),
        'sigma_tot_obs': sigma_tot_obs.tolist(),
        'sigma_pred_galpy': sigma_pred_galpy.tolist(),
        'sigma_pred_sigma_grav': sigma_pred_sigma_grav.tolist(),
        'Sigma': Sigma.tolist(),
        'C_cov': C_cov.tolist(),
        'g_bar_galpy': g_bar_galpy.tolist(),
        'calibration_factor': calib_factor,
    }
    
    result = GalpyTestResult(
        rms_vcirc_obs_vs_galpy=rms_v,
        rms_sigma_obs_vs_pred_galpy=rms_sigma_galpy,
        rms_sigma_obs_vs_pred_sigma_grav=rms_sigma_sigma_grav,
        improvement_over_galpy=improvement,
        mean_sigma_enhancement=Sigma.mean(),
        n_bins=len(df),
        n_stars=n_stars_total,
        details=details,
        message=f"RMS(galpy)={rms_sigma_galpy:.2f}, RMS(Sigma-G)={rms_sigma_sigma_grav:.2f}, improvement={improvement:+.2f} km/s"
    )
    
    print("\n" + "=" * 70)
    print(result.message)
    print("=" * 70)
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BRAVA test with galpy potential")
    parser.add_argument('--input', type=Path, default=None,
                        help="Path to binned parquet file")
    args = parser.parse_args()
    
    result = test_brava_with_galpy(binned_data_path=args.input)
    
    return 0


if __name__ == "__main__":
    exit(main())

