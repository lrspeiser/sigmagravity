#!/usr/bin/env python3
"""
BRAVA 6D Jeans-Field Test: Compare dynamical gravity to predicted gravity

This test uses the Jeans equation to compute the *required* radial gravitational
field from observed velocity dispersions, then compares to Sigma-Gravity predictions.

This is MUCH more sensitive than comparing sigma_tot to V_circ, because it directly
tests whether the kinematics require gravitational enhancement.

The key insight:
- If g_dyn ≈ g_bar: bulge dispersions are explained by baryons alone, no enhancement needed
- If g_dyn > g_bar systematically: bulge requires extra gravity (enhancement or dark matter)
- If g_dyn < g_bar: something is wrong (e.g., model or systematics)

Jeans equation (axisymmetric cylindrical, simplified):
    g_R^dyn = (1/nu) * d(nu * sigma_R^2)/dR + (sigma_R^2 - sigma_phi^2 - v_phi^2)/R

where:
    nu = tracer density (proportional to n_stars / (R * dR * dz))
    sigma_R, sigma_phi = radial and azimuthal velocity dispersions
    v_phi = mean azimuthal streaming velocity

Usage:
    python scripts/test_brava_6d_jeans_field.py
    python scripts/test_brava_6d_jeans_field.py --3d
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Physical constants
G = 6.674e-11  # m^3/kg/s^2
M_sun = 1.989e30  # kg
kpc_to_m = 3.086e19  # m
H0_SI = 2.27e-18  # 1/s
c = 2.998e8  # m/s

# Model parameters
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.60e-11 m/s^2


@dataclass
class JeansFieldResult:
    """Results from Jeans-field bulge test."""
    passed: bool
    rms_g_dyn_vs_g_bar: float  # RMS of (g_dyn - g_bar) in km^2/s^2/kpc
    rms_g_dyn_vs_g_pred: float  # RMS of (g_dyn - g_pred) in km^2/s^2/kpc
    mean_g_ratio_bar: float  # mean(g_dyn / g_bar)
    mean_g_ratio_pred: float  # mean(g_dyn / g_pred)
    improvement: float  # rms_bar - rms_pred (positive = Sigma helps)
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
    Compute baryonic acceleration from MW model (bulge + disk).
    
    Returns g_bar in m/s^2
    """
    M_bulge = 1.0e10 * M_sun  # kg
    M_disk = 4.6e10 * M_sun  # kg
    R_d = 2.5 * kpc_to_m  # m
    
    R_m = R_kpc * kpc_to_m
    M_enc = M_bulge + M_disk * (1.0 - np.exp(-R_m / R_d))
    g_bar = G * M_enc / np.maximum(R_m**2, 1e-9)
    return g_bar  # m/s^2


def C_covariant_coherence(omega2: np.ndarray, rho_kg_m3: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Covariant coherence: C_cov = omega^2/(omega^2 + 4*pi*G*rho + theta^2 + H0^2)"""
    om2 = np.asarray(omega2, dtype=float)
    rho = np.asarray(rho_kg_m3, dtype=float)
    th2 = np.asarray(theta2, dtype=float)
    
    # Convert to consistent units (km/s/kpc)^2
    omega_unit_si = 1000.0 / kpc_to_m  # 1/s
    four_pi_G_rho = 4.0 * np.pi * G * rho  # 1/s^2
    four_pi_G_rho_kms_kpc2 = four_pi_G_rho / (omega_unit_si**2)
    
    H0_kms_per_kpc = H0_SI * (kpc_to_m / 1000.0)
    H0_sq = H0_kms_per_kpc**2
    
    denom = om2 + four_pi_G_rho_kms_kpc2 + th2 + H0_sq
    C = om2 / np.maximum(denom, 1e-30)
    return np.clip(C, 0.0, 1.0)


def compute_jeans_g_dyn(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute dynamical radial acceleration from Jeans equation.
    
    g_R^dyn = (1/nu) * d(nu * sigma_R^2)/dR + (sigma_R^2 - sigma_phi^2 - v_phi^2)/R
    
    Since bins are 2D (R, z), we average over z for each R to get 1D radial profile.
    
    Returns:
        g_dyn: array of dynamical gravity (m/s^2) for each R bin
        df_radial: dataframe with R-averaged quantities
    """
    # Group by R_kpc and compute weighted averages over z
    # Use n_stars as weights
    df = df.copy()
    df['sigma_R_sq'] = (df['vR_std'] * 1000.0)**2  # (m/s)^2
    df['sigma_phi_sq'] = (df['vphi_std'] * 1000.0)**2
    df['v_phi_sq'] = (df['vphi_mean'].abs() * 1000.0)**2
    
    # Round R to avoid floating point grouping issues
    df['R_bin'] = df['R_kpc'].round(3)
    
    # Weighted average over z for each R
    grouped = df.groupby('R_bin').apply(
        lambda g: pd.Series({
            'R_kpc': (g['R_kpc'] * g['n_stars']).sum() / g['n_stars'].sum(),
            'n_stars': g['n_stars'].sum(),
            'sigma_R_sq': (g['sigma_R_sq'] * g['n_stars']).sum() / g['n_stars'].sum(),
            'sigma_phi_sq': (g['sigma_phi_sq'] * g['n_stars']).sum() / g['n_stars'].sum(),
            'v_phi_sq': (g['v_phi_sq'] * g['n_stars']).sum() / g['n_stars'].sum(),
            'omega2': (g['omega2'] * g['n_stars']).sum() / g['n_stars'].sum(),
            'theta2': (g['theta2'] * g['n_stars']).sum() / g['n_stars'].sum(),
            'rho_kg_m3': (g['rho_kg_m3'] * g['n_stars']).sum() / g['n_stars'].sum(),
        }),
        include_groups=False
    ).reset_index(drop=True)
    
    df_radial = grouped.sort_values('R_kpc').reset_index(drop=True)
    print(f"  Collapsed to {len(df_radial)} radial bins (averaged over z)")
    
    if len(df_radial) < 3:
        return np.full(len(df_radial), np.nan), df_radial
    
    R_kpc = df_radial['R_kpc'].values
    R_m = R_kpc * kpc_to_m
    n_stars = df_radial['n_stars'].values.astype(float)
    sigma_R_sq = df_radial['sigma_R_sq'].values
    sigma_phi_sq = df_radial['sigma_phi_sq'].values
    v_phi_sq = df_radial['v_phi_sq'].values
    
    # Tracer density proxy: nu ~ n_stars / R (for cylindrical volume element)
    nu = n_stars / np.maximum(R_m, kpc_to_m * 0.1)
    
    # Compute S = nu * sigma_R^2
    S = nu * sigma_R_sq
    
    # Compute dS/dR using finite differences
    dR = np.diff(R_m)
    dS = np.diff(S)
    dS_dR_centers = dS / np.maximum(dR, 1e-30)
    
    # Interpolate to bin centers
    dS_dR = np.zeros(len(R_m))
    dS_dR[0] = dS_dR_centers[0]  # Forward
    dS_dR[-1] = dS_dR_centers[-1]  # Backward
    for i in range(1, len(dS_dR) - 1):
        dS_dR[i] = 0.5 * (dS_dR_centers[i-1] + dS_dR_centers[i])
    
    # Jeans term 1: (1/nu) * dS/dR
    term1 = dS_dR / np.maximum(nu, 1e-30)
    
    # Jeans term 2: (sigma_R^2 - sigma_phi^2) / R (anisotropy term)
    term2 = (sigma_R_sq - sigma_phi_sq) / np.maximum(R_m, kpc_to_m * 0.1)
    
    # Centrifugal term: v_phi^2 / R (rotation provides centripetal acceleration)
    term3 = v_phi_sq / np.maximum(R_m, kpc_to_m * 0.1)
    
    # Total dynamical gravity: g_R = v_phi^2/R + (1/nu)*d(nu*sigma_R^2)/dR + (sigma_R^2 - sigma_phi^2)/R
    # This is the radial force needed to balance pressure + rotation
    g_dyn = term1 + term2 + term3
    
    # Edge effects
    g_dyn[0] = np.nan
    g_dyn[-1] = np.nan
    
    return g_dyn, df_radial


def test_brava_jeans_field(
    binned_data_path: Optional[Path] = None,
    use_3d_gradients: bool = False,
) -> JeansFieldResult:
    """
    Test BRAVA bulge data using Jeans equation field comparison.
    
    This test computes the dynamical radial gravity from observed dispersions
    and compares to:
    1. Baryonic gravity (g_bar) - baseline
    2. Enhanced gravity (g_pred = g_bar * Sigma) - Sigma-Gravity prediction
    """
    if binned_data_path is None:
        if use_3d_gradients:
            binned_data_path = Path("data/gaia/6d_brava_galcen_3d.parquet")
        else:
            binned_data_path = Path("data/gaia/6d_brava_galcen.parquet")
    
    if not binned_data_path.exists():
        return JeansFieldResult(
            passed=False,
            rms_g_dyn_vs_g_bar=0.0,
            rms_g_dyn_vs_g_pred=0.0,
            mean_g_ratio_bar=0.0,
            mean_g_ratio_pred=0.0,
            improvement=0.0,
            n_bins=0,
            n_stars=0,
            details={},
            message=f"SKIPPED: Binned data not found: {binned_data_path}"
        )
    
    print("=" * 70)
    print("BRAVA 6D JEANS-FIELD TEST")
    print("=" * 70)
    print(f"Using binned data: {binned_data_path}")
    
    df = pd.read_parquet(binned_data_path)
    print(f"Loaded {len(df)} bins")
    
    if len(df) < 3:
        return JeansFieldResult(
            passed=False,
            rms_g_dyn_vs_g_bar=0.0,
            rms_g_dyn_vs_g_pred=0.0,
            mean_g_ratio_bar=0.0,
            mean_g_ratio_pred=0.0,
            improvement=0.0,
            n_bins=len(df),
            n_stars=0,
            details={},
            message="SKIPPED: Not enough bins for gradient computation"
        )
    
    # Compute dynamical gravity from Jeans equation
    print("\nComputing dynamical gravity from Jeans equation...")
    g_dyn, df_radial = compute_jeans_g_dyn(df)
    
    if len(df_radial) < 3:
        return JeansFieldResult(
            passed=False,
            rms_g_dyn_vs_g_bar=0.0,
            rms_g_dyn_vs_g_pred=0.0,
            mean_g_ratio_bar=0.0,
            mean_g_ratio_pred=0.0,
            improvement=0.0,
            n_bins=len(df),
            n_stars=df['n_stars'].sum(),
            details={},
            message="SKIPPED: Not enough radial bins after z-averaging"
        )
    
    # Use radial-averaged data for comparison
    R_kpc = df_radial['R_kpc'].values
    
    # Compute baryonic gravity (fresh, not from stored values)
    g_bar = compute_baryonic_acceleration_mw(R_kpc)
    
    # Recompute Sigma from C_cov (fresh computation for consistency)
    omega2 = df_radial['omega2'].values
    theta2 = df_radial['theta2'].values
    rho_kg_m3 = df_radial['rho_kg_m3'].values
    C_cov = C_covariant_coherence(omega2, rho_kg_m3, theta2)
    h = h_function(g_bar)
    Sigma = 1.0 + A_0 * C_cov * h
    
    # Enhanced gravity
    g_pred = g_bar * Sigma
    
    # Filter valid bins (exclude edge effects)
    valid = np.isfinite(g_dyn) & (g_dyn > 0)  # g_dyn should be positive (inward)
    n_stars_total = df['n_stars'].sum()  # From original df
    if valid.sum() < 2:
        return JeansFieldResult(
            passed=False,
            rms_g_dyn_vs_g_bar=0.0,
            rms_g_dyn_vs_g_pred=0.0,
            mean_g_ratio_bar=0.0,
            mean_g_ratio_pred=0.0,
            improvement=0.0,
            n_bins=len(df),
            n_stars=n_stars_total,
            details={'valid_bins': int(valid.sum())},
            message="SKIPPED: Not enough valid bins after Jeans computation"
        )
    
    g_dyn_valid = g_dyn[valid]
    g_bar_valid = g_bar[valid]
    g_pred_valid = g_pred[valid]
    R_valid = R_kpc[valid]
    Sigma_valid = Sigma[valid]
    C_cov_valid = C_cov[valid]
    
    print(f"Valid bins for Jeans analysis: {valid.sum()}")
    
    # Convert to km^2/s^2/kpc for easier interpretation
    # g in m/s^2, multiply by kpc_to_m/1e6 to get km^2/s^2/kpc
    g_to_kms2_per_kpc = kpc_to_m / 1e6
    g_dyn_kms = g_dyn_valid * g_to_kms2_per_kpc
    g_bar_kms = g_bar_valid * g_to_kms2_per_kpc
    g_pred_kms = g_pred_valid * g_to_kms2_per_kpc
    
    # Compute residuals
    resid_bar = g_dyn_kms - g_bar_kms
    resid_pred = g_dyn_kms - g_pred_kms
    
    rms_bar = np.sqrt((resid_bar**2).mean())
    rms_pred = np.sqrt((resid_pred**2).mean())
    improvement = rms_bar - rms_pred
    
    # Compute ratios
    mean_ratio_bar = (g_dyn_valid / g_bar_valid).mean()
    mean_ratio_pred = (g_dyn_valid / g_pred_valid).mean()
    
    # Diagnostics
    print("\n" + "=" * 70)
    print("JEANS FIELD ANALYSIS")
    print("=" * 70)
    
    print(f"\nGravity comparison (km^2/s^2/kpc):")
    print(f"  g_dyn (from kinematics):  mean={g_dyn_kms.mean():8.2f}, std={g_dyn_kms.std():6.2f}")
    print(f"  g_bar (baryonic):         mean={g_bar_kms.mean():8.2f}, std={g_bar_kms.std():6.2f}")
    print(f"  g_pred (Sigma-enhanced):  mean={g_pred_kms.mean():8.2f}, std={g_pred_kms.std():6.2f}")
    
    print(f"\nGravity ratios:")
    print(f"  g_dyn / g_bar:  mean={mean_ratio_bar:.4f}")
    print(f"  g_dyn / g_pred: mean={mean_ratio_pred:.4f}")
    
    print(f"\nRMS residuals (km^2/s^2/kpc):")
    print(f"  |g_dyn - g_bar|:  {rms_bar:.4f}")
    print(f"  |g_dyn - g_pred|: {rms_pred:.4f}")
    print(f"  Improvement:      {improvement:+.4f} ({'BETTER' if improvement > 0 else 'WORSE' if improvement < 0 else 'SAME'})")
    
    # Convert to equivalent V_circ error for intuition
    # V_circ^2 = R * g, so delta_V_circ ~ delta_g * R / (2 * V_circ)
    V_circ_bar = np.sqrt(g_bar_kms * R_valid)  # km/s (approx)
    V_circ_err_bar = rms_bar * R_valid.mean() / (2 * V_circ_bar.mean())
    V_circ_err_pred = rms_pred * R_valid.mean() / (2 * V_circ_bar.mean())
    
    print(f"\nEquivalent V_circ error (km/s):")
    print(f"  Baseline:     {V_circ_err_bar:.2f}")
    print(f"  Sigma-Grav:   {V_circ_err_pred:.2f}")
    
    print(f"\nSigma enhancement statistics:")
    print(f"  mean(Sigma):   {Sigma_valid.mean():.6f}")
    print(f"  max(Sigma):    {Sigma_valid.max():.6f}")
    print(f"  mean(C_cov):   {C_cov_valid.mean():.4f}")
    
    # Interpret results
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if mean_ratio_bar > 1.1:
        print("\n** g_dyn > g_bar: Bulge kinematics REQUIRE extra gravity **")
        print("   This could be evidence for Sigma enhancement or dark matter.")
        if improvement > 0:
            print(f"   Sigma-Gravity improves fit by {improvement:.4f} km^2/s^2/kpc.")
        else:
            print(f"   But Sigma-Gravity doesn't help ({improvement:+.4f}).")
            print("   May need different enhancement formula for bulges.")
    elif mean_ratio_bar < 0.9:
        print("\n** g_dyn < g_bar: Baryonic model OVERPREDICTS gravity **")
        print("   This suggests systematics in either:")
        print("   - The Jeans computation (tracer gradients, anisotropy)")
        print("   - The baryonic model (M/L, bulge mass)")
    else:
        print("\n** g_dyn ~ g_bar: Baryons explain bulge kinematics **")
        print("   No gravitational enhancement needed for dispersions.")
        print("   SPARC bulge issues are likely M/L or pressure support systematics.")
    
    # Per-bin details
    details = {
        'R_kpc': R_valid.tolist(),
        'g_dyn_kms': g_dyn_kms.tolist(),
        'g_bar_kms': g_bar_kms.tolist(),
        'g_pred_kms': g_pred_kms.tolist(),
        'Sigma': Sigma_valid.tolist(),
        'C_cov': C_cov_valid.tolist(),
        'mean_ratio_bar': mean_ratio_bar,
        'mean_ratio_pred': mean_ratio_pred,
        'V_circ_err_bar': V_circ_err_bar,
        'V_circ_err_pred': V_circ_err_pred,
        'valid_bins': int(valid.sum()),
    }
    
    # Pass if improvement > 0 or if no enhancement needed (ratio ~ 1)
    passed = (improvement > 0) or (0.9 < mean_ratio_bar < 1.1)
    
    result = JeansFieldResult(
        passed=passed,
        rms_g_dyn_vs_g_bar=rms_bar,
        rms_g_dyn_vs_g_pred=rms_pred,
        mean_g_ratio_bar=mean_ratio_bar,
        mean_g_ratio_pred=mean_ratio_pred,
        improvement=improvement,
        n_bins=len(df_radial),
        n_stars=n_stars_total,
        details=details,
        message=f"g_dyn/g_bar ratio = {mean_ratio_bar:.4f}, improvement = {improvement:+.4f}"
    )
    
    print("\n" + "=" * 70)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(result.message)
    print("=" * 70)
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BRAVA Jeans-field test")
    parser.add_argument('--3d', dest='use_3d', action='store_true',
                        help="Use 3D gradient output")
    parser.add_argument('--input', type=Path, default=None,
                        help="Path to binned parquet file")
    args = parser.parse_args()
    
    result = test_brava_jeans_field(
        binned_data_path=args.input,
        use_3d_gradients=args.use_3d,
    )
    
    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())

