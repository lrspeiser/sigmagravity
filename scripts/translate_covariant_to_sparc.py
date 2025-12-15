#!/usr/bin/env python3
"""
Translate Gaia-Calibrated Covariant Coherence to SPARC

This module implements the translation of the covariant coherence scalar
C_cov = ω²/(ω² + 4πGρ + θ² + H₀²) to SPARC galaxies using proxies.

Strategy:
1. From Gaia inner disk, we learned that C_cov improves predictions
2. For SPARC, we approximate:
   - ω² ≈ (V/R)² (vorticity from rotation curve)
   - 4πGρ ≈ ρ_proxy (density proxy from baryonic model)
   - θ² ≈ 0 (steady-state assumption)
3. Use calibrated parameters from Gaia to improve SPARC bulge predictions

Key insight: SPARC bulges need flow information most but have least access to it.
By calibrating on Gaia (where we have 6D flow), we can translate back to SPARC.
"""

import numpy as np
from typing import Optional, Tuple
from scripts.run_regression_experimental import (
    C_covariant_coherence, G, M_sun, kpc_to_m, H0_SI, A_0, L_0, N_EXP
)


def compute_omega2_from_rotation_curve(
    R_kpc: np.ndarray,
    V_kms: np.ndarray,
) -> np.ndarray:
    """Compute ω² proxy from rotation curve.
    
    For axisymmetric rotation: ω ≈ V/R
    ω² in (km/s/kpc)^2
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radius in kpc
    V_kms : np.ndarray
        Rotation velocity in km/s
        
    Returns
    -------
    np.ndarray
        ω² in (km/s/kpc)^2
    """
    R_safe = np.maximum(R_kpc, 0.1)  # Avoid division by zero
    omega = V_kms / R_safe  # km/s/kpc
    omega2 = omega**2
    return omega2


def compute_density_proxy_sparc(
    R_kpc: np.ndarray,
    V_bar_kms: np.ndarray,
    R_d_kpc: float,
    f_bulge: float,
) -> np.ndarray:
    """Compute density proxy for SPARC galaxies.
    
    Approximates 4πGρ from baryonic rotation curve and galaxy structure.
    
    For exponential disk + bulge:
    - Surface density: Σ(R) = Σ₀ exp(-R/R_d)
    - Volume density: ρ(R) ≈ Σ(R) / (2h_z)
    - 4πGρ ≈ 4πG × Σ(R) / (2h_z)
    
    Alternative: Use enclosed mass approximation
    - M_enc(R) ≈ V_bar² R / G
    - ρ(R) ≈ dM/dV ≈ (1/(4πR²)) × dM_enc/dR
    - 4πGρ ≈ (2/R²) × (V_bar²/R) × (dlnV_bar/dlnR + 1)
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radius in kpc
    V_bar_kms : np.ndarray
        Baryonic rotation velocity in km/s
    R_d_kpc : float
        Disk scale length in kpc
    f_bulge : float
        Bulge fraction
        
    Returns
    -------
    np.ndarray
        Density proxy in (km/s/kpc)^2 (same units as ω²)
    """
    R_m = R_kpc * kpc_to_m
    R_safe = np.maximum(R_kpc, 0.1)
    
    # Method 1: Surface density proxy (simpler and more stable)
    # For exponential disk: Σ(R) = Σ₀ exp(-R/R_d)
    # Volume density: ρ(R) ≈ Σ(R) / (2h_z) where h_z ≈ 0.3 kpc
    # 4πGρ ≈ 4πG × Σ(R) / (2h_z)
    
    # Estimate total mass from V_bar at outer radius
    R_max = R_kpc.max()
    V_max = V_bar_kms.max()
    M_total_approx = V_max**2 * (R_max * kpc_to_m) / G  # kg
    
    # Central surface density: M ≈ 2π Σ₀ R_d²
    Sigma0 = M_total_approx / (2.0 * np.pi * (R_d_kpc * kpc_to_m)**2)  # kg/m²
    Sigma = Sigma0 * np.exp(-R_kpc / R_d_kpc)  # kg/m²
    
    # Volume density
    h_z = 0.3  # kpc (typical scale height)
    rho = Sigma / (2.0 * h_z * kpc_to_m)  # kg/m³
    
    # 4πGρ in 1/s²
    four_pi_G_rho_si = 4.0 * np.pi * G * rho  # 1/s²
    
    # Convert to (km/s/kpc)^2
    omega_unit_si = 1000.0 / kpc_to_m  # (km/s/kpc) as 1/s
    four_pi_G_rho_proxy_kms_kpc2 = four_pi_G_rho_si / (omega_unit_si**2)
    
    return np.maximum(four_pi_G_rho_proxy_kms_kpc2, 0.0)
    
    # Method 2: Surface density proxy (alternative)
    # For comparison, also compute from exponential disk
    h_z = 0.3  # kpc (typical scale height)
    # Surface density: Σ(R) = Σ₀ exp(-R/R_d)
    # Central surface density from total mass (approximate)
    # M_total ≈ 2π Σ₀ R_d², so Σ₀ ≈ M_total / (2π R_d²)
    # For now, use V_bar to estimate M_total
    M_enc_approx = V_bar2 * R_m / G  # kg (enclosed mass)
    M_total_approx = M_enc_approx[-1] if len(M_enc_approx) > 0 else 1e10 * M_sun
    Sigma0 = M_total_approx / (2.0 * np.pi * (R_d_kpc * kpc_to_m)**2)  # kg/m²
    Sigma = Sigma0 * np.exp(-R_kpc / R_d_kpc)  # kg/m²
    rho_from_sigma = Sigma / (2.0 * h_z * kpc_to_m)  # kg/m³
    four_pi_G_rho_from_sigma = 4.0 * np.pi * G * rho_from_sigma  # 1/s²
    four_pi_G_rho_from_sigma_kms_kpc2 = four_pi_G_rho_from_sigma / (omega_unit_si**2)
    
    # Use the gradient-based method (more physically motivated)
    # But apply a calibration factor from Gaia
    # Gaia showed that density term is important, so we use the gradient method
    return np.maximum(four_pi_G_rho_proxy_kms_kpc2, 0.0)


def compute_C_cov_proxy_sparc(
    R_kpc: np.ndarray,
    V_kms: np.ndarray,
    V_bar_kms: np.ndarray,
    R_d_kpc: float,
    f_bulge: float,
) -> np.ndarray:
    """Compute C_cov proxy for SPARC galaxies.
    
    Uses rotation curve proxies to approximate the covariant coherence:
    C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radius in kpc
    V_kms : np.ndarray
        Total rotation velocity (observed or predicted) in km/s
    V_bar_kms : np.ndarray
        Baryonic rotation velocity in km/s
    R_d_kpc : float
        Disk scale length in kpc
    f_bulge : float
        Bulge fraction
        
    Returns
    -------
    np.ndarray
        C_cov proxy in [0,1]
    """
    # Compute ω² from rotation curve
    omega2 = compute_omega2_from_rotation_curve(R_kpc, V_kms)
    
    # Compute density proxy
    rho_proxy_kms_kpc2 = compute_density_proxy_sparc(R_kpc, V_bar_kms, R_d_kpc, f_bulge)
    
    # Convert density proxy to "rho_kg_m3" format for C_covariant_coherence
    # This is a bit of a hack - we're using the proxy directly
    # The function expects rho_kg_m3, but we'll pass the proxy in the right units
    # Actually, let's compute C_cov directly here to avoid unit confusion
    
    # θ² ≈ 0 (steady-state assumption)
    theta2 = np.zeros_like(omega2)
    
    # H₀² in (km/s/kpc)^2
    H0_kms_per_kpc = H0_SI * (kpc_to_m / 1000.0)
    H0_sq = H0_kms_per_kpc**2
    
    # C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)
    # We have rho_proxy_kms_kpc2 which is already 4πGρ in (km/s/kpc)^2
    denom = omega2 + rho_proxy_kms_kpc2 + theta2 + H0_sq
    denom = np.maximum(denom, 1e-30)
    C_cov = omega2 / denom
    C_cov = np.nan_to_num(C_cov, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(C_cov, 0.0, 1.0)


def apply_covariant_coherence_to_sparc(
    R_kpc: np.ndarray,
    V_bar_kms: np.ndarray,
    R_d_kpc: float,
    f_bulge: float,
    V_pred_initial: Optional[np.ndarray] = None,
    n_iter: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply covariant coherence to SPARC galaxy prediction.
    
    Uses fixed-point iteration to compute V_pred with C_cov.
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radius in kpc
    V_bar_kms : np.ndarray
        Baryonic rotation velocity in km/s
    R_d_kpc : float
        Disk scale length in kpc
    f_bulge : float
        Bulge fraction
    V_pred_initial : Optional[np.ndarray]
        Initial velocity prediction (default: V_bar)
    n_iter : int
        Number of fixed-point iterations
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (V_pred, C_cov) - predicted velocity and coherence scalar
    """
    from scripts.run_regression_experimental import h_function, A_0, g_dagger
    
    if V_pred_initial is None:
        V_pred = V_bar_kms.copy()
    else:
        V_pred = V_pred_initial.copy()
    
    # Fixed-point iteration
    for i in range(n_iter):
        # Compute C_cov using current V_pred
        C_cov = compute_C_cov_proxy_sparc(R_kpc, V_pred, V_bar_kms, R_d_kpc, f_bulge)
        
        # Compute g_bar
        R_m = R_kpc * kpc_to_m
        M_enc = V_bar_kms**2 * R_m / G  # Approximate enclosed mass
        g_bar = G * M_enc / R_m**2  # m/s²
        g_bar = np.maximum(g_bar, 1e-12)
        
        # Compute h(g_bar)
        h = h_function(g_bar)
        
        # Compute enhancement
        # Use A(L) with L ≈ R (for disk galaxies)
        L = R_kpc
        A_L = A_0 * (L / L_0)**N_EXP
        Sigma = 1.0 + A_L * C_cov * h
        
        # Update V_pred
        V_pred_new = V_bar_kms * np.sqrt(np.maximum(Sigma, 0.0))
        
        # Check convergence
        if np.allclose(V_pred, V_pred_new, rtol=1e-4):
            break
        V_pred = V_pred_new
    
    # Final C_cov
    C_cov_final = compute_C_cov_proxy_sparc(R_kpc, V_pred, V_bar_kms, R_d_kpc, f_bulge)
    
    return V_pred, C_cov_final


if __name__ == "__main__":
    print("Covariant Coherence Translation to SPARC")
    print("=" * 80)
    print()
    print("This module provides functions to translate Gaia-calibrated")
    print("covariant coherence C_cov to SPARC galaxies using rotation curve proxies.")
    print()
    print("Key functions:")
    print("  - compute_C_cov_proxy_sparc(): Compute C_cov from rotation curve")
    print("  - apply_covariant_coherence_to_sparc(): Full prediction with C_cov")
    print()
    print("Next: Integrate into run_regression_experimental.py")

