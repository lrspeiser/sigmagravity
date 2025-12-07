#!/usr/bin/env python3
"""
Wake-Based Coherence Model for Σ-Gravity
=========================================

A toy model that introduces a "wake coherence" order parameter to capture
how kinematic alignment of stellar populations affects gravitational enhancement.

Key Concept:
-----------
Each star generates a "wake" whose strength depends on its mass and velocity.
When wakes are aligned (cold rotating disk), coherence is high (C_wake → 1).
When wakes point in many directions (bulge, counter-rotation), coherence drops.

This naturally explains:
- Why bulge-dominated inner regions show less enhancement
- Why counter-rotating components reduce "dark matter" fractions
- Why dispersion-dominated systems behave differently from cold disks

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s (H0 = 70 km/s/Mpc)
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration (Σ-Gravity)
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.6×10⁻¹¹ m/s²

# Default parameters
A_GALAXY = np.exp(1 / (2 * np.pi))  # ≈ 1.173
XI_SCALE = 1 / (2 * np.pi)  # ξ = R_d/(2π)


# =============================================================================
# WAKE COHERENCE MODEL
# =============================================================================

@dataclass
class WakeParams:
    """Parameters for wake coherence model."""
    alpha: float = 1.5      # Velocity weighting exponent (1-2)
    v0: float = 200.0       # Reference velocity (km/s)
    beta: float = 1.0       # Decoherence sharpness exponent
    sigma_bulge: float = 150.0  # Typical bulge dispersion (km/s)


def C_wake_discrete(
    R: np.ndarray,
    Sigma_d: np.ndarray,
    Sigma_b: np.ndarray,
    v_c: np.ndarray,
    sigma_b: float,
    params: WakeParams = WakeParams()
) -> np.ndarray:
    """
    Compute wake coherence order parameter C_wake(R).
    
    For a disk + bulge system:
    - Disk: ordered rotation → wakes aligned in φ direction
    - Bulge: isotropic dispersion → wakes point randomly → net J ≈ 0
    
    Parameters
    ----------
    R : array
        Radii in kpc
    Sigma_d : array
        Disk surface density at each R (M☉/pc² or relative)
    Sigma_b : array
        Bulge surface density at each R (M☉/pc² or relative)
    v_c : array
        Circular velocity at each R (km/s)
    sigma_b : float
        Bulge velocity dispersion (km/s)
    params : WakeParams
        Model parameters
        
    Returns
    -------
    C_wake : array
        Wake coherence order parameter (0 to 1)
    """
    alpha = params.alpha
    v0 = params.v0
    
    # Ensure arrays
    R = np.atleast_1d(R)
    Sigma_d = np.atleast_1d(Sigma_d)
    Sigma_b = np.atleast_1d(Sigma_b)
    v_c = np.atleast_1d(v_c)
    
    # Disk contribution: aligned wakes
    # J_d = Σ_d × (v_c/v0)^α × φ̂  (all in same direction)
    # N_d = Σ_d × (v_c/v0)^α
    v_c_safe = np.maximum(v_c, 1.0)
    N_d = Sigma_d * (v_c_safe / v0) ** alpha
    J_d = N_d  # Magnitude (all aligned)
    
    # Bulge contribution: random wakes
    # J_b ≈ 0 (vectors cancel)
    # N_b = Σ_b × (σ_b/v0)^α
    sigma_b_safe = max(sigma_b, 10.0)
    N_b = Sigma_b * (sigma_b_safe / v0) ** alpha
    J_b = 0  # Random directions cancel
    
    # Total
    J_total = J_d + J_b  # Only disk contributes to net vector
    N_total = N_d + N_b
    
    # Wake coherence: |J|/N
    N_total_safe = np.maximum(N_total, 1e-10)
    C_wake = J_d / N_total_safe  # J_d is the magnitude of aligned component
    
    # Clip to [0, 1]
    C_wake = np.clip(C_wake, 0.0, 1.0)
    
    return C_wake


def C_wake_continuum(
    R: np.ndarray,
    v_rot: np.ndarray,
    sigma_v: np.ndarray,
    params: WakeParams = WakeParams()
) -> np.ndarray:
    """
    Continuum version using v_rot/σ ratio.
    
    For a distribution with mean rotation v_rot and dispersion σ_v:
    C_wake ≈ v_rot² / (v_rot² + σ_v²)
    
    This is the same structure as the coherence scalar in the main theory.
    
    Parameters
    ----------
    R : array
        Radii (for interface consistency)
    v_rot : array
        Mean rotation velocity (km/s)
    sigma_v : array
        Velocity dispersion (km/s)
    params : WakeParams
        Model parameters
        
    Returns
    -------
    C_wake : array
        Wake coherence order parameter
    """
    v_rot = np.atleast_1d(v_rot)
    sigma_v = np.atleast_1d(sigma_v)
    
    v_rot_sq = np.maximum(v_rot ** 2, 1.0)
    sigma_sq = np.maximum(sigma_v ** 2, 1.0)
    
    # Standard order parameter form
    C_wake = v_rot_sq / (v_rot_sq + sigma_sq)
    
    return C_wake


def D_wake(C_wake: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Decoherence factor from wake coherence.
    
    D_wake = C_wake^β
    
    Higher β makes decoherence kick in sharply when C_wake drops.
    
    Parameters
    ----------
    C_wake : array
        Wake coherence order parameter
    beta : float
        Sharpness exponent (1-2 typical)
        
    Returns
    -------
    D_wake : array
        Decoherence factor (0 to 1)
    """
    return np.power(np.maximum(C_wake, 0.0), beta)


# =============================================================================
# Σ-GRAVITY WITH WAKE CORRECTION
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """Standard Σ-Gravity acceleration function."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_geometric(r: np.ndarray, xi: float) -> np.ndarray:
    """Standard geometric coherence window."""
    xi = max(xi, 0.01)
    return r / (xi + r)


def Sigma_enhancement_baseline(
    R_kpc: np.ndarray,
    g_N: np.ndarray,
    R_d: float,
    A: float = A_GALAXY
) -> np.ndarray:
    """
    Baseline Σ-Gravity enhancement (no wake correction).
    
    Σ = 1 + A × W_geom(r) × h(g_N)
    """
    xi = XI_SCALE * R_d
    W = W_geometric(R_kpc, xi)
    h = h_function(g_N)
    return 1 + A * W * h


def Sigma_enhancement_wake(
    R_kpc: np.ndarray,
    g_N: np.ndarray,
    R_d: float,
    C_wake: np.ndarray,
    A: float = A_GALAXY,
    beta: float = 1.0,
    mode: str = 'multiply'
) -> np.ndarray:
    """
    Wake-corrected Σ-Gravity enhancement.
    
    Two modes:
    1. 'multiply': W_eff = W_geom × C_wake
       Σ = 1 + A × W_eff × h(g_N)
       
    2. 'separate': D_wake as separate factor
       Σ = 1 + A × W_geom × h(g_N) × D_wake
    
    Parameters
    ----------
    R_kpc : array
        Radii in kpc
    g_N : array
        Newtonian acceleration (m/s²)
    R_d : float
        Disk scale length (kpc)
    C_wake : array
        Wake coherence at each radius
    A : float
        Enhancement amplitude
    beta : float
        Decoherence sharpness
    mode : str
        'multiply' or 'separate'
        
    Returns
    -------
    Sigma : array
        Enhancement factor
    """
    xi = XI_SCALE * R_d
    W_geom = W_geometric(R_kpc, xi)
    h = h_function(g_N)
    D = D_wake(C_wake, beta)
    
    if mode == 'multiply':
        W_eff = W_geom * D
        return 1 + A * W_eff * h
    else:  # 'separate'
        return 1 + A * W_geom * h * D


def predict_velocity_wake(
    R_kpc: np.ndarray,
    V_bar: np.ndarray,
    R_d: float,
    C_wake: np.ndarray,
    A: float = A_GALAXY,
    beta: float = 1.0,
    mode: str = 'multiply'
) -> np.ndarray:
    """
    Predict rotation velocity with wake correction.
    
    V_pred = V_bar × √Σ_wake
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_N = np.maximum(V_bar_ms ** 2 / R_m, 1e-15)
    
    Sigma = Sigma_enhancement_wake(R_kpc, g_N, R_d, C_wake, A, beta, mode)
    return V_bar * np.sqrt(Sigma)


def predict_velocity_baseline(
    R_kpc: np.ndarray,
    V_bar: np.ndarray,
    R_d: float,
    A: float = A_GALAXY
) -> np.ndarray:
    """
    Predict rotation velocity (baseline, no wake).
    
    V_pred = V_bar × √Σ
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_N = np.maximum(V_bar_ms ** 2 / R_m, 1e-15)
    
    Sigma = Sigma_enhancement_baseline(R_kpc, g_N, R_d, A)
    return V_bar * np.sqrt(Sigma)


# =============================================================================
# BULGE + DISK PROFILE MODELS
# =============================================================================

def exponential_disk_profile(R: np.ndarray, Sigma_0: float, R_d: float) -> np.ndarray:
    """Exponential disk surface density: Σ(R) = Σ_0 × exp(-R/R_d)"""
    return Sigma_0 * np.exp(-R / R_d)


def sersic_bulge_profile(
    R: np.ndarray,
    Sigma_e: float,
    R_e: float,
    n: float = 4.0
) -> np.ndarray:
    """
    Sérsic bulge profile.
    
    Σ(R) = Σ_e × exp(-b_n × [(R/R_e)^(1/n) - 1])
    
    For n=4 (de Vaucouleurs): b_n ≈ 7.67
    """
    b_n = 2 * n - 1/3 + 4/(405*n)  # Approximation for b_n
    x = (R / R_e) ** (1/n)
    return Sigma_e * np.exp(-b_n * (x - 1))


def estimate_bulge_dispersion(V_flat: float, bulge_frac: float) -> float:
    """
    Estimate bulge velocity dispersion from flat velocity and bulge fraction.
    
    Empirical relation: σ_b ≈ V_flat × (0.5 + 0.3 × f_bulge)
    """
    return V_flat * (0.5 + 0.3 * bulge_frac)


# =============================================================================
# COUNTER-ROTATION EXTENSION
# =============================================================================

def C_wake_counter_rotating(
    R: np.ndarray,
    Sigma_1: np.ndarray,
    Sigma_2: np.ndarray,
    v_1: np.ndarray,
    v_2: np.ndarray,
    sigma_1: np.ndarray,
    sigma_2: np.ndarray,
    params: WakeParams = WakeParams()
) -> np.ndarray:
    """
    Wake coherence for counter-rotating components.
    
    Two stellar populations with opposite rotation directions.
    Their wakes subtract in J but add in N, crushing C_wake.
    
    Parameters
    ----------
    Sigma_1, Sigma_2 : array
        Surface densities of two populations
    v_1, v_2 : array
        Rotation velocities (v_2 < 0 for counter-rotation)
    sigma_1, sigma_2 : array
        Velocity dispersions
    """
    alpha = params.alpha
    v0 = params.v0
    
    # Population 1 (prograde)
    v1_safe = np.maximum(np.abs(v_1), 1.0)
    N_1 = Sigma_1 * (v1_safe / v0) ** alpha
    J_1 = N_1 * np.sign(v_1)  # Direction matters
    
    # Population 2 (retrograde)
    v2_safe = np.maximum(np.abs(v_2), 1.0)
    N_2 = Sigma_2 * (v2_safe / v0) ** alpha
    J_2 = N_2 * np.sign(v_2)  # Negative for counter-rotation
    
    # Dispersion contributions (reduce coherence)
    N_disp_1 = Sigma_1 * (sigma_1 / v0) ** alpha
    N_disp_2 = Sigma_2 * (sigma_2 / v0) ** alpha
    
    # Total
    J_total = np.abs(J_1 + J_2)  # Vectors subtract
    N_total = N_1 + N_2 + N_disp_1 + N_disp_2  # Scalars add
    
    C_wake = J_total / np.maximum(N_total, 1e-10)
    return np.clip(C_wake, 0.0, 1.0)


# =============================================================================
# TEMPORAL WAKE ACCUMULATION (OPTIONAL)
# =============================================================================

def wake_field_steady_state(
    R: np.ndarray,
    rho: np.ndarray,
    v_c: np.ndarray,
    C_wake: np.ndarray,
    tau_damp: float = 1.0  # Damping timescale (Gyr)
) -> np.ndarray:
    """
    Steady-state wake field from balance of source and damping.
    
    dW/dt = S(R) - W/τ_damp = 0
    → W = τ_damp × S(R)
    
    Source: S ∝ ρ × v_c² × (1 - C_wake)
    - Coherent disk: C_wake ≈ 1 → little decohering energy
    - Chaotic bulge: C_wake ≈ 0 → strong decohering energy
    
    Parameters
    ----------
    R : array
        Radii
    rho : array
        Density profile
    v_c : array
        Circular velocity
    C_wake : array
        Wake coherence
    tau_damp : float
        Damping timescale
        
    Returns
    -------
    W_field : array
        Wake field amplitude (normalized)
    """
    # Source term: stronger where coherence is low
    S = rho * v_c ** 2 * (1 - C_wake)
    
    # Steady state
    W_field = tau_damp * S
    
    # Normalize
    W_max = np.max(W_field)
    if W_max > 0:
        W_field = W_field / W_max
    
    return W_field


def D_wake_temporal(W_field: np.ndarray, W_0: float = 0.5) -> np.ndarray:
    """
    Decoherence factor from accumulated wake field.
    
    D_wake = 1 / (1 + W/W_0)
    
    High W_field → strong decoherence → D_wake → 0
    Low W_field → weak decoherence → D_wake → 1
    """
    return 1 / (1 + W_field / W_0)


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def compute_rms(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """Compute RMS velocity residual."""
    return np.sqrt(np.mean((V_obs - V_pred) ** 2))


def compare_models(
    R: np.ndarray,
    V_obs: np.ndarray,
    V_bar: np.ndarray,
    R_d: float,
    C_wake: np.ndarray,
    params: WakeParams = WakeParams()
) -> Dict[str, float]:
    """
    Compare baseline and wake-corrected predictions.
    
    Returns
    -------
    metrics : dict
        RMS for baseline and wake-corrected models
    """
    V_baseline = predict_velocity_baseline(R, V_bar, R_d)
    V_wake_mult = predict_velocity_wake(R, V_bar, R_d, C_wake, 
                                        beta=params.beta, mode='multiply')
    V_wake_sep = predict_velocity_wake(R, V_bar, R_d, C_wake,
                                       beta=params.beta, mode='separate')
    
    return {
        'rms_baseline': compute_rms(V_obs, V_baseline),
        'rms_wake_multiply': compute_rms(V_obs, V_wake_mult),
        'rms_wake_separate': compute_rms(V_obs, V_wake_sep),
        'improvement_multiply': 1 - compute_rms(V_obs, V_wake_mult) / compute_rms(V_obs, V_baseline),
        'improvement_separate': 1 - compute_rms(V_obs, V_wake_sep) / compute_rms(V_obs, V_baseline),
    }


if __name__ == '__main__':
    # Quick test
    R = np.linspace(0.5, 15, 30)
    
    # Mock galaxy: exponential disk + Sérsic bulge
    Sigma_d = exponential_disk_profile(R, Sigma_0=100, R_d=3.0)
    Sigma_b = sersic_bulge_profile(R, Sigma_e=500, R_e=0.5, n=4)
    v_c = 200 * np.tanh(R / 3)  # Rising to flat
    sigma_b = 120  # km/s
    
    # Compute wake coherence
    C = C_wake_discrete(R, Sigma_d, Sigma_b, v_c, sigma_b)
    
    print("Wake Coherence Model Test")
    print("=" * 50)
    print(f"{'R (kpc)':<10} {'Σ_d':<10} {'Σ_b':<10} {'C_wake':<10}")
    print("-" * 50)
    for i in [0, 5, 10, 15, 20, 25, 29]:
        print(f"{R[i]:<10.1f} {Sigma_d[i]:<10.1f} {Sigma_b[i]:<10.1f} {C[i]:<10.3f}")
    
    print("\nInterpretation:")
    print(f"  Inner (R=0.5 kpc): C_wake = {C[0]:.3f} (bulge-dominated → low coherence)")
    print(f"  Outer (R=15 kpc): C_wake = {C[-1]:.3f} (disk-dominated → high coherence)")

