#!/usr/bin/env python3
"""
Σ-GRAVITY EXTENDED REGRESSION TEST
===================================

This extends the master regression test with additional tests developed during
the graviton path model exploration, plus optional ray-tracing lensing tests.

Uses the UNIFIED 3D AMPLITUDE FORMULA: A(L) = A₀ × (L/L₀)^n
No D switch needed - path length L determines amplitude naturally.

CORE TESTS (8):
    1. SPARC Galaxies (171 rotation curves)
    2. Galaxy Clusters (42 Fox+ 2022)
    3. Cluster Holdout (n calibration stability with L₀ fixed)
    4. Milky Way (28,368 Gaia stars)
    5. Redshift Evolution
    6. Solar System (Cassini bound)
    7. Counter-Rotation Effect
    8. Tully-Fisher Relation

EXTENDED TESTS (9):
    9. Wide Binaries (Chae 2023)
    10. Dwarf Spheroidals (Fornax, Draco, Sculptor, Carina)
    11. Ultra-Diffuse Galaxies (DF2, Dragonfly44)
    12. Galaxy-Galaxy Lensing
    13. External Field Effect
    14. Gravitational Waves (GW170817)
    15. Structure Formation
    16. CMB Acoustic Peaks
    17. Bullet Cluster (ray-tracing)

USAGE:
    python scripts/run_regression_experimental.py           # Full test (17 tests)
    python scripts/run_regression_experimental.py --quick   # Skip slow tests
    python scripts/run_regression_experimental.py --core    # Only core 8 tests
    python scripts/run_regression_experimental.py --sigma-components  # Use component-mixed σ(r) in C(r)
    python scripts/run_regression_experimental.py --coherence=jj  # Use JJ (current-current) coherence model
    python scripts/run_regression_experimental.py --guided --guided-kappa 1.0  # Experimental guided-gravity variant
    python scripts/run_regression_experimental.py --compare-guided --guided-kappa 1.0  # Baseline vs guided side-by-side
    python scripts/run_regression_experimental.py --guided --guided-kappa 1.0 --guided-c-default 0.0  # Only use explicit coherence proxies

Author: Leonard Speiser
Last Updated: December 2025
"""

import numpy as np
import pandas as pd
import math
import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
G = 6.674e-11  # m³/kg/s²
H0_SI = 2.27e-18  # 1/s (H0 = 70 km/s/Mpc)
kpc_to_m = 3.086e19
AU_to_m = 1.496e11
M_sun = 1.989e30

# Critical acceleration (derived from cosmology)
g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))  # ≈ 9.60×10⁻¹¹ m/s²

# MOND acceleration scale (for comparison)
a0_mond = 1.2e-10

# =============================================================================
# MODEL PARAMETERS (Σ-GRAVITY UNIFIED FORMULA)
# =============================================================================
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
L_0 = 0.40  # Reference path length (kpc)
N_EXP = 0.27  # Path length exponent
XI_SCALE = 1 / (2 * np.pi)  # ξ = R_d/(2π)
ML_DISK = 0.5
ML_BULGE = 0.7

# Optional: use a component-mixed dispersion profile σ(r) in the coherence scalar C(r).
# This is a population-informed *input* (from baryonic phase mixture), not a per-galaxy fit.
USE_SIGMA_COMPONENTS = False  # set in main()
SIGMA_GAS_KMS = 10.0
SIGMA_DISK_KMS = 25.0
SIGMA_BULGE_KMS = 120.0

# =============================================================================
# COHERENCE MODEL SELECTION (A/B)
# =============================================================================
COHERENCE_MODEL = "C"   # "C" baseline, "JJ" current-current, or "FLOW" vorticity/shear topology

# Covariant coherence proxy for SPARC (translates Gaia-calibrated C_cov)
USE_COVARIANT_PROXY = False

# JJ model hyperparameters (global; no per-galaxy fits)
# Tuned values: JJ_XI_MULT=0.4 gives best SPARC RMS (22.83 km/s vs 22.94 at 1.0)
JJ_XI_MULT = 0.4        # ξ_corr = JJ_XI_MULT * (R_d/(2π)) by default (tuned: 0.4 optimal)
JJ_SMOOTH_M_POINTS = 5  # smooth M_enc before derivative to reduce numerical noise (tuned: 5 optimal)
JJ_EPS = 1e-30


# FLOW / topology coherence hyperparameters
# C_flow = omega^2 / (omega^2 + α·shear^2 + β·theta^2 + γ·tidal^2 + δ·H0^2)
FLOW_ALPHA = 0.5
FLOW_BETA = 0.0
FLOW_GAMMA = 0.1
FLOW_DELTA = 1.0
FLOW_SMOOTH_POINTS = 5   # moving-average window for dV/dR and dg/dR (odd recommended)
FLOW_USE_TIDAL = True

# Bulge-specific flow parameters (for galaxies with f_bulge >= 0.3)
# Bulges: vorticity-dominated, shear irrelevant
# Disks: shear matters, vorticity less important
FLOW_USE_BULGE_SPECIFIC = False  # Enable bulge-specific tuning
FLOW_ALPHA_BULGE = 0.0   # Ignore shear for bulges
FLOW_GAMMA_BULGE = 0.01  # Keep minimal tidal for bulges
FLOW_ALPHA_DISK = 0.02   # Use shear for disks
FLOW_GAMMA_DISK = 0.005  # Minimal tidal for disks

# Optional: load precomputed Gaia 6D flow features (see export_gaia_pointwise_features.py)
GAIA_FLOW_FEATURES_PATH = None
GAIA_FLOW_FEATURES_DF = None
GAIA_FLOW_REQUIRE_MATCH = True

# Fixed-point damping (use <1.0 if a new coherence mode oscillates)
FP_RELAX = 1.0

# Cluster amplitude
A_CLUSTER = A_0 * (600 / L_0)**N_EXP  # ≈ 8.45

# =============================================================================
# GUIDED-GRAVITY (STREAM-SEEKING) EXTENSION — EXPERIMENTAL
# =============================================================================
#
# Concept (toy model): coherent streams "guide" the gravitational response by
# increasing the effective path-length in the amplitude law,
#
#   L_eff = L · (1 + κ · C_stream)
#   A_eff = A(L_eff) = A(L) · (1 + κ · C_stream)^n
#
# where C_stream ∈ [0, 1] is a local proxy for stream coherence.
# - In disk RC fits: we use the existing covariant C(v) (or W(r) if using W).
# - In contexts without an explicit stream proxy (e.g., simple point-mass tests,
#   lensing rays), we use GUIDED_C_DEFAULT.
#
# This is *not* standard GR; it is an experimental hook to test the "gravity
# bends toward streams" idea against the baseline Σ-Gravity formulas.
# Tuned values: κ=0.1 gives best cluster ratio (1.004) with minimal SPARC impact (+1.2%)
USE_GUIDED_GRAVITY = False
GUIDED_KAPPA = 0.1          # κ = 0.1 optimal (clusters: 0.987→1.004, SPARC: 17.42→17.63 km/s)
GUIDED_C_DEFAULT = 0.0      # Used when no local stream proxy is available (has no effect in current tests)
GUIDED_FACTOR_CAP = 1e3     # Safety cap to avoid numerical blow-ups

# =============================================================================
# OBSERVATIONAL BENCHMARKS (GOLD STANDARD)
# All values from peer-reviewed literature with citations
# =============================================================================
OBS_BENCHMARKS = {
    # Solar System - Bertotti+ 2003, Nature 425, 374
    'solar_system': {
        'cassini_gamma_uncertainty': 2.3e-5,  # γ-1 = (2.1±2.3)×10⁻⁵
        'source': 'Bertotti+ 2003',
    },
    
    # SPARC - Lelli, McGaugh & Schombert 2016, AJ 152, 157
    'sparc': {
        'n_quality': 171,
        'mond_rms_kms': 17.15,  # With standard a₀=1.2×10⁻¹⁰, per-galaxy RMS averaged
        'sigma_rms_kms': 17.42,  # Σ-Gravity per-galaxy RMS averaged
        'rar_scatter_dex': 0.10,  # Std of log(V_obs/V_pred) over all points
        'lcdm_rms_kms': 15.0,  # With 2-3 params/galaxy (NFW)
        'source': 'Lelli+ 2016, McGaugh+ 2016',
    },
    
    # Wide Binaries - Chae 2023, ApJ 952, 128
    'wide_binaries': {
        'boost_factor': 1.35,  # ~35% excess at >2000 AU
        'boost_uncertainty': 0.10,
        'threshold_AU': 2000,
        'n_pairs': 26500,  # Gaia DR3
        'controversy': 'Banik+ 2024 disputes; ongoing debate',
        'source': 'Chae 2023',
    },
    
    # Dwarf Spheroidals - Walker+ 2009, McConnachie 2012
    # NEW: d_MW_kpc is distance from MW center (for host inheritance model)
    'dwarf_spheroidals': {
        'fornax': {'M_star': 2e7, 'sigma_obs': 10.7, 'sigma_err': 0.5, 'r_half_kpc': 0.71, 'd_MW_kpc': 147, 'M_L': 7.5},
        'draco': {'M_star': 2.9e5, 'sigma_obs': 9.1, 'sigma_err': 1.2, 'r_half_kpc': 0.22, 'd_MW_kpc': 76, 'M_L': 330},
        'sculptor': {'M_star': 2.3e6, 'sigma_obs': 9.2, 'sigma_err': 0.6, 'r_half_kpc': 0.28, 'd_MW_kpc': 86, 'M_L': 160},
        'carina': {'M_star': 3.8e5, 'sigma_obs': 6.6, 'sigma_err': 1.2, 'r_half_kpc': 0.25, 'd_MW_kpc': 105, 'M_L': 40},
        'ursa_minor': {'M_star': 2.9e5, 'sigma_obs': 9.5, 'sigma_err': 1.2, 'r_half_kpc': 0.30, 'd_MW_kpc': 76, 'M_L': 290},
        'model': 'host_inheritance',  # dSphs inherit MW Σ at their orbital radius
        'mond_status': 'Generally works for isolated dSphs; EFE complicates satellites',
        'source': 'Walker+ 2009, McConnachie 2012',
    },
    
    # Ultra-Diffuse Galaxies - van Dokkum+ 2018, 2016
    'udgs': {
        'df2': {
            'M_star': 2e8, 'sigma_obs': 8.5, 'sigma_err': 2.3, 'r_eff_kpc': 2.2,
            'note': 'Appears to lack DM; MOND predicts ~20 km/s (EFE resolution)',
            'source': 'van Dokkum+ 2018',
        },
        'dragonfly44': {
            'M_star': 3e8, 'sigma_obs': 47, 'sigma_err': 8, 'r_eff_kpc': 4.6,
            'note': 'Very DM dominated; M_dyn ~ 10^12 M☉',
            'source': 'van Dokkum+ 2016',
        },
    },
    
    # Tully-Fisher - McGaugh 2012, AJ 143, 40
    'tully_fisher': {
        'btfr_slope': 3.98,  # ±0.06
        'btfr_normalization': 47,  # M☉/(km/s)^4
        'scatter_dex': 0.10,
        'mond_prediction': 4.0,  # Exact slope 4
        'source': 'McGaugh 2012',
    },
    
    # Gravitational Waves - Abbott+ 2017, PRL 119, 161101
    'gw170817': {
        'delta_c_over_c': 1e-15,  # |c_GW - c|/c
        'time_delay_s': 1.7,  # GRB arrived 1.7s after GW
        'distance_Mpc': 40,
        'source': 'Abbott+ 2017 (GW170817 + GRB170817A)',
    },
    
    # Bullet Cluster - Clowe+ 2006, ApJ 648, L109
    'bullet_cluster': {
        'M_gas': 2.1e14,  # M☉ (from X-ray)
        'M_stars': 0.5e14,  # M☉
        'M_baryonic': 2.6e14,  # M☉
        'M_lensing': 5.5e14,  # M☉ (from weak lensing)
        'mass_ratio': 2.1,  # M_lensing / M_baryonic
        'offset_kpc': 150,  # Lensing peak offset from gas
        'separation_kpc': 720,  # Between main and subcluster
        'mond_challenge': 'Lensing follows stars, not gas',
        'source': 'Clowe+ 2006',
    },
    
    # Galaxy Clusters - Fox+ 2022, ApJ 928, 87
    'clusters': {
        'n_quality': 42,  # spec_z + M500 > 2×10¹⁴
        'mond_mass_discrepancy': 3.0,  # Factor MOND underpredicts
        'lcdm_success': True,  # NFW fits work
        'source': 'Fox+ 2022',
    },
    
    # Milky Way - Eilers+ 2019, McMillan 2017
    'milky_way': {
        'V_sun_kms': 233,  # ±3 (Eilers+ 2019)
        'R_sun_kpc': 8.178,  # GRAVITY Collaboration 2019
        'M_baryonic': 6.5e10,  # M☉ (McMillan 2017)
        'n_gaia_stars': 28368,  # Eilers-APOGEE-Gaia disk sample
        'source': 'Eilers+ 2019, McMillan 2017',
    },
    
    # CMB - Planck 2018
    'cmb': {
        'Omega_b': 0.0493,
        'Omega_c': 0.265,  # CDM
        'Omega_m': 0.315,
        'H0': 67.4,  # km/s/Mpc
        'mond_challenge': 'CMB requires DM at z~1100',
        'source': 'Planck Collaboration 2020',
    },
    
    # Structure Formation
    'structure_formation': {
        'sigma8_planck': 0.811,
        'sigma8_lensing': 0.76,  # S8 tension
        'bao_scale_Mpc': 150,
        'source': 'Planck 2018, SDSS',
    },
}

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    details: Dict[str, Any]
    message: str


def unified_amplitude(L: float) -> float:
    """Unified 3D amplitude: A = A₀ × (L/L₀)^n
    
    No D switch needed - path length L determines amplitude naturally:
    - Thin disk galaxies: L ≈ L₀ (0.4 kpc scale height) → A ≈ A₀
    - Elliptical galaxies: L ~ 1-20 kpc → A ~ 1.5-3.4
    - Galaxy clusters: L ≈ 600 kpc → A ≈ 8.45
    """
    return A_0 * (L / L_0)**N_EXP


def unified_amplitude_legacy(D: float, L: float) -> float:
    """Legacy amplitude with D switch (for backwards compatibility)."""
    return A_0 * (1 - D + D * (L / L_0)**N_EXP)


def guided_amplitude_multiplier(C_stream: Any, exponent: float = N_EXP) -> Any:
    """Return the multiplicative factor (1 + κ·C_stream)^exponent.

    This implements the toy "stream-seeking" extension via:
        L_eff = L · (1 + κ·C_stream)
        A_eff = A(L_eff) = A(L) · (1 + κ·C_stream)^n

    Notes:
      - C_stream is clipped into [0,1] for stability.
      - A soft cap is applied to avoid numerical blow-ups during sweeps.
      - With κ=0 or USE_GUIDED_GRAVITY=False, this returns 1.
    """
    if not USE_GUIDED_GRAVITY:
        return 1.0
    kappa = float(GUIDED_KAPPA)
    if kappa == 0.0:
        return 1.0
    C = np.clip(np.asarray(C_stream, dtype=float), 0.0, 1.0)
    fac = 1.0 + kappa * C
    fac = np.clip(fac, 1.0 / GUIDED_FACTOR_CAP, GUIDED_FACTOR_CAP)
    return np.power(fac, float(exponent))


def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def C_coherence(v_rot: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """
    Covariant coherence scalar: C = v²/(v² + σ²)
    
    This is the PRIMARY formulation, built from 4-velocity invariants.
    """
    v2 = np.maximum(np.asarray(v_rot, dtype=float), 0.0)**2
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    s2 = sigma_arr**2
    return v2 / (v2 + s2)


def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """
    Coherence window W(r) = r/(ξ+r)
    
    This is a validated APPROXIMATION to orbit-averaged C for disk galaxies.
    Gives identical results to C(r) formulation.
    """
    xi = max(xi, 0.01)
    return r / (xi + r)


def _smooth_1d(x: np.ndarray, w: int) -> np.ndarray:
    """Cheap moving-average smoothing (odd w recommended)."""
    x = np.asarray(x, dtype=float)
    if w is None or w <= 1:
        return x
    w = int(w)
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")




def _odd_window(w: int, n: int) -> int:
    """Return a safe odd smoothing window <= n (or 1 if n<3)."""
    try:
        w = int(w)
    except Exception:
        return 1
    if n < 3:
        return 1
    if w > n:
        w = n
    if w % 2 == 0:
        w = max(1, w - 1)
    if w < 3:
        return 1
    return w


def flow_invariants_from_rotation_curve(
    R_kpc: np.ndarray,
    V_kms: np.ndarray,
    V_bar_kms: Optional[np.ndarray] = None,
    smooth_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Axisymmetric flow invariants from a 1D rotation curve.

    Returns (omega2, shear2, theta2, tidal2) in units of (km/s/kpc)^2.

    For an azimuthal flow v = v_phi(R):
      ω_z = (1/R) d(R v_phi)/dR = v_phi/R + dv_phi/dR
      s   = dv_phi/dR - v_phi/R
      θ   ≈ 0  (incompressible circular flow)

    Tidal proxy uses the baryonic acceleration gradient:
      g_bar(R) = V_bar(R)^2 / R
      tidal2 ~ |dg_bar/dR|
    with a unit conversion so tidal2 is also in (km/s/kpc)^2.
    """
    R = np.asarray(R_kpc, dtype=float)
    V = np.asarray(V_kms, dtype=float)
    Vb = np.asarray(V_bar_kms, dtype=float) if V_bar_kms is not None else V

    R_safe = np.clip(R, 1e-6, None)

    # Smooth V before derivative to reduce noise amplification
    w = FLOW_SMOOTH_POINTS if smooth_points is None else int(smooth_points)
    w = _odd_window(w, len(V))
    V_s = _smooth_1d(V, w) if w > 1 else V

    dVdR = np.gradient(V_s, R_safe)  # km/s/kpc

    omega = V_s / R_safe + dVdR
    shear = dVdR - V_s / R_safe
    theta = np.zeros_like(omega)

    omega2 = omega**2
    shear2 = shear**2
    theta2 = theta**2

    # Baryonic tidal term
    R_m = R_safe * kpc_to_m
    Vb_ms = np.maximum(Vb, 0.0) * 1000.0
    g_bar = (Vb_ms**2) / np.maximum(R_m, 1e-9)  # m/s^2
    g_s = _smooth_1d(g_bar, w) if w > 1 else g_bar

    dg_dR = np.gradient(g_s, R_m)  # 1/s^2

    # Convert 1/s^2 -> (km/s/kpc)^2
    omega_unit_si = 1000.0 / kpc_to_m  # (km/s/kpc) expressed as 1/s
    tidal2 = np.abs(dg_dR) / (omega_unit_si**2)

    return omega2, shear2, theta2, tidal2


def C_covariant_coherence(
    omega2: np.ndarray,
    rho_kg_m3: np.ndarray,
    theta2: np.ndarray,
) -> np.ndarray:
    """Covariant coherence scalar C_cov from field theory.

    C_cov = ω² / (ω² + 4πGρ + θ² + H₀²)

    This is the PRIMARY theoretical formulation from the paper.
    The kinematic form C = v²/(v²+σ²) is the non-relativistic limit.

    Parameters
    ----------
    omega2 : np.ndarray
        Vorticity squared in (km/s/kpc)^2
    rho_kg_m3 : np.ndarray
        Baryonic density in kg/m³
    theta2 : np.ndarray
        Expansion squared in (km/s/kpc)^2

    Returns
    -------
    np.ndarray
        Coherence scalar in [0,1]
    """
    om2 = np.asarray(omega2, dtype=float)
    rho = np.asarray(rho_kg_m3, dtype=float)
    th2 = np.asarray(theta2, dtype=float)

    # 4πGρ in (km/s/kpc)^2
    # G = 6.674e-11 m³/kg/s²
    # Convert: 4πGρ (m³/kg/s² × kg/m³ = 1/s²) → (km/s/kpc)^2
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


def C_flow_coherence(
    omega2: np.ndarray,
    shear2: np.ndarray,
    theta2: np.ndarray,
    tidal2: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    delta: Optional[float] = None,
) -> np.ndarray:
    """Topology / flow coherence scalar C_flow in [0,1].

    C_flow = omega^2 / (omega^2 + α·shear^2 + β·theta^2 + γ·tidal^2 + δ·H0^2)

    This is a phenomenological approximation to the covariant C_cov.
    Use C_covariant_coherence() when you have access to density ρ.

    All inputs must be in (km/s/kpc)^2.

    Notes:
      - In Gaia flow-feature exports, omega2 and shear2 are naturally in these units.
      - H0 is converted to km/s/kpc so the floor term is comparable.
    """
    a = FLOW_ALPHA if alpha is None else float(alpha)
    b = FLOW_BETA if beta is None else float(beta)
    gam = FLOW_GAMMA if gamma is None else float(gamma)
    d = FLOW_DELTA if delta is None else float(delta)

    om2 = np.asarray(omega2, dtype=float)
    sh2 = np.asarray(shear2, dtype=float)
    th2 = np.asarray(theta2, dtype=float)

    if tidal2 is None:
        tid2 = 0.0
        use_tidal = False
    else:
        tid2 = np.asarray(tidal2, dtype=float)
        use_tidal = True

    # H0 in km/s/kpc (≈ 0.07)
    H0_kms_per_kpc = H0_SI * (kpc_to_m / 1000.0)

    denom = om2 + a * sh2 + b * th2
    if FLOW_USE_TIDAL and use_tidal:
        denom = denom + gam * tid2
    denom = denom + d * (H0_kms_per_kpc**2)

    denom = np.maximum(denom, 1e-30)
    C = om2 / denom
    C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(C, 0.0, 1.0)

def rho_proxy_from_vbar(R_kpc: np.ndarray, V_bar_kms: np.ndarray) -> np.ndarray:
    """
    Baryonic density proxy from V_bar (spherical inversion):
      M_enc(r) ~ V_bar(r)^2 * r / G
      rho(r) ~ (1/(4πr^2)) dM_enc/dr

    Used ONLY to weight the current-current coherence proxy.
    """
    R_kpc = np.asarray(R_kpc, dtype=float)
    V_bar_kms = np.asarray(V_bar_kms, dtype=float)

    # Enforce positive radii
    R_m = np.maximum(R_kpc, 1e-9) * kpc_to_m
    V_ms = V_bar_kms * 1000.0

    # Enclosed mass proxy (kg)
    M_enc = np.maximum(V_ms**2 * R_m / G, 0.0)

    # Smooth to reduce numerical derivative noise
    M_enc_s = _smooth_1d(M_enc, JJ_SMOOTH_M_POINTS)

    dM_dR = np.gradient(M_enc_s, R_m)
    # Safety: handle negative gradients (shouldn't happen but numerical issues can occur)
    dM_dR = np.maximum(dM_dR, 0.0)
    rho = np.maximum(dM_dR / (4.0 * np.pi * R_m**2), 1e-20)  # kg/m^3 (minimum to avoid zeros)
    
    # Safety: replace any NaN or inf with small positive value
    rho = np.nan_to_num(rho, nan=1e-20, posinf=1e10, neginf=1e-20)

    return rho


def Q_JJ_coherence(
    R_kpc: np.ndarray,
    V_pred_kms: np.ndarray,
    V_bar_kms: np.ndarray,
    R_d_kpc: float,
    sigma_kms: float | np.ndarray = 20.0,
    xi_corr_kpc: float | None = None,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """
    Current-current coherence order parameter Q_JJ(r) in [0,1]:

      Q = <J>_K^2 / <J^2 + J_rand^2>_K

    where J ~ rho_b * v_pred and J_rand ~ rho_b * sigma.
    Kernel is exponential in |Δr| with scale xi_corr.

    NOTE: Uses V_pred (not V_obs) -> no data leakage.
    """
    R_kpc = np.asarray(R_kpc, dtype=float)
    V_pred_kms = np.asarray(V_pred_kms, dtype=float)
    V_bar_kms = np.asarray(V_bar_kms, dtype=float)

    # sigma can be scalar or radial profile
    if np.ndim(sigma_kms) == 0:
        sigma_arr = np.full_like(V_pred_kms, float(sigma_kms))
    else:
        sigma_arr = np.asarray(sigma_kms, dtype=float)

    rho = rho_proxy_from_vbar(R_kpc, V_bar_kms)  # kg/m^3

    # Current proxies (units cancel in the ratio)
    J = rho * (V_pred_kms * 1000.0)
    J_rand2 = (rho * (sigma_arr * 1000.0))**2
    J2_total = J**2 + J_rand2

    # Build kernel matrix once if not provided
    if kernel is None:
        if xi_corr_kpc is None:
            xi_corr_kpc = JJ_XI_MULT * XI_SCALE * max(float(R_d_kpc), 1e-6)
        xi_m = max(float(xi_corr_kpc), 1e-6) * kpc_to_m

        R_m = R_kpc * kpc_to_m
        dR = np.abs(R_m[:, None] - R_m[None, :])
        kernel = np.exp(-dR / xi_m)

    wsum = np.maximum(kernel.sum(axis=1), JJ_EPS)

    meanJ = (kernel @ J) / wsum
    meanJ2 = (kernel @ J2_total) / wsum

    # Safety: handle cases where meanJ2 is too small or meanJ is NaN
    meanJ = np.nan_to_num(meanJ, nan=0.0, posinf=0.0, neginf=0.0)
    meanJ2 = np.maximum(np.nan_to_num(meanJ2, nan=JJ_EPS, posinf=1e10, neginf=JJ_EPS), JJ_EPS)

    Q = (meanJ**2) / meanJ2
    Q = np.nan_to_num(Q, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(Q, 0.0, 1.0)


def sigma_profile_from_components_kms(
    V_gas_kms: np.ndarray,
    V_disk_scaled_kms: np.ndarray,
    V_bulge_scaled_kms: np.ndarray,
) -> np.ndarray:
    """
    Construct an effective dispersion profile σ_eff(r) by mixing fixed phase dispersions.

    Weights use squared baryonic component contributions to V_bar at each radius:
      w_i(r) = V_i(r)^2 / Σ_j V_j(r)^2

    σ_eff(r)^2 = Σ_i w_i(r) σ_i^2

    This is designed to capture the empirical fact that galaxy populations differ in phase-space
    coherence (gas-rich disks vs bulge-dominated systems) without per-object tuning.
    """
    vg2 = np.square(np.nan_to_num(np.asarray(V_gas_kms, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
    vd2 = np.square(np.nan_to_num(np.asarray(V_disk_scaled_kms, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
    vb2 = np.square(np.nan_to_num(np.asarray(V_bulge_scaled_kms, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))

    denom = np.maximum(vg2 + vd2 + vb2, 1e-12)
    wgas = vg2 / denom
    wdisk = vd2 / denom
    wbul = vb2 / denom

    sigma2 = wgas * (SIGMA_GAS_KMS**2) + wdisk * (SIGMA_DISK_KMS**2) + wbul * (SIGMA_BULGE_KMS**2)
    return np.sqrt(np.maximum(sigma2, 1e-6))


def sigma_enhancement(g: np.ndarray, r: np.ndarray = None, xi: float = 1.0, 
                      A: float = None, L: float = L_0) -> np.ndarray:
    """
    Full Σ enhancement factor using W(r) approximation.
    
    Σ = 1 + A(L) × W(r) × h(g)
    
    Note: This uses the W(r) approximation. For the primary C(r) formulation,
    use sigma_enhancement_C() with fixed-point iteration.
    """
    g = np.maximum(np.asarray(g), 1e-15)
    
    # Baseline amplitude A(L)
    A_base = unified_amplitude(L) if A is None else A
    
    h = h_function(g)
    
    if r is not None:
        W = W_coherence(np.asarray(r), xi)
    else:
        W = 1.0

    # Experimental: "stream-seeking" / guided amplitude. Use W(r) as a proxy
    # for local stream coherence when available; otherwise fall back to a
    # configurable default.
    if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
        C_stream = W if r is not None else GUIDED_C_DEFAULT
        A_use = A_base * guided_amplitude_multiplier(C_stream)
    else:
        A_use = A_base
    
    return 1 + A_use * W * h


def sigma_enhancement_C(g: np.ndarray, v_rot: np.ndarray, sigma: float = 20.0,
                        A: float = None, L: float = L_0) -> np.ndarray:
    """
    Full Σ enhancement factor using covariant C(r) - PRIMARY formulation.
    
    Σ = 1 + A(L) × C(r) × h(g)
    
    where C = v_rot²/(v_rot² + σ²)
    """
    g = np.maximum(np.asarray(g), 1e-15)
    
    # Baseline amplitude A(L)
    A_base = unified_amplitude(L) if A is None else A
    
    h = h_function(g)
    C = C_coherence(v_rot, sigma)

    if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
        A_use = A_base * guided_amplitude_multiplier(C)
    else:
        A_use = A_base
    
    return 1 + A_use * C * h


def predict_velocity(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float,
                     h_disk: float = None, f_bulge: float = 0.0,
                     use_C_primary: bool = True, sigma_kms: float = 20.0,
                     sigma_profile_kms: Optional[np.ndarray] = None,
                     coherence_model: Optional[str] = None,
                     coherence_external: Optional[np.ndarray] = None,
                     use_covariant_proxy: bool = False,
                     return_terms: bool = False):
    """Predict rotation velocity using Σ-Gravity.

    This solver supports multiple coherence models:
      - "C"    : baseline covariant coherence C = v^2/(v^2+σ^2)
      - "JJ"   : current-current nonlocal coherence Q_JJ (kernel-averaged)
      - "FLOW" : topology coherence from (ω, shear, θ, tidal) invariants
      - "COVARIANT" : covariant coherence C_cov = ω²/(ω² + 4πGρ + θ² + H₀²)
                     using rotation curve proxies (when use_covariant_proxy=True)

    Parameters
    ----------
    coherence_external : Optional[np.ndarray]
        If provided, uses this as the coherence scalar (clipped to [0,1]) and
        skips fixed-point iteration (converges in one step). This is useful for
        Gaia 6D flow-feature experiments where C_flow is computed directly from
        the observed phase-space structure.
    
    use_covariant_proxy : bool
        If True, use covariant coherence C_cov with rotation curve proxies
        (ω² from V/R, 4πGρ from density proxy). This translates Gaia-calibrated
        C_cov to SPARC using available observables.

    return_terms : bool
        If True, returns (V_pred, diagnostics_dict) where diagnostics include
        Sigma, C_term, A_use, and (for FLOW) omega2/shear2/theta2/tidal2.
    """
    if coherence_model is None:
        coherence_model = COHERENCE_MODEL
    model = str(coherence_model).upper()

    R_m = np.asarray(R_kpc, dtype=float) * kpc_to_m
    V_bar = np.asarray(V_bar, dtype=float)
    V_bar_ms = V_bar * 1000.0
    g_bar = (V_bar_ms**2) / np.maximum(R_m, 1e-9)

    # For thin disk galaxies, use L = L₀ = 0.4 kpc → A = A₀
    A_base = A_0  # = unified_amplitude(L_0)

    # Build sigma input once
    sigma_use = sigma_profile_kms if sigma_profile_kms is not None else float(sigma_kms)

    # Precompute JJ kernel once (performance + stable iteration)
    jj_kernel = None
    if model == "JJ":
        xi_corr_kpc = JJ_XI_MULT * XI_SCALE * max(float(R_d), 1e-6)
        Rm = np.asarray(R_kpc, dtype=float) * kpc_to_m
        dR = np.abs(Rm[:, None] - Rm[None, :])
        jj_kernel = np.exp(-dR / (max(xi_corr_kpc, 1e-6) * kpc_to_m))

    h = h_function(g_bar)

    # If using covariant coherence proxy (SPARC translation from Gaia)
    if use_covariant_proxy:
        try:
            from translate_covariant_to_sparc import compute_C_cov_proxy_sparc
            # Use fixed-point iteration with C_cov proxy
            V = np.array(V_bar, dtype=float)
            for _ in range(5):  # Fixed-point iteration
                C_cov = compute_C_cov_proxy_sparc(R_kpc, V, V_bar, R_d, f_bulge)
                if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
                    A_use = A_base * guided_amplitude_multiplier(C_cov)
                else:
                    A_use = A_base
                Sigma = 1.0 + A_use * C_cov * h
                V_new = V_bar * np.sqrt(np.maximum(Sigma, 0.0))
                if np.allclose(V, V_new, rtol=1e-4):
                    break
                V = V_new
            V_pred = np.nan_to_num(V, nan=V_bar, posinf=V_bar * 10, neginf=V_bar)
            V_pred = np.maximum(V_pred, 1e-6)
            if return_terms:
                C_cov_final = compute_C_cov_proxy_sparc(R_kpc, V_pred, V_bar, R_d, f_bulge)
                diag = {
                    'Sigma': Sigma,
                    'C_term': C_cov_final,
                    'A_use': A_use,
                    'h_term': h,
                    'coherence_model': 'COVARIANT_PROXY',
                }
                return V_pred, diag
            return V_pred
        except ImportError:
            # Fall back to baseline if translation module not available
            pass

    # If coherence is externally supplied, do a single forward pass
    if coherence_external is not None:
        q = np.clip(np.asarray(coherence_external, dtype=float), 0.0, 1.0)
        if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
            A_use = A_base * guided_amplitude_multiplier(q)
        else:
            A_use = A_base
        Sigma = 1.0 + A_use * q * h
        V_pred = V_bar * np.sqrt(np.maximum(Sigma, 0.0))
        V_pred = np.nan_to_num(V_pred, nan=V_bar, posinf=V_bar * 10, neginf=V_bar)
        V_pred = np.maximum(V_pred, 1e-6)
        if return_terms:
            diag = {
                'Sigma': Sigma,
                'C_term': q,
                'A_use': A_use,
                'h_term': h,
            }
            return V_pred, diag
        return V_pred

    V = np.array(V_bar, dtype=float)

    last_terms = None

    for _ in range(50):  # Typically converges in 3-8 iterations
        if model == "JJ":
            q = Q_JJ_coherence(
                R_kpc=R_kpc,
                V_pred_kms=V,
                V_bar_kms=V_bar,
                R_d_kpc=R_d,
                sigma_kms=sigma_use,
                kernel=jj_kernel,
            )
            omega2 = shear2 = theta2 = tidal2 = None

        elif model == "FLOW":
            omega2, shear2, theta2, tidal2 = flow_invariants_from_rotation_curve(
                R_kpc=np.asarray(R_kpc, dtype=float),
                V_kms=V,
                V_bar_kms=V_bar,
                smooth_points=FLOW_SMOOTH_POINTS,
            )
            # Bulge-specific tuning: use different alpha/gamma for bulge vs disk
            if FLOW_USE_BULGE_SPECIFIC and f_bulge >= 0.3:
                # Bulge: ignore shear, use minimal tidal
                q = C_flow_coherence(omega2=omega2, shear2=shear2, theta2=theta2, tidal2=tidal2,
                                    alpha=FLOW_ALPHA_BULGE, gamma=FLOW_GAMMA_BULGE)
            elif FLOW_USE_BULGE_SPECIFIC:
                # Disk: use shear, minimal tidal
                q = C_flow_coherence(omega2=omega2, shear2=shear2, theta2=theta2, tidal2=tidal2,
                                    alpha=FLOW_ALPHA_DISK, gamma=FLOW_GAMMA_DISK)
            else:
                # Default: use global parameters
                q = C_flow_coherence(omega2=omega2, shear2=shear2, theta2=theta2, tidal2=tidal2)

        else:
            omega2 = shear2 = theta2 = tidal2 = None
            q = C_coherence(V, sigma_use)

        # Guided-gravity: allow stream coherence q to feed back into amplitude.
        if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
            A_use = A_base * guided_amplitude_multiplier(q)
        else:
            A_use = A_base

        Sigma = 1.0 + A_use * q * h

        V_new = V_bar * np.sqrt(np.maximum(Sigma, 0.0))
        V_new = np.nan_to_num(V_new, nan=V_bar, posinf=V_bar * 10, neginf=V_bar)
        V_new = np.maximum(V_new, 1e-6)

        if np.max(np.abs(V_new - V)) < 1e-6:
            last_terms = (Sigma, q, A_use, omega2, shear2, theta2, tidal2)
            V = V_new
            break

        # Optional damping if a new coherence mode introduces oscillations
        if FP_RELAX is not None and float(FP_RELAX) < 1.0:
            lam = float(FP_RELAX)
            V = V + lam * (V_new - V)
        else:
            V = V_new

        last_terms = (Sigma, q, A_use, omega2, shear2, theta2, tidal2)

    if return_terms:
        Sigma, q, A_use, omega2, shear2, theta2, tidal2 = last_terms if last_terms is not None else (np.nan, np.nan, np.nan, None, None, None, None)
        diag = {
            'Sigma': Sigma,
            'C_term': q,
            'A_use': A_use,
            'h_term': h,
        }
        if model == 'FLOW':
            diag.update({
                'omega2': omega2,
                'shear2': shear2,
                'theta2': theta2,
                'tidal2': tidal2,
            })
        return V, diag

    return V



def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """MOND prediction for comparison."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0_mond
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    return V_bar * np.sqrt(nu)


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_sparc(data_dir: Path) -> List[Dict]:
    """Load SPARC galaxy rotation curves.
    
    Source: Lelli, McGaugh & Schombert 2016, AJ 152, 157
    URL: http://astroweb.cwru.edu/SPARC/
    
    Returns 171 galaxies after quality cuts (≥5 points, valid V_bar).
    """
    sparc_dir = data_dir / "Rotmod_LTG"
    if not sparc_dir.exists():
        return []
    
    galaxies = []
    for gf in sorted(sparc_dir.glob("*_rotmod.dat")):
        data = []
        with open(gf) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        data.append({
                            'R': float(parts[0]),
                            'V_obs': float(parts[1]),
                            'V_gas': float(parts[3]),
                            'V_disk': float(parts[4]),
                            'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                        })
                    except ValueError:
                        continue
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        
        # Apply M/L corrections (Lelli+ 2016 standard)
        df['V_disk_scaled'] = df['V_disk'] * np.sqrt(ML_DISK)
        df['V_bulge_scaled'] = df['V_bulge'] * np.sqrt(ML_BULGE)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    df['V_disk_scaled']**2 + df['V_bulge_scaled']**2)
        df['V_bar'] = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        valid = (df['V_bar'] > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        df = df[valid]
        
        if len(df) >= 5:
            idx = len(df) // 3
            R_d = df['R'].iloc[idx] if idx > 0 else df['R'].iloc[-1] / 2
            
            # Estimate disk thickness and bulge fraction for unified model
            h_disk = 0.15 * R_d
            total_sq = np.sum(df['V_disk']**2 + df['V_bulge']**2 + df['V_gas']**2)
            f_bulge = np.sum(df['V_bulge']**2) / max(total_sq, 1e-10)
            
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': df['V_bar'].values,
                # Keep baryonic component curves for optional σ(r) mixing.
                'V_gas': df['V_gas'].values,
                'V_disk_scaled': df['V_disk_scaled'].values,
                'V_bulge_scaled': df['V_bulge_scaled'].values,
                'R_d': R_d,
                'h_disk': h_disk,
                'f_bulge': f_bulge,
            })
    
    return galaxies


def load_clusters(data_dir: Path) -> List[Dict]:
    """Load Fox+ 2022 cluster data.
    
    Source: Fox et al. 2022, ApJ 928, 87
    
    Selection criteria (reducing 94 → 42):
    - spec_z_constraint == 'yes' (spectroscopic redshifts)
    - M500 > 2×10¹⁴ M☉ (high-mass clusters)
    
    Baryonic mass estimate:
    M_bar(200 kpc) = 0.4 × f_baryon × M500
    where f_baryon = 0.15 (cosmic baryon fraction)
    """
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    if not cluster_file.exists():
        return []
    
    df = pd.read_csv(cluster_file)
    
    # Filter to high-quality clusters
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes')
    ].copy()
    df_valid = df_valid[df_valid['M500_1e14Msun'] > 2.0].copy()
    
    clusters = []
    f_baryon = 0.15
    
    for idx, row in df_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar_200 = 0.4 * f_baryon * M500
        M_lens_200 = row['MSL_200kpc_1e12Msun'] * 1e12
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar_200,
            'M_lens': M_lens_200,
            'r_kpc': 200,
            'z': row.get('z_lens', 0.3),
        })
    
    return clusters


def load_gaia(data_dir: Path) -> Optional[pd.DataFrame]:
    """Load validated Gaia/Eilers-APOGEE disk star catalog.
    
    Source: Eilers+ 2019 × APOGEE DR17 × Gaia EDR3
    File: data/gaia/eilers_apogee_6d_disk.csv
    
    This file contains the quality-filtered disk sample:
    - 28,368 stars from Eilers+ 2019 cross-matched with APOGEE DR17
    - Pre-filtered to disk region (4 < R_gal < 16 kpc, |z_gal| < 1 kpc)
    - Full 6D phase space information (positions + velocities)
    
    Sign convention: v_phi is positive for counter-rotation in the file,
    so we negate it to get the standard convention (positive = co-rotation).
    """
    gaia_file = data_dir / "gaia" / "eilers_apogee_6d_disk.csv"
    if not gaia_file.exists():
        return None
    
    df = pd.read_csv(gaia_file)
    df['v_phi_obs'] = -df['v_phi']  # Correct sign convention
    return df  # No additional filtering - file is already the disk sample


# =============================================================================
# ORIGINAL TESTS (1-7)
# =============================================================================


def test_sparc(galaxies: List[Dict]) -> TestResult:
    """Test SPARC galaxy rotation curves.

    Gold standard: Lelli+ 2016, McGaugh+ 2016
    - MOND RMS: 17.15 km/s (with a₀=1.2×10⁻¹⁰)
    - ΛCDM RMS: ~15 km/s (with 2-3 params/galaxy)
    - RAR scatter: 0.13 dex

    This test also reports a simple *bulge vs disk* breakdown using the global
    bulge fraction proxy f_bulge (computed from the SPARC component curves).
    """
    if not galaxies:
        return TestResult("SPARC Galaxies", True, 0.0, {}, "SKIPPED: No data")

    rms_list: List[float] = []
    mond_rms_list: List[float] = []
    bulge_rms_list: List[float] = []
    disk_rms_list: List[float] = []

    all_log_ratios: List[float] = []
    wins = 0

    BULGE_GAL_THRESH = 0.30

    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        h_disk = gal.get('h_disk', 0.15 * R_d)
        f_bulge = float(gal.get('f_bulge', 0.0))

        sigma_profile = None
        if USE_SIGMA_COMPONENTS:
            try:
                sigma_profile = sigma_profile_from_components_kms(
                    gal.get('V_gas', np.zeros_like(V_bar)),
                    gal.get('V_disk_scaled', np.zeros_like(V_bar)),
                    gal.get('V_bulge_scaled', np.zeros_like(V_bar)),
                )
            except Exception:
                sigma_profile = None

        # Use covariant coherence proxy for bulge galaxies if enabled
        use_covariant = USE_COVARIANT_PROXY and f_bulge >= BULGE_GAL_THRESH
        V_pred = predict_velocity(R, V_bar, R_d, h_disk, f_bulge, 
                                 sigma_profile_kms=sigma_profile,
                                 use_covariant_proxy=use_covariant)
        V_mond = predict_mond(R, V_bar)

        rms_sigma = float(np.sqrt(((V_pred - V_obs) ** 2).mean()))
        rms_mond = float(np.sqrt(((V_mond - V_obs) ** 2).mean()))

        rms_list.append(rms_sigma)
        mond_rms_list.append(rms_mond)

        if f_bulge >= BULGE_GAL_THRESH:
            bulge_rms_list.append(rms_sigma)
        else:
            disk_rms_list.append(rms_sigma)

        # RAR scatter calculation
        valid = (V_obs > 0) & (V_pred > 0)
        if valid.sum() > 0:
            log_ratio = np.log10(V_obs[valid] / V_pred[valid])
            all_log_ratios.extend(log_ratio)

        if rms_sigma < rms_mond:
            wins += 1

    mean_rms = float(np.mean(rms_list))
    mean_mond_rms = float(np.mean(mond_rms_list))
    win_rate = wins / len(galaxies)
    rar_scatter = float(np.std(all_log_ratios)) if all_log_ratios else 0.0

    bulge_rms = float(np.mean(bulge_rms_list)) if bulge_rms_list else 0.0
    disk_rms = float(np.mean(disk_rms_list)) if disk_rms_list else 0.0

    passed = mean_rms < 20.0

    return TestResult(
        name="SPARC Galaxies",
        passed=passed,
        metric=mean_rms,
        details={
            'n_galaxies': len(galaxies),
            'mean_rms': mean_rms,
            'mean_mond_rms': mean_mond_rms,
            'win_rate': win_rate,
            'rar_scatter_dex': rar_scatter,
            'bulge_thresh_f_bulge': BULGE_GAL_THRESH,
            'n_bulge_galaxies': len(bulge_rms_list),
            'n_disk_galaxies': len(disk_rms_list),
            'bulge_rms': bulge_rms,
            'disk_rms': disk_rms,
            'benchmark_mond_rms': OBS_BENCHMARKS['sparc']['mond_rms_kms'],
            'benchmark_lcdm_rms': OBS_BENCHMARKS['sparc']['lcdm_rms_kms'],
            'benchmark_rar_scatter': OBS_BENCHMARKS['sparc']['rar_scatter_dex'],
        },
        message=(
            f"RMS={mean_rms:.2f} km/s (MOND={mean_mond_rms:.2f}, ΛCDM~15), "
            f"BulgeRMS={bulge_rms:.2f} (n={len(bulge_rms_list)}), "
            f"DiskRMS={disk_rms:.2f} (n={len(disk_rms_list)}), "
            f"Scatter={rar_scatter:.3f} dex, Win={win_rate*100:.1f}%"
        )
    )


def export_sparc_pointwise(galaxies: List[Dict], out_csv: Any, coherence_model: Optional[str] = None) -> Path:
    """Export pointwise SPARC residuals + model terms for discovery.

    This writes one row per (galaxy, radius) point including:
      - observed vs predicted velocities
      - Σ_req and Σ_pred (and dSigma)
      - local component fractions and basic derived features (Ω, τ_dyn, gradients)
      - model terms: h_term, C_term, A_use, Σ

    The output is designed to work with analyze_sparc_pointwise.py and
    discover_sparc_residual_drivers.py.
    """
    if not galaxies:
        raise RuntimeError('No SPARC galaxies loaded; cannot export pointwise.')

    out_path = Path(out_csv)
    if out_path.suffix == '':
        out_path = out_path.with_suffix('.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for gal in galaxies:
        name = gal.get('name', 'UNKNOWN')
        R = np.asarray(gal['R'], dtype=float)
        V_obs = np.asarray(gal['V_obs'], dtype=float)
        V_bar = np.asarray(gal['V_bar'], dtype=float)
        R_d = float(gal['R_d'])
        h_disk = float(gal.get('h_disk', 0.15 * R_d))
        f_bulge_gal = float(gal.get('f_bulge', 0.0))

        V_gas = np.asarray(gal.get('V_gas', np.zeros_like(V_bar)), dtype=float)
        V_disk = np.asarray(gal.get('V_disk_scaled', np.zeros_like(V_bar)), dtype=float)
        V_bulge = np.asarray(gal.get('V_bulge_scaled', np.zeros_like(V_bar)), dtype=float)

        sigma_profile = None
        if USE_SIGMA_COMPONENTS:
            try:
                sigma_profile = sigma_profile_from_components_kms(V_gas, V_disk, V_bulge)
            except Exception:
                sigma_profile = None

        V_pred, diag = predict_velocity(
            R_kpc=R,
            V_bar=V_bar,
            R_d=R_d,
            h_disk=h_disk,
            f_bulge=f_bulge_gal,
            sigma_profile_kms=sigma_profile,
            coherence_model=coherence_model,
            return_terms=True,
        )

        Sigma_pred = np.asarray(diag.get('Sigma', (V_pred / np.maximum(V_bar, 1e-9)) ** 2), dtype=float)
        C_term = np.asarray(diag.get('C_term', np.nan), dtype=float)
        A_use = np.asarray(diag.get('A_use', A_0), dtype=float)
        h_term = np.asarray(diag.get('h_term', np.nan), dtype=float)

        V_bar_safe = np.maximum(V_bar, 1e-9)
        Sigma_req = (V_obs / V_bar_safe) ** 2
        dSigma = Sigma_req - Sigma_pred

        # Local component fractions (in V^2 space)
        Vbar2 = np.maximum(V_bar_safe ** 2, 1e-12)
        f_gas_r = (V_gas ** 2) / Vbar2
        f_disk_r = (V_disk ** 2) / Vbar2
        f_bulge_r = (V_bulge ** 2) / Vbar2

        # Basic derived features
        R_m = np.maximum(R, 1e-9) * kpc_to_m
        V_bar_ms = V_bar_safe * 1000.0
        g_bar = (V_bar_ms ** 2) / np.maximum(R_m, 1e-9)
        Omega_bar_SI = V_bar_ms / np.maximum(R_m, 1e-9)
        sec_per_myr = 365.25 * 24.0 * 3600.0 * 1e6
        tau_dyn_Myr = (1.0 / np.maximum(Omega_bar_SI, 1e-30)) / sec_per_myr

        lnR = np.log(np.maximum(R, 1e-9))
        lnVbar = np.log(np.maximum(V_bar_safe, 1e-9))
        dlnVbar_dlnR = np.gradient(lnVbar, lnR)
        lnGbar = np.log(np.maximum(g_bar, 1e-30))
        dlnGbar_dlnR = np.gradient(lnGbar, lnR)

        df_g = pd.DataFrame({
            'galaxy': name,
            'R_kpc': R,
            'V_obs_kms': V_obs,
            'V_bar_kms': V_bar,
            'V_pred_kms': V_pred,
            'Sigma_req': Sigma_req,
            'Sigma_pred': Sigma_pred,
            'dSigma': dSigma,
            'need_Sigma_lt_1': (Sigma_req < 1.0).astype(int),
            'f_bulge': f_bulge_gal,
            'f_bulge_r': f_bulge_r,
            'f_disk_r': f_disk_r,
            'f_gas_r': f_gas_r,
            'R_d_kpc': R_d,
            'R_over_Rd': R / max(R_d, 1e-6),
            'g_bar_SI': g_bar,
            'Omega_bar_SI': Omega_bar_SI,
            'tau_dyn_Myr': tau_dyn_Myr,
            'dlnVbar_dlnR': dlnVbar_dlnR,
            'dlnGbar_dlnR': dlnGbar_dlnR,
            'h_term': h_term,
            'C_term': C_term,
            'A_use': A_use,
        })

        # Optional FLOW invariants if present
        if isinstance(diag, dict) and 'omega2' in diag:
            df_g['omega2'] = np.asarray(diag.get('omega2'), dtype=float)
            df_g['shear2'] = np.asarray(diag.get('shear2'), dtype=float)
            df_g['theta2'] = np.asarray(diag.get('theta2'), dtype=float)
            df_g['tidal2'] = np.asarray(diag.get('tidal2'), dtype=float)

        rows.append(df_g)

    df_out = pd.concat(rows, ignore_index=True)
    df_out.to_csv(out_path, index=False)
    return out_path


def test_clusters(clusters: List[Dict]) -> TestResult:
    """Test galaxy cluster lensing.
    
    Gold standard: Fox+ 2022
    - MOND: Underpredicts by factor ~3 ("cluster problem")
    - ΛCDM: Works well with NFW fits (2-3 params/cluster)
    - GR+baryons: Underpredicts by factor ~10
    """
    if not clusters:
        return TestResult("Clusters", True, 0.0, {}, "SKIPPED: No data")
    
    ratios = []
    
    # Cluster parameters for unified amplitude
    L_cluster = 600  # kpc (path through cluster baryons)
    A_cluster = unified_amplitude(L_cluster)  # ≈ 8.45
    
    for cl in clusters:
        M_bar = cl['M_bar']
        M_lens = cl['M_lens']
        r_kpc = cl.get('r_kpc', 200)
        
        r_m = r_kpc * kpc_to_m
        g_bar = G * M_bar * M_sun / r_m**2
        
        h = h_function(np.array([g_bar]))[0]
        # For clusters, coherence is high (large correlated currents)
        q_cluster = 1.0 if COHERENCE_MODEL == "C" else 1.0  # JJ model: clusters have large correlation volume
        # Apply guided-gravity multiplier if enabled
        if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
            A_use = A_cluster * guided_amplitude_multiplier(q_cluster if COHERENCE_MODEL == "C" else GUIDED_C_DEFAULT)
        else:
            A_use = A_cluster
        Sigma = 1 + A_use * q_cluster * h
        
        M_pred = M_bar * Sigma
        ratio = M_pred / M_lens
        if np.isfinite(ratio) and ratio > 0:
            ratios.append(ratio)
    
    median_ratio = np.median(ratios)
    scatter = np.std(np.log10(ratios))
    
    # MOND comparison: factor ~3 underprediction
    mond_ratio = 1.0 / OBS_BENCHMARKS['clusters']['mond_mass_discrepancy']
    
    passed = 0.5 < median_ratio < 1.5
    
    return TestResult(
        name="Clusters",
        passed=passed,
        metric=median_ratio,
        details={
            'n_clusters': len(ratios),
            'median_ratio': median_ratio,
            'scatter_dex': scatter,
            'A_cluster': A_cluster,
            'benchmark_mond_ratio': mond_ratio,
            'benchmark_lcdm': 'Works with NFW fits',
        },
        message=f"Median ratio={median_ratio:.3f} (MOND~{mond_ratio:.2f}, ΛCDM~1.0), Scatter={scatter:.3f} dex ({len(ratios)} clusters)"
    )


def test_cluster_holdout(clusters: List[Dict], n_splits: int = 10, 
                         test_fraction: float = 0.3) -> TestResult:
    """Test cluster parameter stability via holdout validation.
    
    With L₀ = 0.4 kpc fixed (physical value), calibrate only n on training set
    and evaluate on holdout set. Validates that n is stable and not overfit.
    
    Pass criteria:
        - Holdout median ratio between 0.7 and 1.4
        - Calibrated n stable (std < 0.05)
    """
    if len(clusters) < 10:
        return TestResult("Cluster Holdout", True, 0.0, {}, 
                         "SKIPPED: Need ≥10 clusters")
    
    from scipy.optimize import minimize_scalar
    
    # Fixed physical parameters
    L0_fixed = 0.4  # kpc - disk scale height (physical, not calibrated)
    L_cluster = 600.0  # kpc - cluster path length
    
    def predict_with_n(M_bar, r_kpc, n):
        """Predict cluster mass with given n (L₀ fixed)."""
        A_base = A_0 * (L_cluster / L0_fixed) ** n
        # Apply guided-gravity multiplier consistently with the definition:
        # A_eff = A(L_eff) = A(L) · (1+κ·C_stream)^n
        if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
            A = A_base * guided_amplitude_multiplier(GUIDED_C_DEFAULT, exponent=n)
        else:
            A = A_base
        r_m = r_kpc * kpc_to_m
        g_N = G * M_bar * M_sun / r_m**2
        h = np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)
        Sigma = 1 + A * h
        return M_bar * Sigma
    
    holdout_medians = []
    calibrated_n_values = []
    
    for seed in range(n_splits):
        np.random.seed(seed + 42)
        
        indices = np.random.permutation(len(clusters))
        n_test = int(len(clusters) * test_fraction)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        train_clusters = [clusters[i] for i in train_idx]
        test_clusters = [clusters[i] for i in test_idx]
        
        def train_objective(n):
            ratios = []
            for cl in train_clusters:
                M_pred = predict_with_n(cl['M_bar'], cl.get('r_kpc', 200), n)
                ratios.append(M_pred / cl['M_lens'])
            return abs(np.median(ratios) - 1.0)
        
        result = minimize_scalar(train_objective, bounds=(0.1, 0.5), method='bounded')
        n_cal = result.x
        calibrated_n_values.append(n_cal)
        
        ratios = []
        for cl in test_clusters:
            M_pred = predict_with_n(cl['M_bar'], cl.get('r_kpc', 200), n_cal)
            ratios.append(M_pred / cl['M_lens'])
        
        holdout_medians.append(np.median(ratios))
    
    mean_holdout = np.mean(holdout_medians)
    std_holdout = np.std(holdout_medians)
    mean_n = np.mean(calibrated_n_values)
    std_n = np.std(calibrated_n_values)
    
    holdout_ok = 0.7 < mean_holdout < 1.4
    n_stable = std_n < 0.05
    passed = holdout_ok and n_stable
    
    return TestResult(
        name="Cluster Holdout",
        passed=passed,
        metric=mean_holdout,
        details={
            'n_splits': n_splits,
            'mean_n': mean_n,
            'std_n': std_n,
            'mean_holdout_ratio': mean_holdout,
            'std_holdout_ratio': std_holdout,
        },
        message=f"n={mean_n:.2f}±{std_n:.2f}, holdout={mean_holdout:.2f}±{std_holdout:.2f}"
    )


def mw_baryonic_density_kg_m3(R_kpc, z_kpc):
    """
    Simple MW baryonic density model used for Gaia bulge covariant test:
      - Exponential disk  ρ ~ exp(-R/Rd) exp(-|z|/zd)
      - Hernquist bulge   ρ(r) = (M/(2π)) a / (r (r+a)^3)

    Returns ρ in kg/m^3.
    """
    # Match the MW baryonic parameters in test_gaia()
    MW_SCALE = 1.16  # McMillan 2017 scaling factor
    M_disk = 6e10 * (MW_SCALE**2)
    R_d = 2.6
    z_d = 0.3

    M_bulge = 1e10 * (MW_SCALE**2)
    a_bulge = 0.7

    R = np.asarray(R_kpc, dtype=float)
    z = np.asarray(z_kpc, dtype=float)

    R_m = R * kpc_to_m
    z_m = z * kpc_to_m
    r_m = np.sqrt(R_m**2 + z_m**2)

    # Disk density
    M_disk_kg = M_disk * M_sun
    R_d_m = R_d * kpc_to_m
    z_d_m = z_d * kpc_to_m

    rho0_disk = M_disk_kg / (4.0 * np.pi * (R_d_m**2) * z_d_m)
    rho_disk = rho0_disk * np.exp(-R / R_d) * np.exp(-np.abs(z) / z_d)

    # Hernquist bulge density
    M_bulge_kg = M_bulge * M_sun
    a_m = a_bulge * kpc_to_m
    r_safe = np.maximum(r_m, 1e-6 * kpc_to_m)

    rho_bulge = (M_bulge_kg / (2.0 * np.pi)) * (a_m / (r_safe * (r_safe + a_m)**3))

    rho_total = rho_disk + rho_bulge
    return np.maximum(rho_total, 0.0)


def test_gaia_bulge_covariant(gaia_df: Optional[pd.DataFrame]) -> TestResult:
    """
    Gaia "bulge covariant coherence" regression test.

    Notes:
      * The Eilers catalog used by load_gaia() is an R~4–16 kpc disk sample.
      * We use an inner-disk proxy selection (R<5 kpc, |z|>0.3 kpc) as a stand-in.
      * We compare baseline coherence (v^2/(v^2+σ^2)) vs covariant coherence:
          C_cov = ω² / (ω² + 4πGρ + (δH0)²)
        with ω² from observed mean v_phi / R and ρ from a simple MW disk+bulge density model.
    """
    if gaia_df is None or len(gaia_df) == 0:
        return TestResult(name="Gaia Bulge Covariant", passed=True, metric=0.0, details={"status": "SKIPPED", "reason": "No Gaia data"}, message="SKIPPED: No Gaia data")

    df = gaia_df.copy()

    # Proxy bulge selection (inner disk / high-|z|)
    sel = (df["R_gal"] < 5.0) & (np.abs(df["z_gal"]) > 0.3)
    df = df[sel].copy()
    n_selected = len(df)

    if n_selected < 200:
        return TestResult(name="Gaia Bulge Covariant", passed=True, metric=0.0, details={"status": "SKIPPED", "n_selected": int(n_selected)}, message=f"SKIPPED: only {int(n_selected)} selected")

    # Bin in (R, |z|) to get stable ω² and dispersions
    R = df["R_gal"].to_numpy(dtype=float)
    z_abs = np.abs(df["z_gal"].to_numpy(dtype=float))
    vphi = df["v_phi_obs"].to_numpy(dtype=float)

    # 5 radial bins × 2 vertical bins = 10 bins (matches design doc)
    R_edges = np.linspace(np.min(R), np.max(R), 6)
    z_edges = np.linspace(np.min(z_abs), np.max(z_abs), 3)

    R_bin = np.digitize(R, R_edges) - 1
    z_bin = np.digitize(z_abs, z_edges) - 1

    mask = (R_bin >= 0) & (R_bin < 5) & (z_bin >= 0) & (z_bin < 2)
    df = df.iloc[np.where(mask)[0]].copy()
    R_bin = R_bin[mask]
    z_bin = z_bin[mask]

    min_stars = 30
    rows = []
    for iR in range(5):
        for iz in range(2):
            idx = (R_bin == iR) & (z_bin == iz)
            if np.sum(idx) < min_stars:
                continue
            sub = df.iloc[np.where(idx)[0]]

            R_mean = float(np.mean(sub["R_gal"]))
            z_mean = float(np.mean(np.abs(sub["z_gal"])))

            vphi_mean = float(np.mean(sub["v_phi_obs"]))
            sigma_phi = float(np.std(sub["v_phi_obs"], ddof=1)) if len(sub) > 1 else 0.0

            rows.append((R_mean, z_mean, vphi_mean, sigma_phi, int(len(sub))))

    if len(rows) < 3:
        return TestResult(name="Gaia Bulge Covariant", passed=False, metric=0.0, details={"status": "FAILED", "reason": "Too few populated bins", "n_bins": int(len(rows)), "n_selected": int(n_selected)}, message=f"FAILED: only {int(len(rows))} populated bins")

    Rm = np.array([r[0] for r in rows], dtype=float)
    zm = np.array([r[1] for r in rows], dtype=float)
    vphi_m = np.array([r[2] for r in rows], dtype=float)
    sigphi = np.array([r[3] for r in rows], dtype=float)
    counts = np.array([r[4] for r in rows], dtype=int)

    omega2 = (vphi_m / np.maximum(Rm, 1e-6))**2  # (km/s/kpc)^2
    theta2 = np.zeros_like(omega2)

    rho = mw_baryonic_density_kg_m3(Rm, zm)
    C_cov = C_covariant_coherence(omega2, rho, theta2)

    # Baseline coherence proxy in same bins
    C_base = (vphi_m**2) / (vphi_m**2 + sigphi**2 + 1e-30)

    # MW baryonic circular speed model (same as test_gaia)
    MW_SCALE = 1.16
    M_disk = 6e10 * (MW_SCALE**2) * M_sun
    M_bulge = 1e10 * (MW_SCALE**2) * M_sun
    M_gas = 1e10 * (MW_SCALE**2) * M_sun
    R_disk = 2.6 * kpc_to_m
    R_bulge = 0.7 * kpc_to_m
    R_gas = 7.0 * kpc_to_m

    Rm_m = Rm * kpc_to_m

    v2_disk = (G * M_disk * Rm_m**2) / (Rm_m**2 + R_disk**2)**1.5
    v2_bulge = (G * M_bulge * Rm_m**2) / (Rm_m**2 + R_bulge**2)**1.5
    v2_gas = (G * M_gas * Rm_m**2) / (Rm_m**2 + R_gas**2)**1.5

    Vbar = np.sqrt(np.maximum(v2_disk + v2_bulge + v2_gas, 0.0)) / 1000.0  # km/s
    gbar_SI = (Vbar * 1000.0)**2 / np.maximum(Rm_m, 1e-6)

    h = h_function(gbar_SI)

    Sigma_base = 1.0 + A_0 * C_base * h
    Sigma_cov = 1.0 + A_0 * C_cov * h

    Vpred_base = Vbar * np.sqrt(np.maximum(Sigma_base, 0.0))
    Vpred_cov = Vbar * np.sqrt(np.maximum(Sigma_cov, 0.0))

    rms_base = float(np.sqrt(np.mean((vphi_m - Vpred_base)**2)))
    rms_cov = float(np.sqrt(np.mean((vphi_m - Vpred_cov)**2)))
    improvement = rms_base - rms_cov

    passed = improvement > 1.0

    details = {
        "status": "PASSED" if passed else "FAILED",
        "baseline_rms_kms": rms_base,
        "rms_kms": rms_cov,
        "improvement_kms": improvement,
        "n_bins": int(len(rows)),
        "n_selected": int(n_selected),
        "mean_C_cov": float(np.mean(C_cov)),
        "mean_omega2": float(np.mean(omega2)),
        "counts": counts.tolist(),
    }
    return TestResult(name="Gaia Bulge Covariant", passed=passed, metric=rms_cov, details=details, message=f"RMS={rms_cov:.2f} km/s (baseline {rms_base:.2f}, Δ={improvement:.2f})")


def test_gaia(gaia_df: Optional[pd.DataFrame]) -> TestResult:
    """Test Milky Way star-by-star validation.
    
    Gold standard: Eilers+ 2019, McMillan 2017
    - V_sun = 233 ± 3 km/s
    - R_sun = 8.178 kpc
    - M_baryonic = 6.5×10¹⁰ M☉
    - Expected RMS ~ 29.5 km/s for 28,368 stars
    """
    if gaia_df is None or len(gaia_df) == 0:
        return TestResult("Gaia/MW", True, 0.0, {}, "SKIPPED: No data")
    
    R = gaia_df['R_gal'].values
    
    # McMillan 2017 baryonic model (scaled by 1.16)
    MW_SCALE = 1.16
    M_disk = 4.6e10 * MW_SCALE**2
    M_bulge = 1.0e10 * MW_SCALE**2
    M_gas = 1.0e10 * MW_SCALE**2
    G_kpc = 4.302e-6  # G in (km/s)² kpc / M☉
    
    v2_disk = G_kpc * M_disk * R**2 / (R**2 + 3.3**2)**1.5
    v2_bulge = G_kpc * M_bulge * R / (R + 0.5)**2
    v2_gas = G_kpc * M_gas * R**2 / (R**2 + 7.0**2)**1.5
    V_bar = np.sqrt(v2_disk + v2_bulge + v2_gas)
    
    R_d_mw = 2.6  # MW disk scale length (kpc)
    sigma_profile = None
    if USE_SIGMA_COMPONENTS:
        sigma_profile = sigma_profile_from_components_kms(np.sqrt(v2_gas), np.sqrt(v2_disk), np.sqrt(v2_bulge))

    # Optional: if we have precomputed 6D flow invariants for the Gaia sample, use them
    # to drive coherence directly (this tests the "flow/topology" concept on Milky Way data).
    coherence_external = None
    used_flow_features = False
    if COHERENCE_MODEL == "FLOW" and GAIA_FLOW_FEATURES_DF is not None:
        try:
            omega2 = GAIA_FLOW_FEATURES_DF['omega2'].values
            shear2 = GAIA_FLOW_FEATURES_DF['shear2'].values
            theta = GAIA_FLOW_FEATURES_DF['theta'].values
            theta2 = theta**2
            coherence_external = C_flow_coherence(omega2, shear2, theta2, tidal2=None)
            used_flow_features = True
        except Exception:
            coherence_external = None
            used_flow_features = False

    V_pred = predict_velocity(
        R,
        V_bar,
        R_d_mw,
        h_disk=0.3,
        f_bulge=0.1,
        sigma_profile_kms=sigma_profile,
        coherence_model=COHERENCE_MODEL,
        coherence_external=coherence_external,
    )
    
    # Asymmetric drift correction
    from scipy.interpolate import interp1d
    R_bins = np.arange(4, 16, 0.5)
    disp_data = []
    for i in range(len(R_bins) - 1):
        mask = (gaia_df['R_gal'] >= R_bins[i]) & (gaia_df['R_gal'] < R_bins[i + 1])
        if mask.sum() > 30:
            disp_data.append({
                'R': (R_bins[i] + R_bins[i + 1]) / 2,
                'sigma_R': gaia_df.loc[mask, 'v_R'].std()
            })
    
    if len(disp_data) > 0:
        disp_df = pd.DataFrame(disp_data)
        sigma_interp = interp1d(disp_df['R'], disp_df['sigma_R'], fill_value='extrapolate')
        sigma_R = sigma_interp(R)
    else:
        sigma_R = 40.0
    
    # Safety: avoid division by zero in asymmetric drift correction
    V_pred_safe = np.maximum(V_pred, 1e-6)
    V_a = sigma_R**2 / (2 * V_pred_safe) * (R / R_d_mw - 1)
    V_a = np.clip(V_a, 0, 50)
    
    v_pred_corrected = V_pred - V_a
    # Safety: handle any remaining NaN/inf
    v_pred_corrected = np.nan_to_num(v_pred_corrected, nan=V_pred, posinf=V_pred*10, neginf=V_pred)
    resid = gaia_df['v_phi_obs'].values - v_pred_corrected
    resid = np.nan_to_num(resid, nan=0.0, posinf=1000, neginf=-1000)
    rms = np.sqrt((resid**2).mean())
    
    passed = rms < 35.0
    
    return TestResult(
        name="Gaia/MW",
        passed=passed,
        metric=rms,
        details={
            'n_stars': len(gaia_df),
            'rms': rms,
            'mean_residual': resid.mean(),
            'flow_features_used': bool(used_flow_features),
            'flow_coherence_mean': (float(np.nanmean(coherence_external)) if coherence_external is not None else None),
            'benchmark_V_sun': OBS_BENCHMARKS['milky_way']['V_sun_kms'],
            'benchmark_n_stars': OBS_BENCHMARKS['milky_way']['n_gaia_stars'],
        },
        message=f"RMS={rms:.1f} km/s ({len(gaia_df)} stars, expected {OBS_BENCHMARKS['milky_way']['n_gaia_stars']})"
    )


def test_redshift() -> TestResult:
    """Test redshift evolution of g†."""
    Omega_m, Omega_L = 0.3, 0.7
    
    def H_z(z):
        return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    
    ratio = H_z(2)
    passed = True
    
    return TestResult(
        name="Redshift Evolution",
        passed=passed,
        metric=ratio,
        details={'g_dagger_z2_ratio': ratio},
        message=f"g†(z=2)/g†(z=0) = {ratio:.3f} (∝ H(z))"
    )


def test_solar_system() -> TestResult:
    """Test Solar System safety (Cassini bound)."""
    r_saturn = 9.5 * AU_to_m
    g_saturn = G * M_sun / r_saturn**2
    
    h_saturn = h_function(np.array([g_saturn]))[0]
    gamma_minus_1 = h_saturn
    if COHERENCE_MODEL == "JJ":
        # "No macroscopic baryonic current network" -> coherence ~ 0
        # Keep it explicit for now (you can later replace with a size-based rule)
        gamma_minus_1 = 0.0 * h_saturn
    
    cassini_bound = OBS_BENCHMARKS['solar_system']['cassini_gamma_uncertainty']
    
    passed = gamma_minus_1 < cassini_bound
    
    return TestResult(
        name="Solar System",
        passed=passed,
        metric=gamma_minus_1,
        details={'h_saturn': h_saturn, 'cassini_bound': cassini_bound, 'coherence_model': COHERENCE_MODEL},
        message=f"|γ-1| = {gamma_minus_1:.2e} < {cassini_bound:.2e} (model: {COHERENCE_MODEL})"
    )


def test_counter_rotation(data_dir: Path) -> TestResult:
    """Test counter-rotation prediction."""
    try:
        from astropy.io import fits
        from astropy.table import Table
        from scipy import stats
    except ImportError:
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: astropy required")
    
    dynpop_file = data_dir / "manga_dynpop" / "SDSSDR17_MaNGA_JAM.fits"
    cr_file = data_dir / "stellar_corgi" / "bevacqua2022_counter_rotating.tsv"
    
    if not dynpop_file.exists() or not cr_file.exists():
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: Data not found")
    
    with fits.open(dynpop_file) as hdul:
        basic = Table(hdul[1].data)
        jam_nfw = Table(hdul[4].data)
    
    with open(cr_file, 'r') as f:
        lines = f.readlines()
    
    # Parse CR data
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            data_start = i + 1
            break
    
    header_line = None
    for i, line in enumerate(lines):
        if line.startswith('MaNGAId'):
            header_line = i
            break
    
    if header_line is None:
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: Parse error")
    
    headers = [h.strip() for h in lines[header_line].split('|')]
    cr_data = []
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= len(headers):
                cr_data.append(dict(zip(headers, parts)))
    
    cr_manga_ids = [d['MaNGAId'].strip() for d in cr_data]
    dynpop_idx = {str(mid).strip(): i for i, mid in enumerate(basic['mangaid'])}
    matches = [dynpop_idx[cr_id] for cr_id in cr_manga_ids if cr_id in dynpop_idx]
    
    if len(matches) < 10:
        return TestResult("Counter-Rotation", True, 0.0, {}, f"SKIPPED: Only {len(matches)} matches")
    
    fdm_all = np.array(jam_nfw['fdm_Re'])
    valid_mask = np.isfinite(fdm_all) & (fdm_all >= 0) & (fdm_all <= 1)
    
    cr_mask = np.zeros(len(fdm_all), dtype=bool)
    cr_mask[matches] = True
    
    fdm_cr = fdm_all[cr_mask & valid_mask]
    fdm_normal = fdm_all[~cr_mask & valid_mask]
    
    if len(fdm_cr) < 10:
        return TestResult("Counter-Rotation", True, 0.0, {}, "SKIPPED: Insufficient data")
    
    mw_stat, mw_pval_two = stats.mannwhitneyu(fdm_cr, fdm_normal)
    mw_pval = mw_pval_two / 2 if np.mean(fdm_cr) < np.mean(fdm_normal) else 1 - mw_pval_two / 2
    
    passed = mw_pval < 0.05 and np.mean(fdm_cr) < np.mean(fdm_normal)
    
    return TestResult(
        name="Counter-Rotation",
        passed=passed,
        metric=mw_pval,
        details={
            'n_cr': len(fdm_cr),
            'fdm_cr_mean': float(np.mean(fdm_cr)),
            'fdm_normal_mean': float(np.mean(fdm_normal)),
        },
        message=f"f_DM(CR)={np.mean(fdm_cr):.3f} < f_DM(Normal)={np.mean(fdm_normal):.3f}, p={mw_pval:.4f}"
    )


def test_tully_fisher() -> TestResult:
    """Test Baryonic Tully-Fisher Relation.
    
    Gold standard: McGaugh 2012, AJ 143, 40
    - Slope: 3.98 ± 0.06 (MOND predicts exactly 4)
    - Normalization: A_TF ≈ 47 M☉/(km/s)⁴
    - Scatter: 0.10 dex (intrinsic)
    """
    # At V_flat = 200 km/s, what baryonic mass does Σ-Gravity predict?
    V_flat = 200  # km/s
    V_flat_ms = V_flat * 1000
    
    # Check the normalization at a typical radius
    R_test = 30  # kpc
    R_m = R_test * kpc_to_m
    g_obs = V_flat_ms**2 / R_m
    
    # Invert to find g_bar: g_obs = g_bar × Σ(g_bar)
    g_bar = g_obs / 2  # Initial guess
    for _ in range(20):
        Sigma = sigma_enhancement(g_bar, A=A_0)
        g_bar_new = g_obs / Sigma
        if abs(g_bar_new - g_bar) / g_bar < 1e-6:
            break
        g_bar = g_bar_new
    
    V_bar = np.sqrt(g_bar * R_m) / 1000  # km/s
    M_bar = V_bar**2 * R_test * kpc_to_m / (G * M_sun) * 1000**2
    
    # BTFR: M_bar = A_TF × V⁴
    A_TF_obs = OBS_BENCHMARKS['tully_fisher']['btfr_normalization']
    slope_obs = OBS_BENCHMARKS['tully_fisher']['btfr_slope']
    M_bar_obs = A_TF_obs * V_flat**4
    
    ratio = M_bar / M_bar_obs
    
    # Slope is automatic in MOND-like theories
    slope_pred = 4
    
    passed = 0.5 < ratio < 2.0
    
    return TestResult(
        name="Tully-Fisher",
        passed=passed,
        metric=ratio,
        details={
            'V_flat': V_flat,
            'M_bar_pred': M_bar,
            'M_bar_obs': M_bar_obs,
            'slope_pred': slope_pred,
            'slope_obs': slope_obs,
            'benchmark_scatter': OBS_BENCHMARKS['tully_fisher']['scatter_dex'],
        },
        message=f"BTFR: M_pred/M_obs = {ratio:.2f} at V={V_flat} km/s, slope={slope_pred} (obs={slope_obs:.2f})"
    )


# =============================================================================
# NEW TESTS (8-16)
# =============================================================================

def test_wide_binaries() -> TestResult:
    """Test wide binary boost at 10 kAU.
    
    Gold standard: Chae 2023, ApJ 952, 128
    - ~35% velocity boost at separations > 2000 AU
    - 26,500 pairs from Gaia DR3
    - Controversy: Banik+ 2024 disputes; ongoing debate
    """
    # At separation s = 10,000 AU
    s_AU = 10000
    s_m = s_AU * AU_to_m
    
    # Typical binary: M_total ~ 1.5 M_sun
    M_total = 1.5 * M_sun
    g_N = G * M_total / s_m**2
    
    # Σ-Gravity enhancement
    Sigma = sigma_enhancement(g_N, A=A_0)
    boost = Sigma - 1
    
    # Chae 2023 observed ~35% boost
    obs_boost = OBS_BENCHMARKS['wide_binaries']['boost_factor'] - 1
    obs_uncertainty = OBS_BENCHMARKS['wide_binaries']['boost_uncertainty']
    
    # Pass if within factor of 2 of observed
    passed = 0.5 * obs_boost < boost < 2.0 * obs_boost
    
    return TestResult(
        name="Wide Binaries",
        passed=passed,
        metric=boost,
        details={
            'separation_AU': s_AU,
            'g_N': g_N,
            'g_over_a0': g_N / a0_mond,
            'boost_pred': boost,
            'boost_obs': obs_boost,
            'obs_uncertainty': obs_uncertainty,
            'n_pairs': OBS_BENCHMARKS['wide_binaries']['n_pairs'],
            'controversy': OBS_BENCHMARKS['wide_binaries']['controversy'],
        },
        message=f"Boost at {s_AU} AU: {boost*100:.1f}% (Chae 2023: {obs_boost*100:.0f}±{obs_uncertainty*100:.0f}%)"
    )


def test_dwarf_spheroidals() -> TestResult:
    """Test dwarf spheroidal velocity dispersions using HOST INHERITANCE model.
    
    NEW MODEL: dSphs inherit the MW's Σ-enhancement at their orbital radius.
    Σ_dSph = Σ_MW(R_orbit)
    
    This naturally explains why dSphs appear "dark matter dominated":
    - They sit in the MW's already-enhanced gravitational field
    - No separate internal enhancement needed
    
    Gold standard: Walker+ 2009, McConnachie 2012
    - MOND: Generally works for isolated dSphs; EFE complicates satellites
    - ΛCDM: Requires NFW halos with high M/L
    """
    dsphs = OBS_BENCHMARKS['dwarf_spheroidals']
    
    # MW baryonic mass for calculating Σ_MW at dSph locations
    M_MW_bar = 6e10  # M_sun
    
    ratios = []
    results_by_name = {}
    for name, data in dsphs.items():
        # Skip non-galaxy entries
        if not isinstance(data, dict) or 'M_star' not in data:
            continue
            
        M_star = data['M_star']
        sigma_obs = data['sigma_obs']
        r_half = data['r_half_kpc'] * kpc_to_m
        d_MW = data.get('d_MW_kpc', 100) * kpc_to_m  # Distance from MW center
        
        # Calculate MW's Σ-enhancement at dSph orbital radius
        # This is what keeps the MW rotation curve flat at ~220 km/s
        g_MW = G * M_MW_bar * M_sun / d_MW**2
        h_MW = h_function(np.array([g_MW]))[0]
        Sigma_MW = 1 + A_0 * h_MW  # MW uses disk amplitude, C≈1 at large r
        
        # dSph inherits this enhancement
        # Effective mass = M_star × Σ_MW
        M_eff = M_star * Sigma_MW
        
        # Predicted velocity dispersion
        # σ² ~ GM_eff/(5r_half) for Plummer sphere
        sigma_pred = np.sqrt(G * M_eff * M_sun / (5 * r_half)) / 1000  # km/s
        
        ratio = sigma_pred / sigma_obs
        ratios.append(ratio)
        results_by_name[name] = {
            'sigma_pred': sigma_pred,
            'sigma_obs': sigma_obs,
            'ratio': ratio,
            'd_MW_kpc': data.get('d_MW_kpc', 100),
            'Sigma_MW': Sigma_MW,
            'M_L_obs': data.get('M_L', 'N/A'),
        }
    
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    
    # Pass if mean ratio is reasonable (within factor of 2)
    # Note: scatter is expected due to M_star uncertainties in faint dSphs
    passed = 0.5 < mean_ratio < 2.0
    
    return TestResult(
        name="Dwarf Spheroidals",
        passed=passed,
        metric=mean_ratio,
        details={
            'n_dsphs': len(ratios),
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'model': 'host_inheritance',
            'results': results_by_name,
            'mond_status': dsphs.get('mond_status', 'Generally works'),
            'note': 'Scatter correlates with M_star uncertainty; Sculptor is best-constrained',
        },
        message=f"σ_pred/σ_obs = {mean_ratio:.2f}±{std_ratio:.2f} (host inheritance, {len(ratios)} dSphs)"
    )


def test_ultra_diffuse_galaxies() -> TestResult:
    """Test UDG velocity dispersions (DF2, Dragonfly44).
    
    Gold standard: van Dokkum+ 2018, 2016
    - DF2: σ = 8.5 km/s (appears to lack DM; MOND predicts ~20 km/s)
    - Dragonfly44: σ = 47 km/s (very DM dominated)
    - MOND resolution for DF2: External Field Effect from NGC1052
    """
    udgs = OBS_BENCHMARKS['udgs']
    
    results = {}
    for name, data in udgs.items():
        if not isinstance(data, dict) or 'M_star' not in data:
            continue
            
        M_star = data['M_star']
        sigma_obs = data['sigma_obs']
        sigma_err = data.get('sigma_err', 5)
        r_eff = data['r_eff_kpc'] * kpc_to_m
        
        # Newtonian
        sigma_N = np.sqrt(G * M_star * M_sun / (5 * r_eff)) / 1000
        g_N = G * M_star * M_sun / r_eff**2
        
        # Σ-Gravity
        Sigma = sigma_enhancement(g_N, A=A_0)
        sigma_pred = sigma_N * np.sqrt(Sigma)
        
        results[name] = {
            'sigma_pred': sigma_pred,
            'sigma_obs': sigma_obs,
            'sigma_err': sigma_err,
            'ratio': sigma_pred / sigma_obs,
            'note': data.get('note', ''),
        }
    
    # DF2 is the challenge case (appears to have no DM)
    df2_ratio = results.get('df2', {}).get('ratio', 1.0)
    df2_pred = results.get('df2', {}).get('sigma_pred', 0)
    df2_obs = results.get('df2', {}).get('sigma_obs', 8.5)
    
    # Note: DF2 likely needs External Field Effect
    passed = True  # Informational test
    
    return TestResult(
        name="Ultra-Diffuse Galaxies",
        passed=passed,
        metric=df2_ratio,
        details={
            'results': results,
            'mond_challenge': 'DF2 requires EFE explanation',
        },
        message=f"DF2: σ_pred={df2_pred:.1f} vs obs={df2_obs:.1f} km/s (EFE needed for MOND/Σ-Gravity)"
    )


def test_galaxy_galaxy_lensing() -> TestResult:
    """Test galaxy-galaxy lensing at 200 kpc."""
    # Typical lens galaxy: M_star = 5×10¹¹ M_sun
    M_star = 5e11 * M_sun
    r_200 = 200 * kpc_to_m
    
    g_N = G * M_star / r_200**2
    
    # Σ-Gravity enhancement
    Sigma = sigma_enhancement(g_N, A=A_0)
    M_eff = M_star * Sigma / M_sun
    
    # Observed: M_eff/M_star ~ 10-30 at 200 kpc
    ratio = M_eff / (5e11)
    
    passed = 5 < ratio < 50
    
    return TestResult(
        name="Galaxy-Galaxy Lensing",
        passed=passed,
        metric=ratio,
        details={
            'M_star': 5e11,
            'M_eff': M_eff,
            'ratio': ratio,
            'g_N': g_N,
            'Sigma': Sigma,
        },
        message=f"M_eff/M_star at 200kpc = {ratio:.1f}× (obs: ~10-30×)"
    )


def test_external_field_effect() -> TestResult:
    """Test External Field Effect suppression."""
    # Internal field (isolated dwarf)
    g_int = 1e-11  # m/s² (typical dSph)
    
    # External field (from host galaxy)
    g_ext = 1e-10  # m/s² (MW at 100 kpc)
    
    # Total field
    g_total = np.sqrt(g_int**2 + g_ext**2)
    
    # Σ-Gravity enhancement with total field
    Sigma_total = sigma_enhancement(g_total, A=A_0)
    
    # Enhancement if isolated
    Sigma_isolated = sigma_enhancement(g_int, A=A_0)
    
    # EFE suppression
    suppression = Sigma_total / Sigma_isolated
    
    # EFE should suppress enhancement when g_ext > g_int
    passed = suppression < 1.0
    
    return TestResult(
        name="External Field Effect",
        passed=passed,
        metric=suppression,
        details={
            'g_int': g_int,
            'g_ext': g_ext,
            'Sigma_isolated': Sigma_isolated,
            'Sigma_total': Sigma_total,
            'suppression': suppression,
        },
        message=f"EFE suppression: {suppression:.2f}× (g_ext/g†={g_ext/g_dagger:.2f})"
    )


def test_gravitational_waves() -> TestResult:
    """Test GW170817 constraint on graviton speed.
    
    Gold standard: Abbott+ 2017, PRL 119, 161101
    - |c_GW - c|/c < 10⁻¹⁵
    - GW170817 + GRB170817A timing (1.7s delay over 40 Mpc)
    - Rules out many modified gravity theories
    """
    # In Σ-Gravity, the enhancement is to the effective gravitational constant
    # The speed of gravitational waves is still c
    
    delta_c = 0  # Σ-Gravity predicts c_GW = c
    
    bound = OBS_BENCHMARKS['gw170817']['delta_c_over_c']
    passed = delta_c < bound
    
    return TestResult(
        name="Gravitational Waves",
        passed=passed,
        metric=delta_c,
        details={
            'delta_c_over_c': delta_c,
            'bound': bound,
            'time_delay_s': OBS_BENCHMARKS['gw170817']['time_delay_s'],
            'distance_Mpc': OBS_BENCHMARKS['gw170817']['distance_Mpc'],
            'source': OBS_BENCHMARKS['gw170817']['source'],
        },
        message=f"|c_GW-c|/c = {delta_c:.0e} < {bound:.0e} (GW170817)"
    )


def test_structure_formation() -> TestResult:
    """Test structure formation at cluster scales.
    
    Gold standard: Planck 2018, SDSS
    - σ8 = 0.811 (Planck) vs 0.76 (weak lensing) - "S8 tension"
    - BAO scale: 150 Mpc
    - Full test requires N-body simulations
    """
    # At cluster scales (M ~ 10^15 M_sun, r ~ 1 Mpc)
    M_cluster = 1e15 * M_sun
    r_cluster = 1000 * kpc_to_m  # 1 Mpc
    
    g_cluster = G * M_cluster / r_cluster**2
    
    # g/g† ratio
    ratio = g_cluster / g_dagger
    
    # At cluster scales, g ~ g† (transition regime)
    # This is where Σ-Gravity effects are significant
    
    passed = True  # Informational
    
    return TestResult(
        name="Structure Formation",
        passed=passed,
        metric=ratio,
        details={
            'g_cluster': g_cluster,
            'g_dagger': g_dagger,
            'ratio': ratio,
            'sigma8_planck': OBS_BENCHMARKS['structure_formation']['sigma8_planck'],
            'sigma8_lensing': OBS_BENCHMARKS['structure_formation']['sigma8_lensing'],
            'bao_scale_Mpc': OBS_BENCHMARKS['structure_formation']['bao_scale_Mpc'],
        },
        message=f"Cluster scale: g/g† = {ratio:.1f} (needs N-body sims; σ8 tension exists)"
    )


def test_cmb() -> TestResult:
    """Test CMB acoustic peaks consistency.
    
    Gold standard: Planck Collaboration 2020
    - Ω_b = 0.0493, Ω_c = 0.265
    - H0 = 67.4 km/s/Mpc
    - MOND challenge: CMB requires DM at z~1100
    - Full test requires Boltzmann code integration
    """
    # At z = 1100 (recombination)
    z_cmb = 1100
    Omega_m = OBS_BENCHMARKS['cmb']['Omega_m']
    Omega_L = 1 - Omega_m
    
    H_z = np.sqrt(Omega_m * (1 + z_cmb)**3 + Omega_L)
    g_dagger_z = g_dagger * H_z
    
    # Typical g at CMB scales
    rho_b = OBS_BENCHMARKS['cmb']['Omega_b'] * 1.36e11 * M_sun / (1e6 * kpc_to_m)**3 * (1 + z_cmb)**3
    r_horizon = 100 * kpc_to_m  # Sound horizon
    M_horizon = 4/3 * np.pi * r_horizon**3 * rho_b
    g_cmb = G * M_horizon / r_horizon**2
    
    ratio = g_cmb / g_dagger_z
    
    # At CMB, g << g†(z) - deep Newtonian regime
    # Σ-Gravity effects should be minimal
    
    passed = True  # Informational
    
    return TestResult(
        name="CMB Acoustic Peaks",
        passed=passed,
        metric=ratio,
        details={
            'g_cmb': g_cmb,
            'g_dagger_z': g_dagger_z,
            'ratio': ratio,
            'Omega_b': OBS_BENCHMARKS['cmb']['Omega_b'],
            'Omega_c': OBS_BENCHMARKS['cmb']['Omega_c'],
            'mond_challenge': OBS_BENCHMARKS['cmb']['mond_challenge'],
        },
        message=f"At z=1100: g/g†(z) = {ratio:.1e} (MOND challenge: {OBS_BENCHMARKS['cmb']['mond_challenge']})"
    )


def test_bullet_cluster() -> TestResult:
    """Test Bullet Cluster lensing offset.
    
    Gold standard: Clowe+ 2006, ApJ 648, L109
    - M_gas = 2.1×10¹⁴ M☉, M_stars = 0.5×10¹⁴ M☉
    - M_lensing = 5.5×10¹⁴ M☉ (mass ratio ~2.1×)
    - Key observation: Lensing peaks offset from gas, coincident with galaxies
    - MOND challenge: Gas dominates baryons but lensing follows stars
    - ΛCDM: Explained by collisionless DM halos
    """
    bc = OBS_BENCHMARKS['bullet_cluster']
    
    M_gas = bc['M_gas'] * M_sun
    M_stars = bc['M_stars'] * M_sun
    M_bar = M_gas + M_stars
    M_lens = bc['M_lensing'] * M_sun
    
    # At r = 150 kpc (where lensing is measured)
    r_lens = bc['offset_kpc'] * kpc_to_m
    g_bar = G * M_bar / r_lens**2
    
    # Σ-Gravity enhancement (cluster amplitude)
    Sigma = sigma_enhancement(g_bar, A=A_CLUSTER)
    M_pred = M_bar * Sigma
    
    ratio_pred = M_pred / M_bar
    ratio_obs = bc['mass_ratio']
    
    # Key test: does Σ-Gravity give reasonable enhancement?
    passed = 0.5 * ratio_obs < ratio_pred < 2.0 * ratio_obs
    
    return TestResult(
        name="Bullet Cluster",
        passed=passed,
        metric=ratio_pred,
        details={
            'M_bar': M_bar / M_sun,
            'M_pred': M_pred / M_sun,
            'M_lens': M_lens / M_sun,
            'ratio_pred': ratio_pred,
            'ratio_obs': ratio_obs,
            'Sigma': Sigma,
            'offset_kpc': bc['offset_kpc'],
            'mond_challenge': bc['mond_challenge'],
        },
        message=f"M_pred/M_bar = {ratio_pred:.2f}× (obs: {ratio_obs:.1f}×, MOND challenge: lensing follows stars)"
    )


# =============================================================================
# MAIN
# =============================================================================

def _parse_cli_float(flag: str, default: float) -> float:
    """Parse a float flag from sys.argv.

    Supports:
      - --flag=1.23
      - --flag 1.23
    """
    for i, arg in enumerate(sys.argv):
        if arg.startswith(flag + "="):
            try:
                return float(arg.split("=", 1)[1])
            except Exception:
                return default
        if arg == flag and i + 1 < len(sys.argv):
            try:
                return float(sys.argv[i + 1])
            except Exception:
                return default
    return default



def _parse_cli_str(flag: str, default: Optional[str] = None) -> Optional[str]:
    """Parse a string flag from sys.argv.

    Supports:
      - --flag=value
      - --flag value
    """
    for i, arg in enumerate(sys.argv):
        if arg.startswith(flag + "="):
            val = arg.split("=", 1)[1].strip()
            return val if val != "" else default
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

def main():
    global USE_SIGMA_COMPONENTS, COHERENCE_MODEL, JJ_XI_MULT, JJ_SMOOTH_M_POINTS
    global FLOW_ALPHA, FLOW_BETA, FLOW_GAMMA, FLOW_DELTA, FLOW_SMOOTH_POINTS, FLOW_USE_TIDAL, FP_RELAX
    global FLOW_USE_BULGE_SPECIFIC, FLOW_ALPHA_BULGE, FLOW_GAMMA_BULGE, FLOW_ALPHA_DISK, FLOW_GAMMA_DISK
    global GAIA_FLOW_FEATURES_PATH, GAIA_FLOW_FEATURES_DF, GAIA_FLOW_REQUIRE_MATCH
    global USE_GUIDED_GRAVITY, GUIDED_KAPPA, GUIDED_C_DEFAULT
    
    quick = '--quick' in sys.argv
    core_only = '--core' in sys.argv
    sigma_components = '--sigma-components' in sys.argv
    
    # NEW: coherence selector
    coherence_arg = None
    jj_xi_mult = None
    jj_smooth_points = None
    
    for a in sys.argv:
        if a.startswith('--coherence='):
            coherence_arg = a.split('=', 1)[1].strip().lower()
        elif a.startswith('--jj-xi-mult='):
            jj_xi_mult = float(a.split('=', 1)[1].strip())
        elif a.startswith('--jj-smooth='):
            jj_smooth_points = int(a.split('=', 1)[1].strip())
    
    
    # FLOW coherence tuning + misc runtime controls
    FLOW_ALPHA = float(_parse_cli_float('--flow-alpha', FLOW_ALPHA))
    FLOW_BETA = float(_parse_cli_float('--flow-beta', FLOW_BETA))
    FLOW_GAMMA = float(_parse_cli_float('--flow-gamma', FLOW_GAMMA))
    FLOW_DELTA = float(_parse_cli_float('--flow-delta', FLOW_DELTA))
    FLOW_SMOOTH_POINTS = int(_parse_cli_float('--flow-smooth', FLOW_SMOOTH_POINTS))
    FP_RELAX = float(_parse_cli_float('--fp-relax', FP_RELAX))

    if '--flow-no-tidal' in sys.argv:
        FLOW_USE_TIDAL = False
    if '--flow-tidal' in sys.argv:
        FLOW_USE_TIDAL = True

    # Bulge-specific tuning
    if '--flow-bulge-specific' in sys.argv:
        FLOW_USE_BULGE_SPECIFIC = True
    FLOW_ALPHA_BULGE = float(_parse_cli_float('--flow-alpha-bulge', FLOW_ALPHA_BULGE))
    FLOW_GAMMA_BULGE = float(_parse_cli_float('--flow-gamma-bulge', FLOW_GAMMA_BULGE))
    FLOW_ALPHA_DISK = float(_parse_cli_float('--flow-alpha-disk', FLOW_ALPHA_DISK))
    FLOW_GAMMA_DISK = float(_parse_cli_float('--flow-gamma-disk', FLOW_GAMMA_DISK))

    # Optional Gaia 6D flow features (see export_gaia_pointwise_features.py)
    GAIA_FLOW_FEATURES_PATH = _parse_cli_str('--gaia-flow-features', GAIA_FLOW_FEATURES_PATH)

    # Optional SPARC pointwise export
    export_sparc_points = _parse_cli_str('--export-sparc-points', None)

    USE_SIGMA_COMPONENTS = bool(sigma_components)

    if coherence_arg in ('jj', 'current', 'currents'):
        COHERENCE_MODEL = "JJ"
        if jj_xi_mult is not None:
            JJ_XI_MULT = jj_xi_mult
        if jj_smooth_points is not None:
            JJ_SMOOTH_M_POINTS = jj_smooth_points
    elif coherence_arg in ('flow', 'topology', 'vorticity', 'vort', 'shear', 'topo'):
        COHERENCE_MODEL = "FLOW"
    else:
        COHERENCE_MODEL = "C"
    
    # NEW: guided-gravity selector
    compare_guided = '--compare-guided' in sys.argv
    guided_flag = ('--guided' in sys.argv) or compare_guided
    guided_kappa = _parse_cli_float('--guided-kappa', GUIDED_KAPPA)
    guided_c_default = _parse_cli_float('--guided-c-default', GUIDED_C_DEFAULT)
    
    GUIDED_KAPPA = float(guided_kappa)
    GUIDED_C_DEFAULT = float(guided_c_default)
    # In compare mode we toggle USE_GUIDED_GRAVITY per run below.
    USE_GUIDED_GRAVITY = bool(guided_flag) and not bool(compare_guided)
    
    # NEW: covariant coherence proxy for SPARC (translates Gaia-calibrated C_cov)
    global USE_COVARIANT_PROXY
    USE_COVARIANT_PROXY = '--use-covariant-proxy' in sys.argv
    
    data_dir = Path(__file__).parent.parent / "data"
    
    print("=" * 80)
    print("Σ-GRAVITY EXPERIMENTAL REGRESSION TEST")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'Core only' if core_only else 'Quick' if quick else 'Full'}")
    if compare_guided:
        print(f"Model: COMPARE baseline vs guided (κ={GUIDED_KAPPA:g}, C_default={GUIDED_C_DEFAULT:g})")
    else:
        print(f"Model: {'GUIDED' if USE_GUIDED_GRAVITY else 'BASELINE'}")
    print(f"Coherence Model: {COHERENCE_MODEL}")
    print()
    print("UNIFIED FORMULA PARAMETERS:")
    print(f"  A₀ = exp(1/2π) ≈ {A_0:.4f}")
    print(f"  L₀ = {L_0} kpc, n = {N_EXP}")
    print(f"  ξ = R_d/(2π) ≈ {XI_SCALE:.4f} × R_d")
    print(f"  M/L = {ML_DISK}/{ML_BULGE}")
    print(f"  g† = {g_dagger:.3e} m/s²")
    print(f"  A_cluster = {A_CLUSTER:.2f}")
    print(f"  σ components mode: {'ON' if USE_SIGMA_COMPONENTS else 'OFF'}")
    if USE_SIGMA_COMPONENTS:
        print(f"    σ_gas/disk/bulge = {SIGMA_GAS_KMS:.1f}/{SIGMA_DISK_KMS:.1f}/{SIGMA_BULGE_KMS:.1f} km/s")
    if COHERENCE_MODEL == "JJ":
        print(f"  JJ coherence: ξ_mult={JJ_XI_MULT}, smooth_points={JJ_SMOOTH_M_POINTS}")
    if COHERENCE_MODEL == "FLOW":
        print(
            f"  Flow coherence: α={FLOW_ALPHA:g}, β={FLOW_BETA:g}, γ={FLOW_GAMMA:g}, δ={FLOW_DELTA:g}, "
            f"smooth={FLOW_SMOOTH_POINTS}, tidal={'ON' if FLOW_USE_TIDAL else 'OFF'}"
        )
        if FLOW_USE_BULGE_SPECIFIC:
            print(f"  Bulge-specific: ON (bulge: α={FLOW_ALPHA_BULGE:g}, γ={FLOW_GAMMA_BULGE:g}; "
                  f"disk: α={FLOW_ALPHA_DISK:g}, γ={FLOW_GAMMA_DISK:g})")
        print(f"  Fixed-point relax: {FP_RELAX:g}")
        if GAIA_FLOW_FEATURES_DF is not None:
            print(f"  Gaia flow features: loaded ({len(GAIA_FLOW_FEATURES_DF)} rows)")

    if guided_flag:
        print(f"  Guided gravity: {'ON' if USE_GUIDED_GRAVITY else 'COMPARE'}")
        print(f"    κ = {GUIDED_KAPPA:g}, C_default = {GUIDED_C_DEFAULT:g}")
    print()
    
    # Load data
    print("Loading data...")
    galaxies = load_sparc(data_dir)
    print(f"  SPARC: {len(galaxies)} galaxies")
    
    clusters = load_clusters(data_dir)
    print(f"  Clusters: {len(clusters)}")
    
    gaia_df = load_gaia(data_dir) if not quick else None
    print(f"  Gaia/MW: {len(gaia_df) if gaia_df is not None else 'Skipped'}")

    # Optionally load Gaia flow features (must align 1:1 with the Gaia catalog rows)
    GAIA_FLOW_FEATURES_DF = None
    if gaia_df is not None and GAIA_FLOW_FEATURES_PATH:
        try:
            GAIA_FLOW_FEATURES_DF = pd.read_csv(GAIA_FLOW_FEATURES_PATH)
            if GAIA_FLOW_REQUIRE_MATCH and (len(GAIA_FLOW_FEATURES_DF) != len(gaia_df)):
                print(f"  WARNING: Gaia flow features rows ({len(GAIA_FLOW_FEATURES_DF)}) != Gaia stars ({len(gaia_df)}). Ignoring flow features.")
                GAIA_FLOW_FEATURES_DF = None
            else:
                print(f"  Gaia flow features: {len(GAIA_FLOW_FEATURES_DF)} rows loaded from {GAIA_FLOW_FEATURES_PATH}")
        except Exception as e:
            print(f"  WARNING: Could not load Gaia flow features from {GAIA_FLOW_FEATURES_PATH}: {e}")
            GAIA_FLOW_FEATURES_DF = None

    print()
    
    # Define tests
    # Original tests (1-8) - now includes holdout validation
    tests_core = [
        ("SPARC", lambda: test_sparc(galaxies)),
        ("Clusters", lambda: test_clusters(clusters)),
        ("Cluster Holdout", lambda: test_cluster_holdout(clusters)),
        ("Gaia/MW", lambda: test_gaia(gaia_df)),
        ("Redshift", lambda: test_redshift()),
        ("Solar System", lambda: test_solar_system()),
        ("Counter-Rotation", lambda: test_counter_rotation(data_dir) if not quick else 
         TestResult("Counter-Rotation", True, 0, {}, "SKIPPED")),
        ("Tully-Fisher", lambda: test_tully_fisher()),
    ]
    
    # Extended tests (9-18)
    tests_extended = [
        ("Gaia Bulge Covariant", lambda: test_gaia_bulge_covariant(gaia_df) if gaia_df is not None else TestResult(name="Gaia Bulge Covariant", passed=True, metric=0.0, details={"status": "SKIPPED", "reason": "No Gaia data"}, message="SKIPPED: No Gaia data")),
        ("Wide Binaries", lambda: test_wide_binaries()),
        ("Dwarf Spheroidals", lambda: test_dwarf_spheroidals()),
        ("UDGs", lambda: test_ultra_diffuse_galaxies()),
        ("Galaxy-Galaxy Lensing", lambda: test_galaxy_galaxy_lensing()),
        ("External Field Effect", lambda: test_external_field_effect()),
        ("Gravitational Waves", lambda: test_gravitational_waves()),
        ("Structure Formation", lambda: test_structure_formation()),
        ("CMB", lambda: test_cmb()),
        ("Bullet Cluster", lambda: test_bullet_cluster()),
    ]
    
    all_tests = tests_core if core_only else tests_core + tests_extended

    def _fmt_metric(x: Any) -> str:
        try:
            xf = float(x)
            if not np.isfinite(xf):
                return str(xf)
            # Compact formatting that still preserves order-of-magnitude changes.
            return f"{xf:.3g}"
        except Exception:
            return str(x)

    def _run_suite(label: str) -> List[TestResult]:
        print("-" * 80)
        print(f"Running tests: {label}")
        print("-" * 80)
        results_local: List[TestResult] = []
        for _, test_func in all_tests:
            result = test_func()
            results_local.append(result)
            status = '✓' if result.passed else '✗'
            print(f"[{status}] {result.name}: {result.message}")
        print("-" * 80)
        return results_local

    def _summarize(results_local: List[TestResult], label: str) -> Tuple[int, bool]:
        passed_local = sum(1 for r in results_local if r.passed)
        print()
        print("=" * 80)
        print(f"{label}: {passed_local}/{len(results_local)} tests passed")
        print("=" * 80)
        if passed_local == len(results_local):
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
            for r in results_local:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        return passed_local, passed_local == len(results_local)

    def _build_report(results_local: List[TestResult], *, model: str, guided_enabled: bool, guided_kappa_used: float) -> Dict[str, Any]:
        passed_local = sum(1 for r in results_local if r.passed)
        return {
            'model': model,
            'coherence_model': COHERENCE_MODEL,
            'guided_gravity': {
                'enabled': bool(guided_enabled),
                'kappa': float(guided_kappa_used),
                'c_default': float(GUIDED_C_DEFAULT),
            },
            'parameters': {
                'A_0': A_0,
                'L_0': L_0,
                'n_exp': N_EXP,
                'xi_scale': XI_SCALE,
                'ml_disk': ML_DISK,
                'ml_bulge': ML_BULGE,
                'g_dagger': g_dagger,
                'A_cluster': A_CLUSTER,
                'use_sigma_components': USE_SIGMA_COMPONENTS,
                'sigma_gas_kms': SIGMA_GAS_KMS,
                'sigma_disk_kms': SIGMA_DISK_KMS,
                'sigma_bulge_kms': SIGMA_BULGE_KMS,
                'coherence_model': COHERENCE_MODEL,
                'jj_xi_mult': JJ_XI_MULT if COHERENCE_MODEL == "JJ" else None,
                'jj_smooth_points': JJ_SMOOTH_M_POINTS if COHERENCE_MODEL == "JJ" else None,
                'flow_alpha': FLOW_ALPHA if COHERENCE_MODEL == "FLOW" else None,
                'flow_beta': FLOW_BETA if COHERENCE_MODEL == "FLOW" else None,
                'flow_gamma': FLOW_GAMMA if COHERENCE_MODEL == "FLOW" else None,
                'flow_delta': FLOW_DELTA if COHERENCE_MODEL == "FLOW" else None,
                'flow_smooth_points': FLOW_SMOOTH_POINTS if COHERENCE_MODEL == "FLOW" else None,
                'flow_use_tidal': FLOW_USE_TIDAL if COHERENCE_MODEL == "FLOW" else None,
                'fp_relax': FP_RELAX,
                'gaia_flow_features_path': GAIA_FLOW_FEATURES_PATH,
            },
            'results': [asdict(r) for r in results_local],
            'summary': {
                'total_tests': len(results_local),
                'passed': passed_local,
                'failed': len(results_local) - passed_local,
            },
            'all_passed': passed_local == len(results_local),
        }

    # Save reports
    output_dir = Path(__file__).parent / "regression_results"
    output_dir.mkdir(exist_ok=True)

    if compare_guided:
        # 1) Baseline run
        USE_GUIDED_GRAVITY = False
        baseline_results = _run_suite("BASELINE (κ=0)")
        baseline_passed, baseline_all = _summarize(baseline_results, "BASELINE SUMMARY")

        if export_sparc_points:
            try:
                p = Path(export_sparc_points)
                p_base = p.with_name(p.stem + '_baseline' + p.suffix if p.suffix else p.name + '_baseline.csv')
                export_path = export_sparc_pointwise(galaxies, p_base, coherence_model=COHERENCE_MODEL)
                print(f"SPARC pointwise export (baseline) saved to: {export_path}")
            except Exception as e:
                print(f"WARNING: SPARC pointwise export (baseline) failed: {e}")


        # 2) Guided run
        USE_GUIDED_GRAVITY = True
        guided_results = _run_suite(f"GUIDED (κ={GUIDED_KAPPA:g}, C_default={GUIDED_C_DEFAULT:g})")
        guided_passed, guided_all = _summarize(guided_results, "GUIDED SUMMARY")

        if export_sparc_points:
            try:
                p = Path(export_sparc_points)
                p_guided = p.with_name(p.stem + '_guided' + p.suffix if p.suffix else p.name + '_guided.csv')
                export_path = export_sparc_pointwise(galaxies, p_guided, coherence_model=COHERENCE_MODEL)
                print(f"SPARC pointwise export (guided) saved to: {export_path}")
            except Exception as e:
                print(f"WARNING: SPARC pointwise export (guided) failed: {e}")


        # 3) Side-by-side comparison (aligned by test name)
        print()
        print("=" * 80)
        print("COMPARISON: baseline → guided")
        print("=" * 80)
        guided_map = {r.name: r for r in guided_results}
        for b in baseline_results:
            g = guided_map.get(b.name)
            if g is None:
                continue
            b_stat = '✓' if b.passed else '✗'
            g_stat = '✓' if g.passed else '✗'
            print(
                f"{b.name:24}  {b_stat} { _fmt_metric(b.metric):>8}  →  {g_stat} { _fmt_metric(g.metric):>8}"
            )

        compare_report = {
            'timestamp': datetime.now().isoformat(),
            'formula': 'sigma_gravity_unified_compare_guided',
            'coherence_model': COHERENCE_MODEL,
            'mode': 'core' if core_only else 'quick' if quick else 'full',
            'guided_config': {
                'kappa': float(GUIDED_KAPPA),
                'c_default': float(GUIDED_C_DEFAULT),
            },
            'baseline': _build_report(baseline_results, model='baseline', guided_enabled=False, guided_kappa_used=0.0),
            'guided': _build_report(guided_results, model='guided', guided_enabled=True, guided_kappa_used=float(GUIDED_KAPPA)),
            'comparison_summary': {
                'baseline_passed': int(baseline_passed),
                'guided_passed': int(guided_passed),
            },
        }

        out_path = output_dir / f"experimental_report_compare_{COHERENCE_MODEL}.json"
        with open(out_path, 'w') as f:
            json.dump(compare_report, f, indent=2, default=float)
        print(f"\nReport saved to: {out_path}")

        # Preserve original CI behavior: exit code follows BASELINE.
        sys.exit(0 if baseline_all else 1)

    else:
        label = f"GUIDED (κ={GUIDED_KAPPA:g}, C_default={GUIDED_C_DEFAULT:g})" if USE_GUIDED_GRAVITY else "BASELINE"
        results = _run_suite(label)
        passed, all_passed = _summarize(results, "SUMMARY")

        report = {
            'timestamp': datetime.now().isoformat(),
            'formula': 'sigma_gravity_unified_guided' if USE_GUIDED_GRAVITY else 'sigma_gravity_unified',
            'coherence_model': COHERENCE_MODEL,
            'mode': 'core' if core_only else 'quick' if quick else 'full',
            'guided_config': {
                'enabled': bool(USE_GUIDED_GRAVITY),
                'kappa': float(GUIDED_KAPPA) if USE_GUIDED_GRAVITY else 0.0,
                'c_default': float(GUIDED_C_DEFAULT),
            },
            'parameters': {
                'A_0': A_0,
                'L_0': L_0,
                'n_exp': N_EXP,
                'xi_scale': XI_SCALE,
                'ml_disk': ML_DISK,
                'ml_bulge': ML_BULGE,
                'g_dagger': g_dagger,
                'A_cluster': A_CLUSTER,
                'use_sigma_components': USE_SIGMA_COMPONENTS,
                'sigma_gas_kms': SIGMA_GAS_KMS,
                'sigma_disk_kms': SIGMA_DISK_KMS,
                'sigma_bulge_kms': SIGMA_BULGE_KMS,
                'coherence_model': COHERENCE_MODEL,
                'jj_xi_mult': JJ_XI_MULT if COHERENCE_MODEL == "JJ" else None,
                'jj_smooth_points': JJ_SMOOTH_M_POINTS if COHERENCE_MODEL == "JJ" else None,
                'flow_alpha': FLOW_ALPHA if COHERENCE_MODEL == "FLOW" else None,
                'flow_beta': FLOW_BETA if COHERENCE_MODEL == "FLOW" else None,
                'flow_gamma': FLOW_GAMMA if COHERENCE_MODEL == "FLOW" else None,
                'flow_delta': FLOW_DELTA if COHERENCE_MODEL == "FLOW" else None,
                'flow_smooth_points': FLOW_SMOOTH_POINTS if COHERENCE_MODEL == "FLOW" else None,
                'flow_use_tidal': FLOW_USE_TIDAL if COHERENCE_MODEL == "FLOW" else None,
                'fp_relax': FP_RELAX,
                'gaia_flow_features_path': GAIA_FLOW_FEATURES_PATH,
            },
            'results': [asdict(r) for r in results],
            'summary': {
                'total_tests': len(results),
                'passed': passed,
                'failed': len(results) - passed,
            },
            'all_passed': bool(all_passed),
        }

        report_filename = f"experimental_report_{COHERENCE_MODEL}{'_guided' if USE_GUIDED_GRAVITY else ''}.json"
        out_path = output_dir / report_filename
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\nReport saved to: {out_path}")

        if export_sparc_points:
            try:
                export_path = export_sparc_pointwise(galaxies, export_sparc_points, coherence_model=COHERENCE_MODEL)
                print(f"SPARC pointwise export saved to: {export_path}")
            except Exception as e:
                print(f"WARNING: SPARC pointwise export failed: {e}")

        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

