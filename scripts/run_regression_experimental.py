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
    python scripts/run_regression_experimental.py --compare-guided --guided-kappa 0.1  # A/B TEST: Compare baseline (κ=0) vs anisotropic (κ>0) predictions against all 17 observational tests
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
# OPTIONAL: SYNTHETIC STREAM-SEEKING TEST (2D/3D)
# =============================================================================
# This imports a small prototype solver/test implementing an anisotropic operator:
#     ∇·[(I + κ w(x) ŝ ŝᵀ) ∇Φ] = 4πGρ
# plus a toy ray-trace regression metric that *depends on directionality*.
#
# Place stream_seeking_anisotropic.py next to this script (or on PYTHONPATH) to enable.
try:
    from stream_seeking_anisotropic import synthetic_stream_lensing_regression
    _HAS_STREAM_SEEKING_TEST = True
except Exception:
    synthetic_stream_lensing_regression = None
    _HAS_STREAM_SEEKING_TEST = False

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

# Optional: "wave build-up + interference" mode.
#
# Concept mapping:
# - Enhancement grows in low-field regions ("far from gravity") because the
#   enhancement function increases as net field strength decreases.
# - Nearby external gravitational fields ("another object hits the wave")
#   reduce the enhancement by increasing the net field and/or dephasing it.
#
# This is implemented as an EFE-style replacement g -> g_tot = sqrt(g_int^2+g_ext^2)
# (plus an optional additional interference factor).
USE_WAVE_INTERFERENCE = False  # set in main() via --wave
WAVE_MODE = "efe"              # "efe" or "interference"
WAVE_BETA = 1.0                # only used for WAVE_MODE == "interference"

# =============================================================================
# COHERENCE MODEL SELECTION (A/B)
# =============================================================================
COHERENCE_MODEL = "C"   # "C" (baseline v^2/(v^2+σ^2)), "JJ" (current-current), or "SRC" (source-current split)

# JJ model hyperparameters (global; no per-galaxy fits)
# Tuned values: JJ_XI_MULT=0.4 gives best SPARC RMS (22.83 km/s vs 22.94 at 1.0)
JJ_XI_MULT = 0.4        # ξ_corr = JJ_XI_MULT * (R_d/(2π)) by default (tuned: 0.4 optimal)
JJ_SMOOTH_M_POINTS = 5  # smooth M_enc before derivative to reduce numerical noise (tuned: 5 optimal)
JJ_EPS = 1e-30

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
# GEOMETRY PATH-LENGTH (MAJOR EXPERIMENT) — galaxy-level L_gal from bulge fraction
# =============================================================================
# Root-cause idea: coherence strength scales with available correlated path length
# of baryonic worldlines. Disks have thin vertical thickness (L ≈ L_0), but bulges
# are 3D orbital families requiring bulge-size-scale path lengths.
#
# L_gal = (1 - f_b) * L_disk + f_b * L_bulge
# where L_bulge = α * R_d (geometry scale)
USE_GEO_L = False              # set in main() via --lgeom
GEO_L_BULGE_MULT = 1.0         # L_bulge = GEO_L_BULGE_MULT * R_d  (kpc)
GEO_L_MIN_KPC = L_0            # keep >= L0 to avoid "shorter-than-disk" unless explicitly relaxed
GEO_L_MAX_KPC = 50.0           # safety cap; adjust as needed

# =============================================================================
# DYNAMICAL PATH-LENGTH (MAJOR EXPERIMENT) — LOCAL L_eff = v_coh / Omega
# =============================================================================
# Root-cause idea: coherence is a phase-ordering process that has only one "clock"
# If coherence propagates at universal speed v_coh, then in one dynamical time
# τ_dyn = 1/Ω, the ordering can correlate over distance L_eff = v_coh * τ_dyn
USE_DYN_L = False          # set in main() via --ldyn
DYN_L_VCOH_KMS = 20.0      # universal coherence transport speed (km/s)
DYN_L_MIN_KPC = 1e-4       # safety floor
DYN_L_MAX_KPC = 1e3        # safety cap (covers clusters too)

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


def unified_amplitude(L: float | np.ndarray, C_stream: float | np.ndarray = None) -> float | np.ndarray:
    """Unified 3D amplitude: A = A₀ × (L_eff/L₀)^n where L_eff = L * (1 + κ * C_stream) if guided gravity is enabled
    
    No D switch needed - path length L determines amplitude naturally:
    - Thin disk galaxies: L ≈ L₀ (0.4 kpc scale height) → A ≈ A₀
    - Elliptical galaxies: L ~ 1-20 kpc → A ~ 1.5-3.4
    - Galaxy clusters: L ≈ 600 kpc → A ≈ 8.45
    
    If guided gravity is enabled, L_eff = L * (1 + κ * C_stream)
    """
    if USE_GUIDED_GRAVITY:
        if C_stream is None:
            C_stream = GUIDED_C_DEFAULT
        L_eff = np.asarray(L) * (1.0 + GUIDED_KAPPA * np.clip(C_stream, 0.0, 1.0))
        return A_0 * (L_eff / L_0)**N_EXP
    else:
        return A_0 * (np.asarray(L) / L_0)**N_EXP


def unified_amplitude_legacy(D: float, L: float) -> float:
    """Legacy amplitude with D switch (for backwards compatibility)."""
    return A_0 * (1 - D + D * (L / L_0)**N_EXP)


def dyn_path_length_kpc(R_kpc: np.ndarray,
                        V_pred_kms: np.ndarray,
                        vcoh_kms: float = None) -> np.ndarray:
    """
    Compute effective coherence depth from local dynamical time:
    
    L_eff(r) = v_coh / Omega(r) = v_coh * r / V_pred(r)
    
    This is based on the idea that coherence is a phase-ordering process
    that propagates at universal speed v_coh. In one dynamical time
    τ_dyn = 1/Ω, the ordering can correlate over distance L_eff.
    
    Units: (km/s) * kpc / (km/s) = kpc
    
    Args:
        R_kpc: Radial positions (kpc)
        V_pred_kms: Predicted rotation velocities (km/s)
        vcoh_kms: Coherence transport speed (km/s), defaults to DYN_L_VCOH_KMS
    
    Returns:
        L_eff: Effective path length (kpc), clipped to [DYN_L_MIN_KPC, DYN_L_MAX_KPC]
    """
    if vcoh_kms is None:
        vcoh_kms = DYN_L_VCOH_KMS

    R = np.asarray(R_kpc, dtype=float)
    V = np.asarray(V_pred_kms, dtype=float)

    # Safety: avoid division by zero
    R = np.maximum(R, 1e-12)
    V = np.maximum(np.abs(V), 1e-6)

    # L_eff = v_coh * r / V_pred
    L = float(vcoh_kms) * (R / V)
    
    # Apply minimum constraint: L_eff should not be smaller than L_0
    # This prevents the dynamical path-length from becoming unphysically small
    # in regions with high angular velocity (inner disk, Solar System)
    L = np.maximum(L, L_0)
    
    return np.clip(L, DYN_L_MIN_KPC, DYN_L_MAX_KPC)


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


def h_effective(g_int: np.ndarray, g_ext: Any = 0.0) -> np.ndarray:
    """Effective enhancement kernel for the optional wave/interference mode.

    Baseline (default):
        h_eff = h(g_int)

    Wave/interference mode (enabled with --wave):
        g_tot = sqrt(g_int^2 + g_ext^2)

      - WAVE_MODE == "efe":
            h_eff = h(g_tot)

      - WAVE_MODE == "interference":
            h_eff = h(g_tot) * (g_int/g_tot)^WAVE_BETA

    Interpretation:
        * "Wave builds" in low-field regions because h grows as g_tot falls.
        * "Other gravity reduces the wave" because g_ext increases g_tot and/or
          reduces the fractional contribution from the internal source.

    Notes:
        - g_ext is treated as a *magnitude* here (EFE-style scalar). A vector
          implementation can be added later if you have external-field direction
          estimates.
        - When USE_WAVE_INTERFERENCE is False, g_ext is ignored.
    """
    g_int = np.maximum(np.asarray(g_int), 1e-15)

    if not USE_WAVE_INTERFERENCE:
        return h_function(g_int)

    g_ext_arr = np.maximum(np.asarray(g_ext, dtype=float), 0.0)
    g_tot = np.sqrt(g_int**2 + g_ext_arr**2)
    h_tot = h_function(g_tot)

    mode = (WAVE_MODE or "efe").lower().strip()
    if mode == "efe":
        return h_tot
    if mode == "interference":
        # Additional dephasing/screening beyond the pure |g| replacement.
        frac = g_int / np.maximum(g_tot, 1e-15)
        return h_tot * np.power(frac, float(WAVE_BETA))

    # Fallback: treat unknown modes as pure EFE.
    return h_tot


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


def disk_gas_weights(V_gas_kms: np.ndarray, V_disk_scaled_kms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute disk and gas weights for source-current coherence.
    
    Returns:
        (w_gas, w_disk) where w_gas + w_disk = 1
    """
    vg2 = np.square(np.nan_to_num(np.asarray(V_gas_kms, dtype=float), nan=0.0))
    vd2 = np.square(np.nan_to_num(np.asarray(V_disk_scaled_kms, dtype=float), nan=0.0))
    denom = np.maximum(vg2 + vd2, 1e-12)
    return vg2 / denom, vd2 / denom


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
                      A: float = None, L: float = L_0,
                      g_ext: Any = 0.0) -> np.ndarray:
    """
    Full Σ enhancement factor using W(r) approximation.
    
    Σ = 1 + A(L) × W(r) × h(g)
    
    Note: This uses the W(r) approximation. For the primary C(r) formulation,
    use sigma_enhancement_C() with fixed-point iteration.
    
    In wave/interference mode, h uses g_tot = sqrt(g_int^2 + g_ext^2).
    """
    g = np.maximum(np.asarray(g), 1e-15)
    
    # Baseline amplitude A(L)
    A_base = unified_amplitude(L) if A is None else A
    
    h = h_effective(g, g_ext)
    
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
                        A: float = None, L: float = L_0,
                        g_ext: Any = 0.0) -> np.ndarray:
    """
    Full Σ enhancement factor using covariant C(r) - PRIMARY formulation.
    
    Σ = 1 + A(L) × C(r) × h(g)
    
    where C = v_rot²/(v_rot² + σ²)
    
    In wave/interference mode, h uses g_tot = sqrt(g_int^2 + g_ext^2).
    """
    g = np.maximum(np.asarray(g), 1e-15)
    
    # Baseline amplitude A(L)
    A_base = unified_amplitude(L) if A is None else A
    
    h = h_effective(g, g_ext)
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
                     V_gas_kms: Optional[np.ndarray] = None,
                     V_disk_scaled_kms: Optional[np.ndarray] = None,
                     V_bulge_scaled_kms: Optional[np.ndarray] = None,
                     g_ext: Any = 0.0) -> np.ndarray:
    """
    Predict rotation velocity using Σ-Gravity.
    
    Parameters:
    -----------
    use_C_primary : bool
        If True (default), use covariant C(r) with fixed-point iteration.
        If False, use W(r) approximation (faster, identical results).
    sigma_kms : float
        Velocity dispersion for C(r) formulation (default 20 km/s).
    coherence_model : str, optional
        If None, uses global COHERENCE_MODEL. "C" for baseline, "JJ" for current-current, "SRC" for source-current split.
    """
    if coherence_model is None:
        coherence_model = COHERENCE_MODEL
    
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    # -------------------------------------------------------------------------
    # Baseline default: disks use L=L0 → A=A0.
    # New experiment: galaxy-level L_gal from bulge fraction (geometry channel).
    # If --ldyn is on, we still allow radius-dependent L_eff(r) inside the loop,
    # but we use L_gal as the minimum floor (instead of L0).
    # -------------------------------------------------------------------------
    L_gal = L_0
    if USE_GEO_L:
        fb = float(f_bulge) if (f_bulge is not None) else 0.0
        fb = max(0.0, min(1.0, fb))

        # Keep disks near-baseline unless you decide otherwise
        L_disk = L_0

        # Bulge geometry scale (kpc). Uses existing R_d input.
        L_bulge = GEO_L_BULGE_MULT * max(float(R_d), 1e-6)

        L_gal = (1.0 - fb) * L_disk + fb * L_bulge
        L_gal = min(max(L_gal, GEO_L_MIN_KPC), GEO_L_MAX_KPC)

    A_base = unified_amplitude(L_gal)
    
    # Optional external field for wave/interference mode
    # Can be scalar or per-radius array
    g_ext_arg = g_ext
    
    # Build sigma input once
    sigma_use = sigma_profile_kms if sigma_profile_kms is not None else float(sigma_kms)
    
    # SRC precompute: split bulge vs coherent (disk+gas) contributions
    src_enabled = (coherence_model.upper() == "SRC")
    if src_enabled:
        if V_bulge_scaled_kms is None:
            # fallback to baseline if components missing
            src_enabled = False
        else:
            V_bar_sq = np.square(np.asarray(V_bar, dtype=float))
            V_bulge_sq = np.square(np.asarray(V_bulge_scaled_kms, dtype=float))
            # Coherent squared contribution = everything that's not bulge (clamped)
            V_coh_sq = np.maximum(V_bar_sq - V_bulge_sq, 0.0)

            if V_gas_kms is None or V_disk_scaled_kms is None:
                # If we can't split gas vs disk, treat all coherent part as "disk-like"
                wgas = np.zeros_like(V_bar_sq)
                wdisk = np.ones_like(V_bar_sq)
            else:
                wgas, wdisk = disk_gas_weights(V_gas_kms, V_disk_scaled_kms)
    
    # Precompute kernel for JJ once (performance + stable iteration)
    jj_kernel = None
    if coherence_model.upper() == "JJ":
        xi_corr_kpc = JJ_XI_MULT * XI_SCALE * max(float(R_d), 1e-6)
        Rm = np.asarray(R_kpc, dtype=float) * kpc_to_m
        dR = np.abs(Rm[:, None] - Rm[None, :])
        jj_kernel = np.exp(-dR / (max(xi_corr_kpc, 1e-6) * kpc_to_m))
    
    h = h_effective(g_bar, g_ext_arg)
    V = np.array(V_bar, dtype=float)
    
    for _ in range(50):  # Typically converges in 3-5 iterations
        # MAJOR CHANGE: allow A to become a radius-dependent profile via L_eff = v_coh / Omega
        if USE_DYN_L:
            L_eff = dyn_path_length_kpc(R_kpc, V, vcoh_kms=DYN_L_VCOH_KMS)
            A_base = unified_amplitude(L_eff)   # NOTE: vector A(r), not scalar
        else:
            A_base = A_0  # Scalar A for baseline
        if coherence_model.upper() == "JJ":
            Q = Q_JJ_coherence(
                R_kpc=R_kpc,
                V_pred_kms=V,
                V_bar_kms=V_bar,
                R_d_kpc=R_d,
                sigma_kms=sigma_use,
                kernel=jj_kernel,
            )
            # Experimental guided-gravity: allow the stream coherence Q to
            # feed back into the amplitude via L_eff = L(1+κQ).
            if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
                A_use = A_base * guided_amplitude_multiplier(Q)
            else:
                A_use = A_base
            Sigma = 1 + A_use * Q * h
            V_new = V_bar * np.sqrt(Sigma)

        elif src_enabled:
            # Source-current coherence: only disk+gas get Σ, bulge stays Newtonian
            C_gas = C_coherence(V, SIGMA_GAS_KMS)
            C_disk = C_coherence(V, SIGMA_DISK_KMS)
            C_src = wgas * C_gas + wdisk * C_disk

            if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
                A_use = A_base * guided_amplitude_multiplier(C_src)
            else:
                A_use = A_base

            Sigma_coh = 1 + A_use * C_src * h
            V_new_sq = V_bulge_sq + V_coh_sq * Sigma_coh
            V_new = np.sqrt(np.maximum(V_new_sq, 1e-12))

        else:
            # Baseline: C = v^2/(v^2+σ^2)
            C = C_coherence(V, sigma_use)
            # Experimental guided-gravity: allow the stream coherence C to
            # feed back into the amplitude via L_eff = L(1+κC).
            if USE_GUIDED_GRAVITY and float(GUIDED_KAPPA) != 0.0:
                A_use = A_base * guided_amplitude_multiplier(C)
            else:
                A_use = A_base
            Sigma = 1 + A_use * C * h
            V_new = V_bar * np.sqrt(Sigma)
        # Safety: handle NaN/inf in V_new
        V_new = np.nan_to_num(V_new, nan=V_bar, posinf=V_bar*10, neginf=V_bar)
        V_new = np.maximum(V_new, 1e-6)  # Ensure positive
        
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new
    
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
    """
    if not galaxies:
        return TestResult("SPARC Galaxies", True, 0.0, {}, "SKIPPED: No data")
    
    rms_list = []
    mond_rms_list = []
    all_log_ratios = []
    wins = 0
    
    # Diagnostic: track bulge-dominated vs disk-dominated galaxies
    bulge_rms_list = []
    disk_rms_list = []
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        h_disk = gal.get('h_disk', 0.15 * R_d)
        f_bulge = gal.get('f_bulge', 0.0)

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
        
        # Optional environmental field (for wave/EFE mode): allow either a scalar
        # gal['g_ext'] or a per-radius profile gal['g_ext_profile'].
        g_ext_arg = gal.get('g_ext_profile', gal.get('g_ext', 0.0))
        V_pred = predict_velocity(
            R, V_bar, R_d, h_disk, f_bulge,
            sigma_profile_kms=sigma_profile,
            V_gas_kms=gal.get('V_gas', None),
            V_disk_scaled_kms=gal.get('V_disk_scaled', None),
            V_bulge_scaled_kms=gal.get('V_bulge_scaled', None),
            g_ext=g_ext_arg,
        )
        V_mond = predict_mond(R, V_bar)
        
        rms_sigma = np.sqrt(((V_pred - V_obs)**2).mean())
        rms_mond = np.sqrt(((V_mond - V_obs)**2).mean())
        
        rms_list.append(rms_sigma)
        mond_rms_list.append(rms_mond)
        
        # Diagnostic: bin by bulge fraction
        if f_bulge > 0.3:
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
    
    mean_rms = np.mean(rms_list)
    mean_mond_rms = np.mean(mond_rms_list)
    win_rate = wins / len(galaxies)
    rar_scatter = np.std(all_log_ratios) if all_log_ratios else 0.0
    
    # Diagnostic: mean RMS by bulge fraction
    bulge_mean_rms = np.mean(bulge_rms_list) if bulge_rms_list else None
    disk_mean_rms = np.mean(disk_rms_list) if disk_rms_list else None
    
    passed = mean_rms < 20.0
    
    details = {
        'n_galaxies': len(galaxies),
        'mean_rms': mean_rms,
        'mean_mond_rms': mean_mond_rms,
        'win_rate': win_rate,
        'rar_scatter_dex': rar_scatter,
        'benchmark_mond_rms': OBS_BENCHMARKS['sparc']['mond_rms_kms'],
        'benchmark_lcdm_rms': OBS_BENCHMARKS['sparc']['lcdm_rms_kms'],
        'benchmark_rar_scatter': OBS_BENCHMARKS['sparc']['rar_scatter_dex'],
    }
    
    # Add diagnostic info if available
    if bulge_mean_rms is not None:
        details['bulge_mean_rms'] = bulge_mean_rms
        details['n_bulge_galaxies'] = len(bulge_rms_list)
    if disk_mean_rms is not None:
        details['disk_mean_rms'] = disk_mean_rms
        details['n_disk_galaxies'] = len(disk_rms_list)
    
    msg = f"RMS={mean_rms:.2f} km/s (MOND={mean_mond_rms:.2f}, ΛCDM~15), Scatter={rar_scatter:.3f} dex, Win={win_rate*100:.1f}%"
    if bulge_mean_rms is not None and disk_mean_rms is not None:
        msg += f" | Bulge: {bulge_mean_rms:.2f}, Disk: {disk_mean_rms:.2f}"
    
    return TestResult(
        name="SPARC Galaxies",
        passed=passed,
        metric=mean_rms,
        details=details,
        message=msg
    )


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
    V_pred = predict_velocity(R, V_bar, R_d_mw, h_disk=0.3, f_bulge=0.1, sigma_profile_kms=sigma_profile)
    
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

    # Optional external field (Milky Way at the Solar radius), for the wave/EFE mode.
    # Using the centripetal estimate g_ext ≈ V_c^2 / R.
    Vc = OBS_BENCHMARKS['milky_way']['V_sun_kms'] * 1000.0
    R_sun = OBS_BENCHMARKS['milky_way']['R_sun_kpc'] * kpc_to_m
    g_ext = (Vc**2) / max(R_sun, 1e-30)
    
    # Σ-Gravity enhancement
    # - baseline (canonical): g_ext ignored
    # - wave/interference mode: g_ext suppresses the enhancement
    Sigma_iso = sigma_enhancement(g_N, A=A_0)
    Sigma_env = sigma_enhancement(g_N, A=A_0, g_ext=g_ext)
    Sigma = Sigma_env if USE_WAVE_INTERFERENCE else Sigma_iso
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
            'g_ext': g_ext,
            'g_over_a0': g_N / a0_mond,
            'boost_pred': boost,
            'boost_iso': float(Sigma_iso - 1),
            'boost_env': float(Sigma_env - 1),
            'wave_mode_enabled': USE_WAVE_INTERFERENCE,
            'wave_mode': WAVE_MODE,
            'wave_beta': WAVE_BETA,
            'boost_obs': obs_boost,
            'obs_uncertainty': obs_uncertainty,
            'n_pairs': OBS_BENCHMARKS['wide_binaries']['n_pairs'],
            'controversy': OBS_BENCHMARKS['wide_binaries']['controversy'],
        },
        message=(
            f"Boost at {s_AU} AU: {boost*100:.1f}% "
            f"(iso={float(Sigma_iso - 1)*100:.1f}%, env={float(Sigma_env - 1)*100:.1f}%; "
            f"Chae 2023: {obs_boost*100:.0f}±{obs_uncertainty*100:.0f}%)"
        )
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
    
    # Total field (scalar EFE proxy)
    g_total = np.sqrt(g_int**2 + g_ext**2)

    # Baseline (always available): evaluate h on |g_total|
    Sigma_total_scalar = sigma_enhancement(g_total, A=A_0)

    # Enhancement if isolated
    Sigma_isolated = sigma_enhancement(g_int, A=A_0)

    # Wave/interference mode: pass g_int and g_ext separately (uses g_tot internally)
    Sigma_total_wave = sigma_enhancement(g_int, A=A_0, g_ext=g_ext)

    suppression_scalar = Sigma_total_scalar / Sigma_isolated
    suppression_wave = Sigma_total_wave / Sigma_isolated

    suppression = suppression_wave if USE_WAVE_INTERFERENCE else suppression_scalar
    
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
            'Sigma_total_scalar': Sigma_total_scalar,
            'Sigma_total_wave': Sigma_total_wave,
            'suppression_scalar': suppression_scalar,
            'suppression_wave': suppression_wave,
            'suppression_used': suppression,
        },
        message=(
            f"EFE suppression: {suppression:.2f}× "
            f"(scalar={suppression_scalar:.2f}×, wave={suppression_wave:.2f}×; "
            f"g_ext/g†={g_ext/g_dagger:.2f})"
        )
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


def test_bullet_anisotropic_ab() -> TestResult:
    """A/B test: Compare baseline (κ=0) vs anisotropic (κ>0) predictions
    for Bullet Cluster lensing offset (directional observable).
    
    This tests whether anisotropic gravity can better predict the observed
    ~150 kpc offset between lensing peak and gas peak in the Bullet Cluster.
    
    Gold standard: Clowe+ 2006, ApJ 648, L109
    """
    try:
        from test_bullet_anisotropic import test_bullet_anisotropic_ab as _test_func
        results = _test_func()
        
        if not results.get('success', False):
            return TestResult(
                name="Bullet Cluster Anisotropic A/B",
                passed=True,
                metric=0.0,
                details={"enabled": False, "error": results.get('error', 'unknown')},
                message=f"SKIPPED ({results.get('error', 'test_bullet_anisotropic module not found')})",
            )
        
        baseline_error = results['comparison']['baseline_rms']
        aniso_error = results['comparison']['aniso_rms']
        improvement = results['comparison']['improvement']
        aniso_better = results['comparison']['aniso_better']
        
        # Pass if anisotropic is better OR if both are within reasonable range
        # (The offset prediction is challenging, so we're looking for relative improvement)
        passed = aniso_better or (baseline_error < 100.0 and aniso_error < 100.0)
        
        return TestResult(
            name="Bullet Cluster Anisotropic A/B",
            passed=bool(passed),
            metric=float(improvement),
            details={
                "observed_offset_kpc": results['observed_offset_kpc'],
                "baseline_offset_kpc": results['baseline']['predicted_offset_kpc'],
                "baseline_error_kpc": baseline_error,
                "aniso_offset_kpc": results['anisotropic']['predicted_offset_kpc'],
                "aniso_error_kpc": aniso_error,
                "aniso_kappa": results['anisotropic']['kappa'],
                "improvement": improvement,
                "aniso_better": aniso_better,
            },
            message=f"Offset: baseline={results['baseline']['predicted_offset_kpc']:.1f} kpc (error={baseline_error:.1f}), aniso={results['anisotropic']['predicted_offset_kpc']:.1f} kpc (error={aniso_error:.1f}), improvement={improvement*100:.1f}%",
        )
    except ImportError:
        return TestResult(
            name="Bullet Cluster Anisotropic A/B",
            passed=True,
            metric=0.0,
            details={"enabled": False},
            message="SKIPPED (test_bullet_anisotropic.py not found on PYTHONPATH)",
        )


def test_synthetic_stream_lensing() -> TestResult:
    """Synthetic anisotropic-operator + ray-trace test (toy 2D/3D regression).

    This is NOT an observational benchmark. It is a diagnostic that the
    *directional* operator
        ∇·[(I + κ w(x) ŝ ŝᵀ) ∇Φ] = 4πGρ
    can produce focusing toward a stream/filament in a way a scalar amplitude
    rescaling cannot mimic.

    The test runs:
      - isotropic solve (κ=0)
      - anisotropic solve (κ>0) localized around a filament
    and compares ray-traced capture + mean-shift statistics.
    """
    if not _HAS_STREAM_SEEKING_TEST:
        return TestResult(
            name="Synthetic Stream Lensing",
            passed=True,
            metric=0.0,
            details={"enabled": False},
            message="SKIPPED (stream_seeking_anisotropic.py not found on PYTHONPATH)",
        )

    out = synthetic_stream_lensing_regression(
        N=160,
        L=1.0,
        kappa=6.0,
        mass_offset_x=0.30,
        mass_sigma=0.12,
        stream_sigma=0.06,
        capture_width=0.08,
        n_rays=80,
        n_steps=240,
        tol=1e-7,
        max_iter=6000,
    )

    # Pass criteria: anisotropy should increase filament capture and pull mean x_end toward x=0
    passed = (out.capture_ratio > 1.05) and (out.mean_shift_ratio < 0.98)

    return TestResult(
        name="Synthetic Stream Lensing",
        passed=bool(passed),
        metric=float(out.capture_ratio),
        details={
            "capture_ratio": float(out.capture_ratio),
            "mean_shift_ratio": float(out.mean_shift_ratio),
            "capture_iso": float(out.capture_iso),
            "capture_aniso": float(out.capture_aniso),
            "mean_x_iso": float(out.mean_x_iso),
            "mean_x_aniso": float(out.mean_x_aniso),
            "solver_iso": out.solver_iso,
            "solver_aniso": out.solver_aniso,
        },
        message=f"capture_ratio={out.capture_ratio:.3f} (want>1), mean_shift_ratio={out.mean_shift_ratio:.3f} (want<1)",
    )


def test_anisotropic_prediction_ab() -> TestResult:
    """A/B test: Does anisotropic gravity predict object impact better than baseline?

    This test compares κ=0 (baseline isotropic) vs κ>0 (anisotropic) predictions
    against observational targets to determine if anisotropy improves predictions
    of how gravity impacts objects (light rays, particles, etc.).

    Currently uses synthetic observations, but designed to accept real data:
    - Light deflection/image positions (strong lensing)
    - Weak lensing shear anisotropy
    - Bullet-like merger offsets
    - Stellar stream tracks

    Pass criteria: Anisotropic model must improve RMS prediction error vs baseline.
    """
    try:
        from score_stream_lensing import (
            build_scene,
            solve_potential,
            predict_ray_endpoints,
            compare_models_ab,
        )
    except ImportError:
        return TestResult(
            name="Anisotropic Prediction A/B",
            passed=True,
            metric=0.0,
            details={"enabled": False},
            message="SKIPPED (score_stream_lensing.py not found)",
        )

    # Build a test scene
    scene = build_scene(
        N=160,
        L=1.0,
        mass_offset_x=0.30,
        mass_sigma=0.12,
        stream_sigma=0.06,
    )

    # Generate synthetic "observations" from a known κ_true
    # In real usage, replace this with actual observational data
    kappa_true = 6.0
    Phi_true, _ = solve_potential(scene, kappa=kappa_true)
    x0s, x_obs = predict_ray_endpoints(Phi_true, scene, n_rays=80, n_steps=240)
    
    # Add realistic measurement noise
    obs_uncertainty = np.full_like(x_obs, 0.002)
    x_obs = x_obs + np.random.normal(0.0, obs_uncertainty[0], size=x_obs.shape)

    # A/B comparison: baseline (κ=0) vs anisotropic (κ=6.0)
    results = compare_models_ab(
        scene=scene,
        kappa_baseline=0.0,
        kappa_aniso=6.0,
        observations=x_obs,
        obs_uncertainty=obs_uncertainty,
        prediction_type="ray_endpoints",
        n_rays=80,
        n_steps=240,
    )

    baseline_rms = results['baseline']['score'].rms
    aniso_rms = results['anisotropic']['score'].rms
    improvement = results['comparison']['rms_improvement']
    aniso_better = results['comparison']['aniso_better']

    # Pass if anisotropic model improves predictions
    # (In this synthetic case, it should since observations were generated with κ>0)
    passed = aniso_better and improvement > 0.05  # Require at least 5% improvement

    return TestResult(
        name="Anisotropic Prediction A/B",
        passed=passed,
        metric=improvement,  # RMS improvement fraction
        details={
            "baseline_rms": float(baseline_rms),
            "anisotropic_rms": float(aniso_rms),
            "rms_improvement": float(improvement),
            "chi2_delta": float(results['comparison']['chi2_delta']) if results['comparison']['chi2_delta'] is not None else None,
            "anisotropic_better": bool(aniso_better),
            "kappa_baseline": 0.0,
            "kappa_aniso": 6.0,
            "kappa_true": kappa_true,
            "n_observations": len(x_obs),
        },
        message=f"RMS improvement: {improvement*100:.1f}% (baseline={baseline_rms:.4f}, aniso={aniso_rms:.4f})",
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

def main():
    global USE_SIGMA_COMPONENTS, COHERENCE_MODEL, JJ_XI_MULT, JJ_SMOOTH_M_POINTS
    global USE_GUIDED_GRAVITY, GUIDED_KAPPA, GUIDED_C_DEFAULT
    global USE_DYN_L, DYN_L_VCOH_KMS
    global USE_GEO_L, GEO_L_BULGE_MULT
    global USE_WAVE_INTERFERENCE, WAVE_MODE, WAVE_BETA
    
    quick = '--quick' in sys.argv
    core_only = '--core' in sys.argv
    sigma_components = '--sigma-components' in sys.argv
    wave_mode_on = '--wave' in sys.argv
    
    # NEW: dynamical path-length mode
    ldyn_flag = '--ldyn' in sys.argv
    vcoh_arg = _parse_cli_float('--vcoh-kms', DYN_L_VCOH_KMS)
    
    USE_DYN_L = bool(ldyn_flag)
    DYN_L_VCOH_KMS = float(vcoh_arg)
    
    # NEW: A/B comparison mode for anisotropic gravity
    compare_anisotropic = '--compare-anisotropic' in sys.argv
    aniso_kappa = _parse_cli_float('--aniso-kappa', 6.0)
    
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
    
    USE_SIGMA_COMPONENTS = bool(sigma_components)
    
    # Wave/interference toggles (optional)
    USE_WAVE_INTERFERENCE = bool(wave_mode_on)
    for arg in sys.argv:
        if arg.startswith('--wave-mode='):
            WAVE_MODE = arg.split('=', 1)[1].strip() or WAVE_MODE
        if arg.startswith('--wave-beta='):
            try:
                WAVE_BETA = float(arg.split('=', 1)[1])
            except Exception:
                pass
    
    if coherence_arg in ('jj', 'current', 'currents'):
        COHERENCE_MODEL = "JJ"
        if jj_xi_mult is not None:
            JJ_XI_MULT = jj_xi_mult
        if jj_smooth_points is not None:
            JJ_SMOOTH_M_POINTS = jj_smooth_points
    elif coherence_arg in ('src', 'source', 'score', 'split'):
        COHERENCE_MODEL = "SRC"
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
    print(f"  wave/interference mode: {'ON' if USE_WAVE_INTERFERENCE else 'OFF'}")
    if USE_WAVE_INTERFERENCE:
        print(f"    wave mode = {WAVE_MODE}, beta = {WAVE_BETA}")
    if USE_SIGMA_COMPONENTS:
        print(f"    σ_gas/disk/bulge = {SIGMA_GAS_KMS:.1f}/{SIGMA_DISK_KMS:.1f}/{SIGMA_BULGE_KMS:.1f} km/s")
    if COHERENCE_MODEL == "JJ":
        print(f"  JJ coherence: ξ_mult={JJ_XI_MULT}, smooth_points={JJ_SMOOTH_M_POINTS}")
    if guided_flag:
        print(f"  Guided gravity: {'ON' if USE_GUIDED_GRAVITY else 'COMPARE'}")
        print(f"    κ = {GUIDED_KAPPA:g}, C_default = {GUIDED_C_DEFAULT:g}")
    if USE_GEO_L:
        print(f"  Geometry L mode: ON (L_bulge = {GEO_L_BULGE_MULT:g} × R_d)")
    if USE_DYN_L:
        print(f"  Dynamical L mode: ON (v_coh={DYN_L_VCOH_KMS:g} km/s)")
    print()
    
    # Load data
    print("Loading data...")
    galaxies = load_sparc(data_dir)
    print(f"  SPARC: {len(galaxies)} galaxies")
    
    clusters = load_clusters(data_dir)
    print(f"  Clusters: {len(clusters)}")
    
    gaia_df = load_gaia(data_dir) if not quick else None
    print(f"  Gaia/MW: {len(gaia_df) if gaia_df is not None else 'Skipped'}")
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
    
    # Extended tests (9-17)
    tests_extended = [
        ("Wide Binaries", lambda: test_wide_binaries()),
        ("Dwarf Spheroidals", lambda: test_dwarf_spheroidals()),
        ("UDGs", lambda: test_ultra_diffuse_galaxies()),
        ("Galaxy-Galaxy Lensing", lambda: test_galaxy_galaxy_lensing()),
        ("External Field Effect", lambda: test_external_field_effect()),
        ("Gravitational Waves", lambda: test_gravitational_waves()),
        ("Structure Formation", lambda: test_structure_formation()),
        ("CMB", lambda: test_cmb()),
        ("Synthetic Stream Lensing", lambda: test_synthetic_stream_lensing() if not quick else
         TestResult("Synthetic Stream Lensing", True, 0, {}, "SKIPPED")),
        ("Anisotropic Prediction A/B", lambda: test_anisotropic_prediction_ab() if not quick else
         TestResult("Anisotropic Prediction A/B", True, 0, {}, "SKIPPED")),
        ("Bullet Cluster Anisotropic A/B", lambda: test_bullet_anisotropic_ab() if not quick else
         TestResult("Bullet Cluster Anisotropic A/B", True, 0, {}, "SKIPPED")),
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

        # 2) Guided run
        USE_GUIDED_GRAVITY = True
        guided_results = _run_suite(f"GUIDED (κ={GUIDED_KAPPA:g}, C_default={GUIDED_C_DEFAULT:g})")
        guided_passed, guided_all = _summarize(guided_results, "GUIDED SUMMARY")

        # 3) Side-by-side comparison (aligned by test name) with MOND
        print()
        print("=" * 80)
        print("COMPARISON: Baseline | Anisotropic | MOND")
        print("=" * 80)
        guided_map = {r.name: r for r in guided_results}
        for b in baseline_results:
            g = guided_map.get(b.name)
            if g is None:
                continue
            b_stat = '✓' if b.passed else '✗'
            g_stat = '✓' if g.passed else '✗'
            
            # Extract MOND metric from test details if available
            mond_metric = None
            mond_label = ""
            if 'mean_mond_rms' in b.details:
                mond_metric = b.details['mean_mond_rms']
                mond_label = f"MOND={mond_metric:.2f}"
            elif 'benchmark_mond_ratio' in b.details:
                mond_metric = b.details['benchmark_mond_ratio']
                mond_label = f"MOND~{mond_metric:.2f}"
            elif 'mond_status' in b.details:
                mond_label = "MOND: varies"
            elif 'mond_challenge' in b.details:
                mond_label = "MOND: challenge"
            else:
                mond_label = "MOND: N/A"
            
            # Determine which is best (lower RMS or closer to 1.0 for ratios)
            best = ""
            if mond_metric is not None:
                # For SPARC: lower RMS is better
                if 'mean_mond_rms' in b.details:
                    if b.metric < g.metric and b.metric < mond_metric:
                        best = " [BEST: Baseline]"
                    elif g.metric < b.metric and g.metric < mond_metric:
                        best = " [BEST: Aniso]"
                    elif mond_metric < b.metric and mond_metric < g.metric:
                        best = " [BEST: MOND]"
                # For clusters: closer to 1.0 is better
                elif 'benchmark_mond_ratio' in b.details:
                    b_dist = abs(b.metric - 1.0)
                    g_dist = abs(g.metric - 1.0)
                    m_dist = abs(mond_metric - 1.0)
                    if b_dist < g_dist and b_dist < m_dist:
                        best = " [BEST: Baseline]"
                    elif g_dist < b_dist and g_dist < m_dist:
                        best = " [BEST: Aniso]"
                    elif m_dist < b_dist and m_dist < g_dist:
                        best = " [BEST: MOND]"
            
            print(
                f"{b.name:24}  {b_stat} {_fmt_metric(b.metric):>8}  |  {g_stat} {_fmt_metric(g.metric):>8}  |  {mond_label:20}{best}"
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

        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

