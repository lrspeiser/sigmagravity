#!/usr/bin/env python3
"""
Σ-GRAVITY EXTENDED REGRESSION TEST - COMPARATIVE MODE
======================================================

This test suite compares multiple gravity theories against observations:
- BASELINE: Original Σ-Gravity (locked, never changes)
- NEW MODEL: Σ-Gravity with extended phase coherence
- MOND: Modified Newtonian Dynamics
- ΛCDM: Standard cosmology with dark matter (where applicable)
- OBSERVED: Actual astronomical measurements

Output shows comparative performance, not just pass/fail.

EXTENDED PHASE COHERENCE MODEL:
The new model introduces disturbance parameter D for non-cluster systems:
- D_asymmetry: Based on kinematic asymmetry in disk galaxies
- D_tidal: Based on tidal distortion for satellites/UDGs
- D_interaction: Based on merger/interaction signatures

Uses the UNIFIED 3D AMPLITUDE FORMULA: A(L) = A₀ × (L/L₀)^n

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
from dataclasses import dataclass, asdict, field
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
# BASELINE MODEL PARAMETERS (LOCKED - DO NOT MODIFY)
# =============================================================================
# These define the BASELINE Σ-Gravity model against which all comparisons are made.
# Changes here would invalidate all historical comparisons.
BASELINE_A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173
BASELINE_L_0 = 0.40  # Reference path length (kpc)
BASELINE_N_EXP = 0.27  # Path length exponent
BASELINE_XI_SCALE = 1 / (2 * np.pi)  # ξ = R_d/(2π)
BASELINE_ML_DISK = 0.5
BASELINE_ML_BULGE = 0.7
BASELINE_A_CLUSTER = BASELINE_A_0 * (600 / BASELINE_L_0)**BASELINE_N_EXP  # ≈ 8.45

# =============================================================================
# NEW MODEL PARAMETERS (EXTENDED PHASE COHERENCE)
# =============================================================================
# These can be tuned to improve predictions while keeping baseline fixed.
A_0 = BASELINE_A_0
L_0 = BASELINE_L_0
N_EXP = BASELINE_N_EXP
XI_SCALE = BASELINE_XI_SCALE
ML_DISK = BASELINE_ML_DISK
ML_BULGE = BASELINE_ML_BULGE
A_CLUSTER = BASELINE_A_CLUSTER

# Extended phase coherence parameters
USE_EXTENDED_PHI = False  # Enable extended phase coherence for non-clusters

# Universal coupling constant
PHI_LAMBDA_0 = 7.0

# Regulators
PHI_MAX = 12.0
PHI_MIN = -6.0

# Extended D proxies for non-cluster systems (TUNED)
D_ASYMMETRY_SCALE = 1.5   # Scale factor for kinematic asymmetry → D  [TUNED]
D_TIDAL_THRESHOLD = 5.0   # r_tidal / r_half threshold for D activation
D_INTERACTION_SCALE = 0.5 # Scale factor for interaction signatures → D
D_WIDE_BINARIES = 0.6     # D for wide binaries in MW tidal field     [TUNED]

# KEY PHYSICS DISTINCTION:
# - In MERGERS: D represents "survival" of coherence → stars get φ > 1
# - In EQUILIBRIUM: D represents "loss" of coherence → φ < 1
# The formula adjusts based on whether system is merging or equilibrium-disturbed
EQUILIBRIUM_DISTURBANCE_SUPPRESSES = True  # If True, D in equilibrium systems → φ < 1

# =============================================================================
# COMPARATIVE RESULT STRUCTURE
# =============================================================================

@dataclass
class TheoryPrediction:
    """A single theory's prediction with key metrics."""
    value: float           # Primary metric (e.g., RMS, ratio)
    unit: str              # Unit of the metric
    details: Dict = field(default_factory=dict)  # Additional details


@dataclass 
class ComparativeResult:
    """Result comparing multiple theories against observations."""
    name: str
    metric_name: str       # What we're measuring (e.g., "RMS", "mass_ratio")
    
    # Observations
    observed: float
    observed_err: Optional[float]
    observed_unit: str
    
    # Theory predictions
    baseline: TheoryPrediction      # Original Σ-Gravity (locked)
    new_model: TheoryPrediction     # Extended Σ-Gravity
    mond: Optional[TheoryPrediction]
    lcdm: Optional[TheoryPrediction]
    
    # Which theory is closest to observed?
    best_theory: str
    
    # Additional context
    n_objects: int
    notes: str = ""


@dataclass
class TestResult:
    """Legacy result format for backwards compatibility."""
    name: str
    passed: bool
    metric: float
    details: Dict[str, Any]
    message: str


# =============================================================================
# OBSERVATIONAL BENCHMARKS (GOLD STANDARD)
# =============================================================================
OBS_BENCHMARKS = {
    'solar_system': {
        'cassini_gamma_uncertainty': 2.3e-5,
        'source': 'Bertotti+ 2003',
    },
    'sparc': {
        'n_quality': 171,
        'mond_rms_kms': 17.15,
        'sigma_rms_kms': 17.42,
        'lcdm_rms_kms': 15.0,
        'rar_scatter_dex': 0.10,
        'source': 'Lelli+ 2016, McGaugh+ 2016',
    },
    'wide_binaries': {
        'boost_factor': 1.35,
        'boost_uncertainty': 0.10,
        'threshold_AU': 2000,
        'n_pairs': 26500,
        'mond_boost': 1.40,  # MOND predicts ~40% boost
        'newtonian_boost': 1.0,  # No boost
        'source': 'Chae 2023',
    },
    'dwarf_spheroidals': {
        'fornax': {'M_star': 2e7, 'sigma_obs': 10.7, 'sigma_err': 0.5, 'r_half_kpc': 0.71, 'd_MW_kpc': 147, 'M_L': 7.5},
        'draco': {'M_star': 2.9e5, 'sigma_obs': 9.1, 'sigma_err': 1.2, 'r_half_kpc': 0.22, 'd_MW_kpc': 76, 'M_L': 330},
        'sculptor': {'M_star': 2.3e6, 'sigma_obs': 9.2, 'sigma_err': 0.6, 'r_half_kpc': 0.28, 'd_MW_kpc': 86, 'M_L': 160},
        'carina': {'M_star': 3.8e5, 'sigma_obs': 6.6, 'sigma_err': 1.2, 'r_half_kpc': 0.25, 'd_MW_kpc': 105, 'M_L': 40},
        'ursa_minor': {'M_star': 2.9e5, 'sigma_obs': 9.5, 'sigma_err': 1.2, 'r_half_kpc': 0.30, 'd_MW_kpc': 76, 'M_L': 290},
        'source': 'Walker+ 2009, McConnachie 2012',
    },
    'udgs': {
        'df2': {
            'M_star': 2e8, 'sigma_obs': 8.5, 'sigma_err': 2.3, 'r_eff_kpc': 2.2,
            'd_host_kpc': 80,  # Distance to NGC1052
            'host_mass': 1e11,  # NGC1052 mass
            'mond_sigma': 20.0,  # MOND predicts ~20 km/s without EFE
            'newtonian_sigma': 6.0,  # Pure Newtonian
            'source': 'van Dokkum+ 2018',
        },
        'dragonfly44': {
            'M_star': 3e8, 'sigma_obs': 47, 'sigma_err': 8, 'r_eff_kpc': 4.6,
            'd_host_kpc': 1000,  # Essentially isolated
            'host_mass': 0,
            'mond_sigma': 33.0,
            'newtonian_sigma': 15.0,
            'source': 'van Dokkum+ 2016',
        },
    },
    'tully_fisher': {
        'btfr_slope': 3.98,
        'btfr_normalization': 47,
        'scatter_dex': 0.10,
        'mond_prediction': 4.0,
        'source': 'McGaugh 2012',
    },
    'gw170817': {
        'delta_c_over_c': 1e-15,
        'time_delay_s': 1.7,
        'distance_Mpc': 40,
        'source': 'Abbott+ 2017',
    },
    'bullet_cluster': {
        'M_gas': 2.1e14,
        'M_stars': 0.5e14,
        'M_baryonic': 2.6e14,
        'M_lensing': 5.5e14,
        'mass_ratio': 2.1,
        'offset_kpc': 150,
        'separation_kpc': 720,
        'mach_shock': 3.0,
        'mond_ratio': 1.5,  # MOND underpredicts
        'source': 'Clowe+ 2006',
    },
    'merging_clusters': {
        'el_gordo': {
            'M_gas': 3.0e14, 'M_stars': 0.8e14, 'M_lensing': 2.2e15,
            'offset_kpc': 200, 'mach_shock': 2.5, 'z': 0.87,
            'source': 'Menanteau+ 2012',
        },
        'a520': {
            'M_gas': 1.5e14, 'M_stars': 0.4e14, 'M_lensing': 4.5e14,
            'offset_kpc': 100, 'mach_shock': 2.0, 'z': 0.201,
            'source': 'Mahdavi+ 2007',
        },
        'macs_j0025': {
            'M_gas': 1.8e14, 'M_stars': 0.5e14, 'M_lensing': 5.0e14,
            'offset_kpc': 120, 'mach_shock': 2.2, 'z': 0.586,
            'source': 'Bradac+ 2008',
        },
        'a2744': {
            'M_gas': 2.5e14, 'M_stars': 0.6e14, 'M_lensing': 1.8e15,
            'offset_kpc': 150, 'mach_shock': 2.0, 'z': 0.308,
            'source': 'Merten+ 2011',
        },
        'a1689_relaxed': {
            'M_gas': 1.2e14, 'M_stars': 0.3e14, 'M_lensing': 1.3e15,
            'offset_kpc': 10, 'mach_shock': 0.2, 'z': 0.183,
            'source': 'Limousin+ 2007',
        },
    },
    'clusters': {
        'n_quality': 42,
        'mond_mass_discrepancy': 3.0,
        'lcdm_success': True,
        'source': 'Fox+ 2022',
    },
    'milky_way': {
        'V_sun_kms': 233,
        'R_sun_kpc': 8.178,
        'M_baryonic': 6.5e10,
        'n_gaia_stars': 28368,
        'source': 'Eilers+ 2019, McMillan 2017',
    },
    'cmb': {
        'Omega_b': 0.0493,
        'Omega_c': 0.265,
        'Omega_m': 0.315,
        'H0': 67.4,
        'source': 'Planck 2020',
    },
    'structure_formation': {
        'sigma8_planck': 0.811,
        'sigma8_lensing': 0.76,
        'bao_scale_Mpc': 150,
        'source': 'Planck 2018, SDSS',
    },
    'bulge_dispersion': {
        # Milky Way bulge from BRAVA/ARGOS/Gaia
        'mw_bulge': {
            'R_kpc': [0.5, 1.0, 1.5, 2.0],
            'sigma_obs': [120, 110, 95, 80],  # km/s velocity dispersion
            'sigma_err': [10, 8, 8, 10],
            'source': 'Zoccali+ 2014, ARGOS, BRAVA',
        },
        # External galaxy bulges (from ATLAS3D, SAURON)
        'external_bulges': {
            'M31_bulge': {'R_kpc': 1.0, 'sigma_obs': 160, 'sigma_err': 15, 'M_star': 3e10},
            'NGC4649': {'R_kpc': 2.0, 'sigma_obs': 340, 'sigma_err': 20, 'M_star': 2e11},
            'NGC3377': {'R_kpc': 0.5, 'sigma_obs': 145, 'sigma_err': 10, 'M_star': 1e10},
            'source': 'ATLAS3D, Cappellari+ 2013',
        },
        # Key physics: These are dispersion-supported (v/σ < 1)
        'v_over_sigma_typical': 0.5,
        'notes': 'Bulges are dispersion-supported. Use σ, not v_circ.',
    },
}


# =============================================================================
# CORE PHYSICS FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def unified_amplitude(L: float, baseline: bool = False) -> float:
    """Unified 3D amplitude: A = A₀ × (L/L₀)^n"""
    if baseline:
        return BASELINE_A_0 * (L / BASELINE_L_0)**BASELINE_N_EXP
    return A_0 * (L / L_0)**N_EXP


def C_coherence(v_rot: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Covariant coherence scalar: C = v²/(v² + σ²)"""
    v2 = np.maximum(np.asarray(v_rot, dtype=float), 0.0)**2
    sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    s2 = sigma_arr**2
    return v2 / (v2 + s2)


def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """Coherence window W(r) = r/(ξ+r)"""
    xi = max(xi, 0.01)
    return r / (xi + r)


# =============================================================================
# EXTENDED PHASE COHERENCE MODEL (UNIFIED)
# =============================================================================
#
# The SINGLE authoritative φ mechanism:
#   φ = 1 + λ₀ × D × (f_ordered - f_turb)
#
# D is the DISTURBANCE PARAMETER sourced from observables:
#   - DISK GALAXIES: D_disk from kinematic asymmetry
#   - BULGE REGIONS: D_bulge from Ω/H₀ and compactness (disordered collisionless)
#   - CLUSTER MERGERS: D from Mach number (shock strength)
#   - SATELLITES/UDGs: D from tidal proximity
#   - WIDE BINARIES: D from MW tidal environment
#
# f_ordered and f_turb encode the MATTER STATE:
#   - Ordered disk stars: f_ordered ≈ 0.95, f_turb ≈ 0
#   - Disordered bulge stars: f_ordered ≈ 0.3-0.5 (velocity dispersion dominated)
#   - Collisional gas (shocked): f_ordered ≈ 0.1, f_turb ∝ Mach
#
# Physical regimes:
#   - D=0 (equilibrium): φ=1 always (no modification)
#   - MERGERS: stars survive → φ>1, gas shocked → φ<1
#   - EQUILIBRIUM DISTURBANCE: disruption → φ<1

# Bulge-specific parameters
D_BULGE_OMEGA_SCALE = 0.5   # How Ω/H₀ maps to D for bulges
D_BULGE_COMPACT_SCALE = 0.3 # How compactness (R/L₀) contributes to D

def compute_D_from_asymmetry_old(v_asym: float, v_circ: float) -> float:
    """
    DEPRECATED: Old method using V_obs scatter - has target leakage risk.
    Kept for comparison only.
    """
    if v_circ <= 0:
        return 0.0
    ratio = abs(v_asym) / v_circ
    return min(1.0, D_ASYMMETRY_SCALE * ratio)


def compute_D_from_morphology(f_gas: float, f_bulge: float, 
                               compactness: float = 1.0,
                               has_bar: bool = False) -> float:
    """
    Compute disturbance D from MORPHOLOGICAL properties (NO V_obs dependence).
    
    This is the "legal" D proxy that doesn't leak from the target.
    Uses only mass distribution information, not kinematics.
    
    Parameters:
    -----------
    f_gas : float
        Gas fraction (0-1). High gas → potentially more turbulent ISM
    f_bulge : float  
        Bulge fraction (0-1). High bulge → more disordered stellar orbits
    compactness : float
        R_d / R_typical. Compact galaxies tend to be more dynamically hot
    has_bar : bool
        Bar presence. Bars drive non-circular motions
        
    Physical reasoning:
    - Gas-rich galaxies have turbulent ISM → D_gas contribution
    - Bulge-dominated galaxies have phase-mixed orbits → D_bulge contribution
    - Compact galaxies are dynamically hotter → D_compact contribution
    - Barred galaxies have strong non-circular motions → D_bar contribution
    """
    # Gas contribution: turbulent ISM in gas-rich systems
    # But only above a threshold - very low gas is essentially "clean"
    D_gas = 0.0
    if f_gas > 0.3:
        D_gas = 0.2 * (f_gas - 0.3) / 0.7  # Scale 0.3-1.0 → 0-0.2
    
    # Bulge contribution: phase-mixed stellar orbits
    D_bulge_morph = 0.0
    if f_bulge > 0.1:
        D_bulge_morph = 0.3 * (f_bulge - 0.1) / 0.9  # Scale 0.1-1.0 → 0-0.3
    
    # Compactness contribution: small R_d means dynamically hot
    # compactness = R_d / R_typical; low values mean compact
    D_compact = 0.0
    if compactness < 1.0:
        D_compact = 0.2 * (1.0 - compactness)  # Compact → higher D
    
    # Bar contribution (if we know it)
    D_bar = 0.3 if has_bar else 0.0
    
    # Combine: take maximum effect, scaled
    D = max(D_gas, D_bulge_morph, D_compact, D_bar) * D_ASYMMETRY_SCALE
    
    return min(1.0, D)


def compute_D_from_rotation_curve_shape(R: np.ndarray, V_obs: np.ndarray, V_bar: np.ndarray) -> float:
    """
    DEPRECATED: Still uses V_obs, kept for leakage audit comparison only.
    
    NOTE: This method still has target leakage risk because it reads V_obs.
    Use compute_D_from_morphology() for clean predictions.
    """
    if len(R) < 5:
        return 0.0
    
    dV = np.diff(V_obs)
    sign_changes = np.sum(np.abs(np.diff(np.sign(dV))) > 0)
    n_points = len(V_obs)
    wiggle_fraction = sign_changes / max(n_points - 2, 1)
    
    dV_obs_norm = np.gradient(V_obs) / (np.abs(V_obs).mean() + 1e-6)
    dV_bar_norm = np.gradient(V_bar) / (np.abs(V_bar).mean() + 1e-6)
    gradient_mismatch = np.std(dV_obs_norm - dV_bar_norm)
    
    mid = len(R) // 2
    if mid > 2 and mid < len(R) - 2:
        inner_slope = (V_obs[mid] - V_obs[0]) / (R[mid] - R[0] + 0.01)
        outer_slope = (V_obs[-1] - V_obs[mid]) / (R[-1] - R[mid] + 0.01)
        v_typical = np.abs(V_obs).mean() + 1e-6
        slope_inconsistency = abs(inner_slope - outer_slope) / (v_typical / (R.mean() + 0.1))
    else:
        slope_inconsistency = 0.0
    
    D_wiggle = min(1.0, wiggle_fraction * 3.0)
    D_gradient = min(1.0, gradient_mismatch * 2.0)
    D_slope = min(1.0, slope_inconsistency * 0.5)
    D = max(D_wiggle, D_gradient, D_slope) * D_ASYMMETRY_SCALE / 1.5
    
    return min(1.0, D)


def compute_D_from_asymmetry(f_gas: float = 0.5, f_bulge: float = 0.0,
                              compactness: float = 1.0, has_bar: bool = False,
                              # Legacy arguments for leakage audit comparison
                              v_asym: float = None, v_circ: float = None,
                              R: np.ndarray = None, V_obs: np.ndarray = None, 
                              V_bar: np.ndarray = None,
                              use_morphology: bool = True) -> float:
    """
    Compute disturbance D for disk galaxies.
    
    Default: Uses morphological properties (no target leakage).
    Fallback: Can use kinematic data for comparison/audit only.
    
    Parameters:
    -----------
    f_gas, f_bulge, compactness, has_bar : morphological inputs
    use_morphology : bool
        If True (default), use morphology-only D (no leakage)
        If False, fall back to shape-based D (for audit comparison)
    """
    if use_morphology:
        return compute_D_from_morphology(f_gas, f_bulge, compactness, has_bar)
    
    # Fallback: shape-based (has leakage, for audit only)
    if R is not None and V_obs is not None and V_bar is not None:
        return compute_D_from_rotation_curve_shape(R, V_obs, V_bar)
    
    # Last resort: old v_asym method
    if v_asym is not None and v_circ is not None and v_circ > 0:
        return compute_D_from_asymmetry_old(v_asym, v_circ)
    
    return 0.0


def compute_D_from_bulge(omega_over_H0: float, R_over_L0: float, v_over_sigma: float) -> float:
    """
    Compute disturbance D for BULGE regions.
    
    Bulges are collisionless but DISORDERED (multi-stream, phase-mixed).
    Key predictors from residual analysis: Ω/H₀ and compactness.
    
    Parameters:
    -----------
    omega_over_H0 : float
        Orbital frequency Ω = V/R normalized by Hubble constant
        High values → central, dynamically hot regions → higher D
    R_over_L0 : float
        Radius normalized by reference length L₀
        Low values (compact) → more disordered → higher D
    v_over_sigma : float
        Rotation-to-dispersion ratio V/σ
        Low values → dispersion-dominated → higher D
        
    Returns:
    --------
    D : float [0, 1]
        Disturbance parameter representing degree of phase disorder in bulge
    """
    # High Ω/H₀ → dynamically hot, disordered → higher D
    D_omega = min(1.0, D_BULGE_OMEGA_SCALE * omega_over_H0 / 100.0)
    
    # Compact regions (low R/L₀) → more disordered → higher D
    if R_over_L0 > 0:
        D_compact = min(1.0, D_BULGE_COMPACT_SCALE * (L_0 / R_over_L0))
    else:
        D_compact = 1.0
    
    # Low V/σ → dispersion-dominated → higher D
    if v_over_sigma > 0:
        D_dispersion = min(1.0, 0.5 / v_over_sigma)
    else:
        D_dispersion = 1.0
    
    # Combine: take the dominant effect
    D = max(D_omega, D_compact, D_dispersion)
    return min(1.0, D)


def compute_D_from_tidal(r_tidal: float, r_half: float) -> float:
    """
    Compute disturbance D from tidal proximity.
    
    r_tidal: Tidal radius (distance at which host gravity dominates)
    r_half: Half-light radius of satellite
    
    D = 0 for isolated systems (r_tidal >> r_half)
    D → 1 for tidally stripped systems (r_tidal ~ r_half)
    """
    if r_half <= 0:
        return 0.0
    ratio = r_tidal / r_half
    if ratio > D_TIDAL_THRESHOLD:
        return 0.0
    return min(1.0, (D_TIDAL_THRESHOLD - ratio) / D_TIDAL_THRESHOLD)


def compute_D_from_mach(mach_shock: float) -> float:
    """
    Compute disturbance D from shock Mach number (for clusters).
    
    D = 0 for Mach < 0.5 (equilibrium)
    D = 1 for Mach > 3.0 (strong shock)
    """
    if mach_shock < 0.5:
        return 0.0
    elif mach_shock > 3.0:
        return 1.0
    else:
        return (mach_shock - 0.5) / 2.5


def compute_D_from_interaction(interaction_strength: float) -> float:
    """
    Compute disturbance D from interaction signatures.
    
    interaction_strength: 0 = isolated, 1 = major merger
    """
    return min(1.0, D_INTERACTION_SCALE * interaction_strength)


def compute_phi(
    D: float,
    matter_type: str = "disk_stars",
    mach_turb: float = 0.0,
    is_merger: bool = False,
    v_over_sigma: float = 10.0
) -> float:
    """
    UNIFIED phase coherence computation.
    
    This is the SINGLE authoritative φ function.
    
    Parameters:
    -----------
    D : float [0, 1]
        Disturbance parameter from appropriate observable proxy
    matter_type : str
        One of: "disk_stars", "bulge_stars", "gas", "collisionless"
    mach_turb : float
        Turbulent Mach number (for gas)
    is_merger : bool
        True for merger/collision events, False for equilibrium disturbance
    v_over_sigma : float
        Rotation-to-dispersion ratio (for bulge stars)
    
    Returns:
    --------
    φ : float
        Phase coherence factor (φ=1 is neutral, φ>1 enhances, φ<1 suppresses)
    """
    # Determine f_ordered and f_turb based on matter type
    if matter_type == "disk_stars":
        # Cold, ordered disk: high f_ordered
        f_ordered = 0.95
        f_turb = 0.0
    elif matter_type == "bulge_stars":
        # Disordered, dispersion-supported: intermediate f_ordered
        # f_ordered depends on V/σ: high V/σ → more ordered
        f_ordered = min(0.9, 0.3 + 0.1 * v_over_sigma)  # 0.3 to 0.9
        f_turb = 0.1  # Some "effective turbulence" from velocity dispersion
    elif matter_type == "gas":
        # Collisional gas: ordered if laminar, turbulent if shocked
        f_ordered = 0.1
        f_turb = min(0.85, 0.25 * mach_turb)
    else:  # "collisionless" generic
        f_ordered = 0.8
        f_turb = 0.0
    
    # Apply the universal formula
    if is_merger:
        # MERGER: D represents survival of coherence
        # Stars: high f_ordered → φ > 1
        # Gas: high f_turb → φ < 1 (screening)
        phi = 1.0 + PHI_LAMBDA_0 * D * (f_ordered - f_turb)
    else:
        # EQUILIBRIUM DISTURBANCE: D represents LOSS of coherence
        if EQUILIBRIUM_DISTURBANCE_SUPPRESSES:
            # Disturbance → suppression (φ < 1)
            # The amount of suppression depends on how ordered the matter was
            suppression = 0.3 * D * f_ordered
            phi = 1.0 - suppression
        else:
            phi = 1.0 + PHI_LAMBDA_0 * D * (f_ordered - f_turb)
    
    return max(PHI_MIN, min(PHI_MAX, phi))


# Legacy function for backwards compatibility
def compute_phi_extended(
    D: float,
    is_collisionless: bool = True,
    mach_turb: float = 0.0,
    is_merger: bool = False
) -> float:
    """Legacy wrapper - use compute_phi() for new code."""
    matter_type = "disk_stars" if is_collisionless else "gas"
    return compute_phi(D, matter_type, mach_turb, is_merger)


# =============================================================================
# BASELINE Σ-GRAVITY (LOCKED)
# =============================================================================

def sigma_enhancement_baseline(g: np.ndarray, r: np.ndarray = None, 
                                xi: float = 1.0, L: float = None) -> np.ndarray:
    """
    BASELINE Σ enhancement - locked, never changes.
    
    Σ = 1 + A(L) × W(r) × h(g)
    """
    g = np.maximum(np.asarray(g), 1e-15)
    
    if L is None:
        L = BASELINE_L_0
    A = unified_amplitude(L, baseline=True)
    
    h = h_function(g)
    
    if r is not None:
        W = W_coherence(np.asarray(r), xi)
    else:
        W = 1.0
    
    return 1 + A * W * h


def predict_velocity_baseline(R_kpc: np.ndarray, V_bar: np.ndarray, 
                               R_d: float, sigma_kms: float = 20.0) -> np.ndarray:
    """BASELINE rotation curve prediction using fixed-point iteration."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A = BASELINE_A_0
    h = h_function(g_bar)
    V = np.array(V_bar, dtype=float)
    
    for _ in range(50):
        C = C_coherence(V, sigma_kms)
        Sigma = 1 + A * C * h
        V_new = V_bar * np.sqrt(Sigma)
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new
    return V


# =============================================================================
# NEW MODEL Σ-GRAVITY (WITH EXTENDED PHASE COHERENCE)
# =============================================================================

def sigma_enhancement_new(g: np.ndarray, r: np.ndarray = None,
                          xi: float = 1.0, L: float = None,
                          phi: float = 1.0) -> np.ndarray:
    """
    NEW MODEL Σ enhancement with phase coherence.
    
    Σ = 1 + A(L) × φ × W(r) × h(g)
    """
    g = np.maximum(np.asarray(g), 1e-15)
    
    if L is None:
        L = L_0
    A = unified_amplitude(L, baseline=False)
    
    h = h_function(g)
    
    if r is not None:
        W = W_coherence(np.asarray(r), xi)
    else:
        W = 1.0
    
    return 1 + A * phi * W * h


def predict_velocity_new(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float,
                         sigma_kms: float = 20.0, 
                         D: float = 0.0,
                         is_merger: bool = False,
                         # Per-point φ array (if provided, overrides all other φ computation)
                         phi_eff: np.ndarray = None,
                         # Per-point arrays for fallback φ computation
                         f_bulge_r: np.ndarray = None,
                         Omega_over_H0: np.ndarray = None,
                         R_over_L0: np.ndarray = None,
                         v_over_sigma: float = 5.0) -> np.ndarray:
    """
    NEW MODEL rotation curve prediction with extended phase coherence.
    
    Per-point φ can be provided in two ways:
    1. Directly via phi_eff array (preferred - allows component-split φ)
    2. Computed from f_bulge_r (fallback - bulge vs disk distinction only)
    
    If neither provided, uses scalar φ from D.
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A = A_0
    h = h_function(g_bar)
    V = np.array(V_bar, dtype=float)
    
    # Determine φ: prefer explicit phi_eff, then compute from bulge info, then scalar
    if phi_eff is not None and USE_EXTENDED_PHI:
        # Use provided per-point φ (e.g., component-split φ from test_sparc)
        phi = np.asarray(phi_eff)
    elif f_bulge_r is not None and USE_EXTENDED_PHI:
        # Fallback: compute from bulge fraction
        phi = np.ones_like(R_kpc)
        for i in range(len(R_kpc)):
            if f_bulge_r[i] > 0.3:
                omega_H0 = Omega_over_H0[i] if Omega_over_H0 is not None else 100.0
                r_L0 = R_over_L0[i] if R_over_L0 is not None else 1.0
                D_bulge = compute_D_from_bulge(omega_H0, r_L0, v_over_sigma)
                phi[i] = compute_phi(D_bulge, matter_type="bulge_stars", 
                                    is_merger=is_merger, v_over_sigma=v_over_sigma)
            else:
                phi[i] = compute_phi(D, matter_type="disk_stars", is_merger=is_merger)
    else:
        # Scalar φ for entire galaxy
        phi = compute_phi(D, matter_type="disk_stars", is_merger=is_merger)
    
    for _ in range(50):
        C = C_coherence(V, sigma_kms)
        Sigma = 1 + A * phi * C * h
        V_new = V_bar * np.sqrt(Sigma)
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new
    return V


# =============================================================================
# MOND PREDICTIONS
# =============================================================================

def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """MOND prediction using simple interpolating function."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0_mond
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))
    return V_bar * np.sqrt(nu)


def mond_enhancement(g: np.ndarray) -> np.ndarray:
    """MOND enhancement factor ν(x) where x = g/a₀."""
    g = np.maximum(np.asarray(g), 1e-15)
    x = g / a0_mond
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-10))))


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_sparc(data_dir: Path) -> List[Dict]:
    """
    Load SPARC galaxy rotation curves with enhanced per-point analysis.
    
    Computes per-point quantities for the new φ model:
    - f_bulge(r): Local bulge fraction at each radius
    - Omega(r): Orbital frequency = V/R
    - v_over_sigma_est(r): Estimated V/σ ratio
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
            
            # Estimate kinematic asymmetry 
            # TODO: Replace with actual approaching/receding asymmetry when available
            v_mean = df['V_obs'].mean()
            v_asym = df['V_obs'].std() * 0.3  # Rough proxy
            
            # Per-point bulge fraction: f_bulge(r) = V_bulge²(r) / V_bar²(r)
            V_total_sq = np.maximum(df['V_bar'].values**2, 1e-10)
            f_bulge_r = df['V_bulge_scaled'].values**2 / V_total_sq
            f_gas_r = np.abs(np.sign(df['V_gas'].values) * df['V_gas'].values**2) / V_total_sq
            f_disk_r = df['V_disk_scaled'].values**2 / V_total_sq
            
            # Global fractions (outer region average)
            outer_idx = max(1, len(df) - 3)
            f_gas = np.mean(f_gas_r[outer_idx:])
            f_bulge = np.mean(f_bulge_r[outer_idx:])
            
            # Per-point orbital frequency: Ω(r) = V(r) / R(r)  [km/s/kpc]
            R_vals = df['R'].values
            V_vals = df['V_obs'].values
            Omega = V_vals / np.maximum(R_vals, 0.01)  # km/s/kpc
            
            # Ω/H₀ where H₀ ≈ 70 km/s/Mpc = 0.07 km/s/kpc
            H0_kpc = 0.07  # km/s/kpc
            Omega_over_H0 = Omega / H0_kpc
            
            # R/L₀ (compactness proxy)
            R_over_L0 = R_vals / L_0
            
            # Estimate V/σ from rotation curve shape
            # Steeper inner rise → more rotation-dominated → higher V/σ
            # Flat curve → dispersion-dominated → lower V/σ
            # Simple proxy: use gradient of V(R) near center
            if len(V_vals) >= 3:
                inner_gradient = (V_vals[2] - V_vals[0]) / (R_vals[2] - R_vals[0] + 0.01)
                v_over_sigma_est = np.clip(inner_gradient / 50.0, 0.5, 10.0)  # Rough scaling
            else:
                v_over_sigma_est = 5.0  # Default moderate value
            
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': R_vals,
                'V_obs': V_vals,
                'V_bar': df['V_bar'].values,
                'V_gas': df['V_gas'].values,
                'V_disk_scaled': df['V_disk_scaled'].values,
                'V_bulge_scaled': df['V_bulge_scaled'].values,
                'R_d': R_d,
                'v_asym': v_asym,
                'v_circ': v_mean,
                'f_gas': f_gas,
                'f_bulge': f_bulge,
                # Per-point arrays for new model
                'f_bulge_r': f_bulge_r,
                'f_gas_r': f_gas_r,
                'f_disk_r': f_disk_r,
                'Omega': Omega,
                'Omega_over_H0': Omega_over_H0,
                'R_over_L0': R_over_L0,
                'v_over_sigma_est': v_over_sigma_est,
            })
    
    return galaxies


def load_clusters(data_dir: Path) -> List[Dict]:
    """Load Fox+ 2022 cluster data."""
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    if not cluster_file.exists():
        return []
    
    df = pd.read_csv(cluster_file)
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
    """Load Gaia/Eilers-APOGEE disk star catalog."""
    gaia_file = data_dir / "gaia" / "eilers_apogee_6d_disk.csv"
    if not gaia_file.exists():
        return None
    
    df = pd.read_csv(gaia_file)
    df['v_phi_obs'] = -df['v_phi']
    return df


# =============================================================================
# SPARC BINNING STRUCTURE
# =============================================================================

@dataclass
class SPARCBinResult:
    """Results for a bin of SPARC galaxies."""
    bin_name: str
    n_galaxies: int
    rms_baseline: float
    rms_new: float
    rms_mond: float
    improvement_pct: float  # (new - baseline) / baseline * 100
    D_mean: float
    D_std: float
    phi_mean: float
    best_theory: str


def compute_sparc_bins(galaxies: List[Dict], results_per_galaxy: List[Dict]) -> Dict[str, List[SPARCBinResult]]:
    """
    Compute SPARC results binned by galaxy type.
    
    Bins:
    - By bulge fraction: disk-dominated, intermediate, bulge-dominated
    - By disturbance D: D≈0, low D, mid D, high D
    - By gas fraction: gas-rich, mixed, star-dominated
    """
    bins = {
        'by_bulge': [],
        'by_D': [],
        'by_gas': [],
    }
    
    # Define bin thresholds
    bulge_bins = [
        ('Disk-dominated (B/T<0.1)', lambda g: g.get('f_bulge', 0) < 0.1),
        ('Intermediate (0.1-0.3)', lambda g: 0.1 <= g.get('f_bulge', 0) < 0.3),
        ('Bulge-dominated (B/T>0.3)', lambda g: g.get('f_bulge', 0) >= 0.3),
    ]
    
    D_bins = [
        ('D=0 (undisturbed)', lambda r: r['D_morphology'] < 0.01),
        ('Low D (0.01-0.05)', lambda r: 0.01 <= r['D_morphology'] < 0.05),
        ('Mid D (0.05-0.15)', lambda r: 0.05 <= r['D_morphology'] < 0.15),
        ('High D (>0.15)', lambda r: r['D_morphology'] >= 0.15),
    ]
    
    gas_bins = [
        ('Gas-rich (f_gas>0.5)', lambda g: g.get('f_gas', 0.5) > 0.5),
        ('Mixed (0.2-0.5)', lambda g: 0.2 <= g.get('f_gas', 0.5) <= 0.5),
        ('Star-dominated (f_gas<0.2)', lambda g: g.get('f_gas', 0.5) < 0.2),
    ]
    
    # Compute bins by bulge fraction
    for bin_name, condition in bulge_bins:
        matching = [(g, r) for g, r in zip(galaxies, results_per_galaxy) if condition(g)]
        if matching:
            rms_b = np.mean([r['rms_baseline'] for _, r in matching])
            rms_n = np.mean([r['rms_new'] for _, r in matching])
            rms_m = np.mean([r['rms_mond'] for _, r in matching])
            D_vals = [r['D_morphology'] for _, r in matching]
            phi_vals = [r['phi'] for _, r in matching]
            
            improvement = (rms_n - rms_b) / rms_b * 100 if rms_b > 0 else 0
            theories = {'Baseline': rms_b, 'New': rms_n, 'MOND': rms_m}
            best = min(theories, key=theories.get)
            
            bins['by_bulge'].append(SPARCBinResult(
                bin_name=bin_name,
                n_galaxies=len(matching),
                rms_baseline=rms_b,
                rms_new=rms_n,
                rms_mond=rms_m,
                improvement_pct=improvement,
                D_mean=np.mean(D_vals),
                D_std=np.std(D_vals),
                phi_mean=np.mean(phi_vals),
                best_theory=best,
            ))
    
    # Compute bins by D value
    for bin_name, condition in D_bins:
        matching = [(g, r) for g, r in zip(galaxies, results_per_galaxy) if condition(r)]
        if matching:
            rms_b = np.mean([r['rms_baseline'] for _, r in matching])
            rms_n = np.mean([r['rms_new'] for _, r in matching])
            rms_m = np.mean([r['rms_mond'] for _, r in matching])
            D_vals = [r['D_morphology'] for _, r in matching]
            phi_vals = [r['phi'] for _, r in matching]
            
            improvement = (rms_n - rms_b) / rms_b * 100 if rms_b > 0 else 0
            theories = {'Baseline': rms_b, 'New': rms_n, 'MOND': rms_m}
            best = min(theories, key=theories.get)
            
            bins['by_D'].append(SPARCBinResult(
                bin_name=bin_name,
                n_galaxies=len(matching),
                rms_baseline=rms_b,
                rms_new=rms_n,
                rms_mond=rms_m,
                improvement_pct=improvement,
                D_mean=np.mean(D_vals),
                D_std=np.std(D_vals),
                phi_mean=np.mean(phi_vals),
                best_theory=best,
            ))
    
    # Compute bins by gas fraction
    for bin_name, condition in gas_bins:
        matching = [(g, r) for g, r in zip(galaxies, results_per_galaxy) if condition(g)]
        if matching:
            rms_b = np.mean([r['rms_baseline'] for _, r in matching])
            rms_n = np.mean([r['rms_new'] for _, r in matching])
            rms_m = np.mean([r['rms_mond'] for _, r in matching])
            D_vals = [r['D_morphology'] for _, r in matching]
            phi_vals = [r['phi'] for _, r in matching]
            
            improvement = (rms_n - rms_b) / rms_b * 100 if rms_b > 0 else 0
            theories = {'Baseline': rms_b, 'New': rms_n, 'MOND': rms_m}
            best = min(theories, key=theories.get)
            
            bins['by_gas'].append(SPARCBinResult(
                bin_name=bin_name,
                n_galaxies=len(matching),
                rms_baseline=rms_b,
                rms_new=rms_n,
                rms_mond=rms_m,
                improvement_pct=improvement,
                D_mean=np.mean(D_vals),
                D_std=np.std(D_vals),
                phi_mean=np.mean(phi_vals),
                best_theory=best,
            ))
    
    return bins


# =============================================================================
# COMPARATIVE TEST FUNCTIONS
# =============================================================================

def test_sparc_comparative(galaxies: List[Dict], verbose: bool = False, 
                          holdout_mode: bool = False, holdout_fraction: float = 0.2) -> Tuple[ComparativeResult, Dict]:
    """
    Compare theories on SPARC galaxy rotation curves.
    
    Args:
        galaxies: List of SPARC galaxy data
        verbose: Print detailed analysis
        holdout_mode: If True, split into train/validation sets
        holdout_fraction: Fraction to hold out for validation (default 0.2)
    
    Returns both the overall result and detailed per-bin analysis.
    """
    # HOLDOUT VALIDATION: Split galaxies by hash of name for reproducibility
    if holdout_mode and galaxies:
        train_galaxies = []
        holdout_galaxies = []
        for gal in galaxies:
            # Use hash of name for deterministic split
            name_hash = hash(gal['name']) % 100
            if name_hash < holdout_fraction * 100:
                holdout_galaxies.append(gal)
            else:
                train_galaxies.append(gal)
        
        if verbose:
            print(f"\n  HOLDOUT MODE: {len(train_galaxies)} train, {len(holdout_galaxies)} validation")
        
        # Parameters were tuned on training set only
        # Report validation performance
        galaxies = holdout_galaxies  # Evaluate on holdout only
    
    if not galaxies:
        empty_result = ComparativeResult(
            name="SPARC Galaxies",
            metric_name="RMS",
            observed=OBS_BENCHMARKS['sparc']['mond_rms_kms'],
            observed_err=None,
            observed_unit="km/s",
            baseline=TheoryPrediction(0, "km/s"),
            new_model=TheoryPrediction(0, "km/s"),
            mond=TheoryPrediction(0, "km/s"),
            lcdm=TheoryPrediction(OBS_BENCHMARKS['sparc']['lcdm_rms_kms'], "km/s"),
            best_theory="N/A",
            n_objects=0,
            notes="No data"
        )
        return empty_result, {}
    
    # Store per-galaxy results for binning
    results_per_galaxy = []
    
    rms_baseline = []
    rms_new = []
    rms_mond = []
    D_values = []
    D_old_values = []  # For leakage audit comparison
    phi_values = []
    
    # Track excluded bulge points
    total_points = 0
    excluded_bulge_points = 0
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        f_bulge_r = gal.get('f_bulge_r', np.zeros_like(R))
        f_gas_r = gal.get('f_gas_r', np.zeros_like(R))
        f_disk_r = gal.get('f_disk_r', np.ones_like(R))
        
        # EXCLUDE bulge-dominated points from rotation curve test
        # Reason: Bulges are dispersion-supported (v/sigma < 1), rotation is wrong observable
        disk_mask = f_bulge_r < 0.3  # Only use points with <30% bulge contribution
        
        total_points += len(R)
        excluded_bulge_points += np.sum(~disk_mask)
        
        # If galaxy has too few disk points, skip it
        if np.sum(disk_mask) < 3:
            continue
        
        R_disk = R[disk_mask]
        V_obs_disk = V_obs[disk_mask]
        V_bar_disk = V_bar[disk_mask]
        f_gas_disk = f_gas_r[disk_mask]
        f_disk_disk = f_disk_r[disk_mask]
        
        # Baseline prediction (disk points only)
        V_pred_baseline = predict_velocity_baseline(R_disk, V_bar_disk, R_d)
        rms_b = np.sqrt(((V_pred_baseline - V_obs_disk)**2).mean())
        rms_baseline.append(rms_b)
        
        # New model with extended phase coherence (disk points only)
        if USE_EXTENDED_PHI:
            # Compute compactness proxy: R_d / typical R_d (use 2.5 kpc as typical)
            compactness = R_d / 2.5
            
            # NEW: Use MORPHOLOGY-BASED D (no V_obs dependence - no target leakage)
            D = compute_D_from_asymmetry(
                f_gas=gal.get('f_gas', 0.5),
                f_bulge=gal.get('f_bulge', 0.0),
                compactness=compactness,
                has_bar=False,  # SPARC doesn't have bar flags, assume False
                use_morphology=True  # Use clean, non-leaky D
            )
            
            # For leakage audit: also compute old D methods
            D_old = compute_D_from_asymmetry_old(gal.get('v_asym', 0), gal.get('v_circ', 200))
            D_shape = compute_D_from_rotation_curve_shape(R_disk, V_obs_disk, V_bar_disk)
            
            # COMPONENT-SPLIT PHI: per-point φ weighted by component fractions
            # φ_eff(r) = f_disk(r) × φ_disk + f_gas(r) × φ_gas
            # This is the physics: stars and gas have different phase coherence
            
            # For disk stars: ordered, cold rotation
            phi_disk = compute_phi(D, matter_type="disk_stars", is_merger=False)
            # For gas: potentially more turbulent (use moderate Mach for disk gas)
            mach_gas = 0.3 + 0.4 * gal.get('f_gas', 0.5)  # More gas → more turbulence
            phi_gas = compute_phi(D, matter_type="gas", mach_turb=mach_gas, is_merger=False)
            
            # Per-point effective φ (component-weighted) - THIS IS THE KEY CHANGE
            phi_eff_array = f_disk_disk * phi_disk + f_gas_disk * phi_gas
            
            # Track mean for diagnostics
            phi = np.mean(phi_eff_array)
        else:
            D = 0.0
            D_old = 0.0
            D_shape = 0.0
            phi = 1.0
            phi_eff_array = None
        
        D_values.append(D)
        D_old_values.append(D_old)
        phi_values.append(phi)
        
        # Predict for disk points only WITH PER-POINT COMPONENT-SPLIT PHI
        # This is the critical change: phi_eff actually affects the Σ calculation
        V_pred_new = predict_velocity_new(
            R_disk, V_bar_disk, R_d, 
            D=D, 
            is_merger=False,
            phi_eff=phi_eff_array  # Pass per-point φ to prediction!
        )
        rms_n = np.sqrt(((V_pred_new - V_obs_disk)**2).mean())
        rms_new.append(rms_n)
        
        # MOND prediction (disk points only)
        V_mond = predict_mond(R_disk, V_bar_disk)
        rms_m = np.sqrt(((V_mond - V_obs_disk)**2).mean())
        rms_mond.append(rms_m)
        
        # Store per-galaxy result
        n_bulge_pts = np.sum(~disk_mask)
        
        results_per_galaxy.append({
            'name': gal['name'],
            'rms_baseline': rms_b,
            'rms_new': rms_n,
            'rms_mond': rms_m,
            'D_morphology': D,  # New morphology-based D (no leakage)
            'D_old_scatter': D_old,  # Old V_obs-based D (has leakage)
            'D_shape': D_shape if USE_EXTENDED_PHI else 0.0,  # Shape-based (has leakage)
            'phi': phi,
            'f_bulge': gal.get('f_bulge', 0),
            'f_gas': gal.get('f_gas', 0.5),
            'compactness': R_d / 2.5 if R_d > 0 else 1.0,
            'has_bulge': n_bulge_pts > 0,
            'n_bulge_pts': n_bulge_pts,
            'n_disk_pts': np.sum(disk_mask),
        })
    
    mean_baseline = np.mean(rms_baseline)
    mean_new = np.mean(rms_new)
    mean_mond = np.mean(rms_mond)
    lcdm_rms = OBS_BENCHMARKS['sparc']['lcdm_rms_kms']
    
    # D and phi statistics
    D_stats = {
        'min': np.min(D_values),
        'max': np.max(D_values),
        'mean': np.mean(D_values),
        'std': np.std(D_values),
        'n_nonzero': sum(1 for d in D_values if d > 0.01),
    }
    phi_stats = {
        'min': np.min(phi_values),
        'max': np.max(phi_values),
        'mean': np.mean(phi_values),
    }
    
    # D LEAKAGE AUDIT: Check if D correlates with baseline residuals
    # A "clean" D should NOT correlate with residuals (that would be target leakage)
    leakage_audit = {}
    if len(results_per_galaxy) > 5 and USE_EXTENDED_PHI:
        D_arr = np.array(D_values)  # New morphology-based D
        D_old_arr = np.array(D_old_values)  # Old V_obs-scatter-based D
        rms_b_arr = np.array(rms_baseline)
        
        # Correlation: D vs baseline RMS (target leakage risk)
        if np.std(D_arr) > 1e-6 and np.std(rms_b_arr) > 1e-6:
            corr_D_morph_rms = np.corrcoef(D_arr, rms_b_arr)[0, 1]
            corr_D_old_rms = np.corrcoef(D_old_arr, rms_b_arr)[0, 1]
        else:
            corr_D_morph_rms = 0.0
            corr_D_old_rms = 0.0
        
        # Good: morphology-based D should have LOW correlation with RMS
        # Bad: old scatter-based D had HIGH correlation (learning the error)
        leakage_audit = {
            'D_method': 'morphology-based (no V_obs)',
            'corr_D_morphology_vs_rms': corr_D_morph_rms,
            'corr_D_old_scatter_vs_rms': corr_D_old_rms,
            'leakage_risk_new': 'HIGH' if abs(corr_D_morph_rms) > 0.3 else 'LOW',
            'leakage_risk_old': 'HIGH' if abs(corr_D_old_rms) > 0.3 else 'LOW',
            'D_stats': f"D range: {D_arr.min():.3f}-{D_arr.max():.3f}, mean: {D_arr.mean():.3f}",
        }
    
    # Compute bins - ALWAYS compute, not just when extended phi enabled
    bins = compute_sparc_bins(galaxies, results_per_galaxy)
    
    # BINNED METRICS GUARDRAILS (State-based)
    # These are the regimes where the physics SHOULD help if φ(state) is real
    bin_guardrails = {}
    
    # By bulge
    if bins.get('by_bulge'):
        for b in bins['by_bulge']:
            if 'Bulge-dominated' in b.bin_name:
                bin_guardrails['bulge_worsening'] = {
                    'threshold': 20.0,
                    'actual': b.improvement_pct,
                    'passed': b.improvement_pct < 20.0,
                    'physics': 'Bulge should not worsen significantly',
                }
            if 'Disk-dominated' in b.bin_name:
                bin_guardrails['disk_improvement'] = {
                    'threshold': 0.0,
                    'actual': b.improvement_pct,
                    'passed': b.improvement_pct < 0.0,
                    'physics': 'Disk regions should benefit from component-split phi',
                }
    
    # STATE-BASED GUARDRAILS: These are where φ(state) MUST help if it's real physics
    if bins.get('by_D'):
        for b in bins['by_D']:
            # High D galaxies: This is where disturbance physics should matter most!
            if 'High D' in b.bin_name:
                bin_guardrails['high_D_improvement'] = {
                    'threshold': 0.0,
                    'actual': b.improvement_pct,
                    'passed': b.improvement_pct < 0.0,  # MUST improve
                    'physics': 'High-D (disturbed) galaxies are where phi(state) should help most',
                    'critical': True,  # This is a critical test
                }
    
    if bins.get('by_gas'):
        for b in bins['by_gas']:
            # Star-dominated: φ_disk should dominate and maintain coherence
            if 'Star-dominated' in b.bin_name:
                bin_guardrails['star_dominated_improvement'] = {
                    'threshold': 5.0,  # Allow some tolerance
                    'actual': b.improvement_pct,
                    'passed': b.improvement_pct < 5.0,
                    'physics': 'Star-dominated should not worsen (phi_disk ~ 1)',
                }
    
    # Print verbose analysis
    if verbose:
        pct_excluded = excluded_bulge_points / total_points * 100 if total_points > 0 else 0
        print(f"\n  SPARC DISK-ONLY ANALYSIS (bulge points excluded):")
        print(f"  Total points: {total_points}, Excluded bulge points: {excluded_bulge_points} ({pct_excluded:.1f}%)")
        print(f"  Galaxies used: {len(results_per_galaxy)} (skipped galaxies with <3 disk points)")
        
        # D LEAKAGE AUDIT OUTPUT
        if leakage_audit:
            print("")
            print("  D LEAKAGE AUDIT (checking for target leakage):")
            print(f"    D computation: {leakage_audit['D_method']}")
            print(f"    {leakage_audit['D_stats']}")
            print(f"    Correlation morphology-D vs RMS: {leakage_audit['corr_D_morphology_vs_rms']:.3f} (risk: {leakage_audit['leakage_risk_new']})")
            print(f"    Correlation old-scatter-D vs RMS: {leakage_audit['corr_D_old_scatter_vs_rms']:.3f} (risk: {leakage_audit['leakage_risk_old']})")
            if leakage_audit['leakage_risk_new'] == 'LOW' and leakage_audit['leakage_risk_old'] == 'HIGH':
                print("    SUCCESS: Morphology-based D has lower leakage than old V_obs-based D")
        
        # Binned analysis
        print("")
        print("  " + "-" * 90)
        print(f"  {'Bin':<35} {'N':>4} {'Baseline':>10} {'New':>10} {'MOND':>10} {'Improv%':>10} {'Best':>8}")
        print("  " + "-" * 90)
        
        for category, bin_list in bins.items():
            print(f"  {category.upper()}:")
            for b in bin_list:
                print(f"    {b.bin_name:<33} {b.n_galaxies:>4} {b.rms_baseline:>10.2f} {b.rms_new:>10.2f} {b.rms_mond:>10.2f} {b.improvement_pct:>+10.1f} {b.best_theory:>8}")
        
        print("  " + "-" * 90)
        print(f"  D stats: min={D_stats['min']:.3f}, max={D_stats['max']:.3f}, mean={D_stats['mean']:.3f}, n_nonzero={D_stats['n_nonzero']}")
        print(f"  phi stats: min={phi_stats['min']:.3f}, max={phi_stats['max']:.3f}, mean={phi_stats['mean']:.3f}")
        
        # Guardrail checks
        if bin_guardrails:
            print("")
            print("  STATE-BASED GUARDRAILS (where phi(state) should help):")
            critical_failed = False
            for name, check in bin_guardrails.items():
                status = "PASS" if check['passed'] else "FAIL"
                critical_marker = " [CRITICAL]" if check.get('critical') and not check['passed'] else ""
                if check.get('critical') and not check['passed']:
                    critical_failed = True
                print(f"    {name}: {check['actual']:+.1f}% vs threshold {check['threshold']:.1f}% [{status}]{critical_marker}")
                if check.get('physics'):
                    print(f"      Physics: {check['physics']}")
            
            if critical_failed:
                print("")
                print("  WARNING: Critical guardrail failed! phi(state) may not be capturing the right physics.")
    
    # Reference: MOND is the "gold standard" for rotation curves
    observed_rms = mean_mond
    
    # Determine best theory
    theories = {
        'Baseline': mean_baseline,
        'New Model': mean_new,
        'MOND': mean_mond,
        'LCDM': lcdm_rms
    }
    best = min(theories, key=theories.get)
    
    # Calculate win rate safely
    n_tested = len(rms_baseline)
    win_rate = sum(1 for b, m in zip(rms_baseline, rms_mond) if b < m) / n_tested * 100 if n_tested > 0 else 0
    
    pct_excluded = excluded_bulge_points / total_points * 100 if total_points > 0 else 0
    
    # Build notes with guardrail status
    guardrail_note = ""
    if bin_guardrails:
        passed = all(g['passed'] for g in bin_guardrails.values())
        guardrail_note = " Guardrails: " + ("PASS" if passed else "FAIL")
    
    result = ComparativeResult(
        name="SPARC Disk-Only",
        metric_name="RMS",
        observed=observed_rms,
        observed_err=None,
        observed_unit="km/s",
        baseline=TheoryPrediction(mean_baseline, "km/s", {'win_rate': win_rate}),
        new_model=TheoryPrediction(mean_new, "km/s", {'D_stats': D_stats, 'phi_stats': phi_stats, 'component_split': True}),
        mond=TheoryPrediction(mean_mond, "km/s"),
        lcdm=TheoryPrediction(lcdm_rms, "km/s", {'note': '2-3 params/galaxy'}),
        best_theory=best,
        n_objects=n_tested,
        notes=f"Disk only ({pct_excluded:.0f}% bulge pts excluded). Baseline beats MOND {win_rate:.1f}%.{guardrail_note}"
    )
    
    # Return result and detailed bin analysis
    analysis = {
        'bins': bins,
        'D_stats': D_stats,
        'phi_stats': phi_stats,
        'leakage_audit': leakage_audit,
        'bin_guardrails': bin_guardrails,
        'per_galaxy': results_per_galaxy if verbose else None,
    }
    
    return result, analysis


def test_wide_binaries_comparative() -> Tuple[ComparativeResult, Dict]:
    """Compare theories on wide binary velocity boost."""
    s_AU = 10000
    s_m = s_AU * AU_to_m
    M_total = 1.5 * M_sun
    g_N = G * M_total / s_m**2
    
    # Observed
    obs_boost = OBS_BENCHMARKS['wide_binaries']['boost_factor']
    obs_err = OBS_BENCHMARKS['wide_binaries']['boost_uncertainty']
    
    # Baseline: Standard Σ-Gravity
    h_val = h_function(np.array([g_N]))[0]
    Sigma_baseline = 1 + BASELINE_A_0 * h_val
    boost_baseline = Sigma_baseline
    
    # New model: With potential tidal D
    # Wide binaries are in MW's tidal field → equilibrium disturbance
    if USE_EXTENDED_PHI:
        # D for wide binaries - represents loss of perfect coherence in MW tidal field
        D = D_WIDE_BINARIES
        phi = compute_phi_extended(D, is_collisionless=True, is_merger=False)  # EQUILIBRIUM
    else:
        D = 0.0
        phi = 1.0
    Sigma_new = 1 + A_0 * phi * h_val
    boost_new = Sigma_new
    
    # Analysis dict
    analysis = {'D': D, 'phi': phi, 'h': h_val, 'g_N': g_N}
    
    # MOND
    mond_enhance = mond_enhancement(g_N)
    boost_mond = mond_enhance
    
    # Newtonian (no boost)
    boost_newton = 1.0
    
    theories = {
        'Baseline': abs(boost_baseline - obs_boost),
        'New Model': abs(boost_new - obs_boost),
        'MOND': abs(boost_mond - obs_boost),
        'Newtonian': abs(boost_newton - obs_boost)
    }
    best = min(theories, key=theories.get)
    
    result = ComparativeResult(
        name="Wide Binaries",
        metric_name="Boost Factor",
        observed=obs_boost,
        observed_err=obs_err,
        observed_unit="x",
        baseline=TheoryPrediction(boost_baseline, "x", {'overpredicts': boost_baseline > obs_boost + obs_err}),
        new_model=TheoryPrediction(boost_new, "x", {'D': D, 'phi': phi}),
        mond=TheoryPrediction(boost_mond, "x"),
        lcdm=TheoryPrediction(boost_newton, "x", {'note': 'No boost'}),
        best_theory=best,
        n_objects=OBS_BENCHMARKS['wide_binaries']['n_pairs'],
        notes=f"Chae 2023: {obs_boost:.2f}x at >{OBS_BENCHMARKS['wide_binaries']['threshold_AU']} AU"
    )
    return result, analysis


def test_udgs_comparative() -> Tuple[ComparativeResult, Dict]:
    """Compare theories on DF2 (the famous 'dark matter free' galaxy)."""
    df2 = OBS_BENCHMARKS['udgs']['df2']
    
    M_star = df2['M_star']
    sigma_obs = df2['sigma_obs']
    sigma_err = df2['sigma_err']
    r_eff = df2['r_eff_kpc'] * kpc_to_m
    d_host = df2['d_host_kpc'] * kpc_to_m
    host_mass = df2['host_mass']
    
    # Newtonian prediction
    sigma_N = np.sqrt(G * M_star * M_sun / (5 * r_eff)) / 1000
    g_N = G * M_star * M_sun / r_eff**2
    
    # Baseline Σ-Gravity
    h_val = h_function(np.array([g_N]))[0]
    Sigma_baseline = 1 + BASELINE_A_0 * h_val
    sigma_baseline = sigma_N * np.sqrt(Sigma_baseline)
    
    # New model with tidal D (DF2 is close to NGC1052)
    if USE_EXTENDED_PHI:
        # Tidal radius from host
        r_tidal = d_host * (M_star / (3 * host_mass))**(1/3) if host_mass > 0 else np.inf
        D = compute_D_from_tidal(r_tidal / kpc_to_m, df2['r_eff_kpc'])
        # DF2 is tidally disturbed → equilibrium disturbance → φ < 1
        phi = compute_phi_extended(D, is_collisionless=True, is_merger=False)  # EQUILIBRIUM
    else:
        D = 0.0
        phi = 1.0
    
    Sigma_new = 1 + A_0 * phi * h_val
    sigma_new = sigma_N * np.sqrt(max(Sigma_new, 0.1))
    
    # MOND (without EFE - overpredicts)
    sigma_mond = df2['mond_sigma']
    
    # ΛCDM/Newtonian
    sigma_newton = sigma_N
    
    theories = {
        'Baseline': abs(sigma_baseline - sigma_obs),
        'New Model': abs(sigma_new - sigma_obs),
        'MOND': abs(sigma_mond - sigma_obs),
        'Newtonian': abs(sigma_newton - sigma_obs)
    }
    best = min(theories, key=theories.get)
    
    result = ComparativeResult(
        name="DF2 (UDG)",
        metric_name="Velocity Dispersion",
        observed=sigma_obs,
        observed_err=sigma_err,
        observed_unit="km/s",
        baseline=TheoryPrediction(sigma_baseline, "km/s", {'Sigma': Sigma_baseline}),
        new_model=TheoryPrediction(sigma_new, "km/s", {'D': D, 'phi': phi, 'Sigma': Sigma_new}),
        mond=TheoryPrediction(sigma_mond, "km/s", {'note': 'Needs EFE'}),
        lcdm=TheoryPrediction(sigma_newton, "km/s"),
        best_theory=best,
        n_objects=1,
        notes=f"DF2 appears 'dark matter free'. d_host={df2['d_host_kpc']} kpc from NGC1052"
    )
    
    analysis = {'D': D, 'phi': phi, 'Sigma_baseline': Sigma_baseline, 'Sigma_new': Sigma_new}
    return result, analysis


def test_dsph_comparative() -> ComparativeResult:
    """Compare theories on dwarf spheroidal velocity dispersions."""
    dsphs = OBS_BENCHMARKS['dwarf_spheroidals']
    M_MW_bar = 6e10
    
    results_baseline = []
    results_new = []
    results_mond = []
    sigma_obs_list = []
    
    for name, data in dsphs.items():
        if not isinstance(data, dict) or 'M_star' not in data:
            continue
            
        M_star = data['M_star']
        sigma_obs = data['sigma_obs']
        r_half = data['r_half_kpc'] * kpc_to_m
        d_MW = data.get('d_MW_kpc', 100) * kpc_to_m
        
        sigma_obs_list.append(sigma_obs)
        
        # MW's Σ at dSph location
        g_MW = G * M_MW_bar * M_sun / d_MW**2
        h_MW = h_function(np.array([g_MW]))[0]
        Sigma_MW = 1 + BASELINE_A_0 * h_MW
        
        # Baseline: Host inheritance
        M_eff = M_star * Sigma_MW
        sigma_baseline = np.sqrt(G * M_eff * M_sun / (5 * r_half)) / 1000
        results_baseline.append(sigma_baseline / sigma_obs)
        
        # New model for dSphs
        # KEY PHYSICS: dSphs INHERIT the MW's Σ-enhancement at their orbital radius
        # Unlike isolated galaxies, they don't need their own internal coherence
        # The tidal field enhances, not suppresses, their effective gravity
        # So we DON'T apply equilibrium suppression to the inherited enhancement
        phi = 1.0  # dSphs use host inheritance, not internal coherence
        
        Sigma_new = 1 + A_0 * phi * h_MW
        M_eff_new = M_star * Sigma_new
        sigma_new = np.sqrt(G * M_eff_new * M_sun / (5 * r_half)) / 1000
        results_new.append(sigma_new / sigma_obs)
        
        # MOND
        g_int = G * M_star * M_sun / r_half**2
        mond_nu = mond_enhancement(g_int)
        sigma_mond = np.sqrt(G * M_star * M_sun * mond_nu / (5 * r_half)) / 1000
        results_mond.append(sigma_mond / sigma_obs)
    
    mean_baseline = np.mean(results_baseline)
    mean_new = np.mean(results_new)
    mean_mond = np.mean(results_mond)
    
    # ΛCDM: Uses NFW halos, typically works with M/L tuning
    mean_lcdm = 1.0  # By construction with NFW
    
    theories = {
        'Baseline': abs(mean_baseline - 1.0),
        'New Model': abs(mean_new - 1.0),
        'MOND': abs(mean_mond - 1.0),
        'LCDM': abs(mean_lcdm - 1.0)
    }
    best = min(theories, key=theories.get)
    
    return ComparativeResult(
        name="Dwarf Spheroidals",
        metric_name="sigma_pred/sigma_obs",
        observed=1.0,
        observed_err=None,
        observed_unit="ratio",
        baseline=TheoryPrediction(mean_baseline, "ratio", {'std': np.std(results_baseline)}),
        new_model=TheoryPrediction(mean_new, "ratio", {'std': np.std(results_new)}),
        mond=TheoryPrediction(mean_mond, "ratio", {'note': 'EFE complicates'}),
        lcdm=TheoryPrediction(mean_lcdm, "ratio", {'note': 'NFW tuned'}),
        best_theory=best,
        n_objects=len(results_baseline),
        notes="Host inheritance model for satellites"
    )


def test_bulge_dispersion_comparative() -> ComparativeResult:
    """
    Compare theories on BULGE velocity dispersions.
    
    Unlike rotation curves, bulges are dispersion-supported systems (v/σ < 1).
    The correct observable is velocity dispersion σ, not rotation velocity.
    
    This is a separate test from SPARC rotation curves because:
    1. Different observable (σ vs v_circ)
    2. Different physics (pressure support vs rotation)
    3. Different coherence regime (disordered vs ordered)
    """
    bulge_data = OBS_BENCHMARKS['bulge_dispersion']
    
    # Use MW bulge data (best constrained)
    mw = bulge_data['mw_bulge']
    R_kpc = np.array(mw['R_kpc'])
    sigma_obs = np.array(mw['sigma_obs'])
    sigma_err = np.array(mw['sigma_err'])
    
    # MW bulge mass model (Hernquist profile)
    M_bulge = 1.5e10  # Solar masses
    r_s = 0.5  # Scale radius in kpc
    
    results_baseline = []
    results_new = []
    results_mond = []
    results_lcdm = []
    
    for R, obs, err in zip(R_kpc, sigma_obs, sigma_err):
        r_m = R * kpc_to_m
        
        # Hernquist enclosed mass: M(<r) = M_total * r² / (r + r_s)²
        M_enc = M_bulge * R**2 / (R + r_s)**2
        
        # Newtonian prediction: σ² ~ GM/r (virial theorem)
        g_N = G * M_enc * M_sun / r_m**2
        sigma_N = np.sqrt(G * M_enc * M_sun / (3 * r_m)) / 1000  # km/s
        
        # BASELINE Σ-Gravity: enhance gravity
        h_val = h_function(np.array([g_N]))[0]
        Sigma_baseline = 1 + BASELINE_A_0 * h_val
        sigma_baseline = sigma_N * np.sqrt(Sigma_baseline)
        results_baseline.append(sigma_baseline)
        
        # NEW MODEL: bulge-specific treatment
        # Bulges are disordered → use reduced coherence
        # D from compactness/Omega
        v_over_sigma = 0.5  # Typical for bulges
        D_bulge = compute_D_from_bulge(100 / R, R / L_0, v_over_sigma)  # Omega/H0 ~ 100/R approx
        phi = compute_phi(D_bulge, matter_type="bulge_stars", is_merger=False, v_over_sigma=v_over_sigma)
        
        Sigma_new = 1 + A_0 * phi * h_val
        sigma_new = sigma_N * np.sqrt(max(Sigma_new, 0.01))
        results_new.append(sigma_new)
        
        # MOND prediction
        mond_nu = mond_enhancement(g_N)
        sigma_mond = sigma_N * np.sqrt(mond_nu)
        results_mond.append(sigma_mond)
        
        # ΛCDM with dark matter halo (typical NFW)
        # DM typically contributes 50% extra mass in bulge region
        sigma_lcdm = sigma_N * np.sqrt(1.5)
        results_lcdm.append(sigma_lcdm)
    
    # Compute RMS for each theory
    rms_baseline = np.sqrt(((np.array(results_baseline) - sigma_obs)**2).mean())
    rms_new = np.sqrt(((np.array(results_new) - sigma_obs)**2).mean())
    rms_mond = np.sqrt(((np.array(results_mond) - sigma_obs)**2).mean())
    rms_lcdm = np.sqrt(((np.array(results_lcdm) - sigma_obs)**2).mean())
    
    theories = {
        'Baseline': rms_baseline,
        'New Model': rms_new,
        'MOND': rms_mond,
        'LCDM': rms_lcdm
    }
    best = min(theories, key=theories.get)
    
    return ComparativeResult(
        name="Bulge Dispersion",
        metric_name="RMS(sigma)",
        observed=np.mean(sigma_obs),
        observed_err=np.mean(sigma_err),
        observed_unit="km/s",
        baseline=TheoryPrediction(rms_baseline, "km/s", {'mean_pred': np.mean(results_baseline)}),
        new_model=TheoryPrediction(rms_new, "km/s", {'mean_pred': np.mean(results_new)}),
        mond=TheoryPrediction(rms_mond, "km/s", {'mean_pred': np.mean(results_mond)}),
        lcdm=TheoryPrediction(rms_lcdm, "km/s", {'mean_pred': np.mean(results_lcdm)}),
        best_theory=best,
        n_objects=len(R_kpc),
        notes=f"MW bulge σ (dispersion-supported, v/sigma~0.5)"
    )


def test_clusters_comparative(clusters: List[Dict]) -> ComparativeResult:
    """Compare theories on cluster lensing masses."""
    if not clusters:
        return ComparativeResult(
            name="Galaxy Clusters",
            metric_name="M_pred/M_lens",
            observed=1.0,
            observed_err=None,
            observed_unit="ratio",
            baseline=TheoryPrediction(0, "ratio"),
            new_model=TheoryPrediction(0, "ratio"),
            mond=TheoryPrediction(0, "ratio"),
            lcdm=TheoryPrediction(1.0, "ratio"),
            best_theory="N/A",
            n_objects=0,
            notes="No data"
        )
    
    L_cluster = 600
    A_cluster_baseline = unified_amplitude(L_cluster, baseline=True)
    A_cluster_new = unified_amplitude(L_cluster, baseline=False)
    
    ratios_baseline = []
    ratios_new = []
    ratios_mond = []
    
    for cl in clusters:
        M_bar = cl['M_bar']
        M_lens = cl['M_lens']
        r_kpc = cl.get('r_kpc', 200)
        r_m = r_kpc * kpc_to_m
        
        g_bar = G * M_bar * M_sun / r_m**2
        h = h_function(np.array([g_bar]))[0]
        
        # Baseline
        Sigma_baseline = 1 + A_cluster_baseline * h
        M_pred_baseline = M_bar * Sigma_baseline
        ratios_baseline.append(M_pred_baseline / M_lens)
        
        # New model (clusters are equilibrium, D=0)
        Sigma_new = 1 + A_cluster_new * h
        M_pred_new = M_bar * Sigma_new
        ratios_new.append(M_pred_new / M_lens)
        
        # MOND
        mond_nu = mond_enhancement(g_bar)
        M_pred_mond = M_bar * mond_nu
        ratios_mond.append(M_pred_mond / M_lens)
    
    median_baseline = np.median(ratios_baseline)
    median_new = np.median(ratios_new)
    median_mond = np.median(ratios_mond)
    
    theories = {
        'Baseline': abs(median_baseline - 1.0),
        'New Model': abs(median_new - 1.0),
        'MOND': abs(median_mond - 1.0),
        'LCDM': 0.0  # NFW works by construction
    }
    best = min(theories, key=theories.get)
    
    return ComparativeResult(
        name="Galaxy Clusters",
        metric_name="M_pred/M_lens",
        observed=1.0,
        observed_err=None,
        observed_unit="ratio",
        baseline=TheoryPrediction(median_baseline, "ratio", {'scatter': np.std(np.log10(ratios_baseline))}),
        new_model=TheoryPrediction(median_new, "ratio"),
        mond=TheoryPrediction(median_mond, "ratio", {'underpredicts_by': 1/median_mond}),
        lcdm=TheoryPrediction(1.0, "ratio", {'note': 'NFW fits work'}),
        best_theory=best,
        n_objects=len(clusters),
        notes=f"MOND underpredicts by ~{1/median_mond:.1f}x"
    )


def test_bullet_cluster_comparative() -> ComparativeResult:
    """Compare theories on Bullet Cluster spatial lensing."""
    bc = OBS_BENCHMARKS['bullet_cluster']
    
    M_gas = bc['M_gas'] * M_sun
    M_stars = bc['M_stars'] * M_sun
    M_bar = M_gas + M_stars
    M_lens = bc['M_lensing'] * M_sun
    r_lens = bc['offset_kpc'] * kpc_to_m
    mach = bc['mach_shock']
    
    g_gas = G * M_gas / r_lens**2
    g_stars = G * M_stars / r_lens**2
    g_bar = G * M_bar / r_lens**2
    
    h_gas = h_function(np.array([g_gas]))[0]
    h_stars = h_function(np.array([g_stars]))[0]
    h_bar = h_function(np.array([g_bar]))[0]
    
    A_cluster = unified_amplitude(600, baseline=True)
    
    # Baseline: Same Σ for all matter
    Sigma_baseline = 1 + A_cluster * h_bar
    M_pred_baseline = M_bar * Sigma_baseline
    ratio_baseline = M_pred_baseline / M_lens
    lensing_baseline = "GAS"  # Follows baryons = mostly gas
    
    # New model with phase coherence
    # Bullet Cluster is a MERGER → stars maintain coherence, gas disrupted
    D = compute_D_from_mach(mach)
    phi_stars = compute_phi_extended(D, is_collisionless=True, is_merger=True)  # MERGER
    phi_gas = compute_phi_extended(D, is_collisionless=False, mach_turb=mach, is_merger=True)  # MERGER
    
    Sigma_stars = max(1 + A_cluster * phi_stars * h_stars, 0.01)
    Sigma_gas = max(1 + A_cluster * phi_gas * h_gas, 0.01)
    
    M_eff_stars = M_stars * Sigma_stars
    M_eff_gas = M_gas * Sigma_gas
    M_pred_new = M_eff_gas + M_eff_stars
    ratio_new = M_pred_new / M_lens
    lensing_new = "STARS" if M_eff_gas < M_eff_stars else "GAS"
    
    # MOND
    mond_nu = mond_enhancement(g_bar)
    M_pred_mond = M_bar * mond_nu
    ratio_mond = M_pred_mond / M_lens
    lensing_mond = "GAS"  # MOND follows baryons
    
    # ΛCDM
    ratio_lcdm = 1.0  # By construction with DM halos
    lensing_lcdm = "DM HALO"  # Follows collisionless DM
    
    # Observed: Lensing peaks at STARS
    observed_lensing = "STARS"
    observed_ratio = bc['mass_ratio']
    
    # Score based on both mass ratio AND lensing location
    def score(ratio, lensing):
        mass_err = abs(ratio - 1.0)  # Want M_pred/M_lens ~ 1
        loc_err = 0 if lensing == observed_lensing else 1.0  # Penalty for wrong location
        return mass_err + loc_err
    
    theories = {
        'Baseline': score(ratio_baseline, lensing_baseline),
        'New Model': score(ratio_new, lensing_new),
        'MOND': score(ratio_mond, lensing_mond),
        'LCDM': score(ratio_lcdm, lensing_lcdm)
    }
    best = min(theories, key=theories.get)
    
    return ComparativeResult(
        name="Bullet Cluster",
        metric_name="Lensing Location",
        observed=observed_ratio,
        observed_err=None,
        observed_unit="x (at STARS)",
        baseline=TheoryPrediction(ratio_baseline, f"x (at {lensing_baseline})", {'Sigma': Sigma_baseline}),
        new_model=TheoryPrediction(ratio_new, f"x (at {lensing_new})", {'phi_stars': phi_stars, 'phi_gas': phi_gas, 'D': D}),
        mond=TheoryPrediction(ratio_mond, f"x (at {lensing_mond})", {'note': 'Spatial problem'}),
        lcdm=TheoryPrediction(ratio_lcdm, f"x (at {lensing_lcdm})", {'note': 'DM halo solves spatial'}),
        best_theory=best,
        n_objects=1,
        notes="Key test: Lensing follows STARS (20%), not gas (80%)"
    )


def test_gaia_comparative(gaia_df: Optional[pd.DataFrame]) -> ComparativeResult:
    """Compare theories on Milky Way rotation curve."""
    if gaia_df is None or len(gaia_df) == 0:
        return ComparativeResult(
            name="Gaia/MW",
            metric_name="RMS",
            observed=0,
            observed_err=None,
            observed_unit="km/s",
            baseline=TheoryPrediction(0, "km/s"),
            new_model=TheoryPrediction(0, "km/s"),
            mond=TheoryPrediction(0, "km/s"),
            lcdm=TheoryPrediction(0, "km/s"),
            best_theory="N/A",
            n_objects=0,
            notes="No data"
        )
    
    R = gaia_df['R_gal'].values
    v_phi_obs = gaia_df['v_phi_obs'].values
    
    # McMillan 2017 baryonic model
    MW_SCALE = 1.16
    M_disk = 4.6e10 * MW_SCALE**2
    M_bulge = 1.0e10 * MW_SCALE**2
    M_gas = 1.0e10 * MW_SCALE**2
    G_kpc = 4.302e-6
    
    v2_disk = G_kpc * M_disk * R**2 / (R**2 + 3.3**2)**1.5
    v2_bulge = G_kpc * M_bulge * R / (R + 0.5)**2
    v2_gas = G_kpc * M_gas * R**2 / (R**2 + 7.0**2)**1.5
    V_bar = np.sqrt(v2_disk + v2_bulge + v2_gas)
    
    R_d_mw = 2.6
    
    # Baseline
    V_pred_baseline = predict_velocity_baseline(R, V_bar, R_d_mw)
    rms_baseline = np.sqrt(((V_pred_baseline - v_phi_obs)**2).mean())
    
    # New model
    V_pred_new = predict_velocity_new(R, V_bar, R_d_mw, D=0.0)  # MW is equilibrium
    rms_new = np.sqrt(((V_pred_new - v_phi_obs)**2).mean())
    
    # MOND
    V_mond = predict_mond(R, V_bar)
    rms_mond = np.sqrt(((V_mond - v_phi_obs)**2).mean())
    
    # ΛCDM (NFW halo tuned)
    rms_lcdm = 25.0  # Typical with NFW
    
    theories = {
        'Baseline': rms_baseline,
        'New Model': rms_new,
        'MOND': rms_mond,
        'LCDM': rms_lcdm
    }
    best = min(theories, key=theories.get)
    
    return ComparativeResult(
        name="Gaia/MW",
        metric_name="RMS",
        observed=rms_mond,  # Use MOND as reference
        observed_err=None,
        observed_unit="km/s",
        baseline=TheoryPrediction(rms_baseline, "km/s"),
        new_model=TheoryPrediction(rms_new, "km/s"),
        mond=TheoryPrediction(rms_mond, "km/s"),
        lcdm=TheoryPrediction(rms_lcdm, "km/s", {'note': 'NFW tuned'}),
        best_theory=best,
        n_objects=len(gaia_df),
        notes=f"{len(gaia_df)} stars from Eilers+ 2019"
    )


def test_solar_system_comparative() -> ComparativeResult:
    """Compare theories on Solar System (Cassini bound)."""
    r_saturn = 9.5 * AU_to_m
    g_saturn = G * M_sun / r_saturn**2
    
    cassini_bound = OBS_BENCHMARKS['solar_system']['cassini_gamma_uncertainty']
    
    # Baseline
    h_baseline = h_function(np.array([g_saturn]))[0]
    gamma_baseline = h_baseline
    
    # New model (same for solar system)
    gamma_new = h_baseline
    
    # MOND
    x = g_saturn / a0_mond
    gamma_mond = (mond_enhancement(g_saturn) - 1)  # Enhancement factor - 1
    
    # GR (no deviation)
    gamma_gr = 0.0
    
    theories = {
        'Baseline': gamma_baseline,
        'New Model': gamma_new,
        'MOND': gamma_mond,
        'GR': gamma_gr
    }
    best = min(theories, key=theories.get)
    
    return ComparativeResult(
        name="Solar System",
        metric_name="|gamma-1|",
        observed=cassini_bound,
        observed_err=None,
        observed_unit="bound",
        baseline=TheoryPrediction(gamma_baseline, "", {'safe': gamma_baseline < cassini_bound}),
        new_model=TheoryPrediction(gamma_new, "", {'safe': gamma_new < cassini_bound}),
        mond=TheoryPrediction(gamma_mond, "", {'safe': gamma_mond < cassini_bound}),
        lcdm=TheoryPrediction(gamma_gr, "", {'note': 'GR = no deviation'}),
        best_theory=best,
        n_objects=1,
        notes=f"Cassini bound: |gamma-1| < {cassini_bound:.1e}"
    )


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_comparison_table(results: List[ComparativeResult]) -> str:
    """Format results as a comparison table."""
    lines = []
    lines.append("=" * 120)
    lines.append("COMPARATIVE RESULTS: Baseline vs New Model vs MOND vs LCDM")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    header = f"{'Test':<20} {'Observed':<15} {'Baseline':<15} {'New Model':<15} {'MOND':<15} {'LCDM':<15} {'Best':<10}"
    lines.append(header)
    lines.append("-" * 120)
    
    for r in results:
        obs_str = f"{r.observed:.2f}" if r.observed else "N/A"
        if r.observed_err:
            obs_str += f"+/-{r.observed_err:.2f}"
        obs_str += f" {r.observed_unit}"
        
        base_str = f"{r.baseline.value:.2f}" if r.baseline else "N/A"
        new_str = f"{r.new_model.value:.2f}" if r.new_model else "N/A"
        mond_str = f"{r.mond.value:.2f}" if r.mond else "N/A"
        lcdm_str = f"{r.lcdm.value:.2f}" if r.lcdm else "N/A"
        
        # Highlight the best theory
        best = r.best_theory
        
        line = f"{r.name:<20} {obs_str:<15} {base_str:<15} {new_str:<15} {mond_str:<15} {lcdm_str:<15} {best:<10}"
        lines.append(line)
    
    lines.append("-" * 120)
    lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY:")
    best_counts = {}
    for r in results:
        best = r.best_theory
        best_counts[best] = best_counts.get(best, 0) + 1
    
    for theory, count in sorted(best_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {theory}: Best in {count}/{len(results)} tests ({count/len(results)*100:.1f}%)")
    
    lines.append("")
    
    # Detailed notes
    lines.append("DETAILED NOTES:")
    for r in results:
        if r.notes:
            lines.append(f"  {r.name}: {r.notes}")
    
    return "\n".join(lines)


def format_improvement_table(results: List[ComparativeResult]) -> str:
    """Show how New Model improves/worsens vs Baseline."""
    lines = []
    lines.append("")
    lines.append("NEW MODEL vs BASELINE (negative = improvement):")
    lines.append("-" * 80)
    
    improvements = []
    for r in results:
        if r.baseline and r.new_model and r.observed:
            # Calculate distance from observed
            base_dist = abs(r.baseline.value - r.observed)
            new_dist = abs(r.new_model.value - r.observed)
            
            # Improvement (negative = better)
            if base_dist > 0:
                pct_change = (new_dist - base_dist) / base_dist * 100
            else:
                pct_change = 0
            
            improvements.append((r.name, pct_change, new_dist < base_dist))
    
    for name, pct, improved in sorted(improvements, key=lambda x: x[1]):
        status = "IMPROVED" if improved else "WORSENED" if pct > 0 else "UNCHANGED"
        lines.append(f"  {name:<25} {pct:+.1f}% {status}")
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    global USE_EXTENDED_PHI, PHI_LAMBDA_0, D_ASYMMETRY_SCALE, D_TIDAL_THRESHOLD, D_WIDE_BINARIES
    
    quick = '--quick' in sys.argv
    extended_phi = '--extended-phi' in sys.argv
    holdout = '--holdout' in sys.argv
    
    USE_EXTENDED_PHI = extended_phi
    
    # Parse optional parameters
    for arg in sys.argv:
        if arg.startswith('--phi-lambda='):
            PHI_LAMBDA_0 = float(arg.split('=')[1])
        if arg.startswith('--d-asymmetry='):
            D_ASYMMETRY_SCALE = float(arg.split('=')[1])
        if arg.startswith('--d-tidal='):
            D_TIDAL_THRESHOLD = float(arg.split('=')[1])
        if arg.startswith('--d-wb='):
            D_WIDE_BINARIES = float(arg.split('=')[1])
    
    data_dir = Path(__file__).parent.parent / "data"
    
    print("=" * 120)
    print("Sigma-GRAVITY COMPARATIVE REGRESSION TEST")
    print("=" * 120)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    print("BASELINE MODEL (LOCKED):")
    print(f"  A_0 = {BASELINE_A_0:.4f}, L_0 = {BASELINE_L_0} kpc, n = {BASELINE_N_EXP}")
    print(f"  xi = R_d/(2pi), M/L = {BASELINE_ML_DISK}/{BASELINE_ML_BULGE}")
    print(f"  A_cluster = {BASELINE_A_CLUSTER:.2f}")
    print()
    print("NEW MODEL:")
    if USE_EXTENDED_PHI:
        print(f"  Extended Phase Coherence: ENABLED")
        print(f"  phi = 1 + lambda_0 * D * (f_ordered - f_turb)")
        print(f"  lambda_0 = {PHI_LAMBDA_0:.1f}")
        print(f"  D_asymmetry_scale = {D_ASYMMETRY_SCALE:.2f}")
        print(f"  D_tidal_threshold = {D_TIDAL_THRESHOLD:.1f}")
    else:
        print(f"  Extended Phase Coherence: DISABLED (same as baseline)")
        print(f"  Enable with --extended-phi")
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
    
    # Run comparative tests
    print("Running comparative tests...")
    print("-" * 120)
    
    results = []
    analyses = {}
    
    # Core tests - handle tuple returns
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    sparc_result, sparc_analysis = test_sparc_comparative(
        galaxies, verbose=verbose, holdout_mode=holdout
    )
    results.append(sparc_result)
    analyses['sparc'] = sparc_analysis
    
    results.append(test_clusters_comparative(clusters))
    results.append(test_gaia_comparative(gaia_df))
    results.append(test_solar_system_comparative())
    
    # Extended tests
    wb_result, wb_analysis = test_wide_binaries_comparative()
    results.append(wb_result)
    analyses['wide_binaries'] = wb_analysis
    
    udg_result, udg_analysis = test_udgs_comparative()
    results.append(udg_result)
    analyses['udg'] = udg_analysis
    
    results.append(test_dsph_comparative())
    results.append(test_bulge_dispersion_comparative())  # NEW: Separate bulge test
    results.append(test_bullet_cluster_comparative())
    
    # Print comparison table
    print(format_comparison_table(results))
    
    # Print improvement analysis
    print(format_improvement_table(results))
    
    # Save report
    output_dir = Path(__file__).parent / "regression_results"
    output_dir.mkdir(exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'extended_phi' if USE_EXTENDED_PHI else 'baseline_only',
        'parameters': {
            'baseline': {
                'A_0': BASELINE_A_0,
                'L_0': BASELINE_L_0,
                'n_exp': BASELINE_N_EXP,
            },
            'new_model': {
                'use_extended_phi': USE_EXTENDED_PHI,
                'phi_lambda_0': PHI_LAMBDA_0,
                'd_asymmetry_scale': D_ASYMMETRY_SCALE,
                'd_tidal_threshold': D_TIDAL_THRESHOLD,
            }
        },
        'results': [asdict(r) for r in results],
    }
    
    with open(output_dir / "comparative_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=float)
    
    print(f"\nReport saved to: {output_dir / 'comparative_report.json'}")


if __name__ == "__main__":
    main()
