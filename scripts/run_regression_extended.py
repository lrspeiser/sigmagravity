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
# EXTENDED PHASE COHERENCE MODEL
# =============================================================================

def compute_D_from_asymmetry(v_asym: float, v_circ: float) -> float:
    """
    Compute disturbance D from kinematic asymmetry.
    
    v_asym: Asymmetry in velocity field (e.g., lopsidedness, warp amplitude)
    v_circ: Circular velocity
    
    D = 0 for perfectly symmetric galaxies
    D → 1 for highly disturbed/asymmetric systems
    """
    if v_circ <= 0:
        return 0.0
    ratio = abs(v_asym) / v_circ
    return min(1.0, D_ASYMMETRY_SCALE * ratio)


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
    # D increases as tidal radius approaches the system size
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


def phi_universal(D: float, f_ordered: float, f_turb: float) -> float:
    """
    UNIVERSAL phase coherence formula.
    
    φ = 1 + λ₀ × D × (f_ordered - f_turb)
    
    KEY: For D = 0, φ = 1 ALWAYS (equilibrium systems unchanged)
    """
    phi = 1.0 + PHI_LAMBDA_0 * D * (f_ordered - f_turb)
    return max(PHI_MIN, min(PHI_MAX, phi))


def compute_phi_extended(
    D: float,
    is_collisionless: bool = True,
    mach_turb: float = 0.0,
    is_merger: bool = False
) -> float:
    """
    Compute φ for extended model (applies to non-cluster systems too).
    
    KEY PHYSICS:
    - MERGER systems (is_merger=True): D represents survival of coherence
      Stars maintain order through collision → φ > 1
      Gas gets shocked/turbulent → φ < 1
    - EQUILIBRIUM systems (is_merger=False): D represents LOSS of coherence
      Disturbance (tidal, asymmetry) disrupts phase coherence → φ < 1
    
    Parameters:
    -----------
    D : float [0, 1]
        Disturbance parameter from observable proxies
    is_collisionless : bool
        True for stars, False for gas
    mach_turb : float
        Turbulent Mach number (for gas)
    is_merger : bool
        True for merger/collision events, False for equilibrium disturbance
    """
    if is_merger:
        # MERGER: Stars maintain order, gas gets disrupted
        if is_collisionless:
            f_ordered = 0.95
            f_turb = 0.0
        else:
            f_ordered = 0.1
            f_turb = min(0.85, 0.25 * mach_turb)
        return phi_universal(D, f_ordered, f_turb)
    else:
        # EQUILIBRIUM DISTURBANCE: Both stars and gas lose coherence
        # D represents loss of order → φ < 1
        if EQUILIBRIUM_DISTURBANCE_SUPPRESSES:
            # Disturbance reduces coherence → suppress enhancement
            # φ = 1 - λ₀ × D × suppression_factor
            suppression = 0.3 * D  # Mild suppression for equilibrium
            return max(PHI_MIN, 1.0 - suppression)
        else:
            # Original formula
            if is_collisionless:
                f_ordered = 0.95
                f_turb = 0.0
            else:
                f_ordered = 0.1
                f_turb = min(0.85, 0.25 * mach_turb)
            return phi_universal(D, f_ordered, f_turb)


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
                         is_merger: bool = False) -> np.ndarray:
    """NEW MODEL rotation curve prediction with extended phase coherence."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A = A_0
    h = h_function(g_bar)
    V = np.array(V_bar, dtype=float)
    
    # Compute φ based on D and system type
    # For disk galaxies: mostly ordered stellar motion, equilibrium
    phi = compute_phi_extended(D, is_collisionless=True, mach_turb=0.0, is_merger=is_merger)
    
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
    """Load SPARC galaxy rotation curves."""
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
            
            # Estimate kinematic asymmetry (simple proxy: velocity residual scatter)
            # This would ideally come from actual 2D velocity field analysis
            v_mean = df['V_obs'].mean()
            v_asym = df['V_obs'].std() * 0.3  # Rough proxy for asymmetry
            
            # Compute component fractions (at outermost radius for representative value)
            V_gas_sq = np.sign(df['V_gas'].values[-1]) * df['V_gas'].values[-1]**2
            V_disk_sq = df['V_disk_scaled'].values[-1]**2
            V_bulge_sq = df['V_bulge_scaled'].values[-1]**2
            V_total_sq = max(abs(V_gas_sq) + V_disk_sq + V_bulge_sq, 1e-10)
            
            f_gas = abs(V_gas_sq) / V_total_sq
            f_bulge = V_bulge_sq / V_total_sq
            
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': df['V_bar'].values,
                'V_gas': df['V_gas'].values,
                'V_disk_scaled': df['V_disk_scaled'].values,
                'V_bulge_scaled': df['V_bulge_scaled'].values,
                'R_d': R_d,
                'v_asym': v_asym,
                'v_circ': v_mean,
                'f_gas': f_gas,
                'f_bulge': f_bulge,
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
        ('D=0 (undisturbed)', lambda r: r['D'] < 0.01),
        ('Low D (0.01-0.05)', lambda r: 0.01 <= r['D'] < 0.05),
        ('Mid D (0.05-0.15)', lambda r: 0.05 <= r['D'] < 0.15),
        ('High D (>0.15)', lambda r: r['D'] >= 0.15),
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
            D_vals = [r['D'] for _, r in matching]
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
            D_vals = [r['D'] for _, r in matching]
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
            D_vals = [r['D'] for _, r in matching]
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

def test_sparc_comparative(galaxies: List[Dict], verbose: bool = False) -> Tuple[ComparativeResult, Dict]:
    """
    Compare theories on SPARC galaxy rotation curves.
    
    Returns both the overall result and detailed per-bin analysis.
    """
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
    phi_values = []
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        R_d = gal['R_d']
        
        # Baseline prediction
        V_pred_baseline = predict_velocity_baseline(R, V_bar, R_d)
        rms_b = np.sqrt(((V_pred_baseline - V_obs)**2).mean())
        rms_baseline.append(rms_b)
        
        # New model with extended phase coherence
        if USE_EXTENDED_PHI:
            D = compute_D_from_asymmetry(gal.get('v_asym', 0), gal.get('v_circ', 200))
            phi = compute_phi_extended(D, is_collisionless=True, is_merger=False)
        else:
            D = 0.0
            phi = 1.0
        
        D_values.append(D)
        phi_values.append(phi)
        
        V_pred_new = predict_velocity_new(R, V_bar, R_d, D=D, is_merger=False)
        rms_n = np.sqrt(((V_pred_new - V_obs)**2).mean())
        rms_new.append(rms_n)
        
        # MOND prediction
        V_mond = predict_mond(R, V_bar)
        rms_m = np.sqrt(((V_mond - V_obs)**2).mean())
        rms_mond.append(rms_m)
        
        # Store per-galaxy result for binning
        results_per_galaxy.append({
            'name': gal['name'],
            'rms_baseline': rms_b,
            'rms_new': rms_n,
            'rms_mond': rms_m,
            'D': D,
            'phi': phi,
            'f_bulge': gal.get('f_bulge', 0),
            'f_gas': gal.get('f_gas', 0.5),
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
    
    # Compute bins
    bins = compute_sparc_bins(galaxies, results_per_galaxy) if USE_EXTENDED_PHI else {}
    
    # Print verbose bin analysis
    if verbose and USE_EXTENDED_PHI:
        print("\n  SPARC BINNED ANALYSIS:")
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
    
    result = ComparativeResult(
        name="SPARC Galaxies",
        metric_name="RMS",
        observed=observed_rms,
        observed_err=None,
        observed_unit="km/s",
        baseline=TheoryPrediction(mean_baseline, "km/s", {'win_rate': sum(1 for b, m in zip(rms_baseline, rms_mond) if b < m) / len(galaxies)}),
        new_model=TheoryPrediction(mean_new, "km/s", {'D_stats': D_stats, 'phi_stats': phi_stats}),
        mond=TheoryPrediction(mean_mond, "km/s"),
        lcdm=TheoryPrediction(lcdm_rms, "km/s", {'note': '2-3 params/galaxy'}),
        best_theory=best,
        n_objects=len(galaxies),
        notes=f"Win rates: Baseline beats MOND {sum(1 for b, m in zip(rms_baseline, rms_mond) if b < m)/len(galaxies)*100:.1f}%"
    )
    
    # Return result and detailed bin analysis
    analysis = {
        'bins': bins,
        'D_stats': D_stats,
        'phi_stats': phi_stats,
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
    
    sparc_result, sparc_analysis = test_sparc_comparative(galaxies, verbose=verbose)
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
