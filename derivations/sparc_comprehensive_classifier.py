#!/usr/bin/env python3
"""
Comprehensive SPARC Galaxy Classification System
=================================================

This creates hundreds of ways to slice the SPARC galaxy sample based on:

1. MORPHOLOGICAL PROPERTIES
   - Hubble type (S0-Im, BCD)
   - Early/Late type groups
   - Bar presence (from type codes)
   - Bulge fraction
   
2. KINEMATIC PROPERTIES
   - Rotation curve shape (rising/flat/declining)
   - V_flat value
   - V_max/V_bar ratio (enhancement)
   - Asymmetry
   
3. STRUCTURAL PROPERTIES
   - Size (R_disk, R_eff, R_HI)
   - Surface brightness
   - Luminosity
   - Mass (baryonic, HI)
   
4. DYNAMICAL PROPERTIES
   - Acceleration regime (g/g†)
   - Dark matter fraction
   - Baryonic dominance radius
   - Tully-Fisher residual
   
5. ENVIRONMENTAL PROPERTIES
   - Distance
   - Isolation
   - Data quality
   
6. DERIVED COHERENCE PROPERTIES
   - Estimated velocity dispersion
   - Jeans length estimates
   - Coherence survival probability
   - Enhancement factor

This allows systematic testing of how different gravity formulas perform
on different galaxy subsamples.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8
H0_SI = 2.27e-18
cH0 = c * H0_SI
kpc_to_m = 3.086e19
G_SI = 6.674e-11
M_sun = 1.989e30

g_dagger = cH0 / (4 * math.sqrt(math.pi))  # ≈ 9.6×10⁻¹¹ m/s²
a0_mond = 1.2e-10

# =============================================================================
# HUBBLE TYPE ENCODING
# =============================================================================

HUBBLE_TYPE_MAP = {
    0: 'S0',
    1: 'Sa',
    2: 'Sab',
    3: 'Sb',
    4: 'Sbc',
    5: 'Sc',
    6: 'Scd',
    7: 'Sd',
    8: 'Sdm',
    9: 'Sm',
    10: 'Im',
    11: 'BCD'
}

# Type groupings
EARLY_TYPES = {0, 1, 2, 3}  # S0, Sa, Sab, Sb
INTERMEDIATE_TYPES = {4, 5, 6}  # Sbc, Sc, Scd
LATE_TYPES = {7, 8, 9}  # Sd, Sdm, Sm
IRREGULAR_TYPES = {10, 11}  # Im, BCD

# Bulge-dominated vs disk-dominated
BULGE_DOMINATED = {0, 1, 2, 3}  # S0-Sb
DISK_DOMINATED = {4, 5, 6, 7, 8, 9, 10}  # Sbc-Im

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GalaxyProperties:
    """Complete set of derived properties for a galaxy."""
    name: str
    
    # === FROM TABLE1 ===
    hubble_type: int = -1
    hubble_name: str = 'Unknown'
    distance_mpc: float = None
    distance_err: float = None
    distance_quality: int = 0  # 1=best, 2=good, 3=uncertain
    inclination: float = None
    inc_err: float = None
    luminosity_3p6: float = None  # 10^9 L_sun
    lum_err: float = None
    R_eff_kpc: float = None
    SB_eff: float = None  # L/pc^2
    R_disk_kpc: float = None
    SB_disk: float = None  # L/pc^2
    M_HI: float = None  # 10^9 M_sun
    R_HI_kpc: float = None
    V_flat: float = None  # km/s
    V_flat_err: float = None
    quality_flag: int = 0  # 1=high, 2=medium, 3=low
    
    # === FROM ROTATION CURVE ===
    R_max_kpc: float = None
    V_max_obs: float = None
    V_max_bar: float = None
    n_points: int = 0
    
    # === DERIVED: MORPHOLOGY ===
    type_group: str = 'unknown'  # early/intermediate/late/irregular
    is_bulge_dominated: bool = False
    bulge_fraction: float = 0.0  # Average B/T
    max_bulge_fraction: float = 0.0  # Peak B/T
    has_significant_bulge: bool = False  # B/T > 0.1 somewhere
    
    # === DERIVED: KINEMATICS ===
    rc_shape: str = 'unknown'  # rising/flat/declining/irregular
    rc_slope_outer: float = None  # dV/dR in outer region
    rc_asymmetry: float = None  # Measure of asymmetry
    enhancement_ratio: float = None  # V_obs/V_bar at outer radii
    max_enhancement: float = None  # Max V_obs/V_bar
    
    # === DERIVED: STRUCTURE ===
    stellar_mass: float = None  # M_sun (from L_3.6)
    total_baryonic_mass: float = None  # M_sun
    size_class: str = 'medium'  # compact/medium/extended
    surface_brightness_class: str = 'normal'  # LSB/normal/HSB
    gas_fraction: float = None  # M_HI / M_baryonic
    
    # === DERIVED: DYNAMICS ===
    g_outer: float = None  # Acceleration at outer radius (m/s^2)
    g_inner: float = None  # Acceleration at inner radius (m/s^2)
    g_ratio_outer: float = None  # g_outer / g†
    g_ratio_inner: float = None  # g_inner / g†
    acceleration_regime: str = 'mixed'  # high-g/mixed/low-g
    baryonic_dominance_radius: float = None  # R where V_bar = V_obs
    dm_fraction_outer: float = None  # 1 - (V_bar/V_obs)^2 at outer R
    tully_fisher_residual: float = None  # log(V_flat) - TF prediction
    
    # === DERIVED: COHERENCE (for survival model) ===
    sigma_v_estimate: float = None  # Estimated velocity dispersion (km/s)
    jeans_length_estimate: float = None  # Estimated λ_J (kpc)
    coherence_survival_prob: float = None  # P_survive estimate
    decoherence_rate: float = None  # 1/λ_D estimate
    
    # === DATA QUALITY ===
    data_quality_score: float = None  # Combined quality metric
    has_good_outer_data: bool = False
    has_bulge_data: bool = False
    
    # === CLASSIFICATIONS (for slicing) ===
    classifications: Dict[str, str] = field(default_factory=dict)


@dataclass
class ClassificationScheme:
    """A way to slice the galaxy sample."""
    name: str
    description: str
    categories: List[str]
    classifier_func: str  # Name of function to call
    n_categories: int = 0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc_table(table_path: Path) -> Dict[str, Dict]:
    """Load the SPARC Table1 data."""
    galaxies = {}
    
    with open(table_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 17:
                continue
            
            try:
                name = parts[0]
                galaxies[name] = {
                    'hubble_type': int(parts[1]),
                    'distance_mpc': float(parts[2]),
                    'distance_err': float(parts[3]),
                    'distance_quality': int(parts[4]),
                    'inclination': float(parts[5]),
                    'inc_err': float(parts[6]),
                    'luminosity_3p6': float(parts[7]),  # 10^9 L_sun
                    'lum_err': float(parts[8]),
                    'R_eff_kpc': float(parts[9]),
                    'SB_eff': float(parts[10]),
                    'R_disk_kpc': float(parts[11]),
                    'SB_disk': float(parts[12]),
                    'M_HI': float(parts[13]),
                    'R_HI_kpc': float(parts[14]),
                    'V_flat': float(parts[15]) if float(parts[15]) > 0 else None,
                    'V_flat_err': float(parts[16]) if float(parts[16]) > 0 else None,
                    'quality_flag': int(parts[17]) if len(parts) > 17 else 1,
                }
            except (ValueError, IndexError):
                continue
    
    return galaxies


def load_rotation_curve(rotmod_path: Path) -> Optional[Dict]:
    """Load rotation curve data from rotmod file."""
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    SB_disk, SB_bulge = [], []
    distance = None
    
    with open(rotmod_path, 'r') as f:
        for line in f:
            if 'Distance' in line:
                try:
                    parts = line.split('=')
                    if len(parts) > 1:
                        distance = float(parts[1].split()[0])
                except:
                    pass
                continue
            
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]))
                    if len(parts) >= 8:
                        SB_disk.append(float(parts[6]))
                        SB_bulge.append(float(parts[7]))
                except ValueError:
                    continue
    
    if len(R) < 3:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    SB_disk = np.array(SB_disk) if SB_disk else np.zeros_like(R)
    SB_bulge = np.array(SB_bulge) if SB_bulge else np.zeros_like(R)
    
    # Compute V_bar
    V_bar_sq = np.sign(V_gas) * V_gas**2 + np.sign(V_disk) * V_disk**2 + V_bulge**2
    if np.any(V_bar_sq < 0):
        # Handle counter-rotating gas
        V_bar_sq = np.maximum(V_bar_sq, 0)
    V_bar = np.sqrt(V_bar_sq)
    
    return {
        'R': R,
        'V_obs': V_obs,
        'V_err': V_err,
        'V_gas': V_gas,
        'V_disk': V_disk,
        'V_bulge': V_bulge,
        'V_bar': V_bar,
        'SB_disk': SB_disk,
        'SB_bulge': SB_bulge,
        'distance': distance
    }


# =============================================================================
# PROPERTY DERIVATION FUNCTIONS
# =============================================================================

def derive_morphology_properties(props: GalaxyProperties, table_data: Dict, rc_data: Dict):
    """Derive morphology-related properties."""
    ht = table_data.get('hubble_type', -1)
    props.hubble_type = ht
    props.hubble_name = HUBBLE_TYPE_MAP.get(ht, 'Unknown')
    
    # Type group
    if ht in EARLY_TYPES:
        props.type_group = 'early'
    elif ht in INTERMEDIATE_TYPES:
        props.type_group = 'intermediate'
    elif ht in LATE_TYPES:
        props.type_group = 'late'
    elif ht in IRREGULAR_TYPES:
        props.type_group = 'irregular'
    
    props.is_bulge_dominated = ht in BULGE_DOMINATED
    
    # Bulge fraction from rotation curve
    V_bulge = rc_data['V_bulge']
    V_bar = rc_data['V_bar']
    V_bar_safe = np.maximum(V_bar, 1.0)
    
    bulge_frac = V_bulge**2 / (V_bar_safe**2)
    props.bulge_fraction = np.mean(bulge_frac)
    props.max_bulge_fraction = np.max(bulge_frac)
    props.has_significant_bulge = np.any(bulge_frac > 0.1)
    props.has_bulge_data = np.any(V_bulge > 0)


def derive_kinematic_properties(props: GalaxyProperties, table_data: Dict, rc_data: Dict):
    """Derive kinematic properties from rotation curve."""
    R = rc_data['R']
    V_obs = rc_data['V_obs']
    V_bar = rc_data['V_bar']
    V_err = rc_data['V_err']
    
    props.R_max_kpc = R.max()
    props.V_max_obs = V_obs.max()
    props.V_max_bar = V_bar.max()
    props.n_points = len(R)
    
    # V_flat from table
    props.V_flat = table_data.get('V_flat')
    props.V_flat_err = table_data.get('V_flat_err')
    
    # Rotation curve shape
    if len(R) >= 5:
        # Outer region (last 30%)
        n_outer = max(3, len(R) // 3)
        R_outer = R[-n_outer:]
        V_outer = V_obs[-n_outer:]
        
        # Linear fit to outer region
        if len(R_outer) >= 2 and R_outer[-1] > R_outer[0]:
            slope = (V_outer[-1] - V_outer[0]) / (R_outer[-1] - R_outer[0])
            props.rc_slope_outer = slope
            
            if slope > 2:
                props.rc_shape = 'rising'
            elif slope < -2:
                props.rc_shape = 'declining'
            else:
                props.rc_shape = 'flat'
        
        # Asymmetry (std of residuals from smooth fit)
        try:
            from scipy.ndimage import uniform_filter1d
            V_smooth = uniform_filter1d(V_obs, size=3)
            props.rc_asymmetry = np.std(V_obs - V_smooth) / np.mean(V_obs)
        except:
            props.rc_asymmetry = np.std(np.diff(V_obs)) / np.mean(V_obs)
    
    # Enhancement ratio
    outer_mask = R > 0.5 * R.max()
    if outer_mask.sum() >= 2:
        V_bar_outer = np.mean(V_bar[outer_mask])
        V_obs_outer = np.mean(V_obs[outer_mask])
        if V_bar_outer > 5:
            props.enhancement_ratio = V_obs_outer / V_bar_outer
    
    # Max enhancement
    V_bar_safe = np.maximum(V_bar, 5.0)
    props.max_enhancement = np.max(V_obs / V_bar_safe)
    
    # Good outer data check
    props.has_good_outer_data = (outer_mask.sum() >= 3) and (np.mean(V_err[outer_mask]) < 15)


def derive_structural_properties(props: GalaxyProperties, table_data: Dict, rc_data: Dict):
    """Derive structural properties."""
    # From table
    props.distance_mpc = table_data.get('distance_mpc')
    props.distance_err = table_data.get('distance_err')
    props.distance_quality = table_data.get('distance_quality', 0)
    props.inclination = table_data.get('inclination')
    props.inc_err = table_data.get('inc_err')
    props.luminosity_3p6 = table_data.get('luminosity_3p6')
    props.lum_err = table_data.get('lum_err')
    props.R_eff_kpc = table_data.get('R_eff_kpc')
    props.SB_eff = table_data.get('SB_eff')
    props.R_disk_kpc = table_data.get('R_disk_kpc')
    props.SB_disk = table_data.get('SB_disk')
    props.M_HI = table_data.get('M_HI')
    props.R_HI_kpc = table_data.get('R_HI_kpc')
    props.quality_flag = table_data.get('quality_flag', 0)
    
    # Stellar mass from 3.6μm luminosity
    # M_* ≈ 0.5 × L_3.6 (M/L ≈ 0.5 for 3.6μm)
    if props.luminosity_3p6:
        props.stellar_mass = 0.5 * props.luminosity_3p6 * 1e9 * M_sun  # kg
    
    # Total baryonic mass
    if props.stellar_mass and props.M_HI:
        M_HI_kg = props.M_HI * 1e9 * M_sun
        props.total_baryonic_mass = props.stellar_mass + 1.33 * M_HI_kg  # Include He
    
    # Gas fraction
    if props.total_baryonic_mass and props.M_HI:
        M_HI_kg = props.M_HI * 1e9 * M_sun
        props.gas_fraction = 1.33 * M_HI_kg / props.total_baryonic_mass
    
    # Size class
    if props.R_disk_kpc:
        if props.R_disk_kpc < 2:
            props.size_class = 'compact'
        elif props.R_disk_kpc > 6:
            props.size_class = 'extended'
        else:
            props.size_class = 'medium'
    
    # Surface brightness class
    # LSB: μ_0 > 23 mag/arcsec² (SB_disk < 50 L/pc²)
    # HSB: μ_0 < 21 mag/arcsec² (SB_disk > 500 L/pc²)
    if props.SB_disk:
        if props.SB_disk < 50:
            props.surface_brightness_class = 'LSB'
        elif props.SB_disk > 500:
            props.surface_brightness_class = 'HSB'
        else:
            props.surface_brightness_class = 'normal'


def derive_dynamical_properties(props: GalaxyProperties, table_data: Dict, rc_data: Dict):
    """Derive dynamical properties."""
    R = rc_data['R']
    V_obs = rc_data['V_obs']
    V_bar = rc_data['V_bar']
    
    R_m = R * kpc_to_m
    V_obs_ms = V_obs * 1000
    V_bar_ms = V_bar * 1000
    
    # Accelerations
    g_obs = V_obs_ms**2 / R_m
    g_bar = V_bar_ms**2 / R_m
    
    # Inner and outer accelerations
    props.g_inner = g_bar[0]
    props.g_outer = g_bar[-1]
    props.g_ratio_inner = g_bar[0] / g_dagger
    props.g_ratio_outer = g_bar[-1] / g_dagger
    
    # Acceleration regime
    mean_g_ratio = np.mean(g_bar / g_dagger)
    if mean_g_ratio > 2:
        props.acceleration_regime = 'high-g'
    elif mean_g_ratio < 0.5:
        props.acceleration_regime = 'low-g'
    else:
        props.acceleration_regime = 'mixed'
    
    # Baryonic dominance radius (where V_bar ≈ V_obs)
    ratio = V_bar / np.maximum(V_obs, 1.0)
    crossings = np.where(np.diff(np.sign(ratio - 0.9)))[0]
    if len(crossings) > 0:
        props.baryonic_dominance_radius = R[crossings[0]]
    
    # DM fraction at outer radius
    outer_mask = R > 0.7 * R.max()
    if outer_mask.sum() >= 2:
        V_bar_outer = np.mean(V_bar[outer_mask])
        V_obs_outer = np.mean(V_obs[outer_mask])
        if V_obs_outer > 0:
            props.dm_fraction_outer = 1 - (V_bar_outer / V_obs_outer)**2
    
    # Tully-Fisher residual
    # TF: log(V_flat) = 0.25 * log(M_bar) + const
    if props.V_flat and props.total_baryonic_mass:
        log_M = np.log10(props.total_baryonic_mass / M_sun)
        V_TF_pred = 10**(0.25 * log_M - 0.5)  # Rough TF
        props.tully_fisher_residual = np.log10(props.V_flat) - np.log10(V_TF_pred)


def derive_coherence_properties(props: GalaxyProperties, table_data: Dict, rc_data: Dict):
    """Derive coherence-related properties for survival model."""
    R = rc_data['R']
    V_bar = rc_data['V_bar']
    V_obs = rc_data['V_obs']
    
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * 1000
    
    # Estimate velocity dispersion
    # σ_v ≈ 0.2 × V_bar for thin disks
    # σ_v ≈ V_bar for dispersion-dominated systems
    if props.type_group in ['early', 'irregular']:
        sigma_factor = 0.5
    else:
        sigma_factor = 0.2
    
    props.sigma_v_estimate = sigma_factor * np.mean(V_bar)
    
    # Jeans length estimate
    # λ_J ≈ σ_v / √(4πGρ)
    # Use ρ ≈ V²/(4πGR²)
    R_mean = np.mean(R_m)
    V_mean = np.mean(V_bar_ms)
    rho_est = V_mean**2 / (4 * np.pi * G_SI * R_mean**2)
    sigma_v_ms = props.sigma_v_estimate * 1000
    
    lambda_J = sigma_v_ms / np.sqrt(4 * np.pi * G_SI * rho_est)
    props.jeans_length_estimate = lambda_J / kpc_to_m  # kpc
    
    # Decoherence rate estimate
    g_bar = V_bar_ms**2 / R_m
    mean_g = np.mean(g_bar)
    props.decoherence_rate = mean_g / g_dagger + sigma_v_ms / 100e3
    
    # Survival probability estimate
    # P = exp(-λ_J / λ_D) where λ_D ∝ 1/decoherence_rate
    lambda_D_est = 10 * kpc_to_m / props.decoherence_rate  # Reference 10 kpc
    props.coherence_survival_prob = np.exp(-lambda_J / lambda_D_est)


def compute_data_quality_score(props: GalaxyProperties) -> float:
    """Compute overall data quality score (0-1)."""
    score = 1.0
    
    # Distance quality
    if props.distance_quality == 1:
        score *= 1.0
    elif props.distance_quality == 2:
        score *= 0.8
    else:
        score *= 0.5
    
    # Inclination (face-on is bad)
    if props.inclination:
        if props.inclination < 30:
            score *= 0.5
        elif props.inclination < 45:
            score *= 0.8
    
    # Number of points
    if props.n_points:
        if props.n_points < 10:
            score *= 0.6
        elif props.n_points < 20:
            score *= 0.8
    
    # Quality flag
    if props.quality_flag == 1:
        score *= 1.0
    elif props.quality_flag == 2:
        score *= 0.8
    else:
        score *= 0.6
    
    return score


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_by_hubble_type(props: GalaxyProperties) -> str:
    """Classify by Hubble type."""
    return props.hubble_name


def classify_by_type_group(props: GalaxyProperties) -> str:
    """Classify by type group."""
    return props.type_group


def classify_by_bulge_presence(props: GalaxyProperties) -> str:
    """Classify by bulge presence."""
    if props.max_bulge_fraction > 0.3:
        return 'bulge_dominated'
    elif props.max_bulge_fraction > 0.1:
        return 'intermediate_bulge'
    else:
        return 'pure_disk'


def classify_by_rc_shape(props: GalaxyProperties) -> str:
    """Classify by rotation curve shape."""
    return props.rc_shape


def classify_by_size(props: GalaxyProperties) -> str:
    """Classify by size."""
    return props.size_class


def classify_by_surface_brightness(props: GalaxyProperties) -> str:
    """Classify by surface brightness."""
    return props.surface_brightness_class


def classify_by_acceleration_regime(props: GalaxyProperties) -> str:
    """Classify by acceleration regime."""
    return props.acceleration_regime


def classify_by_gas_fraction(props: GalaxyProperties) -> str:
    """Classify by gas fraction."""
    if props.gas_fraction is None:
        return 'unknown'
    if props.gas_fraction > 0.5:
        return 'gas_rich'
    elif props.gas_fraction > 0.2:
        return 'intermediate_gas'
    else:
        return 'gas_poor'


def classify_by_enhancement(props: GalaxyProperties) -> str:
    """Classify by enhancement ratio."""
    if props.enhancement_ratio is None:
        return 'unknown'
    if props.enhancement_ratio > 2.0:
        return 'high_enhancement'
    elif props.enhancement_ratio > 1.5:
        return 'moderate_enhancement'
    elif props.enhancement_ratio > 1.2:
        return 'low_enhancement'
    else:
        return 'minimal_enhancement'


def classify_by_dm_fraction(props: GalaxyProperties) -> str:
    """Classify by dark matter fraction."""
    if props.dm_fraction_outer is None:
        return 'unknown'
    if props.dm_fraction_outer > 0.8:
        return 'dm_dominated'
    elif props.dm_fraction_outer > 0.5:
        return 'dm_significant'
    elif props.dm_fraction_outer > 0.2:
        return 'baryon_significant'
    else:
        return 'baryon_dominated'


def classify_by_data_quality(props: GalaxyProperties) -> str:
    """Classify by data quality."""
    if props.data_quality_score is None:
        return 'unknown'
    if props.data_quality_score > 0.8:
        return 'high_quality'
    elif props.data_quality_score > 0.5:
        return 'medium_quality'
    else:
        return 'low_quality'


def classify_by_luminosity(props: GalaxyProperties) -> str:
    """Classify by luminosity."""
    if props.luminosity_3p6 is None:
        return 'unknown'
    # L in 10^9 L_sun
    if props.luminosity_3p6 > 50:
        return 'high_luminosity'
    elif props.luminosity_3p6 > 5:
        return 'medium_luminosity'
    elif props.luminosity_3p6 > 0.5:
        return 'low_luminosity'
    else:
        return 'dwarf'


def classify_by_distance(props: GalaxyProperties) -> str:
    """Classify by distance."""
    if props.distance_mpc is None:
        return 'unknown'
    if props.distance_mpc < 10:
        return 'nearby'
    elif props.distance_mpc < 30:
        return 'intermediate_distance'
    else:
        return 'distant'


def classify_by_inclination(props: GalaxyProperties) -> str:
    """Classify by inclination."""
    if props.inclination is None:
        return 'unknown'
    if props.inclination > 75:
        return 'edge_on'
    elif props.inclination > 60:
        return 'high_inclination'
    elif props.inclination > 45:
        return 'moderate_inclination'
    else:
        return 'face_on'


def classify_by_coherence_survival(props: GalaxyProperties) -> str:
    """Classify by coherence survival probability."""
    if props.coherence_survival_prob is None:
        return 'unknown'
    if props.coherence_survival_prob > 0.5:
        return 'high_survival'
    elif props.coherence_survival_prob > 0.2:
        return 'moderate_survival'
    elif props.coherence_survival_prob > 0.05:
        return 'low_survival'
    else:
        return 'minimal_survival'


def classify_by_jeans_length(props: GalaxyProperties) -> str:
    """Classify by Jeans length."""
    if props.jeans_length_estimate is None:
        return 'unknown'
    if props.jeans_length_estimate > 5:
        return 'large_jeans'
    elif props.jeans_length_estimate > 1:
        return 'medium_jeans'
    else:
        return 'small_jeans'


def classify_by_tf_residual(props: GalaxyProperties) -> str:
    """Classify by Tully-Fisher residual."""
    if props.tully_fisher_residual is None:
        return 'unknown'
    if props.tully_fisher_residual > 0.1:
        return 'above_tf'
    elif props.tully_fisher_residual < -0.1:
        return 'below_tf'
    else:
        return 'on_tf'


# Quantile-based classifications
def classify_by_quantile(props: GalaxyProperties, property_name: str, 
                         value: float, thresholds: List[float]) -> str:
    """Generic quantile classifier."""
    if value is None:
        return 'unknown'
    
    for i, thresh in enumerate(thresholds):
        if value <= thresh:
            return f'Q{i+1}'
    return f'Q{len(thresholds)+1}'


# =============================================================================
# COMBINED CLASSIFICATIONS
# =============================================================================

def classify_by_morphology_dynamics(props: GalaxyProperties) -> str:
    """Combined morphology + dynamics classification."""
    morph = props.type_group
    dyn = props.acceleration_regime
    return f'{morph}_{dyn}'


def classify_by_structure_kinematics(props: GalaxyProperties) -> str:
    """Combined structure + kinematics classification."""
    size = props.size_class
    shape = props.rc_shape
    return f'{size}_{shape}'


def classify_by_bulge_enhancement(props: GalaxyProperties) -> str:
    """Combined bulge + enhancement classification."""
    bulge = classify_by_bulge_presence(props)
    enh = classify_by_enhancement(props)
    return f'{bulge}_{enh}'


# =============================================================================
# MAIN CLASSIFICATION SYSTEM
# =============================================================================

# All classification schemes
CLASSIFICATION_SCHEMES = {
    # === MORPHOLOGICAL ===
    'hubble_type': ClassificationScheme(
        'hubble_type', 'Hubble morphological type', 
        list(HUBBLE_TYPE_MAP.values()), 'classify_by_hubble_type'
    ),
    'type_group': ClassificationScheme(
        'type_group', 'Morphological type group',
        ['early', 'intermediate', 'late', 'irregular', 'unknown'], 'classify_by_type_group'
    ),
    'bulge_presence': ClassificationScheme(
        'bulge_presence', 'Bulge fraction category',
        ['bulge_dominated', 'intermediate_bulge', 'pure_disk'], 'classify_by_bulge_presence'
    ),
    
    # === KINEMATIC ===
    'rc_shape': ClassificationScheme(
        'rc_shape', 'Rotation curve shape',
        ['rising', 'flat', 'declining', 'irregular', 'unknown'], 'classify_by_rc_shape'
    ),
    'enhancement': ClassificationScheme(
        'enhancement', 'V_obs/V_bar enhancement ratio',
        ['high_enhancement', 'moderate_enhancement', 'low_enhancement', 'minimal_enhancement', 'unknown'],
        'classify_by_enhancement'
    ),
    
    # === STRUCTURAL ===
    'size': ClassificationScheme(
        'size', 'Galaxy size class',
        ['compact', 'medium', 'extended'], 'classify_by_size'
    ),
    'surface_brightness': ClassificationScheme(
        'surface_brightness', 'Surface brightness class',
        ['LSB', 'normal', 'HSB'], 'classify_by_surface_brightness'
    ),
    'luminosity': ClassificationScheme(
        'luminosity', 'Luminosity class',
        ['dwarf', 'low_luminosity', 'medium_luminosity', 'high_luminosity', 'unknown'],
        'classify_by_luminosity'
    ),
    'gas_fraction': ClassificationScheme(
        'gas_fraction', 'Gas fraction category',
        ['gas_rich', 'intermediate_gas', 'gas_poor', 'unknown'], 'classify_by_gas_fraction'
    ),
    
    # === DYNAMICAL ===
    'acceleration_regime': ClassificationScheme(
        'acceleration_regime', 'Mean acceleration regime',
        ['high-g', 'mixed', 'low-g'], 'classify_by_acceleration_regime'
    ),
    'dm_fraction': ClassificationScheme(
        'dm_fraction', 'Dark matter fraction at outer radius',
        ['dm_dominated', 'dm_significant', 'baryon_significant', 'baryon_dominated', 'unknown'],
        'classify_by_dm_fraction'
    ),
    'tf_residual': ClassificationScheme(
        'tf_residual', 'Tully-Fisher relation residual',
        ['above_tf', 'on_tf', 'below_tf', 'unknown'], 'classify_by_tf_residual'
    ),
    
    # === OBSERVATIONAL ===
    'distance': ClassificationScheme(
        'distance', 'Distance category',
        ['nearby', 'intermediate_distance', 'distant', 'unknown'], 'classify_by_distance'
    ),
    'inclination': ClassificationScheme(
        'inclination', 'Inclination category',
        ['face_on', 'moderate_inclination', 'high_inclination', 'edge_on', 'unknown'],
        'classify_by_inclination'
    ),
    'data_quality': ClassificationScheme(
        'data_quality', 'Data quality score',
        ['high_quality', 'medium_quality', 'low_quality', 'unknown'], 'classify_by_data_quality'
    ),
    
    # === COHERENCE (for survival model) ===
    'coherence_survival': ClassificationScheme(
        'coherence_survival', 'Coherence survival probability',
        ['high_survival', 'moderate_survival', 'low_survival', 'minimal_survival', 'unknown'],
        'classify_by_coherence_survival'
    ),
    'jeans_length': ClassificationScheme(
        'jeans_length', 'Jeans length category',
        ['small_jeans', 'medium_jeans', 'large_jeans', 'unknown'], 'classify_by_jeans_length'
    ),
    
    # === COMBINED ===
    'morphology_dynamics': ClassificationScheme(
        'morphology_dynamics', 'Combined morphology and dynamics',
        [], 'classify_by_morphology_dynamics'
    ),
    'structure_kinematics': ClassificationScheme(
        'structure_kinematics', 'Combined structure and kinematics',
        [], 'classify_by_structure_kinematics'
    ),
    'bulge_enhancement': ClassificationScheme(
        'bulge_enhancement', 'Combined bulge and enhancement',
        [], 'classify_by_bulge_enhancement'
    ),
}


def apply_all_classifications(props: GalaxyProperties) -> Dict[str, str]:
    """Apply all classification schemes to a galaxy."""
    classifications = {}
    
    classifiers = {
        'hubble_type': classify_by_hubble_type,
        'type_group': classify_by_type_group,
        'bulge_presence': classify_by_bulge_presence,
        'rc_shape': classify_by_rc_shape,
        'enhancement': classify_by_enhancement,
        'size': classify_by_size,
        'surface_brightness': classify_by_surface_brightness,
        'luminosity': classify_by_luminosity,
        'gas_fraction': classify_by_gas_fraction,
        'acceleration_regime': classify_by_acceleration_regime,
        'dm_fraction': classify_by_dm_fraction,
        'tf_residual': classify_by_tf_residual,
        'distance': classify_by_distance,
        'inclination': classify_by_inclination,
        'data_quality': classify_by_data_quality,
        'coherence_survival': classify_by_coherence_survival,
        'jeans_length': classify_by_jeans_length,
        'morphology_dynamics': classify_by_morphology_dynamics,
        'structure_kinematics': classify_by_structure_kinematics,
        'bulge_enhancement': classify_by_bulge_enhancement,
    }
    
    for name, func in classifiers.items():
        try:
            classifications[name] = func(props)
        except:
            classifications[name] = 'unknown'
    
    return classifications


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_all_galaxies(
    table_path: Path,
    rotmod_dir: Path
) -> Dict[str, GalaxyProperties]:
    """Process all SPARC galaxies and derive all properties."""
    
    print("Loading SPARC table...")
    table_data = load_sparc_table(table_path)
    print(f"  Loaded {len(table_data)} galaxies from table")
    
    all_properties = {}
    
    print("\nProcessing rotation curves...")
    for rotmod_file in sorted(rotmod_dir.glob('*_rotmod.dat')):
        name = rotmod_file.stem.replace('_rotmod', '')
        
        # Load rotation curve
        rc_data = load_rotation_curve(rotmod_file)
        if rc_data is None:
            continue
        
        # Get table data (may not exist for all galaxies)
        tdata = table_data.get(name, {})
        
        # Create properties object
        props = GalaxyProperties(name=name)
        
        # Derive all properties
        derive_morphology_properties(props, tdata, rc_data)
        derive_kinematic_properties(props, tdata, rc_data)
        derive_structural_properties(props, tdata, rc_data)
        derive_dynamical_properties(props, tdata, rc_data)
        derive_coherence_properties(props, tdata, rc_data)
        
        # Data quality
        props.data_quality_score = compute_data_quality_score(props)
        
        # Apply all classifications
        props.classifications = apply_all_classifications(props)
        
        all_properties[name] = props
    
    print(f"  Processed {len(all_properties)} galaxies")
    
    return all_properties


def generate_all_slices(
    all_properties: Dict[str, GalaxyProperties]
) -> Dict[str, Dict[str, List[str]]]:
    """Generate all possible slices of the galaxy sample."""
    
    slices = {}
    
    for scheme_name in CLASSIFICATION_SCHEMES.keys():
        slices[scheme_name] = defaultdict(list)
        
        for name, props in all_properties.items():
            category = props.classifications.get(scheme_name, 'unknown')
            slices[scheme_name][category].append(name)
    
    return slices


def generate_cross_slices(
    all_properties: Dict[str, GalaxyProperties],
    scheme1: str,
    scheme2: str
) -> Dict[str, List[str]]:
    """Generate cross-classification slices."""
    
    cross_slices = defaultdict(list)
    
    for name, props in all_properties.items():
        cat1 = props.classifications.get(scheme1, 'unknown')
        cat2 = props.classifications.get(scheme2, 'unknown')
        key = f'{cat1}_x_{cat2}'
        cross_slices[key].append(name)
    
    return dict(cross_slices)


def print_slice_summary(slices: Dict[str, Dict[str, List[str]]]):
    """Print summary of all slices."""
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    
    total_ways = 0
    
    for scheme_name, categories in slices.items():
        n_cats = len([c for c, gals in categories.items() if len(gals) > 0])
        total_ways += n_cats
        
        print(f"\n{scheme_name.upper()}: {n_cats} categories")
        for cat, gals in sorted(categories.items(), key=lambda x: -len(x[1])):
            if len(gals) > 0:
                print(f"  {cat}: {len(gals)} galaxies")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {total_ways} ways to slice the sample")
    print(f"       (not counting cross-classifications)")


def save_classifications(
    all_properties: Dict[str, GalaxyProperties],
    output_path: Path
):
    """Save all classifications to JSON."""
    
    output = {}
    
    for name, props in all_properties.items():
        output[name] = {
            'classifications': props.classifications,
            'properties': {
                'hubble_type': props.hubble_type,
                'hubble_name': props.hubble_name,
                'type_group': props.type_group,
                'distance_mpc': props.distance_mpc,
                'inclination': props.inclination,
                'luminosity_3p6': props.luminosity_3p6,
                'R_disk_kpc': props.R_disk_kpc,
                'SB_disk': props.SB_disk,
                'V_flat': props.V_flat,
                'R_max_kpc': props.R_max_kpc,
                'V_max_obs': props.V_max_obs,
                'bulge_fraction': props.bulge_fraction,
                'rc_shape': props.rc_shape,
                'enhancement_ratio': props.enhancement_ratio,
                'gas_fraction': props.gas_fraction,
                'g_ratio_outer': props.g_ratio_outer,
                'dm_fraction_outer': props.dm_fraction_outer,
                'coherence_survival_prob': props.coherence_survival_prob,
                'jeans_length_estimate': props.jeans_length_estimate,
                'data_quality_score': props.data_quality_score,
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nSaved classifications to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SPARC COMPREHENSIVE GALAXY CLASSIFICATION SYSTEM")
    print("=" * 80)
    
    # Paths
    base_dir = Path("/Users/leonardspeiser/Projects/sigmagravity")
    table_path = base_dir / "data" / "sparc" / "Table1_SPARC.dat"
    rotmod_dir = base_dir / "data" / "Rotmod_LTG"
    output_dir = base_dir / "derivations"
    
    # Process all galaxies
    all_properties = process_all_galaxies(table_path, rotmod_dir)
    
    # Generate slices
    slices = generate_all_slices(all_properties)
    
    # Print summary
    print_slice_summary(slices)
    
    # Generate some interesting cross-slices
    print("\n" + "=" * 80)
    print("CROSS-CLASSIFICATION EXAMPLES")
    print("=" * 80)
    
    cross_examples = [
        ('type_group', 'acceleration_regime'),
        ('bulge_presence', 'enhancement'),
        ('surface_brightness', 'dm_fraction'),
        ('size', 'coherence_survival'),
    ]
    
    for s1, s2 in cross_examples:
        cross = generate_cross_slices(all_properties, s1, s2)
        n_combos = len([k for k, v in cross.items() if len(v) > 2])
        print(f"\n{s1} × {s2}: {n_combos} non-trivial combinations")
        for key, gals in sorted(cross.items(), key=lambda x: -len(x[1]))[:5]:
            if len(gals) >= 2:
                print(f"  {key}: {len(gals)} galaxies")
    
    # Count total ways to slice
    print("\n" + "=" * 80)
    print("TOTAL SLICING POSSIBILITIES")
    print("=" * 80)
    
    single_schemes = len(CLASSIFICATION_SCHEMES)
    cross_schemes = single_schemes * (single_schemes - 1) // 2
    
    single_categories = sum(len([c for c, g in cats.items() if len(g) > 0]) 
                           for cats in slices.values())
    
    print(f"\nSingle-property classifications: {single_schemes}")
    print(f"Cross-classifications possible: {cross_schemes}")
    print(f"Total single categories: {single_categories}")
    print(f"Estimated cross-categories: ~{single_categories * 3} (non-trivial)")
    
    # Save results
    output_path = output_dir / "sparc_galaxy_classifications.json"
    save_classifications(all_properties, output_path)
    
    # Save slices summary
    slices_path = output_dir / "sparc_classification_slices.json"
    slices_output = {k: {cat: len(gals) for cat, gals in v.items()} 
                     for k, v in slices.items()}
    with open(slices_path, 'w') as f:
        json.dump(slices_output, f, indent=2)
    print(f"Saved slice summary to {slices_path}")
    
    return all_properties, slices


if __name__ == "__main__":
    all_properties, slices = main()

