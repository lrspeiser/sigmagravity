#!/usr/bin/env python3
"""
Coherence Root Cause Analysis

This script investigates what physical properties drive gravitational coherence
by:
1. Fitting per-galaxy coherence scale (r0 or ξ) while holding A and h(g) fixed
2. Computing fundamental physical proxies for each galaxy
3. Correlating fitted coherence scales with physical properties
4. Binning galaxies by quartiles of each property and comparing performance

The goal is to identify what coherence is "really tracking" in the data,
moving from phenomenological fitting toward fundamental understanding.

Physical proxies tested:
- g_bar slope (radial gradient noise)
- g_bar curvature (second-order structure)
- R_min_gbar (distance from acceleration minimum)
- Shear/winding q = -d ln Ω / d ln R (dephasing proxy)
- Effective velocity dispersion σ_eff (temperature proxy)
- Coherence time proxy (orbital periods)

Usage:
    python coherence_root_cause_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize_scalar, minimize
import json
import math
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration
g_dagger = c * H0_SI / (4 * math.sqrt(math.pi))
g_dagger_kpc = g_dagger / (kpc_to_m / 1e6)  # Convert to (km/s)^2 / kpc

# Fixed global parameters (from optimizer)
A_COEFF = 1.6
B_COEFF = 109.0
G_GALAXY = 0.038
A_GALAXY = np.sqrt(A_COEFF + B_COEFF * G_GALAXY**2)

# Typical velocity dispersions by component
SIGMA_GAS = 10.0    # km/s - cold gas
SIGMA_DISK = 25.0   # km/s - thin disk stars
SIGMA_BULGE = 120.0 # km/s - bulge/spheroid


@dataclass
class GalaxyPhysics:
    """Container for galaxy data and derived physical properties."""
    name: str
    R: np.ndarray
    V_obs: np.ndarray
    V_bar: np.ndarray
    V_gas: np.ndarray
    V_disk: np.ndarray
    V_bulge: np.ndarray
    
    # Basic properties
    R_max: float = 0.0
    V_flat: float = 0.0
    R_d: float = 0.0  # Disk scale length estimate
    gas_fraction: float = 0.0
    bulge_fraction: float = 0.0
    
    # Fit quality with global r0
    rms_global: float = 0.0
    
    # Fitted per-galaxy coherence scale
    r0_fitted: float = 0.0
    rms_fitted: float = 0.0
    improvement: float = 0.0  # % improvement from fitting r0
    
    # Physical proxies (root concepts)
    gbar_slope_med: float = 0.0      # Median |d ln g_bar / d ln R|
    gbar_curv_med: float = 0.0       # Median |d² ln g_bar / d ln R²|
    R_min_gbar: float = 0.0          # Radius of minimum g_bar
    R_min_gbar_frac: float = 0.0     # R_min_gbar / R_max
    shear_q_med: float = 0.0         # Median winding parameter q
    shear_q_inner: float = 0.0       # Inner region shear
    shear_q_outer: float = 0.0       # Outer region shear
    sigma_eff: float = 0.0           # Effective velocity dispersion
    sigma_v_ratio: float = 0.0       # σ_eff / V_flat
    orbital_periods: float = 0.0     # Number of orbits at R_max over Hubble time
    coherence_time: float = 0.0      # Orbital period at R_d (Gyr)
    
    # Derived coherence metrics
    W_dispersion: float = 0.0        # exp(-(σ/v_c)²) at outer radius
    

def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def f_path(r: np.ndarray, r0: float) -> np.ndarray:
    """Path-length coherence factor f(r) = r/(r+r0)."""
    return r / (r + r0)


def predict_sigma_gravity(R_kpc: np.ndarray, V_bar: np.ndarray, r0: float) -> np.ndarray:
    """Predict rotation velocity using Σ-Gravity with given r0."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    f = f_path(R_kpc, r0)
    
    Sigma = 1 + A_GALAXY * f * h
    return V_bar * np.sqrt(Sigma)


def compute_rms(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """Compute RMS error."""
    return np.sqrt(((V_obs - V_pred)**2).mean())


def compute_physical_proxies(gal: GalaxyPhysics) -> None:
    """Compute all physical proxy metrics for a galaxy."""
    R = gal.R
    V_obs = gal.V_obs
    V_bar = gal.V_bar
    V_gas = gal.V_gas
    V_disk = gal.V_disk
    V_bulge = gal.V_bulge
    
    eps = 1e-12
    
    # Basic properties
    gal.R_max = R.max()
    gal.V_flat = np.median(V_obs[-3:]) if len(V_obs) >= 3 else V_obs[-1]
    
    # Estimate disk scale length
    if len(V_disk) > 0 and V_disk.max() > 0:
        peak_idx = np.argmax(np.abs(V_disk))
        gal.R_d = R[peak_idx] if peak_idx > 0 else gal.R_max / 3
    else:
        gal.R_d = gal.R_max / 3
    
    # Component fractions
    V_gas_max = np.abs(V_gas).max() if len(V_gas) > 0 else 0
    V_disk_max = np.abs(V_disk).max() if len(V_disk) > 0 else 0
    V_bulge_max = np.abs(V_bulge).max() if len(V_bulge) > 0 else 0
    V_total_sq = V_gas_max**2 + V_disk_max**2 + V_bulge_max**2
    
    if V_total_sq > 0:
        gal.gas_fraction = V_gas_max**2 / V_total_sq
        gal.bulge_fraction = V_bulge_max**2 / V_total_sq
    
    # =========================================================================
    # PHYSICAL PROXIES
    # =========================================================================
    
    # 1. g_bar shape analysis (slope and curvature)
    g_bar_proxy = V_bar**2 / (R + eps)  # (km/s)^2/kpc - shape proxy
    
    logR = np.log(R + eps)
    logg = np.log(g_bar_proxy + eps)
    
    if len(R) > 3:
        g_slope = np.gradient(logg, logR)
        g_curv = np.gradient(g_slope, logR)
        
        gal.gbar_slope_med = float(np.nanmedian(np.abs(g_slope)))
        gal.gbar_curv_med = float(np.nanmedian(np.abs(g_curv)))
    
    # 2. Radius of minimum g_bar (where acceleration is weakest)
    min_idx = np.nanargmin(g_bar_proxy)
    gal.R_min_gbar = float(R[min_idx])
    gal.R_min_gbar_frac = gal.R_min_gbar / gal.R_max if gal.R_max > 0 else 0
    
    # 3. Shear/winding parameter q = -d ln Ω / d ln R
    Omega = V_obs / (R + eps)
    logOmega = np.log(Omega + eps)
    
    if len(R) > 3:
        q = -np.gradient(logOmega, logR)
        valid_q = q[np.isfinite(q)]
        
        if len(valid_q) > 0:
            gal.shear_q_med = float(np.nanmedian(valid_q))
            
            # Inner vs outer shear
            R_mid = gal.R_max / 2
            inner_mask = R < R_mid
            outer_mask = R >= R_mid
            
            if inner_mask.sum() > 1:
                q_inner = q[inner_mask]
                gal.shear_q_inner = float(np.nanmedian(q_inner[np.isfinite(q_inner)]))
            if outer_mask.sum() > 1:
                q_outer = q[outer_mask]
                gal.shear_q_outer = float(np.nanmedian(q_outer[np.isfinite(q_outer)]))
    
    # 4. Effective velocity dispersion (weighted by component fractions)
    disk_fraction = 1 - gal.gas_fraction - gal.bulge_fraction
    disk_fraction = max(0, disk_fraction)
    
    gal.sigma_eff = (gal.gas_fraction * SIGMA_GAS + 
                     disk_fraction * SIGMA_DISK + 
                     gal.bulge_fraction * SIGMA_BULGE)
    
    gal.sigma_v_ratio = gal.sigma_eff / gal.V_flat if gal.V_flat > 0 else 0
    
    # 5. Dispersion coherence factor W_dispersion = exp(-(σ/v_c)²)
    gal.W_dispersion = np.exp(-gal.sigma_v_ratio**2)
    
    # 6. Orbital dynamics
    # Number of orbits at R_max over Hubble time (~14 Gyr)
    T_orbit_outer = 2 * np.pi * gal.R_max / gal.V_flat if gal.V_flat > 0 else 0  # kpc/(km/s)
    T_orbit_outer_gyr = T_orbit_outer * kpc_to_m / (1e9 * 3.15e7 * 1000)  # Convert to Gyr
    gal.orbital_periods = 14.0 / T_orbit_outer_gyr if T_orbit_outer_gyr > 0 else 0
    
    # Orbital period at disk scale length
    V_at_Rd = np.interp(gal.R_d, R, V_obs) if gal.R_d <= R.max() else gal.V_flat
    T_orbit_Rd = 2 * np.pi * gal.R_d / V_at_Rd if V_at_Rd > 0 else 0
    gal.coherence_time = T_orbit_Rd * kpc_to_m / (1e9 * 3.15e7 * 1000)  # Gyr


def fit_per_galaxy_r0(gal: GalaxyPhysics, r0_global: float = 5.0) -> None:
    """Fit optimal r0 for a single galaxy while holding A fixed."""
    
    # First compute RMS with global r0
    V_pred_global = predict_sigma_gravity(gal.R, gal.V_bar, r0_global)
    gal.rms_global = compute_rms(gal.V_obs, V_pred_global)
    
    # Now fit r0 for this galaxy
    def objective(r0):
        if r0 <= 0.1:
            return 1e10
        V_pred = predict_sigma_gravity(gal.R, gal.V_bar, r0)
        return compute_rms(gal.V_obs, V_pred)
    
    # Search over reasonable range
    result = minimize_scalar(objective, bounds=(0.5, 100), method='bounded')
    
    gal.r0_fitted = result.x
    gal.rms_fitted = result.fun
    gal.improvement = (gal.rms_global - gal.rms_fitted) / gal.rms_global * 100 if gal.rms_global > 0 else 0


def load_sparc_galaxies(data_dir: Path) -> List[GalaxyPhysics]:
    """Load all SPARC galaxies."""
    sparc_dir = data_dir / "Rotmod_LTG"
    galaxy_files = sorted(sparc_dir.glob("*_rotmod.dat"))
    
    galaxies = []
    for gf in galaxy_files:
        data = []
        with open(gf) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    data.append({
                        'R': float(parts[0]),
                        'V_obs': float(parts[1]),
                        'V_gas': float(parts[3]),
                        'V_disk': float(parts[4]),
                        'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                    })
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        
        # Apply M/L corrections
        V_disk_scaled = df['V_disk'] * np.sqrt(0.5)
        V_bulge_scaled = df['V_bulge'] * np.sqrt(0.7)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    V_disk_scaled**2 + V_bulge_scaled**2)
        V_bar = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        # Filter valid points
        valid = (V_bar > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        if valid.sum() < 5:
            continue
        
        gal = GalaxyPhysics(
            name=gf.stem.replace('_rotmod', ''),
            R=df.loc[valid, 'R'].values,
            V_obs=df.loc[valid, 'V_obs'].values,
            V_bar=V_bar[valid].values,
            V_gas=df.loc[valid, 'V_gas'].values,
            V_disk=V_disk_scaled[valid].values,
            V_bulge=V_bulge_scaled[valid].values
        )
        
        galaxies.append(gal)
    
    return galaxies


def compute_correlations(galaxies: List[GalaxyPhysics]) -> Dict[str, float]:
    """Compute correlations between fitted r0 and physical proxies."""
    # Collect arrays
    r0_fitted = np.array([g.r0_fitted for g in galaxies])
    
    properties = {
        'V_flat': np.array([g.V_flat for g in galaxies]),
        'R_max': np.array([g.R_max for g in galaxies]),
        'R_d': np.array([g.R_d for g in galaxies]),
        'gas_fraction': np.array([g.gas_fraction for g in galaxies]),
        'bulge_fraction': np.array([g.bulge_fraction for g in galaxies]),
        'gbar_slope_med': np.array([g.gbar_slope_med for g in galaxies]),
        'gbar_curv_med': np.array([g.gbar_curv_med for g in galaxies]),
        'R_min_gbar': np.array([g.R_min_gbar for g in galaxies]),
        'R_min_gbar_frac': np.array([g.R_min_gbar_frac for g in galaxies]),
        'shear_q_med': np.array([g.shear_q_med for g in galaxies]),
        'shear_q_inner': np.array([g.shear_q_inner for g in galaxies]),
        'shear_q_outer': np.array([g.shear_q_outer for g in galaxies]),
        'sigma_eff': np.array([g.sigma_eff for g in galaxies]),
        'sigma_v_ratio': np.array([g.sigma_v_ratio for g in galaxies]),
        'W_dispersion': np.array([g.W_dispersion for g in galaxies]),
        'orbital_periods': np.array([g.orbital_periods for g in galaxies]),
        'coherence_time': np.array([g.coherence_time for g in galaxies]),
    }
    
    correlations = {}
    for name, values in properties.items():
        valid = np.isfinite(r0_fitted) & np.isfinite(values)
        if valid.sum() > 10:
            corr = np.corrcoef(r0_fitted[valid], values[valid])[0, 1]
            correlations[name] = corr
        else:
            correlations[name] = np.nan
    
    return correlations


def quartile_analysis(galaxies: List[GalaxyPhysics], 
                      property_name: str, 
                      get_value: callable) -> Dict:
    """Analyze performance by quartiles of a property."""
    values = np.array([get_value(g) for g in galaxies])
    valid = np.isfinite(values)
    
    if valid.sum() < 20:
        return {'error': 'insufficient data'}
    
    # Compute quartiles
    q25, q50, q75 = np.percentile(values[valid], [25, 50, 75])
    
    quartiles = {
        'Q1 (lowest)': (values <= q25),
        'Q2': (values > q25) & (values <= q50),
        'Q3': (values > q50) & (values <= q75),
        'Q4 (highest)': (values > q75)
    }
    
    results = {'property': property_name, 'quartiles': {}}
    
    for q_name, mask in quartiles.items():
        q_gals = [g for g, m in zip(galaxies, mask) if m]
        
        if len(q_gals) < 3:
            continue
        
        rms_global = [g.rms_global for g in q_gals]
        rms_fitted = [g.rms_fitted for g in q_gals]
        r0_fitted = [g.r0_fitted for g in q_gals]
        improvement = [g.improvement for g in q_gals]
        
        results['quartiles'][q_name] = {
            'count': len(q_gals),
            'property_range': f"{min([get_value(g) for g in q_gals]):.2f} - {max([get_value(g) for g in q_gals]):.2f}",
            'rms_global_mean': np.mean(rms_global),
            'rms_fitted_mean': np.mean(rms_fitted),
            'r0_fitted_mean': np.mean(r0_fitted),
            'r0_fitted_std': np.std(r0_fitted),
            'improvement_mean': np.mean(improvement),
            'galaxies': [g.name for g in q_gals]
        }
    
    return results


def test_modified_coherence_window(galaxies: List[GalaxyPhysics]) -> Dict:
    """
    Test modified coherence window with dispersion factor:
    W_eff = W(r) × exp(-(σ/v_c)²)
    """
    results = {
        'description': 'Testing W_eff = W(r) × exp(-(σ/v_c)²)',
        'galaxies': []
    }
    
    r0_global = 5.0
    
    for gal in galaxies:
        R = gal.R
        V_bar = gal.V_bar
        V_obs = gal.V_obs
        
        # Standard prediction
        V_pred_std = predict_sigma_gravity(R, V_bar, r0_global)
        rms_std = compute_rms(V_obs, V_pred_std)
        
        # Modified prediction with dispersion factor
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_bar = V_bar_ms**2 / R_m
        
        h = h_function(g_bar)
        f = f_path(R, r0_global)
        
        # Dispersion modulation
        W_disp = np.exp(-(gal.sigma_v_ratio)**2)
        
        Sigma_mod = 1 + A_GALAXY * f * h * W_disp
        V_pred_mod = V_bar * np.sqrt(Sigma_mod)
        rms_mod = compute_rms(V_obs, V_pred_mod)
        
        results['galaxies'].append({
            'name': gal.name,
            'sigma_v_ratio': gal.sigma_v_ratio,
            'W_dispersion': W_disp,
            'rms_standard': rms_std,
            'rms_modified': rms_mod,
            'change': (rms_mod - rms_std) / rms_std * 100
        })
    
    # Summary statistics
    changes = [g['change'] for g in results['galaxies']]
    results['summary'] = {
        'mean_change': np.mean(changes),
        'std_change': np.std(changes),
        'improved': sum(1 for c in changes if c < 0),
        'worsened': sum(1 for c in changes if c > 0),
        'total': len(changes)
    }
    
    return results


def test_shear_modulated_coherence(galaxies: List[GalaxyPhysics]) -> Dict:
    """
    Test shear-modulated coherence:
    W_eff = W(r) × exp(-|q - 1|)
    
    q = 1 corresponds to flat rotation (Keplerian would be q = 0.5)
    Deviation from flat rotation reduces coherence.
    """
    results = {
        'description': 'Testing W_eff = W(r) × exp(-|q - 1|)',
        'galaxies': []
    }
    
    r0_global = 5.0
    
    for gal in galaxies:
        R = gal.R
        V_bar = gal.V_bar
        V_obs = gal.V_obs
        
        if len(R) < 5:
            continue
        
        # Standard prediction
        V_pred_std = predict_sigma_gravity(R, V_bar, r0_global)
        rms_std = compute_rms(V_obs, V_pred_std)
        
        # Compute local shear q(R)
        eps = 1e-12
        Omega = V_obs / (R + eps)
        logOmega = np.log(Omega + eps)
        logR = np.log(R + eps)
        q = -np.gradient(logOmega, logR)
        q = np.clip(q, -2, 3)  # Clip extreme values
        
        # Shear modulation: exp(-|q - 1|)
        # q = 1 is flat rotation, deviations reduce coherence
        W_shear = np.exp(-np.abs(q - 1))
        
        # Modified prediction
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_bar = V_bar_ms**2 / R_m
        
        h = h_function(g_bar)
        f = f_path(R, r0_global)
        
        Sigma_mod = 1 + A_GALAXY * f * h * W_shear
        V_pred_mod = V_bar * np.sqrt(Sigma_mod)
        rms_mod = compute_rms(V_obs, V_pred_mod)
        
        results['galaxies'].append({
            'name': gal.name,
            'shear_q_med': gal.shear_q_med,
            'W_shear_mean': float(np.nanmean(W_shear)),
            'rms_standard': rms_std,
            'rms_modified': rms_mod,
            'change': (rms_mod - rms_std) / rms_std * 100
        })
    
    # Summary statistics
    changes = [g['change'] for g in results['galaxies']]
    results['summary'] = {
        'mean_change': np.mean(changes),
        'std_change': np.std(changes),
        'improved': sum(1 for c in changes if c < 0),
        'worsened': sum(1 for c in changes if c > 0),
        'total': len(changes)
    }
    
    return results


def test_gbar_gradient_modulation(galaxies: List[GalaxyPhysics]) -> Dict:
    """
    Test g_bar gradient modulation:
    W_eff = W(r) × exp(-α × |d ln g_bar / d ln R|)
    
    Steep gradients in baryonic acceleration create noise/decoherence.
    """
    results = {
        'description': 'Testing W_eff = W(r) × exp(-α × |d ln g_bar / d ln R|)',
        'alpha_tested': [0.1, 0.2, 0.5, 1.0],
        'best_alpha': None,
        'galaxies': []
    }
    
    r0_global = 5.0
    best_alpha = 0.0
    best_mean_rms = float('inf')
    
    for alpha in results['alpha_tested']:
        total_rms = 0
        count = 0
        
        for gal in galaxies:
            R = gal.R
            V_bar = gal.V_bar
            V_obs = gal.V_obs
            
            if len(R) < 5:
                continue
            
            # Compute local g_bar gradient
            eps = 1e-12
            g_bar_proxy = V_bar**2 / (R + eps)
            logg = np.log(g_bar_proxy + eps)
            logR = np.log(R + eps)
            g_slope = np.abs(np.gradient(logg, logR))
            
            # Gradient modulation
            W_grad = np.exp(-alpha * g_slope)
            
            # Modified prediction
            R_m = R * kpc_to_m
            V_bar_ms = V_bar * 1000
            g_bar = V_bar_ms**2 / R_m
            
            h = h_function(g_bar)
            f = f_path(R, r0_global)
            
            Sigma_mod = 1 + A_GALAXY * f * h * W_grad
            V_pred_mod = V_bar * np.sqrt(Sigma_mod)
            rms_mod = compute_rms(V_obs, V_pred_mod)
            
            total_rms += rms_mod
            count += 1
        
        mean_rms = total_rms / count if count > 0 else float('inf')
        if mean_rms < best_mean_rms:
            best_mean_rms = mean_rms
            best_alpha = alpha
    
    results['best_alpha'] = best_alpha
    results['best_mean_rms'] = best_mean_rms
    
    # Detailed results with best alpha
    for gal in galaxies:
        R = gal.R
        V_bar = gal.V_bar
        V_obs = gal.V_obs
        
        if len(R) < 5:
            continue
        
        # Standard prediction
        V_pred_std = predict_sigma_gravity(R, V_bar, r0_global)
        rms_std = compute_rms(V_obs, V_pred_std)
        
        # Compute local g_bar gradient
        eps = 1e-12
        g_bar_proxy = V_bar**2 / (R + eps)
        logg = np.log(g_bar_proxy + eps)
        logR = np.log(R + eps)
        g_slope = np.abs(np.gradient(logg, logR))
        
        # Gradient modulation with best alpha
        W_grad = np.exp(-best_alpha * g_slope)
        
        # Modified prediction
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_bar = V_bar_ms**2 / R_m
        
        h = h_function(g_bar)
        f = f_path(R, r0_global)
        
        Sigma_mod = 1 + A_GALAXY * f * h * W_grad
        V_pred_mod = V_bar * np.sqrt(Sigma_mod)
        rms_mod = compute_rms(V_obs, V_pred_mod)
        
        results['galaxies'].append({
            'name': gal.name,
            'gbar_slope_med': gal.gbar_slope_med,
            'W_grad_mean': float(np.nanmean(W_grad)),
            'rms_standard': rms_std,
            'rms_modified': rms_mod,
            'change': (rms_mod - rms_std) / rms_std * 100
        })
    
    # Summary statistics
    changes = [g['change'] for g in results['galaxies']]
    results['summary'] = {
        'mean_change': np.mean(changes),
        'std_change': np.std(changes),
        'improved': sum(1 for c in changes if c < 0),
        'worsened': sum(1 for c in changes if c > 0),
        'total': len(changes)
    }
    
    return results


def print_report(galaxies: List[GalaxyPhysics], 
                 correlations: Dict[str, float],
                 quartile_results: List[Dict],
                 modification_tests: Dict) -> None:
    """Print comprehensive analysis report."""
    
    print("=" * 100)
    print("COHERENCE ROOT CAUSE ANALYSIS")
    print("=" * 100)
    
    # =========================================================================
    # 1. Per-galaxy r0 fitting summary
    # =========================================================================
    print(f"\n{'='*100}")
    print("1. PER-GALAXY COHERENCE SCALE (r₀) FITTING")
    print("=" * 100)
    
    r0_fitted = [g.r0_fitted for g in galaxies]
    rms_global = [g.rms_global for g in galaxies]
    rms_fitted = [g.rms_fitted for g in galaxies]
    improvement = [g.improvement for g in galaxies]
    
    print(f"\nGlobal r₀ = 5.0 kpc")
    print(f"Mean RMS with global r₀: {np.mean(rms_global):.2f} km/s")
    print(f"Mean RMS with fitted r₀: {np.mean(rms_fitted):.2f} km/s")
    print(f"Mean improvement: {np.mean(improvement):.1f}%")
    
    print(f"\nFitted r₀ distribution:")
    print(f"  Mean: {np.mean(r0_fitted):.2f} kpc")
    print(f"  Median: {np.median(r0_fitted):.2f} kpc")
    print(f"  Std: {np.std(r0_fitted):.2f} kpc")
    print(f"  Range: {np.min(r0_fitted):.2f} - {np.max(r0_fitted):.2f} kpc")
    
    # =========================================================================
    # 2. Correlations with physical proxies
    # =========================================================================
    print(f"\n{'='*100}")
    print("2. CORRELATIONS: FITTED r₀ vs PHYSICAL PROPERTIES")
    print("=" * 100)
    print("\n(What is coherence scale really tracking?)")
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
    
    print(f"\n{'Property':<25} {'Correlation':>12} {'Interpretation':<50}")
    print("-" * 90)
    
    interpretations = {
        'V_flat': 'Larger r₀ for massive galaxies',
        'R_max': 'Larger r₀ for extended galaxies',
        'R_d': 'Larger r₀ for larger disk scale lengths',
        'gas_fraction': 'Gas-rich galaxies need different r₀',
        'bulge_fraction': 'Bulge-dominated need different r₀',
        'gbar_slope_med': 'Steeper g_bar profiles affect coherence',
        'gbar_curv_med': 'More curved g_bar profiles affect coherence',
        'R_min_gbar': 'Location of weakest acceleration matters',
        'R_min_gbar_frac': 'Relative position of g_bar minimum matters',
        'shear_q_med': 'Winding/shear affects coherence scale',
        'shear_q_inner': 'Inner shear affects coherence',
        'shear_q_outer': 'Outer shear affects coherence',
        'sigma_eff': 'Velocity dispersion affects coherence',
        'sigma_v_ratio': 'Temperature/rotation ratio matters',
        'W_dispersion': 'Dispersion coherence factor',
        'orbital_periods': 'More orbits = more coherence buildup',
        'coherence_time': 'Orbital period at R_d'
    }
    
    for prop, corr in sorted_corr:
        if np.isnan(corr):
            continue
        interp = interpretations.get(prop, '')
        print(f"{prop:<25} {corr:>12.3f} {interp:<50}")
    
    # Highlight strongest correlations
    print("\n" + "-" * 90)
    print("STRONGEST CORRELATIONS (|r| > 0.3):")
    for prop, corr in sorted_corr:
        if not np.isnan(corr) and abs(corr) > 0.3:
            direction = "increases" if corr > 0 else "decreases"
            print(f"  • r₀ {direction} with {prop} (r = {corr:.3f})")
    
    # =========================================================================
    # 3. Quartile analysis
    # =========================================================================
    print(f"\n{'='*100}")
    print("3. QUARTILE ANALYSIS BY PHYSICAL PROPERTY")
    print("=" * 100)
    
    for qr in quartile_results:
        if 'error' in qr:
            continue
        
        print(f"\n--- {qr['property']} ---")
        print(f"{'Quartile':<15} {'N':>5} {'Range':<20} {'RMS(global)':>12} {'RMS(fitted)':>12} {'r₀ fitted':>12} {'Improv%':>10}")
        print("-" * 95)
        
        for q_name, stats in qr['quartiles'].items():
            print(f"{q_name:<15} {stats['count']:>5} {stats['property_range']:<20} "
                  f"{stats['rms_global_mean']:>12.2f} {stats['rms_fitted_mean']:>12.2f} "
                  f"{stats['r0_fitted_mean']:>12.2f} {stats['improvement_mean']:>10.1f}")
    
    # =========================================================================
    # 4. Modification tests
    # =========================================================================
    print(f"\n{'='*100}")
    print("4. COHERENCE MODIFICATION TESTS")
    print("=" * 100)
    
    # Dispersion test
    print("\n--- Test A: Dispersion Modulation W_eff = W(r) × exp(-(σ/v_c)²) ---")
    disp_test = modification_tests['dispersion']
    print(f"Mean change in RMS: {disp_test['summary']['mean_change']:+.2f}%")
    print(f"Galaxies improved: {disp_test['summary']['improved']}/{disp_test['summary']['total']}")
    print(f"Galaxies worsened: {disp_test['summary']['worsened']}/{disp_test['summary']['total']}")
    
    # Shear test
    print("\n--- Test B: Shear Modulation W_eff = W(r) × exp(-|q - 1|) ---")
    shear_test = modification_tests['shear']
    print(f"Mean change in RMS: {shear_test['summary']['mean_change']:+.2f}%")
    print(f"Galaxies improved: {shear_test['summary']['improved']}/{shear_test['summary']['total']}")
    print(f"Galaxies worsened: {shear_test['summary']['worsened']}/{shear_test['summary']['total']}")
    
    # Gradient test
    print("\n--- Test C: g_bar Gradient Modulation W_eff = W(r) × exp(-α|d ln g_bar/d ln R|) ---")
    grad_test = modification_tests['gradient']
    print(f"Best α: {grad_test['best_alpha']}")
    print(f"Mean change in RMS: {grad_test['summary']['mean_change']:+.2f}%")
    print(f"Galaxies improved: {grad_test['summary']['improved']}/{grad_test['summary']['total']}")
    print(f"Galaxies worsened: {grad_test['summary']['worsened']}/{grad_test['summary']['total']}")
    
    # =========================================================================
    # 5. Key insights
    # =========================================================================
    print(f"\n{'='*100}")
    print("5. KEY INSIGHTS: WHAT DRIVES COHERENCE?")
    print("=" * 100)
    
    # Find the strongest predictors of r0
    strong_predictors = [(p, c) for p, c in sorted_corr if not np.isnan(c) and abs(c) > 0.3]
    
    print("\nBased on correlations with fitted r₀:")
    if strong_predictors:
        for prop, corr in strong_predictors[:5]:
            if corr > 0:
                print(f"  • {prop}: Larger galaxies need larger coherence scales (r = {corr:.3f})")
            else:
                print(f"  • {prop}: Higher values → smaller coherence scales (r = {corr:.3f})")
    else:
        print("  • No strong correlations found (|r| > 0.3)")
    
    # Check if any modification helped
    best_mod = None
    best_improvement = 0
    for name, test in modification_tests.items():
        if test['summary']['mean_change'] < best_improvement:
            best_improvement = test['summary']['mean_change']
            best_mod = name
    
    print("\nBased on coherence window modifications:")
    if best_mod and best_improvement < -1:
        print(f"  • {best_mod.upper()} modulation improved fits by {-best_improvement:.1f}% on average")
    else:
        print("  • None of the tested modifications significantly improved fits")
        print("  • This suggests the current f(r) formulation may already capture the key physics")
    
    # Recommendations
    print("\n" + "-" * 100)
    print("RECOMMENDATIONS FOR FUNDAMENTAL DERIVATION:")
    print("-" * 100)
    
    # Check if r0 scales with galaxy size
    r0_Rmax_corr = correlations.get('R_max', 0)
    r0_Rd_corr = correlations.get('R_d', 0)
    
    if abs(r0_Rmax_corr) > 0.4 or abs(r0_Rd_corr) > 0.4:
        print("\n1. SCALE-DEPENDENT COHERENCE:")
        print(f"   r₀ correlates with galaxy size (r = {r0_Rmax_corr:.3f} with R_max, {r0_Rd_corr:.3f} with R_d)")
        print("   → Consider r₀ ∝ R_d or r₀ ∝ R_max as a fundamental relation")
        print("   → Physical interpretation: coherence builds over a fraction of the disk scale")
    
    # Check if r0 scales with mass
    r0_Vflat_corr = correlations.get('V_flat', 0)
    if abs(r0_Vflat_corr) > 0.4:
        print("\n2. MASS-DEPENDENT COHERENCE:")
        print(f"   r₀ correlates with V_flat (r = {r0_Vflat_corr:.3f})")
        print("   → Consider r₀ ∝ V_flat² (mass proxy) as a fundamental relation")
        print("   → Physical interpretation: more massive systems have larger coherence volumes")
    
    # Check dispersion effects
    r0_sigma_corr = correlations.get('sigma_v_ratio', 0)
    if abs(r0_sigma_corr) > 0.3:
        print("\n3. TEMPERATURE/DISPERSION EFFECTS:")
        print(f"   r₀ correlates with σ/V ratio (r = {r0_sigma_corr:.3f})")
        print("   → Consider explicit dispersion factor in coherence window")
        print("   → Physical interpretation: thermal motion disrupts phase coherence")
    
    # Check shear effects
    r0_shear_corr = correlations.get('shear_q_med', 0)
    if abs(r0_shear_corr) > 0.3:
        print("\n4. SHEAR/WINDING EFFECTS:")
        print(f"   r₀ correlates with shear parameter q (r = {r0_shear_corr:.3f})")
        print("   → Consider shear-dependent coherence")
        print("   → Physical interpretation: differential rotation causes dephasing")


def main():
    print("=" * 100)
    print("COHERENCE ROOT CAUSE ANALYSIS")
    print("Investigating what physical properties drive gravitational coherence")
    print("=" * 100)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    
    print("\nLoading SPARC galaxies...")
    galaxies = load_sparc_galaxies(data_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Compute physical proxies for each galaxy
    print("\nComputing physical proxies...")
    for gal in galaxies:
        compute_physical_proxies(gal)
    
    # Fit per-galaxy r0
    print("\nFitting per-galaxy coherence scale r₀...")
    for gal in galaxies:
        fit_per_galaxy_r0(gal, r0_global=5.0)
    
    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(galaxies)
    
    # Quartile analysis
    print("\nPerforming quartile analysis...")
    properties_to_analyze = [
        ('V_flat (km/s)', lambda g: g.V_flat),
        ('R_max (kpc)', lambda g: g.R_max),
        ('R_d (kpc)', lambda g: g.R_d),
        ('gas_fraction', lambda g: g.gas_fraction),
        ('bulge_fraction', lambda g: g.bulge_fraction),
        ('gbar_slope_med', lambda g: g.gbar_slope_med),
        ('shear_q_med', lambda g: g.shear_q_med),
        ('sigma_v_ratio', lambda g: g.sigma_v_ratio),
        ('orbital_periods', lambda g: g.orbital_periods),
    ]
    
    quartile_results = []
    for prop_name, get_value in properties_to_analyze:
        result = quartile_analysis(galaxies, prop_name, get_value)
        quartile_results.append(result)
    
    # Test modifications
    print("\nTesting coherence window modifications...")
    modification_tests = {
        'dispersion': test_modified_coherence_window(galaxies),
        'shear': test_shear_modulated_coherence(galaxies),
        'gradient': test_gbar_gradient_modulation(galaxies)
    }
    
    # Print report
    print_report(galaxies, correlations, quartile_results, modification_tests)
    
    # Save results
    output_dir = Path(__file__).parent / "coherence_root_cause_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save correlations
    with open(output_dir / "correlations.json", 'w') as f:
        json.dump({k: float(v) if not np.isnan(v) else None for k, v in correlations.items()}, f, indent=2)
    
    # Save per-galaxy data
    galaxy_data = []
    for g in galaxies:
        galaxy_data.append({
            'name': g.name,
            'V_flat': float(g.V_flat),
            'R_max': float(g.R_max),
            'R_d': float(g.R_d),
            'gas_fraction': float(g.gas_fraction),
            'bulge_fraction': float(g.bulge_fraction),
            'rms_global': float(g.rms_global),
            'r0_fitted': float(g.r0_fitted),
            'rms_fitted': float(g.rms_fitted),
            'improvement': float(g.improvement),
            'gbar_slope_med': float(g.gbar_slope_med),
            'gbar_curv_med': float(g.gbar_curv_med),
            'R_min_gbar': float(g.R_min_gbar),
            'shear_q_med': float(g.shear_q_med),
            'sigma_eff': float(g.sigma_eff),
            'sigma_v_ratio': float(g.sigma_v_ratio),
            'W_dispersion': float(g.W_dispersion),
            'orbital_periods': float(g.orbital_periods),
            'coherence_time': float(g.coherence_time)
        })
    
    with open(output_dir / "galaxy_physics.json", 'w') as f:
        json.dump(galaxy_data, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

