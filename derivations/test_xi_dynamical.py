#!/usr/bin/env python3
"""
Test Dynamical Coherence Scale ξ_dyn as Drop-in Alternative

This script compares:
1. Canonical: ξ = R_d/(2π) ≈ 0.159 × R_d
2. Dynamical: ξ_dyn = k × σ_eff / Ω_d with k ≈ 0.24

Both use the same W(r) = r/(ξ+r) form.

The dynamical scale has physical motivation:
- ξ_dyn ∝ σ/Ω captures the ratio of random to ordered motion
- Connects to the covariant coherence scalar C = ω²/(ω² + σ²/r² + ...)
- The transition C = 1/2 occurs when v_rot = σ, giving r_transition ~ σ/Ω

Usage:
    python derivations/test_xi_dynamical.py [--verbose]
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s (H0 = 70 km/s/Mpc)
cH0 = c * H0_SI
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration (derived from cosmology)
g_dagger = cH0 / (4 * math.sqrt(math.pi))

# =============================================================================
# CANONICAL PARAMETERS (from README)
# =============================================================================
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173, base amplitude
L_0 = 0.40  # Reference path length (kpc)
N_EXP = 0.27  # Path length exponent
XI_SCALE_CANONICAL = 1 / (2 * np.pi)  # ≈ 0.159

# M/L ratios (Lelli+ 2016)
ML_DISK = 0.5
ML_BULGE = 0.7

# Dynamical ξ parameters
K_DYNAMICAL = 0.24  # Calibrated constant

# Velocity dispersions for dynamical ξ (km/s)
SIGMA_GAS = 10.0    # Cold gas
SIGMA_DISK = 25.0   # Thin disk stars
SIGMA_BULGE = 120.0 # Bulge stars


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)
    
    Standard form with α = 0.5
    """
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def W_coherence(r_kpc: np.ndarray, xi_kpc: float) -> np.ndarray:
    """Coherence window W(r) = r/(ξ+r)
    
    This is the canonical form from the README.
    """
    xi = max(xi_kpc, 0.01)
    return r_kpc / (xi + r_kpc)


def xi_canonical(R_d_kpc: float) -> float:
    """Canonical coherence scale: ξ = R_d/(2π)"""
    return XI_SCALE_CANONICAL * R_d_kpc


def xi_dyn_kpc(R_d_kpc: float, V_bar_at_Rd_kms: float, sigma_eff_kms: float, k: float = K_DYNAMICAL) -> float:
    """Dynamical coherence scale: ξ = k × σ_eff / Ω_d
    
    Parameters:
    -----------
    R_d_kpc : float
        Disk scale length in kpc
    V_bar_at_Rd_kms : float
        Baryonic velocity at R_d in km/s
    sigma_eff_kms : float
        Effective velocity dispersion in km/s
    k : float
        Calibrated constant ≈ 0.24
        
    Returns:
    --------
    xi : float
        Coherence scale in kpc
    """
    Omega = V_bar_at_Rd_kms / np.maximum(R_d_kpc, 1e-6)  # (km/s)/kpc
    return k * sigma_eff_kms / np.maximum(Omega, 1e-12)  # kpc


def compute_sigma_eff(V_gas: np.ndarray, V_disk: np.ndarray, V_bulge: np.ndarray) -> float:
    """Compute effective velocity dispersion from component fractions.
    
    Uses mass-weighted average of component dispersions.
    """
    V_gas_max = np.abs(V_gas).max() if len(V_gas) > 0 else 0
    V_disk_max = np.abs(V_disk).max() if len(V_disk) > 0 else 0
    V_bulge_max = np.abs(V_bulge).max() if len(V_bulge) > 0 else 0
    
    V_total_sq = V_gas_max**2 + V_disk_max**2 + V_bulge_max**2
    
    if V_total_sq > 0:
        gas_frac = V_gas_max**2 / V_total_sq
        bulge_frac = V_bulge_max**2 / V_total_sq
        disk_frac = max(0, 1 - gas_frac - bulge_frac)
    else:
        gas_frac, disk_frac, bulge_frac = 0.3, 0.7, 0.0
    
    sigma_eff = gas_frac * SIGMA_GAS + disk_frac * SIGMA_DISK + bulge_frac * SIGMA_BULGE
    return sigma_eff


def predict_velocity(R_kpc: np.ndarray, V_bar: np.ndarray, xi_kpc: float,
                     A: float = A_0) -> np.ndarray:
    """Predict rotation velocity using Σ-Gravity.
    
    Σ = 1 + A × W(r) × h(g_N)
    V_pred = V_bar × √Σ
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    W = W_coherence(R_kpc, xi_kpc)
    
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """Predict rotation velocity using MOND.
    
    Uses simple interpolation function: ν(x) = 1/(1 - e^(-√x))
    """
    a0 = 1.2e-10  # MOND acceleration scale
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    
    return V_bar * np.sqrt(nu)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sparc_data(data_dir: Path) -> List[Dict]:
    """Load SPARC galaxy rotation curves."""
    sparc_dir = data_dir / "Rotmod_LTG"
    if not sparc_dir.exists():
        print(f"ERROR: SPARC data not found at {sparc_dir}")
        return []
    
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
        df['V_disk_scaled'] = df['V_disk'] * np.sqrt(ML_DISK)
        df['V_bulge_scaled'] = df['V_bulge'] * np.sqrt(ML_BULGE)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    df['V_disk_scaled']**2 + df['V_bulge_scaled']**2)
        df['V_bar'] = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        valid = (df['V_bar'] > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        df = df[valid]
        
        if len(df) >= 5:
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': df['V_bar'].values,
                'V_disk': df['V_disk_scaled'].values,
                'V_bulge': df['V_bulge_scaled'].values,
                'V_gas': df['V_gas'].values
            })
    
    return galaxies


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_comparison(verbose: bool = False):
    """Run comparison between canonical and dynamical ξ."""
    
    # Find data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found at {data_dir}")
        return
    
    print("=" * 70)
    print("DYNAMICAL COHERENCE SCALE COMPARISON")
    print("=" * 70)
    print()
    print("Comparing:")
    print(f"  1. Canonical: ξ = R_d/(2π) ≈ {XI_SCALE_CANONICAL:.4f} × R_d")
    print(f"  2. Dynamical: ξ_dyn = k × σ_eff / Ω_d with k = {K_DYNAMICAL}")
    print()
    print(f"Both use W(r) = r/(ξ+r) and A = {A_0:.4f}")
    print()
    
    # Load data
    print("Loading SPARC data...")
    galaxies = load_sparc_data(data_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    print()
    
    if len(galaxies) == 0:
        print("ERROR: No galaxies loaded!")
        return
    
    # Run comparison
    results_canonical = []
    results_dynamical = []
    results_mond = []
    
    xi_canonical_values = []
    xi_dynamical_values = []
    
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        V_disk = gal.get('V_disk', V_bar)
        V_bulge = gal.get('V_bulge', np.zeros_like(V_bar))
        V_gas = gal.get('V_gas', np.zeros_like(V_bar))
        
        # Estimate R_d from disk velocity profile
        if len(V_disk) > 0 and np.abs(V_disk).max() > 0:
            peak_idx = np.argmax(np.abs(V_disk))
            R_d = R[peak_idx] if peak_idx > 0 else R.max() / 3
        else:
            R_d = R.max() / 3
        
        # Compute sigma_eff
        sigma_eff = compute_sigma_eff(V_gas, V_disk, V_bulge)
        
        # Compute V_bar at R_d by interpolation
        V_bar_at_Rd = np.interp(R_d, R, V_bar)
        
        # Compute both ξ values
        xi_can = xi_canonical(R_d)
        xi_dyn = xi_dyn_kpc(R_d, V_bar_at_Rd, sigma_eff)
        
        xi_canonical_values.append(xi_can)
        xi_dynamical_values.append(xi_dyn)
        
        # Predictions
        V_pred_can = predict_velocity(R, V_bar, xi_can)
        V_pred_dyn = predict_velocity(R, V_bar, xi_dyn)
        V_pred_mond = predict_mond(R, V_bar)
        
        # RMS errors
        rms_can = np.sqrt(((V_obs - V_pred_can)**2).mean())
        rms_dyn = np.sqrt(((V_obs - V_pred_dyn)**2).mean())
        rms_mond = np.sqrt(((V_obs - V_pred_mond)**2).mean())
        
        results_canonical.append({
            'name': gal['name'],
            'rms': rms_can,
            'xi': xi_can,
            'R_d': R_d
        })
        results_dynamical.append({
            'name': gal['name'],
            'rms': rms_dyn,
            'xi': xi_dyn,
            'sigma_eff': sigma_eff,
            'V_bar_at_Rd': V_bar_at_Rd
        })
        results_mond.append({
            'name': gal['name'],
            'rms': rms_mond
        })
        
        if verbose:
            print(f"{gal['name']}: R_d={R_d:.2f} kpc, ξ_can={xi_can:.3f}, ξ_dyn={xi_dyn:.3f}, "
                  f"RMS_can={rms_can:.1f}, RMS_dyn={rms_dyn:.1f}, RMS_MOND={rms_mond:.1f}")
    
    # Summary statistics
    rms_can_all = [r['rms'] for r in results_canonical]
    rms_dyn_all = [r['rms'] for r in results_dynamical]
    rms_mond_all = [r['rms'] for r in results_mond]
    
    mean_rms_can = np.mean(rms_can_all)
    mean_rms_dyn = np.mean(rms_dyn_all)
    mean_rms_mond = np.mean(rms_mond_all)
    
    # Win rates
    wins_can_vs_mond = sum(1 for rc, rm in zip(rms_can_all, rms_mond_all) if rc < rm)
    wins_dyn_vs_mond = sum(1 for rd, rm in zip(rms_dyn_all, rms_mond_all) if rd < rm)
    wins_dyn_vs_can = sum(1 for rd, rc in zip(rms_dyn_all, rms_can_all) if rd < rc)
    
    n_gal = len(galaxies)
    
    # RAR scatter
    def compute_rar_scatter(galaxies, results, use_dynamical=False):
        log_ratios = []
        for gal, res in zip(galaxies, results):
            R = gal['R']
            V_obs = gal['V_obs']
            V_bar = gal['V_bar']
            
            R_m = R * kpc_to_m
            g_obs = (V_obs * 1000)**2 / R_m
            g_bar = (V_bar * 1000)**2 / R_m
            
            xi = res['xi']
            V_pred = predict_velocity(R, V_bar, xi)
            g_pred = (V_pred * 1000)**2 / R_m
            
            valid = (g_obs > 0) & (g_pred > 0)
            log_ratios.extend(np.log10(g_obs[valid] / g_pred[valid]))
        
        return np.std(log_ratios)
    
    scatter_can = compute_rar_scatter(galaxies, results_canonical)
    scatter_dyn = compute_rar_scatter(galaxies, results_dynamical)
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Canonical':<15} {'Dynamical':<15} {'MOND':<15}")
    print("-" * 70)
    print(f"{'Mean RMS (km/s)':<30} {mean_rms_can:<15.2f} {mean_rms_dyn:<15.2f} {mean_rms_mond:<15.2f}")
    print(f"{'RAR scatter (dex)':<30} {scatter_can:<15.3f} {scatter_dyn:<15.3f} {'—':<15}")
    print(f"{'Win rate vs MOND':<30} {wins_can_vs_mond/n_gal*100:<14.1f}% {wins_dyn_vs_mond/n_gal*100:<14.1f}% {'—':<15}")
    print()
    
    improvement = (mean_rms_can - mean_rms_dyn) / mean_rms_can * 100
    print(f"Dynamical vs Canonical:")
    print(f"  - RMS improvement: {improvement:+.1f}%")
    print(f"  - Win rate: {wins_dyn_vs_can}/{n_gal} = {wins_dyn_vs_can/n_gal*100:.1f}%")
    print()
    
    # ξ statistics
    print(f"Coherence scale statistics:")
    print(f"  - ξ_canonical: mean={np.mean(xi_canonical_values):.3f} kpc, std={np.std(xi_canonical_values):.3f}")
    print(f"  - ξ_dynamical: mean={np.mean(xi_dynamical_values):.3f} kpc, std={np.std(xi_dynamical_values):.3f}")
    print(f"  - Ratio ξ_dyn/ξ_can: mean={np.mean(np.array(xi_dynamical_values)/np.array(xi_canonical_values)):.2f}")
    print()
    
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if improvement > 0:
        print(f"✓ Dynamical ξ IMPROVES fit by {improvement:.1f}%")
    else:
        print(f"✗ Dynamical ξ WORSENS fit by {-improvement:.1f}%")
    print()
    
    # Return results for further analysis
    return {
        'canonical': {
            'mean_rms': mean_rms_can,
            'scatter': scatter_can,
            'win_rate_vs_mond': wins_can_vs_mond / n_gal
        },
        'dynamical': {
            'mean_rms': mean_rms_dyn,
            'scatter': scatter_dyn,
            'win_rate_vs_mond': wins_dyn_vs_mond / n_gal
        },
        'improvement_percent': improvement,
        'n_galaxies': n_gal
    }


def sweep_k_parameter():
    """Sweep over k values to find optimal."""
    
    # Find data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    print("Loading SPARC data...")
    galaxies = load_sparc_data(data_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    print()
    
    print("Sweeping k parameter...")
    print(f"{'k':<10} {'Mean RMS':<15} {'Win vs Canon':<15} {'Win vs MOND':<15}")
    print("-" * 55)
    
    k_values = np.linspace(0.10, 0.50, 21)
    best_k = None
    best_rms = float('inf')
    
    for k in k_values:
        rms_dyn_all = []
        rms_can_all = []
        rms_mond_all = []
        
        for gal in galaxies:
            R = gal['R']
            V_obs = gal['V_obs']
            V_bar = gal['V_bar']
            V_disk = gal.get('V_disk', V_bar)
            V_bulge = gal.get('V_bulge', np.zeros_like(V_bar))
            V_gas = gal.get('V_gas', np.zeros_like(V_bar))
            
            # Estimate R_d
            if len(V_disk) > 0 and np.abs(V_disk).max() > 0:
                peak_idx = np.argmax(np.abs(V_disk))
                R_d = R[peak_idx] if peak_idx > 0 else R.max() / 3
            else:
                R_d = R.max() / 3
            
            sigma_eff = compute_sigma_eff(V_gas, V_disk, V_bulge)
            V_bar_at_Rd = np.interp(R_d, R, V_bar)
            
            xi_can = xi_canonical(R_d)
            xi_dyn = xi_dyn_kpc(R_d, V_bar_at_Rd, sigma_eff, k=k)
            
            V_pred_can = predict_velocity(R, V_bar, xi_can)
            V_pred_dyn = predict_velocity(R, V_bar, xi_dyn)
            V_pred_mond = predict_mond(R, V_bar)
            
            rms_can_all.append(np.sqrt(((V_obs - V_pred_can)**2).mean()))
            rms_dyn_all.append(np.sqrt(((V_obs - V_pred_dyn)**2).mean()))
            rms_mond_all.append(np.sqrt(((V_obs - V_pred_mond)**2).mean()))
        
        mean_rms = np.mean(rms_dyn_all)
        wins_vs_can = sum(1 for rd, rc in zip(rms_dyn_all, rms_can_all) if rd < rc)
        wins_vs_mond = sum(1 for rd, rm in zip(rms_dyn_all, rms_mond_all) if rd < rm)
        n = len(galaxies)
        
        print(f"{k:<10.3f} {mean_rms:<15.2f} {wins_vs_can/n*100:<14.1f}% {wins_vs_mond/n*100:<14.1f}%")
        
        if mean_rms < best_rms:
            best_rms = mean_rms
            best_k = k
    
    print()
    print(f"Optimal k = {best_k:.3f} with RMS = {best_rms:.2f} km/s")
    print(f"Canonical RMS = {np.mean(rms_can_all):.2f} km/s")
    return best_k


if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    sweep = "--sweep" in sys.argv or "-s" in sys.argv
    
    if sweep:
        sweep_k_parameter()
    else:
        run_comparison(verbose=verbose)

