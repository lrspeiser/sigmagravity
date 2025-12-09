#!/usr/bin/env python3
"""
Test: Replacing W(r) with Local Coherence Scalar C(r)

This script evaluates the proposal from feedback items 5-8:
  - Replace W(r) = r/(ξ+r) with W_C(r) = C(r) = v²/(v² + σ²)
  - Add fixed-point iteration since W_C depends on V_pred
  - Test different velocity dispersion models
  - Compare orbit-averaged vs local C

Key insight: W(r) ≈ ⟨C⟩_orbit was always the theoretical motivation.
This change makes that explicit in the code.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s (H0 = 70 km/s/Mpc)
cH0 = c * H0_SI
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

# Critical acceleration
g_dagger = cH0 / (4 * math.sqrt(math.pi))

# Canonical amplitude
A_GALAXY = np.exp(1 / (2 * np.pi))  # ≈ 1.173

print("=" * 80)
print("COHERENCE SCALAR W(r) REPLACEMENT TEST")
print("=" * 80)
print(f"\nPhysical constants:")
print(f"  g† = {g_dagger:.3e} m/s²")
print(f"  A₀ = {A_GALAXY:.4f}")

# =============================================================================
# PART 1: DEFINE THE NEW FUNCTIONS
# =============================================================================

def C_local(v_rot_kms: np.ndarray, sigma_kms: np.ndarray) -> np.ndarray:
    """
    Local coherence scalar: C = v²/(v² + σ²)
    
    This is the non-relativistic limit of the covariant expression:
    C = ω²/(ω² + 4πGρ + θ² + H₀²)
    """
    v2 = np.maximum(v_rot_kms, 0.0)**2
    s2 = np.maximum(sigma_kms, 1e-6)**2
    return v2 / (v2 + s2)


def W_geometric(r_kpc: np.ndarray, R_d_kpc: float) -> np.ndarray:
    """Original geometric coherence window: W(r) = r/(ξ+r), ξ = R_d/(2π)"""
    xi = R_d_kpc / (2 * np.pi)
    xi = max(xi, 0.01)
    return r_kpc / (xi + r_kpc)


def h_function(g: np.ndarray) -> np.ndarray:
    """Acceleration function: h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


# =============================================================================
# PART 2: VELOCITY DISPERSION MODELS
# =============================================================================

def sigma_constant(r_kpc: np.ndarray, V_bar: np.ndarray, R_d: float, 
                   sigma0: float = 20.0) -> np.ndarray:
    """Constant dispersion (simplest model)"""
    return np.full_like(r_kpc, sigma0)


def sigma_exponential(r_kpc: np.ndarray, V_bar: np.ndarray, R_d: float,
                      sigma0: float = 80.0, sigma_disk: float = 15.0) -> np.ndarray:
    """
    Exponential falloff: σ(r) = σ_disk + (σ_0 - σ_disk) × exp(-r/2R_d)
    
    Physical: High dispersion in bulge, low in outer disk
    """
    return sigma_disk + (sigma0 - sigma_disk) * np.exp(-r_kpc / (2 * R_d))


def sigma_mixed(r_kpc: np.ndarray, V_bar: np.ndarray, R_d: float,
                floor: float = 8.0, frac: float = 0.15) -> np.ndarray:
    """
    Mixed model: σ = √(floor² + (frac × V_bar)²)
    
    Physical: Floor from gas turbulence + fraction of rotation
    """
    return np.sqrt(floor**2 + (frac * V_bar)**2)


SIGMA_MODELS = {
    "constant": (sigma_constant, {"sigma0": 20.0}),
    "exponential": (sigma_exponential, {"sigma0": 80.0, "sigma_disk": 15.0}),
    "mixed": (sigma_mixed, {"floor": 8.0, "frac": 0.15}),
}

# =============================================================================
# PART 3: PREDICTION FUNCTIONS
# =============================================================================

def predict_velocity_geometric(R_kpc: np.ndarray, V_bar: np.ndarray, 
                               R_d: float, A: float = A_GALAXY) -> np.ndarray:
    """Original prediction with W(r) = r/(ξ+r)"""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    W = W_geometric(R_kpc, R_d)
    
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


def predict_velocity_C_local(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float,
                             sigma_fn, sigma_params: dict,
                             A: float = A_GALAXY,
                             max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Prediction with W_C(r) = C(r) using fixed-point iteration.
    
    Since C depends on V_pred, we iterate:
    1. Initialize V = V_bar
    2. Compute σ(r)
    3. Compute C = V²/(V² + σ²)
    4. Compute Σ = 1 + A × C × h(g_N)
    5. V_new = V_bar × √Σ
    6. Repeat until convergence
    
    CRITICAL: We use V_pred (not V_obs) to avoid data leakage!
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m  # Baryonic acceleration (fixed)
    
    h = h_function(g_bar)
    sigma = sigma_fn(R_kpc, V_bar, R_d, **sigma_params)
    
    # Initialize with V_bar
    V = np.array(V_bar, dtype=float)
    
    for i in range(max_iter):
        # C depends on predicted V, not observed!
        C = C_local(V, sigma)
        
        Sigma = 1 + A * C * h
        V_new = V_bar * np.sqrt(Sigma)
        
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    
    return V


def orbit_avg_C(r_grid: np.ndarray, C_grid: np.ndarray, 
                weight_grid: np.ndarray, ell_kpc: float) -> np.ndarray:
    """
    Orbit-averaged coherence: ⟨C⟩ at each radius.
    
    W(r) = ∫ C(r') Σ_b(r') K(r,r') r' dr' / ∫ Σ_b(r') K(r,r') r' dr'
    
    where K(r,r') ~ exp(-|r-r'|/ℓ) is a mixing kernel.
    """
    r = r_grid[:, None]
    rp = r_grid[None, :]
    
    K = np.exp(-np.abs(r - rp) / np.maximum(ell_kpc, 1e-6))
    
    num = (K * (C_grid[None, :] * weight_grid[None, :]) * rp).sum(axis=1)
    den = (K * (weight_grid[None, :]) * rp).sum(axis=1)
    
    return num / np.maximum(den, 1e-30)


def predict_velocity_C_orbit_avg(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float,
                                  sigma_fn, sigma_params: dict,
                                  A: float = A_GALAXY,
                                  max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """
    Prediction with orbit-averaged C.
    
    Uses V_bar² as baryonic weight proxy (proportional to enclosed baryonic influence).
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    sigma = sigma_fn(R_kpc, V_bar, R_d, **sigma_params)
    
    # Baryonic weight proxy: V_bar²
    weight = V_bar**2
    
    # Mixing scale: ~R_d
    ell = R_d
    
    V = np.array(V_bar, dtype=float)
    
    for i in range(max_iter):
        C = C_local(V, sigma)
        W = orbit_avg_C(R_kpc, C, weight, ell)
        
        Sigma = 1 + A * W * h
        V_new = V_bar * np.sqrt(Sigma)
        
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    
    return V


# =============================================================================
# PART 4: LOAD SPARC DATA
# =============================================================================

def load_sparc_galaxies(data_dir: str = "data/Rotmod_LTG") -> List[Dict]:
    """Load SPARC galaxy data."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {data_path}")
        return []
    
    # Load master sheet
    master_file = data_path / "MasterSheet_SPARC.mrt"
    if not master_file.exists():
        print(f"ERROR: Master sheet not found: {master_file}")
        return []
    
    # Parse master sheet
    master_data = {}
    with open(master_file, 'r') as f:
        in_data = False
        for line in f:
            if line.startswith('Galaxy'):
                in_data = True
                continue
            if not in_data or line.strip() == '' or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                name = parts[0]
                master_data[name] = {
                    'Hubtype': float(parts[1]) if parts[1] != '...' else 5.0,
                    'D': float(parts[2]) if parts[2] != '...' else 10.0,
                    'Inc': float(parts[3]) if parts[3] != '...' else 60.0,
                    'L36': float(parts[4]) if parts[4] != '...' else 1e9,
                    'Vflat': float(parts[5]) if parts[5] != '...' else 100.0,
                    'Rdisk': float(parts[6]) if parts[6] != '...' else 3.0,
                    'SBdisk': float(parts[7]) if parts[7] != '...' else 100.0,
                }
    
    # Load individual galaxy files
    galaxies = []
    dat_files = sorted(data_path.glob("*_rotmod.dat"))
    
    for dat_file in dat_files:
        name = dat_file.stem.replace("_rotmod", "")
        
        try:
            data = np.loadtxt(dat_file)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            if data.shape[1] >= 7 and len(data) >= 3:
                R = data[:, 0]
                V_obs = data[:, 1]
                V_err = data[:, 2]
                V_gas = data[:, 3]
                V_disk = data[:, 4]
                V_bulge = data[:, 5]
                
                # Skip if any NaN
                if np.any(np.isnan(R)) or np.any(np.isnan(V_obs)):
                    continue
                
                # Get R_d from master sheet
                R_d = master_data.get(name, {}).get('Rdisk', 3.0)
                
                galaxies.append({
                    'name': name,
                    'R': R,
                    'V_obs': V_obs,
                    'V_err': V_err,
                    'V_gas': V_gas,
                    'V_disk': V_disk,
                    'V_bulge': V_bulge,
                    'R_d': R_d,
                    'n_points': len(R),
                })
        except Exception as e:
            continue
    
    return galaxies


# =============================================================================
# PART 5: RUN COMPARISON
# =============================================================================

def compute_V_bar(V_gas, V_disk, V_bulge, ml_disk=0.5, ml_bulge=0.7):
    """Compute baryonic velocity from components."""
    V_bar_sq = V_gas**2 + ml_disk * V_disk**2 + ml_bulge * V_bulge**2
    return np.sqrt(np.maximum(V_bar_sq, 0))


def compute_rms(V_pred, V_obs, V_err):
    """Compute RMS error."""
    residuals = V_pred - V_obs
    return np.sqrt(np.mean(residuals**2))


def run_comparison():
    """Run full comparison on SPARC galaxies."""
    
    print("\n" + "=" * 80)
    print("LOADING SPARC DATA")
    print("=" * 80)
    
    galaxies = load_sparc_galaxies()
    print(f"Loaded {len(galaxies)} galaxies")
    
    if len(galaxies) == 0:
        print("ERROR: No galaxies loaded!")
        return
    
    # Filter to high-quality galaxies
    galaxies = [g for g in galaxies if g['n_points'] >= 5]
    print(f"Using {len(galaxies)} galaxies with ≥5 data points")
    
    # Methods to compare
    methods = {
        "Geometric W(r)": lambda R, V_bar, R_d, **kw: predict_velocity_geometric(R, V_bar, R_d),
        "C_local (σ=20)": lambda R, V_bar, R_d, **kw: predict_velocity_C_local(
            R, V_bar, R_d, sigma_constant, {"sigma0": 20.0}),
        "C_local (σ=30)": lambda R, V_bar, R_d, **kw: predict_velocity_C_local(
            R, V_bar, R_d, sigma_constant, {"sigma0": 30.0}),
        "C_local (σ=40)": lambda R, V_bar, R_d, **kw: predict_velocity_C_local(
            R, V_bar, R_d, sigma_constant, {"sigma0": 40.0}),
        "C_local (exp)": lambda R, V_bar, R_d, **kw: predict_velocity_C_local(
            R, V_bar, R_d, sigma_exponential, {"sigma0": 80.0, "sigma_disk": 15.0}),
        "C_local (mixed)": lambda R, V_bar, R_d, **kw: predict_velocity_C_local(
            R, V_bar, R_d, sigma_mixed, {"floor": 8.0, "frac": 0.15}),
        "C_orbit_avg (exp)": lambda R, V_bar, R_d, **kw: predict_velocity_C_orbit_avg(
            R, V_bar, R_d, sigma_exponential, {"sigma0": 80.0, "sigma_disk": 15.0}),
    }
    
    # Results storage
    results = {name: {"rms_list": [], "total_sq_err": 0, "total_points": 0} 
               for name in methods}
    
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON")
    print("=" * 80)
    
    for i, gal in enumerate(galaxies):
        R = gal['R']
        V_obs = gal['V_obs']
        V_err = gal['V_err']
        R_d = gal['R_d']
        
        V_bar = compute_V_bar(gal['V_gas'], gal['V_disk'], gal['V_bulge'])
        
        for method_name, method_fn in methods.items():
            try:
                V_pred = method_fn(R, V_bar, R_d)
                rms = compute_rms(V_pred, V_obs, V_err)
                
                results[method_name]["rms_list"].append(rms)
                results[method_name]["total_sq_err"] += np.sum((V_pred - V_obs)**2)
                results[method_name]["total_points"] += len(V_obs)
            except Exception as e:
                pass
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(galaxies)} galaxies...")
    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print(f"{'Method':<25} {'Mean RMS':>12} {'Median RMS':>12} {'Global RMS':>12}")
    print("-" * 80)
    
    baseline_rms = None
    
    for method_name in methods:
        r = results[method_name]
        if len(r["rms_list"]) > 0:
            mean_rms = np.mean(r["rms_list"])
            median_rms = np.median(r["rms_list"])
            global_rms = np.sqrt(r["total_sq_err"] / r["total_points"])
            
            if baseline_rms is None:
                baseline_rms = global_rms
            
            change = (global_rms - baseline_rms) / baseline_rms * 100
            change_str = f"({change:+.1f}%)" if method_name != "Geometric W(r)" else "(baseline)"
            
            print(f"{method_name:<25} {mean_rms:>10.2f} km/s {median_rms:>10.2f} km/s {global_rms:>10.2f} km/s {change_str}")
    
    print("-" * 80)
    
    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: C_local vs Geometric W(r)")
    print("=" * 80)
    
    # Compare on a few example galaxies
    example_galaxies = ["NGC2403", "NGC3198", "NGC6946", "UGC128", "DDO154"]
    
    print("\nExample galaxy comparison (σ_exp model):")
    print("-" * 90)
    print(f"{'Galaxy':<12} {'R_d':>6} {'N_pts':>6} {'Geom RMS':>12} {'C_local RMS':>12} {'Change':>10}")
    print("-" * 90)
    
    for gal in galaxies:
        if gal['name'] in example_galaxies:
            R = gal['R']
            V_obs = gal['V_obs']
            V_err = gal['V_err']
            R_d = gal['R_d']
            V_bar = compute_V_bar(gal['V_gas'], gal['V_disk'], gal['V_bulge'])
            
            V_geom = predict_velocity_geometric(R, V_bar, R_d)
            V_C = predict_velocity_C_local(R, V_bar, R_d, sigma_exponential, 
                                           {"sigma0": 80.0, "sigma_disk": 15.0})
            
            rms_geom = compute_rms(V_geom, V_obs, V_err)
            rms_C = compute_rms(V_C, V_obs, V_err)
            change = (rms_C - rms_geom) / rms_geom * 100
            
            print(f"{gal['name']:<12} {R_d:>6.2f} {len(R):>6} {rms_geom:>10.2f} km/s {rms_C:>10.2f} km/s {change:>+8.1f}%")
    
    print("-" * 90)
    
    # Physical insight
    print("\n" + "=" * 80)
    print("PHYSICAL INTERPRETATION")
    print("=" * 80)
    
    print("""
The key difference between W_geometric and W_C:

1. GEOMETRIC W(r) = r/(ξ+r):
   - Only depends on radius r and disk scale length R_d
   - Same for all galaxies with same R_d
   - No dependence on actual kinematics
   
2. LOCAL C(r) = v²/(v² + σ²):
   - Depends on actual rotation velocity
   - Depends on velocity dispersion (ordered vs random)
   - Self-consistent: uses V_pred not V_obs (no data leakage)
   - Requires fixed-point iteration
   
3. ORBIT-AVERAGED ⟨C⟩:
   - Accounts for gravitational mixing
   - More physically motivated
   - Computationally more expensive

The counter-rotating galaxy test (44% lower f_DM) already validates that
coherence depends on kinematics. The C_local formulation makes this explicit.
""")
    
    return results


if __name__ == "__main__":
    results = run_comparison()

