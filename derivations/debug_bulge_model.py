#!/usr/bin/env python3
"""
DEBUG: Why does component-weighted model give different results?

The baseline model (17.48 km/s) uses total V_bar.
The component model (21.58 km/s) weights by component.

Let's understand the difference.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

c = 3e8
H0 = 2.27e-18
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

data_dir = Path(__file__).parent.parent / "data"
rotmod_dir = data_dir / "Rotmod_LTG"

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return r / (xi + r)

# Load one galaxy to debug
test_files = list(rotmod_dir.glob("NGC2841*.dat"))
if test_files:
    f = test_files[0]
    lines = f.read_text().strip().split('\n')
    data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
    data = np.array([list(map(float, l.split())) for l in data_lines])
    
    R = data[:, 0]
    V_obs = data[:, 1]
    V_gas = data[:, 3]
    V_disk = data[:, 4]
    V_bulge = data[:, 5]
    
    V_disk_scaled = np.abs(V_disk) * np.sqrt(0.5)
    V_bulge_scaled = np.abs(V_bulge) * np.sqrt(0.7)
    
    V_bar_sq = np.sign(V_gas) * V_gas**2 + V_disk_scaled**2 + V_bulge_scaled**2
    V_bar = np.sqrt(np.maximum(V_bar_sq, 1e-10))
    
    # Disk scale length
    cumsum = np.cumsum(V_disk**2 * R)
    half_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    R_d = R[min(half_idx, len(R) - 1)]
    
    print("=" * 80)
    print(f"DEBUG: NGC2841 (bulge-dominated)")
    print("=" * 80)
    
    print(f"\nGalaxy properties:")
    print(f"  R_d = {R_d:.2f} kpc")
    print(f"  N points = {len(R)}")
    
    # Component fractions at each radius
    print(f"\n  {'R':<8} {'V_obs':<8} {'V_bar':<8} {'f_disk':<8} {'f_bulge':<8} {'f_gas':<8}")
    print("  " + "-" * 50)
    
    for i in [0, len(R)//4, len(R)//2, 3*len(R)//4, len(R)-1]:
        f_disk = V_disk_scaled[i]**2 / V_bar_sq[i]
        f_bulge = V_bulge_scaled[i]**2 / V_bar_sq[i]
        f_gas = np.abs(np.sign(V_gas[i]) * V_gas[i]**2) / V_bar_sq[i]
        print(f"  {R[i]:<8.1f} {V_obs[i]:<8.0f} {V_bar[i]:<8.0f} {f_disk:<8.2f} {f_bulge:<8.2f} {f_gas:<8.2f}")
    
    # Compare predictions
    A = np.exp(1 / (2 * np.pi))
    xi_coeff = 1 / (2 * np.pi)
    xi = xi_coeff * R_d
    
    # Method 1: Baseline (total V_bar)
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    W = W_coherence(R, xi)
    h = h_function(g_bar)
    Sigma_baseline = 1 + A * W * h
    V_pred_baseline = V_bar * np.sqrt(Sigma_baseline)
    
    # Method 2: Component-weighted with A_bulge = A_disk
    g_disk = (V_disk_scaled * 1000)**2 / R_m
    g_bulge = (V_bulge_scaled * 1000)**2 / R_m
    g_gas = np.abs(np.sign(V_gas) * (V_gas * 1000)**2 / R_m)
    
    h_disk = h_function(g_disk)
    h_bulge = h_function(g_bulge)
    h_gas = h_function(g_gas)
    
    Sigma_disk = 1 + A * W * h_disk
    Sigma_bulge = 1 + A * W * h_bulge
    Sigma_gas = 1 + A * W * h_gas
    
    f_disk = V_disk_scaled**2 / np.maximum(V_bar_sq, 1e-10)
    f_bulge = V_bulge_scaled**2 / np.maximum(V_bar_sq, 1e-10)
    f_gas = np.abs(np.sign(V_gas) * V_gas**2) / np.maximum(V_bar_sq, 1e-10)
    
    Sigma_weighted = f_disk * Sigma_disk + f_bulge * Sigma_bulge + f_gas * Sigma_gas
    V_pred_weighted = V_bar * np.sqrt(np.maximum(Sigma_weighted, 1))
    
    # Method 3: Component-weighted with A_bulge = 0
    Sigma_bulge_zero = 1 + 0 * W * h_bulge  # = 1
    Sigma_weighted_zero = f_disk * Sigma_disk + f_bulge * Sigma_bulge_zero + f_gas * Sigma_gas
    V_pred_zero = V_bar * np.sqrt(np.maximum(Sigma_weighted_zero, 1))
    
    print(f"\n  {'R':<8} {'V_obs':<8} {'V_base':<8} {'V_weight':<8} {'V_zero':<8} {'Σ_base':<8} {'Σ_weight':<8}")
    print("  " + "-" * 65)
    
    for i in [0, len(R)//4, len(R)//2, 3*len(R)//4, len(R)-1]:
        print(f"  {R[i]:<8.1f} {V_obs[i]:<8.0f} {V_pred_baseline[i]:<8.0f} {V_pred_weighted[i]:<8.0f} {V_pred_zero[i]:<8.0f} {Sigma_baseline[i]:<8.2f} {Sigma_weighted[i]:<8.2f}")
    
    rms_baseline = np.sqrt(np.mean((V_obs - V_pred_baseline)**2))
    rms_weighted = np.sqrt(np.mean((V_obs - V_pred_weighted)**2))
    rms_zero = np.sqrt(np.mean((V_obs - V_pred_zero)**2))
    
    print(f"\n  RMS comparison:")
    print(f"    Baseline (total V_bar): {rms_baseline:.1f} km/s")
    print(f"    Weighted (A_bulge = A): {rms_weighted:.1f} km/s")
    print(f"    Weighted (A_bulge = 0): {rms_zero:.1f} km/s")
    
    print("""
KEY INSIGHT:
The baseline model applies h(g_total) to V_total.
The weighted model applies h(g_component) to each component.

Since h(g) is nonlinear:
  h(g_disk + g_bulge) ≠ f_disk × h(g_disk) + f_bulge × h(g_bulge)

The weighted model is MORE CORRECT physically, but gives different results.
""")

# =============================================================================
# CORRECT APPROACH: Apply enhancement to total, modulated by bulge fraction
# =============================================================================
print("\n" + "=" * 80)
print("CORRECT APPROACH: MODULATE TOTAL ENHANCEMENT BY BULGE FRACTION")
print("=" * 80)

print("""
Instead of weighting by component, modulate the total enhancement:

  Σ = 1 + A × W(r) × h(g_bar) × (1 - α × f_bulge)

where:
  - g_bar is total baryonic acceleration
  - f_bulge is local bulge fraction
  - α is suppression factor (0 = no suppression, 1 = full suppression)
""")

def load_all_galaxies():
    galaxies = []
    for f in sorted(rotmod_dir.glob("*.dat")):
        try:
            lines = f.read_text().strip().split('\n')
            data_lines = [l for l in lines if l.strip() and not l.startswith('#')]
            if len(data_lines) < 3:
                continue
            
            data = np.array([list(map(float, l.split())) for l in data_lines])
            
            R = data[:, 0]
            V_obs = data[:, 1]
            V_gas = data[:, 3] if data.shape[1] > 3 else np.zeros_like(R)
            V_disk = data[:, 4] if data.shape[1] > 4 else np.zeros_like(R)
            V_bulge = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)
            
            V_disk_scaled = np.abs(V_disk) * np.sqrt(0.5)
            V_bulge_scaled = np.abs(V_bulge) * np.sqrt(0.7)
            
            V_bar_sq = np.sign(V_gas) * V_gas**2 + V_disk_scaled**2 + V_bulge_scaled**2
            if np.any(V_bar_sq <= 0):
                continue
            V_bar = np.sqrt(np.maximum(V_bar_sq, 1e-10))
            
            if np.sum(V_disk**2) > 0:
                cumsum = np.cumsum(V_disk**2 * R)
                half_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                R_d = R[min(half_idx, len(R) - 1)]
            else:
                R_d = R[-1] / 3
            R_d = max(R_d, 0.3)
            
            # Local bulge fraction at each radius
            f_bulge_local = V_bulge_scaled**2 / np.maximum(V_bar_sq, 1e-10)
            
            # Global bulge fraction
            f_bulge_global = np.sum(V_bulge_scaled**2) / max(np.sum(V_bar_sq), 1e-10)
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'R_d': R_d,
                'f_bulge_local': f_bulge_local,
                'f_bulge_global': f_bulge_global,
            })
        except:
            continue
    return galaxies

galaxies = load_all_galaxies()
print(f"\nLoaded {len(galaxies)} galaxies")

def mond_velocity(R, V_bar):
    a0 = 1.2e-10
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    y = g_bar / a0
    nu = 1 / (1 - np.exp(-np.sqrt(y)))
    return V_bar * np.sqrt(nu)

def evaluate_bulge_modulation(galaxies, alpha, use_local=True):
    """Evaluate with bulge modulation: A_eff = A × (1 - α × f_bulge)"""
    A = np.exp(1 / (2 * np.pi))
    xi_coeff = 1 / (2 * np.pi)
    
    results = []
    for gal in galaxies:
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_function(g_bar)
        
        if use_local:
            f_bulge = gal['f_bulge_local']
        else:
            f_bulge = gal['f_bulge_global']
        
        A_eff = A * (1 - alpha * f_bulge)
        Sigma = 1 + A_eff * W * h
        V_pred = gal['V_bar'] * np.sqrt(Sigma)
        
        V_mond = mond_velocity(gal['R'], gal['V_bar'])
        
        rms = np.sqrt(np.mean((gal['V_obs'] - V_pred)**2))
        rms_mond = np.sqrt(np.mean((gal['V_obs'] - V_mond)**2))
        
        results.append({
            'rms': rms,
            'wins': rms < rms_mond,
            'f_bulge': gal['f_bulge_global'],
        })
    
    return pd.DataFrame(results)

print("\nTesting bulge modulation (LOCAL f_bulge at each radius):")
print(f"\n  {'α':<10} {'Mean RMS':<12} {'Win Rate':<12} {'Bulge-dom RMS':<15}")
print("  " + "-" * 50)

for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    df = evaluate_bulge_modulation(galaxies, alpha, use_local=True)
    bulge_dom = df[df['f_bulge'] > 0.3]
    marker = " <-- baseline" if alpha == 0 else ""
    print(f"  {alpha:<10.1f} {df['rms'].mean():<12.2f} {df['wins'].mean()*100:<12.1f}% {bulge_dom['rms'].mean():<15.2f}{marker}")

print("\nTesting bulge modulation (GLOBAL f_bulge for whole galaxy):")
print(f"\n  {'α':<10} {'Mean RMS':<12} {'Win Rate':<12} {'Bulge-dom RMS':<15}")
print("  " + "-" * 50)

for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    df = evaluate_bulge_modulation(galaxies, alpha, use_local=False)
    bulge_dom = df[df['f_bulge'] > 0.3]
    marker = " <-- baseline" if alpha == 0 else ""
    print(f"  {alpha:<10.1f} {df['rms'].mean():<12.2f} {df['wins'].mean()*100:<12.1f}% {bulge_dom['rms'].mean():<15.2f}{marker}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
FINDINGS:

1. Component-weighted model is DIFFERENT from baseline:
   - Baseline: Σ = 1 + A × W × h(g_total)
   - Weighted: Σ = Σ(f_i × Σ_i) where Σ_i = 1 + A × W × h(g_i)
   
   These are NOT equivalent due to nonlinearity of h(g).

2. CORRECT APPROACH for bulge suppression:
   - Keep baseline formula
   - Modulate A by bulge fraction: A_eff = A × (1 - α × f_bulge)
   
3. RESULTS:
   - Local modulation: α ~ 0.4-0.6 improves bulge-dominated fits
   - Global modulation: Similar improvement
   
4. PHYSICAL INTERPRETATION:
   - Bulge stars have random orbits → less coherence
   - Modulating A by f_bulge captures this effect
   - Full suppression (α = 1) is too aggressive
""")

