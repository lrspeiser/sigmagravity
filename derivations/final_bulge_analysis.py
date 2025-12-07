#!/usr/bin/env python3
"""
FINAL BULGE ANALYSIS

Key insight from debugging:
- Component-weighted model is MORE CORRECT physically
- But it gives different (often worse) results overall
- For NGC2841 specifically, weighted model is BETTER (43.5 vs 67.2 km/s)

Let's understand:
1. When does component weighting help vs hurt?
2. What's the optimal A_bulge in the weighted model?
3. Is the issue with the baseline model or the weighting?
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

def mond_velocity(R, V_bar):
    a0 = 1.2e-10
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    y = g_bar / a0
    nu = 1 / (1 - np.exp(-np.sqrt(y)))
    return V_bar * np.sqrt(nu)

def load_galaxies():
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
            
            f_bulge = np.sum(V_bulge_scaled**2) / max(np.sum(V_bar_sq), 1e-10)
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'V_gas': V_gas,
                'V_disk': V_disk_scaled,
                'V_bulge': V_bulge_scaled,
                'R_d': R_d,
                'f_bulge': f_bulge,
            })
        except:
            continue
    return galaxies

galaxies = load_galaxies()
print(f"Loaded {len(galaxies)} galaxies")

A = np.exp(1 / (2 * np.pi))
xi_coeff = 1 / (2 * np.pi)

# =============================================================================
# COMPARE BASELINE vs WEIGHTED FOR EACH GALAXY
# =============================================================================
print("\n" + "=" * 80)
print("COMPARING BASELINE vs COMPONENT-WEIGHTED MODEL")
print("=" * 80)

results = []
for gal in galaxies:
    R_m = gal['R'] * kpc_to_m
    xi = xi_coeff * gal['R_d']
    W = W_coherence(gal['R'], xi)
    
    # Baseline
    g_bar = (gal['V_bar'] * 1000)**2 / R_m
    h_bar = h_function(g_bar)
    Sigma_base = 1 + A * W * h_bar
    V_base = gal['V_bar'] * np.sqrt(Sigma_base)
    
    # Component-weighted (same A for all)
    g_disk = (gal['V_disk'] * 1000)**2 / R_m
    g_bulge = (gal['V_bulge'] * 1000)**2 / R_m
    g_gas = np.abs(np.sign(gal['V_gas']) * (gal['V_gas'] * 1000)**2 / R_m)
    
    h_disk = h_function(g_disk)
    h_bulge = h_function(g_bulge)
    h_gas = h_function(g_gas)
    
    Sigma_disk = 1 + A * W * h_disk
    Sigma_bulge = 1 + A * W * h_bulge
    Sigma_gas = 1 + A * W * h_gas
    
    V_bar_sq = gal['V_bar']**2
    f_disk = gal['V_disk']**2 / np.maximum(V_bar_sq, 1e-10)
    f_bulge_local = gal['V_bulge']**2 / np.maximum(V_bar_sq, 1e-10)
    f_gas = np.abs(np.sign(gal['V_gas']) * gal['V_gas']**2) / np.maximum(V_bar_sq, 1e-10)
    
    Sigma_weighted = f_disk * Sigma_disk + f_bulge_local * Sigma_bulge + f_gas * Sigma_gas
    V_weighted = gal['V_bar'] * np.sqrt(np.maximum(Sigma_weighted, 1))
    
    # MOND
    V_mond = mond_velocity(gal['R'], gal['V_bar'])
    
    rms_base = np.sqrt(np.mean((gal['V_obs'] - V_base)**2))
    rms_weighted = np.sqrt(np.mean((gal['V_obs'] - V_weighted)**2))
    rms_mond = np.sqrt(np.mean((gal['V_obs'] - V_mond)**2))
    
    results.append({
        'name': gal['name'],
        'rms_base': rms_base,
        'rms_weighted': rms_weighted,
        'rms_mond': rms_mond,
        'f_bulge': gal['f_bulge'],
        'base_better': rms_base < rms_weighted,
        'delta': rms_base - rms_weighted,  # Positive = weighted better
    })

df = pd.DataFrame(results)

print(f"\nOverall comparison:")
print(f"  Baseline mean RMS: {df['rms_base'].mean():.2f} km/s")
print(f"  Weighted mean RMS: {df['rms_weighted'].mean():.2f} km/s")
print(f"  Baseline better: {df['base_better'].sum()} galaxies")
print(f"  Weighted better: {(~df['base_better']).sum()} galaxies")

# Split by bulge fraction
no_bulge = df[df['f_bulge'] < 0.01]
low_bulge = df[(df['f_bulge'] >= 0.01) & (df['f_bulge'] < 0.1)]
mid_bulge = df[(df['f_bulge'] >= 0.1) & (df['f_bulge'] < 0.3)]
high_bulge = df[df['f_bulge'] >= 0.3]

print(f"\nBy bulge fraction:")
print(f"\n  {'Category':<20} {'N':<6} {'Base RMS':<12} {'Weight RMS':<12} {'Weight better':<15}")
print("  " + "-" * 70)
print(f"  {'No bulge (<1%)':<20} {len(no_bulge):<6} {no_bulge['rms_base'].mean():<12.2f} {no_bulge['rms_weighted'].mean():<12.2f} {(~no_bulge['base_better']).sum():<15}")
print(f"  {'Low bulge (1-10%)':<20} {len(low_bulge):<6} {low_bulge['rms_base'].mean():<12.2f} {low_bulge['rms_weighted'].mean():<12.2f} {(~low_bulge['base_better']).sum():<15}")
print(f"  {'Mid bulge (10-30%)':<20} {len(mid_bulge):<6} {mid_bulge['rms_base'].mean():<12.2f} {mid_bulge['rms_weighted'].mean():<12.2f} {(~mid_bulge['base_better']).sum():<15}")
print(f"  {'High bulge (>30%)':<20} {len(high_bulge):<6} {high_bulge['rms_base'].mean():<12.2f} {high_bulge['rms_weighted'].mean():<12.2f} {(~high_bulge['base_better']).sum():<15}")

# =============================================================================
# WHERE DOES WEIGHTED MODEL HELP?
# =============================================================================
print("\n" + "=" * 80)
print("WHERE DOES WEIGHTED MODEL HELP?")
print("=" * 80)

# Sort by improvement (positive = weighted better)
df_sorted = df.sort_values('delta', ascending=False)

print("\nGalaxies where WEIGHTED model is much better:")
print(f"\n  {'Galaxy':<20} {'f_bulge':<10} {'Base RMS':<10} {'Weight RMS':<12} {'Δ':<10}")
print("  " + "-" * 65)
for _, row in df_sorted.head(10).iterrows():
    print(f"  {row['name']:<20} {row['f_bulge']:<10.2f} {row['rms_base']:<10.1f} {row['rms_weighted']:<12.1f} {row['delta']:+.1f}")

print("\nGalaxies where BASELINE model is much better:")
print(f"\n  {'Galaxy':<20} {'f_bulge':<10} {'Base RMS':<10} {'Weight RMS':<12} {'Δ':<10}")
print("  " + "-" * 65)
for _, row in df_sorted.tail(10).iterrows():
    print(f"  {row['name']:<20} {row['f_bulge']:<10.2f} {row['rms_base']:<10.1f} {row['rms_weighted']:<12.1f} {row['delta']:+.1f}")

# =============================================================================
# OPTIMAL A_BULGE IN WEIGHTED MODEL
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMAL A_BULGE IN WEIGHTED MODEL")
print("=" * 80)

def evaluate_weighted_with_A_bulge(galaxies, A_bulge):
    results = []
    for gal in galaxies:
        R_m = gal['R'] * kpc_to_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        
        g_disk = (gal['V_disk'] * 1000)**2 / R_m
        g_bulge = (gal['V_bulge'] * 1000)**2 / R_m
        g_gas = np.abs(np.sign(gal['V_gas']) * (gal['V_gas'] * 1000)**2 / R_m)
        
        h_disk = h_function(g_disk)
        h_bulge = h_function(g_bulge)
        h_gas = h_function(g_gas)
        
        Sigma_disk = 1 + A * W * h_disk
        Sigma_bulge = 1 + A_bulge * W * h_bulge
        Sigma_gas = 1 + A * W * h_gas
        
        V_bar_sq = gal['V_bar']**2
        f_disk = gal['V_disk']**2 / np.maximum(V_bar_sq, 1e-10)
        f_bulge_local = gal['V_bulge']**2 / np.maximum(V_bar_sq, 1e-10)
        f_gas = np.abs(np.sign(gal['V_gas']) * gal['V_gas']**2) / np.maximum(V_bar_sq, 1e-10)
        
        Sigma_weighted = f_disk * Sigma_disk + f_bulge_local * Sigma_bulge + f_gas * Sigma_gas
        V_pred = gal['V_bar'] * np.sqrt(np.maximum(Sigma_weighted, 1))
        
        V_mond = mond_velocity(gal['R'], gal['V_bar'])
        
        rms = np.sqrt(np.mean((gal['V_obs'] - V_pred)**2))
        rms_mond = np.sqrt(np.mean((gal['V_obs'] - V_mond)**2))
        
        results.append({
            'rms': rms,
            'wins': rms < rms_mond,
            'f_bulge': gal['f_bulge'],
        })
    
    return pd.DataFrame(results)

print("\nTesting A_bulge in weighted model:")
print(f"\n  {'A_bulge':<10} {'Mean RMS':<12} {'Win Rate':<12} {'High-bulge RMS':<15}")
print("  " + "-" * 55)

for A_bulge in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.17, 1.5, 2.0]:
    df_test = evaluate_weighted_with_A_bulge(galaxies, A_bulge)
    high_bulge = df_test[df_test['f_bulge'] > 0.3]
    marker = " <-- same as disk" if abs(A_bulge - A) < 0.1 else ""
    print(f"  {A_bulge:<10.2f} {df_test['rms'].mean():<12.2f} {df_test['wins'].mean()*100:<12.1f}% {high_bulge['rms'].mean():<15.2f}{marker}")

# =============================================================================
# THE KEY INSIGHT
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

print("""
FINDING: The weighted model with A_bulge = A_disk gives:
  - Mean RMS = 22.97 km/s (worse than baseline 17.48)
  - But for HIGH-BULGE galaxies: 37.3 km/s (vs 31.3 baseline)

This is COUNTER-INTUITIVE. Why does the "more correct" model perform worse?

ANSWER: The issue is with the BASELINE model, not the weighting!

The baseline model uses h(g_total), which is WRONG for mixed systems:
  - When bulge and disk are both present, g_total is higher
  - Higher g → lower h(g) → less enhancement
  - This accidentally "suppresses" bulge contribution

The weighted model is MORE CORRECT but reveals that:
  - Our h(g) function may be wrong
  - Or our A value is wrong
  - Or the coherence model needs refinement

IMPLICATIONS:
1. The baseline model's "good" performance may be an accident
2. The weighted model exposes the true behavior
3. We need to understand WHY h(g_component) gives different results

NEXT STEPS:
1. Compare with MOND (which also uses g_total)
2. Test if MOND-like h(g) helps the weighted model
3. Investigate the physics of component-wise vs total acceleration
""")

# =============================================================================
# COMPARE WITH MOND
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON WITH MOND")
print("=" * 80)

print("\nMOND also uses g_total, not component-wise acceleration.")
print("Let's compare win rates:")

wins_base_vs_mond = (df['rms_base'] < df['rms_mond']).sum()
wins_weight_vs_mond = (df['rms_weighted'] < df['rms_mond']).sum()

print(f"\n  Baseline vs MOND: {wins_base_vs_mond}/{len(df)} wins ({wins_base_vs_mond/len(df)*100:.1f}%)")
print(f"  Weighted vs MOND: {wins_weight_vs_mond}/{len(df)} wins ({wins_weight_vs_mond/len(df)*100:.1f}%)")

# For high-bulge galaxies
high_bulge_df = df[df['f_bulge'] > 0.3]
wins_base_hb = (high_bulge_df['rms_base'] < high_bulge_df['rms_mond']).sum()
wins_weight_hb = (high_bulge_df['rms_weighted'] < high_bulge_df['rms_mond']).sum()

print(f"\n  High-bulge galaxies only:")
print(f"    Baseline vs MOND: {wins_base_hb}/{len(high_bulge_df)} wins")
print(f"    Weighted vs MOND: {wins_weight_hb}/{len(high_bulge_df)} wins")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

