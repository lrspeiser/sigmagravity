#!/usr/bin/env python3
"""
BULGE INNER REGION ANALYSIS

Key finding: Inner regions have RMS = 37.7 km/s vs Outer = 22.4 km/s
The model UNDER-predicts in inner regions (+18.7 km/s residual)

Questions:
1. Why does the model under-predict in bulge-dominated inner regions?
2. Is it because W(r) is too low at small r?
3. Or is it because h(g) is wrong at high g (inner regions have high g)?
4. What if we use different parameters for inner regions?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
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
            f_bulge_local = V_bulge_scaled**2 / np.maximum(V_bar_sq, 1e-10)
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'V_bulge': V_bulge_scaled,
                'V_disk': V_disk_scaled,
                'R_d': R_d,
                'f_bulge': f_bulge,
                'f_bulge_local': f_bulge_local,
            })
        except:
            continue
    return galaxies

galaxies = load_galaxies()
bulge_galaxies = [g for g in galaxies if g['f_bulge'] > 0.1]

print("=" * 80)
print("BULGE INNER REGION ANALYSIS")
print("=" * 80)
print(f"\nAnalyzing {len(bulge_galaxies)} bulge galaxies")

A = np.exp(1 / (2 * np.pi))
xi_coeff = 1 / (2 * np.pi)

# =============================================================================
# DIAGNOSE: WHAT'S HAPPENING IN INNER REGIONS
# =============================================================================
print("\n" + "=" * 80)
print("DIAGNOSING INNER REGION UNDER-PREDICTION")
print("=" * 80)

print("\nFor inner regions, model under-predicts by +18.7 km/s on average.")
print("This means V_obs > V_pred, so Σ is too low.")
print("\nPossible causes:")
print("  1. W(r) too low at small r → need larger ξ or different W form")
print("  2. h(g) too low at high g → need different h(g) for high-g regime")
print("  3. A too low → need higher amplitude")

# Collect inner region data
inner_data = []
for gal in bulge_galaxies:
    R = gal['R']
    R_max = R.max()
    inner_mask = R < R_max / 3
    
    if inner_mask.sum() == 0:
        continue
    
    R_inner = R[inner_mask]
    V_obs_inner = gal['V_obs'][inner_mask]
    V_bar_inner = gal['V_bar'][inner_mask]
    f_bulge_inner = gal['f_bulge_local'][inner_mask]
    
    R_m = R_inner * kpc_to_m
    g_bar = (V_bar_inner * 1000)**2 / R_m
    xi = xi_coeff * gal['R_d']
    W = W_coherence(R_inner, xi)
    h = h_function(g_bar)
    
    for i in range(len(R_inner)):
        inner_data.append({
            'name': gal['name'],
            'R': R_inner[i],
            'V_obs': V_obs_inner[i],
            'V_bar': V_bar_inner[i],
            'f_bulge': f_bulge_inner[i],
            'g_bar': g_bar[i],
            'g_norm': g_bar[i] / g_dagger,
            'W': W[i],
            'h': h[i],
            'Sigma': 1 + A * W[i] * h[i],
            'V_pred': V_bar_inner[i] * np.sqrt(1 + A * W[i] * h[i]),
            'residual': V_obs_inner[i] - V_bar_inner[i] * np.sqrt(1 + A * W[i] * h[i]),
        })

df = pd.DataFrame(inner_data)

print(f"\nInner region statistics ({len(df)} data points):")
print(f"  Mean g/g†: {df['g_norm'].mean():.2f}")
print(f"  Mean W: {df['W'].mean():.3f}")
print(f"  Mean h: {df['h'].mean():.3f}")
print(f"  Mean Σ: {df['Sigma'].mean():.3f}")
print(f"  Mean residual: {df['residual'].mean():+.1f} km/s")

# =============================================================================
# TEST 1: WHAT IF W WERE HIGHER?
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: WHAT IF W WERE HIGHER?")
print("=" * 80)

print("\nCurrent: W = r/(ξ+r) with ξ = R_d/(2π)")
print("Testing: What W value would give zero residual?")

# For each point, compute required W
df['W_required'] = np.nan
for i, row in df.iterrows():
    # V_obs = V_bar × √(1 + A × W × h)
    # (V_obs/V_bar)² = 1 + A × W × h
    # W = ((V_obs/V_bar)² - 1) / (A × h)
    ratio_sq = (row['V_obs'] / row['V_bar'])**2
    if ratio_sq > 1 and row['h'] > 0:
        W_req = (ratio_sq - 1) / (A * row['h'])
        df.loc[i, 'W_required'] = min(W_req, 10)  # Cap at 10

valid = df[df['W_required'].notna() & (df['W_required'] < 5)]
print(f"\n  Current W: {valid['W'].mean():.3f}")
print(f"  Required W: {valid['W_required'].mean():.3f}")
print(f"  Ratio: {valid['W_required'].mean() / valid['W'].mean():.2f}x")

# =============================================================================
# TEST 2: WHAT IF h WERE HIGHER?
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: WHAT IF h WERE HIGHER?")
print("=" * 80)

print("\nCurrent: h = √(g†/g) × g†/(g†+g)")
print("Testing: What h value would give zero residual?")

df['h_required'] = np.nan
for i, row in df.iterrows():
    ratio_sq = (row['V_obs'] / row['V_bar'])**2
    if ratio_sq > 1 and row['W'] > 0:
        h_req = (ratio_sq - 1) / (A * row['W'])
        df.loc[i, 'h_required'] = min(h_req, 50)

valid = df[df['h_required'].notna() & (df['h_required'] < 20)]
print(f"\n  Current h: {valid['h'].mean():.3f}")
print(f"  Required h: {valid['h_required'].mean():.3f}")
print(f"  Ratio: {valid['h_required'].mean() / valid['h'].mean():.2f}x")

# =============================================================================
# TEST 3: WHAT IF A WERE HIGHER?
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: WHAT IF A WERE HIGHER?")
print("=" * 80)

print(f"\nCurrent: A = e^(1/2π) ≈ {A:.3f}")
print("Testing: What A value would give zero mean residual?")

df['A_required'] = np.nan
for i, row in df.iterrows():
    ratio_sq = (row['V_obs'] / row['V_bar'])**2
    if ratio_sq > 1 and row['W'] > 0 and row['h'] > 0:
        A_req = (ratio_sq - 1) / (row['W'] * row['h'])
        df.loc[i, 'A_required'] = min(A_req, 50)

valid = df[df['A_required'].notna() & (df['A_required'] < 20)]
print(f"\n  Current A: {A:.3f}")
print(f"  Required A: {valid['A_required'].mean():.3f}")
print(f"  Ratio: {valid['A_required'].mean() / A:.2f}x")

# =============================================================================
# TEST 4: CORRELATION WITH g/g†
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: DOES UNDER-PREDICTION CORRELATE WITH g/g†?")
print("=" * 80)

r, p = stats.pearsonr(df['g_norm'], df['residual'])
print(f"\nCorrelation between g/g† and residual: r = {r:.3f}, p = {p:.4f}")

# Bin by g/g†
print("\nResiduals by g/g† bin:")
for g_min, g_max in [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 20)]:
    mask = (df['g_norm'] >= g_min) & (df['g_norm'] < g_max)
    if mask.sum() > 0:
        subset = df[mask]
        print(f"  g/g† [{g_min:.1f}-{g_max:.1f}]: residual = {subset['residual'].mean():+.1f} km/s, h = {subset['h'].mean():.3f}, N = {mask.sum()}")

# =============================================================================
# TEST 5: DIFFERENT h(g) FOR HIGH-g REGIME
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: ALTERNATIVE h(g) FOR HIGH-g REGIME")
print("=" * 80)

def h_mond_standard(g):
    """MOND standard interpolation"""
    g = np.maximum(np.asarray(g), 1e-15)
    y = g / g_dagger
    nu = 0.5 * (1 + np.sqrt(1 + 4/y))
    return nu - 1

def h_slower_decay(g, power=0.3):
    """Slower decay at high g"""
    g = np.maximum(np.asarray(g), 1e-15)
    return (g_dagger / g)**power * g_dagger / (g_dagger + g)

# Compare h values for inner region data
df['h_mond'] = h_mond_standard(df['g_bar'])
df['h_slow'] = h_slower_decay(df['g_bar'], 0.3)

print("\nComparing h(g) functions for inner regions:")
print(f"  Current h: {df['h'].mean():.3f}")
print(f"  MOND h: {df['h_mond'].mean():.3f} ({df['h_mond'].mean() / df['h'].mean():.2f}x)")
print(f"  Slow-decay h (p=0.3): {df['h_slow'].mean():.3f} ({df['h_slow'].mean() / df['h'].mean():.2f}x)")

# Compute predictions with alternative h
df['V_pred_mond'] = df['V_bar'] * np.sqrt(1 + A * df['W'] * df['h_mond'])
df['V_pred_slow'] = df['V_bar'] * np.sqrt(1 + A * df['W'] * df['h_slow'])

df['residual_mond'] = df['V_obs'] - df['V_pred_mond']
df['residual_slow'] = df['V_obs'] - df['V_pred_slow']

print(f"\nInner region residuals with alternative h(g):")
print(f"  Current: {df['residual'].mean():+.1f} km/s")
print(f"  MOND h: {df['residual_mond'].mean():+.1f} km/s")
print(f"  Slow-decay h: {df['residual_slow'].mean():+.1f} km/s")

# =============================================================================
# TEST 6: HIGHER A FOR BULGE-DOMINATED REGIONS
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: HIGHER A FOR BULGE-DOMINATED INNER REGIONS")
print("=" * 80)

print("\nWhat if A increases where bulge dominates?")
print("A_eff = A × (1 + α × f_bulge)")

for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
    A_eff = A * (1 + alpha * df['f_bulge'])
    V_pred = df['V_bar'] * np.sqrt(1 + A_eff * df['W'] * df['h'])
    residual = df['V_obs'] - V_pred
    print(f"  α = {alpha:.1f}: mean residual = {residual.mean():+.1f} km/s")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
INNER REGION UNDER-PREDICTION DIAGNOSIS:

1. CURRENT SITUATION:
   - Mean g/g† = {df['g_norm'].mean():.2f} (HIGH acceleration regime)
   - Mean W = {df['W'].mean():.3f} (coherence window)
   - Mean h = {df['h'].mean():.3f} (enhancement factor)
   - Mean residual = {df['residual'].mean():+.1f} km/s (UNDER-prediction)

2. REQUIRED CHANGES TO FIX:
   - W needs to be {valid['W_required'].mean() / valid['W'].mean():.1f}x higher
   - OR h needs to be {df[df['h_required'].notna() & (df['h_required'] < 20)]['h_required'].mean() / df['h'].mean():.1f}x higher
   - OR A needs to be {valid['A_required'].mean() / A:.1f}x higher

3. ROOT CAUSE ANALYSIS:
   - High g/g† → low h(g) → insufficient enhancement
   - The h(g) function decays too fast at high accelerations
   - MOND h(g) helps: residual goes from +{df['residual'].mean():.0f} to +{df['residual_mond'].mean():.0f} km/s

4. PHYSICAL INTERPRETATION:
   - Inner bulge regions have HIGH baryonic acceleration
   - But stars still move faster than baryons predict
   - This suggests enhancement should NOT decay as fast at high g
   - OR bulge-dominated regions need higher A

5. IMPLICATIONS FOR CLUSTER vs GALAXY DIFFERENCE:
   - Clusters have A = 8.0, galaxies have A = 1.17
   - Bulge inner regions may need intermediate A
   - This supports the idea that 3D geometry needs different A
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

