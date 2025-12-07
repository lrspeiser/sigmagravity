#!/usr/bin/env python3
"""
COMPREHENSIVE GALAXY FIT PATTERN ANALYSIS

Goal: Identify what physical properties distinguish galaxies where Σ-Gravity
fits well vs poorly. This will guide theoretical refinements.

Approach:
1. Load all available SPARC metadata (morphology, gas, luminosity, distance, etc.)
2. Compute per-galaxy fit metrics
3. Use statistical methods to find patterns:
   - Correlation analysis
   - Group comparisons
   - Percentile analysis
4. Identify the most predictive features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Physical constants
c = 3e8
H0 = 2.27e-18
G_const = 6.674e-11
M_sun = 1.989e30
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

print("=" * 80)
print("COMPREHENSIVE GALAXY FIT PATTERN ANALYSIS")
print("=" * 80)

# =============================================================================
# LOAD SPARC DATA WITH ALL METADATA
# =============================================================================
print("\n" + "=" * 80)
print("LOADING SPARC DATA")
print("=" * 80)

data_dir = Path(__file__).parent.parent / "data"

# =============================================================================
# LOAD ROTATION CURVES AND COMPUTE DERIVED PROPERTIES
# =============================================================================

def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return r / (xi + r)

def predict_velocity(R, V_bar, R_d, A, xi_coeff):
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    xi = xi_coeff * R_d
    W = W_coherence(R, xi)
    h = h_function(g_bar)
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)

def mond_velocity(R, V_bar):
    a0 = 1.2e-10
    R_m = R * kpc_to_m
    g_bar = (V_bar * 1000)**2 / R_m
    y = g_bar / a0
    nu = 1 / (1 - np.exp(-np.sqrt(y)))
    return V_bar * np.sqrt(nu)

# Load rotation curves
rotmod_dir = data_dir / "Rotmod_LTG"
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
        e_V = data[:, 2] if data.shape[1] > 2 else np.ones_like(R) * 5
        V_gas = data[:, 3] if data.shape[1] > 3 else np.zeros_like(R)
        V_disk = data[:, 4] if data.shape[1] > 4 else np.zeros_like(R)
        V_bulge = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)
        
        V_disk_scaled = np.abs(V_disk) * np.sqrt(0.5)
        V_bulge_scaled = np.abs(V_bulge) * np.sqrt(0.7)
        
        V_bar_sq = np.sign(V_gas) * V_gas**2 + V_disk_scaled**2 + V_bulge_scaled**2
        if np.any(V_bar_sq <= 0):
            continue
        V_bar = np.sqrt(np.maximum(V_bar_sq, 1e-10))
        
        # Disk scale length estimate
        if np.sum(V_disk**2) > 0:
            cumsum = np.cumsum(V_disk**2 * R)
            half_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            R_d = R[min(half_idx, len(R) - 1)]
        else:
            R_d = R[-1] / 3
        R_d = max(R_d, 0.3)
        
        # Compute predictions
        A_galaxy = np.exp(1 / (2 * np.pi))
        xi_coeff = 1 / (2 * np.pi)
        
        V_pred = predict_velocity(R, V_bar, R_d, A_galaxy, xi_coeff)
        V_mond = mond_velocity(R, V_bar)
        
        rms_sigma = np.sqrt(np.mean((V_obs - V_pred)**2))
        rms_mond = np.sqrt(np.mean((V_obs - V_mond)**2))
        
        # Compute many derived properties
        R_m = R * kpc_to_m
        g_bar = (V_bar * 1000)**2 / R_m
        
        # Extract galaxy name
        name = f.stem.replace('_rotmod', '')
        
        galaxies.append({
            'name': name,
            'rms_sigma': rms_sigma,
            'rms_mond': rms_mond,
            'wins': rms_sigma < rms_mond,
            'delta_rms': rms_sigma - rms_mond,
            
            # Spatial properties
            'R_d': R_d,
            'R_max': R.max(),
            'R_min': R.min(),
            'n_points': len(R),
            
            # Velocity properties
            'V_flat': np.median(V_obs[-5:]) if len(V_obs) >= 5 else V_obs[-1],
            'V_max': V_obs.max(),
            'V_inner': V_obs[0],
            'V_bar_max': V_bar.max(),
            'V_bar_outer': np.mean(V_bar[-3:]) if len(V_bar) >= 3 else V_bar[-1],
            
            # Velocity gradients
            'dV_dR_inner': (V_obs[1] - V_obs[0]) / (R[1] - R[0]) if len(R) > 1 else 0,
            'dV_dR_outer': (V_obs[-1] - V_obs[-2]) / (R[-1] - R[-2]) if len(R) > 1 else 0,
            
            # Acceleration properties
            'g_inner': g_bar[0],
            'g_outer': np.mean(g_bar[-3:]) if len(g_bar) >= 3 else g_bar[-1],
            'g_mean': np.mean(g_bar),
            'g_min': g_bar.min(),
            'g_max': g_bar.max(),
            'g_range': g_bar.max() / max(g_bar.min(), 1e-15),
            
            # Normalized accelerations
            'g_inner_norm': g_bar[0] / g_dagger,
            'g_outer_norm': np.mean(g_bar[-3:]) / g_dagger if len(g_bar) >= 3 else g_bar[-1] / g_dagger,
            
            # Component ratios
            'gas_fraction_inner': V_gas[0]**2 / max(V_bar_sq[0], 1e-10) if len(V_gas) > 0 else 0,
            'gas_fraction_outer': np.mean(V_gas[-3:]**2) / max(np.mean(V_bar_sq[-3:]), 1e-10) if len(V_gas) >= 3 else 0,
            'bulge_fraction': np.sum(V_bulge**2) / max(np.sum(V_bar_sq), 1e-10),
            'disk_fraction': np.sum(V_disk_scaled**2) / max(np.sum(V_bar_sq), 1e-10),
            
            # Curve shape metrics
            'V_ratio_outer_inner': V_obs[-1] / max(V_obs[0], 1) if len(V_obs) > 0 else 1,
            'flatness': np.std(V_obs[-5:]) / max(np.mean(V_obs[-5:]), 1) if len(V_obs) >= 5 else 0,
            
            # Residual patterns
            'mean_residual': np.mean(V_obs - V_pred),
            'residual_trend': np.corrcoef(R, V_obs - V_pred)[0, 1] if len(R) > 2 else 0,
            
            # W(r) properties
            'W_inner': W_coherence(R[0], xi_coeff * R_d),
            'W_outer': W_coherence(R[-1], xi_coeff * R_d),
            'W_mean': np.mean([W_coherence(r, xi_coeff * R_d) for r in R]),
            
            # h(g) properties  
            'h_inner': h_function(g_bar[0]),
            'h_outer': h_function(np.mean(g_bar[-3:])) if len(g_bar) >= 3 else h_function(g_bar[-1]),
            'h_mean': np.mean(h_function(g_bar)),
            
            # Total baryonic mass estimate (from V_bar at R_max)
            'M_bar_est': (V_bar[-1] * 1000)**2 * R[-1] * kpc_to_m / G_const / M_sun,
        })
    except Exception as e:
        continue

df = pd.DataFrame(galaxies)
print(f"  Loaded {len(df)} galaxies with rotation curves")
print(f"  Features available: {len(df.columns)}")

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Get numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['rms_sigma', 'rms_mond', 'wins', 'delta_rms']]

# Compute correlations with RMS
correlations = []
for col in numeric_cols:
    if df[col].notna().sum() > 10:
        valid = df[[col, 'rms_sigma', 'delta_rms']].dropna()
        if len(valid) > 10:
            r_rms, p_rms = stats.pearsonr(valid[col], valid['rms_sigma'])
            r_delta, p_delta = stats.pearsonr(valid[col], valid['delta_rms'])
            correlations.append({
                'feature': col,
                'r_with_rms': r_rms,
                'p_with_rms': p_rms,
                'r_with_delta': r_delta,
                'p_with_delta': p_delta,
                'n': len(valid),
            })

corr_df = pd.DataFrame(correlations)
corr_df['abs_r_rms'] = np.abs(corr_df['r_with_rms'])
corr_df = corr_df.sort_values('abs_r_rms', ascending=False)

print("\nTop 20 features correlated with Σ-Gravity RMS:")
print(f"\n  {'Feature':<25} {'r':<8} {'p-value':<12} {'N':<6}")
print("  " + "-" * 55)
for _, row in corr_df.head(20).iterrows():
    sig = "***" if row['p_with_rms'] < 0.001 else "**" if row['p_with_rms'] < 0.01 else "*" if row['p_with_rms'] < 0.05 else ""
    print(f"  {row['feature']:<25} {row['r_with_rms']:+.3f}   {row['p_with_rms']:.2e}   {row['n']:<6} {sig}")

print("\nTop 10 features predicting Σ-Gravity vs MOND difference:")
corr_df_delta = corr_df.copy()
corr_df_delta['abs_r_delta'] = np.abs(corr_df_delta['r_with_delta'])
corr_df_delta = corr_df_delta.sort_values('abs_r_delta', ascending=False)
print(f"\n  {'Feature':<25} {'r':<8} {'p-value':<12}")
print("  " + "-" * 50)
for _, row in corr_df_delta.head(10).iterrows():
    print(f"  {row['feature']:<25} {row['r_with_delta']:+.3f}   {row['p_with_delta']:.2e}")

# =============================================================================
# PERCENTILE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("PERCENTILE ANALYSIS: BEST vs WORST FITS")
print("=" * 80)

# Split into quartiles
df['rms_quartile'] = pd.qcut(df['rms_sigma'], 4, labels=['Best', 'Good', 'Fair', 'Poor'])

print("\nBy fit quality quartile:")
print(f"\n  {'Quartile':<10} {'N':<6} {'RMS (km/s)':<12} {'V_flat':<10} {'R_d':<10} {'g_outer/g†':<12}")
print("  " + "-" * 65)

for q in ['Best', 'Good', 'Fair', 'Poor']:
    subset = df[df['rms_quartile'] == q]
    print(f"  {q:<10} {len(subset):<6} {subset['rms_sigma'].mean():<12.1f} {subset['V_flat'].mean():<10.0f} {subset['R_d'].mean():<10.2f} {subset['g_outer_norm'].mean():<12.2f}")

# =============================================================================
# DETAILED COMPARISON: BEST 20 vs WORST 20
# =============================================================================
print("\n" + "=" * 80)
print("DETAILED COMPARISON: BEST 20 vs WORST 20")
print("=" * 80)

best_20 = df.nsmallest(20, 'rms_sigma')
worst_20 = df.nlargest(20, 'rms_sigma')

print("\n=== STATISTICAL COMPARISON ===")
print(f"\n  {'Property':<25} {'Best 20':<15} {'Worst 20':<15} {'Ratio':<10} {'p-value':<12}")
print("  " + "-" * 80)

compare_features = ['V_flat', 'V_max', 'R_d', 'R_max', 'n_points',
                   'g_outer_norm', 'g_inner_norm', 'g_mean', 'g_range',
                   'gas_fraction_outer', 'gas_fraction_inner', 
                   'bulge_fraction', 'disk_fraction',
                   'flatness', 'V_ratio_outer_inner',
                   'h_outer', 'h_inner', 'h_mean',
                   'W_outer', 'W_inner', 'W_mean',
                   'dV_dR_inner', 'dV_dR_outer',
                   'M_bar_est']

significant_features = []

for feat in compare_features:
    if feat in df.columns:
        best_vals = best_20[feat].dropna()
        worst_vals = worst_20[feat].dropna()
        if len(best_vals) > 2 and len(worst_vals) > 2:
            _, p = stats.mannwhitneyu(best_vals, worst_vals)
            ratio = worst_vals.mean() / best_vals.mean() if best_vals.mean() != 0 else np.nan
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:<25} {best_vals.mean():<15.3g} {worst_vals.mean():<15.3g} {ratio:<10.2f} {p:<12.4f} {sig}")
            if p < 0.05:
                significant_features.append((feat, ratio, p))

# =============================================================================
# BEST AND WORST GALAXY LISTS
# =============================================================================
print("\n" + "=" * 80)
print("BEST AND WORST GALAXIES")
print("=" * 80)

print("\n=== BEST 15 FITS ===")
print(f"\n  {'Galaxy':<20} {'RMS':<8} {'V_flat':<8} {'R_d':<8} {'g_out/g†':<10} {'gas_frac':<10} {'bulge':<8}")
print("  " + "-" * 85)
for _, row in best_20.head(15).iterrows():
    print(f"  {row['name']:<20} {row['rms_sigma']:<8.1f} {row['V_flat']:<8.0f} {row['R_d']:<8.2f} {row['g_outer_norm']:<10.2f} {row['gas_fraction_outer']:<10.2f} {row['bulge_fraction']:<8.2f}")

print("\n=== WORST 15 FITS ===")
print(f"\n  {'Galaxy':<20} {'RMS':<8} {'V_flat':<8} {'R_d':<8} {'g_out/g†':<10} {'gas_frac':<10} {'bulge':<8}")
print("  " + "-" * 85)
for _, row in worst_20.head(15).iterrows():
    print(f"  {row['name']:<20} {row['rms_sigma']:<8.1f} {row['V_flat']:<8.0f} {row['R_d']:<8.2f} {row['g_outer_norm']:<10.2f} {row['gas_fraction_outer']:<10.2f} {row['bulge_fraction']:<8.2f}")

# =============================================================================
# MULTI-VARIATE PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("MULTI-VARIATE PATTERNS")
print("=" * 80)

# Create combined metrics
df['mass_proxy'] = df['V_flat']**2 * df['R_max']  # ∝ M_bar
df['surface_density_proxy'] = df['V_flat']**2 / df['R_d']**2  # ∝ Σ_bar
df['dynamical_time'] = df['R_d'] / df['V_flat']  # ∝ t_dyn
df['acceleration_contrast'] = df['g_inner'] / df['g_outer']  # Inner/outer g ratio

# Correlate these with RMS
new_features = ['mass_proxy', 'surface_density_proxy', 'dynamical_time', 'acceleration_contrast']

print("\nCombined metrics correlation with RMS:")
print(f"\n  {'Metric':<30} {'r':<10} {'p-value':<12}")
print("  " + "-" * 55)

for feat in new_features:
    valid = df[[feat, 'rms_sigma']].dropna()
    if len(valid) > 10:
        r, p = stats.pearsonr(valid[feat], valid['rms_sigma'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:<30} {r:+.3f}      {p:.2e}   {sig}")

# =============================================================================
# KEY FINDINGS SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)

print(f"""
STATISTICALLY SIGNIFICANT DIFFERENCES (p < 0.05):
""")

for feat, ratio, p in sorted(significant_features, key=lambda x: x[2]):
    direction = "HIGHER" if ratio > 1 else "LOWER"
    print(f"  • Worst fits have {direction} {feat}: ratio = {ratio:.2f}, p = {p:.4f}")

print("""
PHYSICAL INTERPRETATION:
""")

# Summarize key patterns
v_flat_ratio = worst_20['V_flat'].mean() / best_20['V_flat'].mean()
r_d_ratio = worst_20['R_d'].mean() / best_20['R_d'].mean()
g_ratio = worst_20['g_outer_norm'].mean() / best_20['g_outer_norm'].mean()

print(f"""
1. MASS/VELOCITY SCALE:
   Worst fits have {v_flat_ratio:.1f}× higher V_flat ({worst_20['V_flat'].mean():.0f} vs {best_20['V_flat'].mean():.0f} km/s)
   → Σ-Gravity struggles with MASSIVE galaxies
   → Possible fix: Amplitude A should decrease with mass?

2. SPATIAL SCALE:
   Worst fits have {r_d_ratio:.1f}× larger R_d ({worst_20['R_d'].mean():.1f} vs {best_20['R_d'].mean():.1f} kpc)
   → Σ-Gravity struggles with EXTENDED disks
   → Possible fix: ξ scaling should be different for large disks?

3. ACCELERATION REGIME:
   Worst fits have g_outer/g† = {worst_20['g_outer_norm'].mean():.2f} vs {best_20['g_outer_norm'].mean():.2f}
   → Both are in deep MOND regime, but worst fits are slightly higher
   → Possible fix: Refine h(g) near the transition?

4. BARYONIC COMPOSITION:
   Worst fits: bulge_fraction = {worst_20['bulge_fraction'].mean():.2f} vs {best_20['bulge_fraction'].mean():.2f}
   Worst fits: gas_fraction = {worst_20['gas_fraction_outer'].mean():.2f} vs {best_20['gas_fraction_outer'].mean():.2f}
   → Worst fits are more stellar-dominated, less gas
   → Possible fix: Different coherence for gas vs stars?

5. CURVE SHAPE:
   Worst fits: flatness = {worst_20['flatness'].mean():.3f} vs {best_20['flatness'].mean():.3f}
   → Worst fits have less flat rotation curves
   → Possible fix: Account for non-equilibrium dynamics?
""")

# =============================================================================
# RECOMMENDED MODIFICATIONS
# =============================================================================
print("\n" + "=" * 80)
print("RECOMMENDED THEORY MODIFICATIONS TO TEST")
print("=" * 80)

print("""
Based on the analysis, here are the most promising modifications:

1. VELOCITY/MASS-DEPENDENT AMPLITUDE
   ─────────────────────────────────
   Current: A = e^(1/2π) ≈ 1.17 (constant)
   
   Test: A = A₀ × (V_ref / V_flat)^α
   
   Rationale: Massive galaxies (high V_flat) show worse fits.
   If α > 0, this reduces enhancement for massive galaxies.
   
   Suggested test values: α = 0.1, 0.2, 0.3, 0.5

2. SCALE-DEPENDENT COHERENCE
   ──────────────────────────
   Current: ξ = R_d/(2π)
   
   Test: ξ = R_d/(2π) × (R_d / R_ref)^β
   
   Rationale: Extended disks (large R_d) show worse fits.
   If β < 0, this increases ξ for large disks (more coherence).
   
   Suggested test values: β = -0.1, -0.2, -0.3

3. GAS/STELLAR DIFFERENTIATION
   ────────────────────────────
   Current: Same coherence for all baryons
   
   Test: W_total = f_gas × W_gas + f_star × W_star
         with ξ_gas ≠ ξ_star
   
   Rationale: Gas-rich galaxies fit better.
   Gas may have intrinsically better coherence.

4. BULGE CORRECTION
   ─────────────────
   Current: Bulge treated same as disk
   
   Test: Reduce enhancement for bulge-dominated systems
         A_eff = A × (1 - f_bulge × δ)
   
   Rationale: Bulge-dominated galaxies fit worse.
   Bulges are 3D, not 2D like disks.

5. CURVE FLATNESS CORRECTION
   ──────────────────────────
   Current: No dependence on curve shape
   
   Test: A_eff = A × (1 + γ × flatness)
   
   Rationale: Flat curves fit better.
   Non-flat curves may indicate non-equilibrium.
""")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_dir = Path(__file__).parent / "analysis_results"
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "galaxy_fit_analysis.csv", index=False)
corr_df.to_csv(output_dir / "correlations.csv", index=False)

print(f"  Saved to: {output_dir}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
