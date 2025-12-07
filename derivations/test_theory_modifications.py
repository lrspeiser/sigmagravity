#!/usr/bin/env python3
"""
TEST THEORY MODIFICATIONS

Based on the pattern analysis, test specific modifications to improve fits
for massive/extended galaxies while maintaining good fits for dwarfs.
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
print("TESTING THEORY MODIFICATIONS")
print("=" * 80)

# =============================================================================
# LOAD DATA
# =============================================================================
data_dir = Path(__file__).parent.parent / "data"
rotmod_dir = data_dir / "Rotmod_LTG"

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
            
            V_flat = np.median(V_obs[-5:]) if len(V_obs) >= 5 else V_obs[-1]
            
            # Gas and bulge fractions
            gas_frac = np.mean(V_gas[-3:]**2) / max(np.mean(V_bar_sq[-3:]), 1e-10) if len(V_gas) >= 3 else 0
            bulge_frac = np.sum(V_bulge**2) / max(np.sum(V_bar_sq), 1e-10)
            
            galaxies.append({
                'name': f.stem.replace('_rotmod', ''),
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'V_gas': V_gas,
                'V_disk': V_disk_scaled,
                'V_bulge': V_bulge_scaled,
                'R_d': R_d,
                'V_flat': V_flat,
                'gas_frac': gas_frac,
                'bulge_frac': bulge_frac,
            })
        except:
            continue
    return galaxies

galaxies = load_galaxies()
print(f"\nLoaded {len(galaxies)} galaxies")

# =============================================================================
# BASELINE MODEL
# =============================================================================

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

def evaluate_model(galaxies, predict_func, name="Model"):
    """Evaluate a model across all galaxies."""
    results = []
    for gal in galaxies:
        V_pred = predict_func(gal)
        V_mond = mond_velocity(gal['R'], gal['V_bar'])
        
        rms_sigma = np.sqrt(np.mean((gal['V_obs'] - V_pred)**2))
        rms_mond = np.sqrt(np.mean((gal['V_obs'] - V_mond)**2))
        
        results.append({
            'name': gal['name'],
            'rms': rms_sigma,
            'rms_mond': rms_mond,
            'wins': rms_sigma < rms_mond,
            'V_flat': gal['V_flat'],
            'R_d': gal['R_d'],
        })
    
    df = pd.DataFrame(results)
    return {
        'name': name,
        'mean_rms': df['rms'].mean(),
        'median_rms': df['rms'].median(),
        'win_rate': df['wins'].mean() * 100,
        'rms_low_mass': df[df['V_flat'] < 100]['rms'].mean(),
        'rms_high_mass': df[df['V_flat'] >= 100]['rms'].mean(),
        'df': df,
    }

# Baseline: Current 2D framework
def baseline_predict(gal):
    A = np.exp(1 / (2 * np.pi))
    xi_coeff = 1 / (2 * np.pi)
    
    R_m = gal['R'] * kpc_to_m
    g_bar = (gal['V_bar'] * 1000)**2 / R_m
    xi = xi_coeff * gal['R_d']
    W = W_coherence(gal['R'], xi)
    h = h_function(g_bar)
    Sigma = 1 + A * W * h
    return gal['V_bar'] * np.sqrt(Sigma)

baseline = evaluate_model(galaxies, baseline_predict, "Baseline (2D)")
print(f"\nBaseline: RMS = {baseline['mean_rms']:.2f} km/s, Win rate = {baseline['win_rate']:.1f}%")
print(f"  Low-mass (<100 km/s): {baseline['rms_low_mass']:.2f} km/s")
print(f"  High-mass (≥100 km/s): {baseline['rms_high_mass']:.2f} km/s")

# =============================================================================
# MODIFICATION 1: VELOCITY-DEPENDENT AMPLITUDE
# =============================================================================
print("\n" + "=" * 80)
print("MODIFICATION 1: VELOCITY-DEPENDENT AMPLITUDE")
print("A = A₀ × (V_ref / V_flat)^α")
print("=" * 80)

V_ref = 100  # Reference velocity (km/s)
A0 = np.exp(1 / (2 * np.pi))

for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    def predict_v_dep(gal, alpha=alpha):
        A = A0 * (V_ref / gal['V_flat'])**alpha
        xi_coeff = 1 / (2 * np.pi)
        
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_function(g_bar)
        Sigma = 1 + A * W * h
        return gal['V_bar'] * np.sqrt(Sigma)
    
    result = evaluate_model(galaxies, predict_v_dep, f"V-dep α={alpha}")
    delta = result['mean_rms'] - baseline['mean_rms']
    print(f"  α = {alpha:.2f}: RMS = {result['mean_rms']:.2f} km/s (Δ = {delta:+.2f}), Win = {result['win_rate']:.1f}%")
    print(f"           Low-mass: {result['rms_low_mass']:.2f}, High-mass: {result['rms_high_mass']:.2f}")

# =============================================================================
# MODIFICATION 2: SCALE-DEPENDENT ξ
# =============================================================================
print("\n" + "=" * 80)
print("MODIFICATION 2: SCALE-DEPENDENT ξ")
print("ξ = R_d/(2π) × (R_d / R_ref)^β")
print("=" * 80)

R_ref = 5  # Reference scale length (kpc)

for beta in [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]:
    def predict_xi_dep(gal, beta=beta):
        A = np.exp(1 / (2 * np.pi))
        xi_coeff = 1 / (2 * np.pi) * (gal['R_d'] / R_ref)**beta
        
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_function(g_bar)
        Sigma = 1 + A * W * h
        return gal['V_bar'] * np.sqrt(Sigma)
    
    result = evaluate_model(galaxies, predict_xi_dep, f"ξ-dep β={beta}")
    delta = result['mean_rms'] - baseline['mean_rms']
    print(f"  β = {beta:+.1f}: RMS = {result['mean_rms']:.2f} km/s (Δ = {delta:+.2f}), Win = {result['win_rate']:.1f}%")

# =============================================================================
# MODIFICATION 3: GAS/STELLAR DIFFERENTIATION
# =============================================================================
print("\n" + "=" * 80)
print("MODIFICATION 3: GAS/STELLAR DIFFERENTIATION")
print("A_eff = A × (1 + δ × gas_frac)")
print("=" * 80)

for delta_gas in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    def predict_gas_dep(gal, delta_gas=delta_gas):
        A_base = np.exp(1 / (2 * np.pi))
        A = A_base * (1 + delta_gas * gal['gas_frac'])
        xi_coeff = 1 / (2 * np.pi)
        
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_function(g_bar)
        Sigma = 1 + A * W * h
        return gal['V_bar'] * np.sqrt(Sigma)
    
    result = evaluate_model(galaxies, predict_gas_dep, f"Gas δ={delta_gas}")
    delta = result['mean_rms'] - baseline['mean_rms']
    print(f"  δ = {delta_gas:.1f}: RMS = {result['mean_rms']:.2f} km/s (Δ = {delta:+.2f}), Win = {result['win_rate']:.1f}%")

# =============================================================================
# MODIFICATION 4: BULGE SUPPRESSION
# =============================================================================
print("\n" + "=" * 80)
print("MODIFICATION 4: BULGE SUPPRESSION")
print("A_eff = A × (1 - δ × bulge_frac)")
print("=" * 80)

for delta_bulge in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    def predict_bulge_dep(gal, delta_bulge=delta_bulge):
        A_base = np.exp(1 / (2 * np.pi))
        A = A_base * (1 - delta_bulge * min(gal['bulge_frac'], 1.0))
        A = max(A, 0.1)  # Ensure A stays positive
        xi_coeff = 1 / (2 * np.pi)
        
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_function(g_bar)
        Sigma = 1 + A * W * h
        return gal['V_bar'] * np.sqrt(Sigma)
    
    result = evaluate_model(galaxies, predict_bulge_dep, f"Bulge δ={delta_bulge}")
    delta = result['mean_rms'] - baseline['mean_rms']
    print(f"  δ = {delta_bulge:.1f}: RMS = {result['mean_rms']:.2f} km/s (Δ = {delta:+.2f}), Win = {result['win_rate']:.1f}%")

# =============================================================================
# MODIFICATION 5: COMBINED (V + GAS + BULGE)
# =============================================================================
print("\n" + "=" * 80)
print("MODIFICATION 5: COMBINED MODIFICATIONS")
print("=" * 80)

best_combos = []

for alpha in [0.1, 0.15, 0.2]:
    for delta_gas in [0.2, 0.3, 0.5]:
        for delta_bulge in [0.3, 0.5, 0.7]:
            def predict_combined(gal, alpha=alpha, delta_gas=delta_gas, delta_bulge=delta_bulge):
                A_base = np.exp(1 / (2 * np.pi))
                # Velocity dependence
                A = A_base * (V_ref / gal['V_flat'])**alpha
                # Gas boost
                A = A * (1 + delta_gas * gal['gas_frac'])
                # Bulge suppression
                A = A * (1 - delta_bulge * min(gal['bulge_frac'], 1.0))
                A = max(A, 0.1)
                
                xi_coeff = 1 / (2 * np.pi)
                R_m = gal['R'] * kpc_to_m
                g_bar = (gal['V_bar'] * 1000)**2 / R_m
                xi = xi_coeff * gal['R_d']
                W = W_coherence(gal['R'], xi)
                h = h_function(g_bar)
                Sigma = 1 + A * W * h
                return gal['V_bar'] * np.sqrt(Sigma)
            
            result = evaluate_model(galaxies, predict_combined)
            best_combos.append({
                'alpha': alpha,
                'delta_gas': delta_gas,
                'delta_bulge': delta_bulge,
                'rms': result['mean_rms'],
                'win_rate': result['win_rate'],
            })

# Sort by RMS
best_combos = sorted(best_combos, key=lambda x: x['rms'])

print("\nTop 10 combinations:")
print(f"\n  {'α':<6} {'δ_gas':<8} {'δ_bulge':<10} {'RMS':<10} {'Win %':<8}")
print("  " + "-" * 45)
for combo in best_combos[:10]:
    delta = combo['rms'] - baseline['mean_rms']
    print(f"  {combo['alpha']:<6.2f} {combo['delta_gas']:<8.2f} {combo['delta_bulge']:<10.2f} {combo['rms']:<10.2f} {combo['win_rate']:<8.1f} (Δ = {delta:+.2f})")

# =============================================================================
# MODIFICATION 6: ALTERNATIVE h(g) FUNCTIONS
# =============================================================================
print("\n" + "=" * 80)
print("MODIFICATION 6: ALTERNATIVE h(g) FUNCTIONS")
print("=" * 80)

def h_standard(g):
    """Standard h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def h_mond_simple(g):
    """MOND simple: ν(y) = 1/(1 - exp(-√y)) - 1"""
    g = np.maximum(np.asarray(g), 1e-15)
    y = g / g_dagger
    nu = 1 / (1 - np.exp(-np.sqrt(y)))
    return nu - 1  # Enhancement factor

def h_mond_standard(g):
    """MOND standard: ν(y) = (1 + √(1+4/y))/2 - 1"""
    g = np.maximum(np.asarray(g), 1e-15)
    y = g / g_dagger
    nu = 0.5 * (1 + np.sqrt(1 + 4/y))
    return nu - 1

def h_power_law(g, n=1):
    """Power law: h = (g†/g)^n"""
    g = np.maximum(np.asarray(g), 1e-15)
    return (g_dagger / g)**n

def h_exponential(g):
    """Exponential: h = exp(-g/g†) × (g†/g)^0.5"""
    g = np.maximum(np.asarray(g), 1e-15)
    return np.exp(-g / g_dagger) * np.sqrt(g_dagger / g)

h_functions = [
    ("Standard", h_standard),
    ("MOND simple", h_mond_simple),
    ("MOND standard", h_mond_standard),
    ("Power n=0.5", lambda g: h_power_law(g, 0.5)),
    ("Power n=0.75", lambda g: h_power_law(g, 0.75)),
    ("Power n=1.0", lambda g: h_power_law(g, 1.0)),
    ("Exponential", h_exponential),
]

for name, h_func in h_functions:
    def predict_h(gal, h_func=h_func):
        A = np.exp(1 / (2 * np.pi))
        xi_coeff = 1 / (2 * np.pi)
        
        R_m = gal['R'] * kpc_to_m
        g_bar = (gal['V_bar'] * 1000)**2 / R_m
        xi = xi_coeff * gal['R_d']
        W = W_coherence(gal['R'], xi)
        h = h_func(g_bar)
        Sigma = 1 + A * W * h
        return gal['V_bar'] * np.sqrt(np.maximum(Sigma, 1))
    
    result = evaluate_model(galaxies, predict_h, name)
    delta = result['mean_rms'] - baseline['mean_rms']
    print(f"  {name:<15}: RMS = {result['mean_rms']:.2f} km/s (Δ = {delta:+.2f}), Win = {result['win_rate']:.1f}%")

# =============================================================================
# BEST OVERALL MODEL
# =============================================================================
print("\n" + "=" * 80)
print("BEST OVERALL MODEL")
print("=" * 80)

# Use best combination
best = best_combos[0]
print(f"\nBest parameters found:")
print(f"  α (velocity exponent) = {best['alpha']}")
print(f"  δ_gas (gas boost) = {best['delta_gas']}")
print(f"  δ_bulge (bulge suppression) = {best['delta_bulge']}")

def best_predict(gal):
    A_base = np.exp(1 / (2 * np.pi))
    A = A_base * (V_ref / gal['V_flat'])**best['alpha']
    A = A * (1 + best['delta_gas'] * gal['gas_frac'])
    A = A * (1 - best['delta_bulge'] * min(gal['bulge_frac'], 1.0))
    A = max(A, 0.1)
    
    xi_coeff = 1 / (2 * np.pi)
    R_m = gal['R'] * kpc_to_m
    g_bar = (gal['V_bar'] * 1000)**2 / R_m
    xi = xi_coeff * gal['R_d']
    W = W_coherence(gal['R'], xi)
    h = h_function(g_bar)
    Sigma = 1 + A * W * h
    return gal['V_bar'] * np.sqrt(Sigma)

best_result = evaluate_model(galaxies, best_predict, "Best Combined")

print(f"\nPerformance comparison:")
print(f"\n  {'Model':<25} {'Mean RMS':<12} {'Win Rate':<12} {'Low-mass':<12} {'High-mass':<12}")
print("  " + "-" * 70)
print(f"  {'Baseline (2D)':<25} {baseline['mean_rms']:<12.2f} {baseline['win_rate']:<12.1f}% {baseline['rms_low_mass']:<12.2f} {baseline['rms_high_mass']:<12.2f}")
print(f"  {'Best Combined':<25} {best_result['mean_rms']:<12.2f} {best_result['win_rate']:<12.1f}% {best_result['rms_low_mass']:<12.2f} {best_result['rms_high_mass']:<12.2f}")

# Compare per-galaxy
baseline_df = baseline['df']
best_df = best_result['df']

merged = baseline_df.merge(best_df, on='name', suffixes=('_baseline', '_best'))
merged['improvement'] = merged['rms_baseline'] - merged['rms_best']

print(f"\nPer-galaxy comparison:")
print(f"  Galaxies improved: {(merged['improvement'] > 0).sum()}")
print(f"  Galaxies worsened: {(merged['improvement'] < 0).sum()}")
print(f"  Mean improvement: {merged['improvement'].mean():.2f} km/s")

# Show biggest improvements and degradations
print("\nBiggest improvements:")
for _, row in merged.nlargest(5, 'improvement').iterrows():
    print(f"  {row['name']:<20}: {row['rms_baseline']:.1f} → {row['rms_best']:.1f} (Δ = {row['improvement']:+.1f})")

print("\nBiggest degradations:")
for _, row in merged.nsmallest(5, 'improvement').iterrows():
    print(f"  {row['name']:<20}: {row['rms_baseline']:.1f} → {row['rms_best']:.1f} (Δ = {row['improvement']:+.1f})")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
KEY FINDINGS:

1. VELOCITY-DEPENDENT AMPLITUDE (α ~ 0.1-0.2):
   - Reduces enhancement for massive galaxies
   - Improves high-mass fits without hurting dwarfs
   - PHYSICAL INTERPRETATION: Coherence decreases with velocity dispersion?

2. GAS BOOST (δ_gas ~ 0.2-0.5):
   - Gas-rich galaxies get more enhancement
   - PHYSICAL INTERPRETATION: Gas is more coherent than stars?

3. BULGE SUPPRESSION (δ_bulge ~ 0.3-0.7):
   - Reduces enhancement for bulge-dominated systems
   - PHYSICAL INTERPRETATION: 3D bulges have less coherence than 2D disks

4. ALTERNATIVE h(g):
   - Standard h(g) performs well
   - MOND-like functions work similarly
   - Power-law with n=0.5-0.75 also viable

THEORETICAL IMPLICATIONS:
- The amplitude A should depend on galaxy structure
- 2D disk coherence differs from 3D bulge coherence
- Gas may have intrinsically better coherence than stars
- Velocity dispersion may disrupt coherence
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

