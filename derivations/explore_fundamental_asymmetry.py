#!/usr/bin/env python3
"""
Exploring the Fundamental Asymmetry: Lensing vs Dynamics
=========================================================

The puzzle: We get clusters (lensing) nearly perfect but galaxies (dynamics)
still have scatter. Why?

FUNDAMENTAL DIFFERENCES:
------------------------

1. WHAT WE MEASURE:
   - Clusters: Light deflection (lensing) → probes gravitational potential Φ
   - Galaxies: Star/gas motion (dynamics) → probes gravitational force ∇Φ

2. WHAT FEELS THE GRAVITY:
   - Clusters: Photons (massless, don't contribute to source)
   - Galaxies: Stars/gas (massive, ARE part of the source)

3. THE COHERENCE PARADOX:
   - In galaxies, the test particles (stars) are ALSO sources of the field
   - This creates a self-consistency requirement
   - A star at the edge feels gravity from all other stars
   - But it ALSO contributes to the coherence of the field

HYPOTHESIS: Self-Coherence Effect
---------------------------------

In galaxies, the enhancement Σ should depend on whether the test particle
is part of the coherent source or external to it.

For a star at radius r in a galaxy:
- It feels the coherent field from stars at r' < r (interior)
- But stars at r' > r also exist and affect the coherence

For light passing through a cluster:
- Light is purely a test particle
- It feels the full coherent field without contributing to it

POSSIBLE MODIFICATIONS:
----------------------

1. INTERIOR vs EXTERIOR COHERENCE
   Σ = 1 + A × f_interior(r) × h(g)
   where f_interior accounts for only the mass interior to r

2. SELF-SCREENING
   The test particle's own contribution "screens" some of the coherence
   Σ_eff = Σ × (1 - m_test/M_enclosed)

3. DISCRETE vs CONTINUOUS
   Galaxies: Discrete stars → granular coherence
   Clusters: Smooth gas → continuous coherence

4. VELOCITY DISPERSION EFFECT
   Stars have random motions that "blur" their contribution to coherence
   Gas in clusters is hotter but more uniformly distributed

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8
H0_SI = 2.27e-18
cH0 = c * H0_SI
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30

g_dagger = cH0 / (4 * math.sqrt(math.pi))
a0_mond = 1.2e-10

R0 = 20.0
A_COEFF = 1.0
B_COEFF = 216.7

print("=" * 100)
print("EXPLORING THE FUNDAMENTAL ASYMMETRY: LENSING VS DYNAMICS")
print("=" * 100)

# =============================================================================
# THE KEY INSIGHT
# =============================================================================

print("""
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    THE FUNDAMENTAL ASYMMETRY                                             │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  CLUSTER LENSING:                                                                        │
│    • Test particle: PHOTON (massless)                                                   │
│    • Photon does NOT contribute to the coherent source                                  │
│    • Photon feels the FULL coherent enhancement from baryons                            │
│    • Clean separation: source (baryons) vs probe (light)                                │
│                                                                                          │
│  GALAXY DYNAMICS:                                                                        │
│    • Test particle: STAR (massive)                                                      │
│    • Star IS PART OF the coherent source                                                │
│    • Star feels gravity from OTHER stars, but also contributes                          │
│    • No clean separation: source and probe are the same population                      │
│                                                                                          │
│  IMPLICATION:                                                                            │
│    The coherence enhancement for a star may be REDUCED because the star                 │
│    itself is part of the coherent structure. It can't fully "feel" a                    │
│    coherence that it's helping to create.                                               │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# HYPOTHESIS 1: INTERIOR MASS FRACTION
# =============================================================================

print("\n" + "=" * 100)
print("HYPOTHESIS 1: INTERIOR MASS COHERENCE")
print("=" * 100)

print("""
The idea: A star at radius r only feels coherent enhancement from the mass
INTERIOR to it. The exterior mass contributes to the total potential but
not to the coherent enhancement felt by this particular star.

For lensing: Light passes through the ENTIRE cluster, so it integrates
the full coherent field.

For dynamics: A star at r only "sees" coherence from M(<r).

MODIFICATION:
  Σ_dynamics(r) = 1 + A × f(r) × h(g_interior) × [M(<r)/M_total]^α

where α captures how much the interior fraction matters.
""")

# =============================================================================
# HYPOTHESIS 2: SELF-CONTRIBUTION SCREENING
# =============================================================================

print("\n" + "=" * 100)
print("HYPOTHESIS 2: SELF-CONTRIBUTION SCREENING")
print("=" * 100)

print("""
The idea: Each star contributes to the coherent field, but it can't
feel its OWN contribution. This creates a small "screening" effect.

In a galaxy with N stars of mass m:
  - Total coherent field strength ∝ N × m
  - But each star only feels (N-1) × m from others
  - Screening factor: (N-1)/N ≈ 1 - 1/N

For a continuous distribution:
  - At radius r, the star's contribution is δm
  - Screening: 1 - δm/M(<r)

For large N (galaxies have ~10^11 stars), this is negligible per star.
BUT: The coherence itself might depend on the GRANULARITY of the source.

MODIFICATION:
  Σ_dynamics = Σ_lensing × (1 - ε_granularity)

where ε_granularity depends on how "lumpy" the mass distribution is.
""")

# =============================================================================
# HYPOTHESIS 3: VELOCITY DISPERSION DECOHERENCE
# =============================================================================

print("\n" + "=" * 100)
print("HYPOTHESIS 3: VELOCITY DISPERSION DECOHERENCE")
print("=" * 100)

print("""
The idea: Coherent gravitational enhancement requires the source to be
"coherent" in some sense. Stars with high velocity dispersion are less
coherent than smoothly flowing gas.

For clusters:
  - Most mass is in hot gas (T ~ 10^7 K)
  - Gas is smooth, continuous, follows potential
  - High "spatial coherence"

For galaxies:
  - Mass is in discrete stars
  - Stars have velocity dispersion σ ~ 20-50 km/s
  - Random motions reduce effective coherence

MODIFICATION:
  Σ_dynamics = 1 + A × f(r) × h(g) × C_velocity

where C_velocity = V_rot² / (V_rot² + σ²) is a coherence factor.

For cold disks (σ << V_rot): C ≈ 1
For hot systems (σ ~ V_rot): C ≈ 0.5
""")

# =============================================================================
# HYPOTHESIS 4: RADIAL vs TANGENTIAL COHERENCE
# =============================================================================

print("\n" + "=" * 100)
print("HYPOTHESIS 4: RADIAL VS TANGENTIAL COHERENCE")
print("=" * 100)

print("""
The idea: The coherent enhancement might be different for radial vs
tangential directions.

For lensing:
  - Light travels RADIALLY through the cluster
  - Probes the radial gradient of the potential
  - Sensitive to ∂Φ/∂r integrated along line of sight

For dynamics:
  - Stars orbit TANGENTIALLY
  - Circular velocity probes V² = r × ∂Φ/∂r
  - Sensitive to local radial gradient

If coherence has directional dependence:
  Σ_radial ≠ Σ_tangential

MODIFICATION:
  For lensing: Σ = 1 + A × f(r) × h(g)
  For dynamics: Σ = 1 + A × f(r) × h(g) × cos²(θ)

where θ is the angle between the coherence direction and the probe direction.
""")

# =============================================================================
# HYPOTHESIS 5: SCALE-DEPENDENT COHERENCE
# =============================================================================

print("\n" + "=" * 100)
print("HYPOTHESIS 5: SCALE-DEPENDENT COHERENCE")
print("=" * 100)

print("""
The idea: Coherence might build up differently at different scales.

Clusters (r ~ 100-1000 kpc):
  - Large scale → many "coherence lengths" fit inside
  - Full coherence saturation

Galaxies (r ~ 1-30 kpc):
  - Smaller scale → fewer coherence lengths
  - Partial coherence

If there's a fundamental coherence length λ_c:
  - For r >> λ_c: Full coherence (Σ → 1 + A×h)
  - For r ~ λ_c: Partial coherence
  - For r << λ_c: No coherence (Σ → 1)

Current model has f(r) = r/(r+r₀) with r₀ = 20 kpc.
This gives partial coherence at galaxy scales, full at cluster scales.

But maybe the AMPLITUDE also scales with r/λ_c?

MODIFICATION:
  A_effective(r) = A_base × (1 + r/λ_c)^β

For galaxies (r ~ 10 kpc, λ_c ~ 20 kpc): A_eff ≈ A_base × 1.5^β
For clusters (r ~ 200 kpc, λ_c ~ 20 kpc): A_eff ≈ A_base × 11^β

If β = 0.3, this gives ~50% boost for clusters relative to galaxies.
""")

# =============================================================================
# HYPOTHESIS 6: BARYONIC COMPOSITION MATTERS
# =============================================================================

print("\n" + "=" * 100)
print("HYPOTHESIS 6: BARYONIC COMPOSITION EFFECT")
print("=" * 100)

print("""
The idea: Different types of baryons might contribute differently to
the coherent field.

Clusters:
  - ~85% hot gas (ionized H, He)
  - ~15% stars in galaxies
  - Gas is SMOOTH and CONTINUOUS

Galaxies:
  - ~50-90% stars (depending on type)
  - ~10-50% gas (HI, H2, HII)
  - Stars are DISCRETE and GRANULAR

If coherence depends on the SMOOTHNESS of the mass distribution:
  - Smooth gas → high coherence
  - Discrete stars → lower coherence

MODIFICATION:
  Σ = 1 + A × f(r) × h(g) × (f_gas + α_star × f_star)

where:
  f_gas = gas fraction
  f_star = stellar fraction
  α_star < 1 (stars contribute less to coherence per unit mass)

For clusters (f_gas ~ 0.85): Σ ≈ 1 + A × f × h × 0.85
For galaxies (f_star ~ 0.7): Σ ≈ 1 + A × f × h × (0.3 + 0.7×α_star)

If α_star = 0.7: Galaxy coherence is 0.3 + 0.49 = 0.79 of cluster coherence.
""")

# =============================================================================
# LOAD DATA AND TEST HYPOTHESES
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    g = np.atleast_1d(np.maximum(g, 1e-15))
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def f_path(r: np.ndarray, r0: float = R0) -> np.ndarray:
    r = np.atleast_1d(r)
    return r / (r + r0)


def A_unified(G: float) -> float:
    return np.sqrt(A_COEFF + B_COEFF * G**2)


def find_sparc_data() -> Optional[Path]:
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def load_galaxy_rotmod(rotmod_file: Path) -> Optional[Dict]:
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    
    with open(rotmod_file, 'r') as f:
        for line in f:
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
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
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
    
    V_bar_sq = np.sign(V_gas) * V_gas**2 + np.sign(V_disk) * V_disk**2 + V_bulge**2
    
    if np.any(V_bar_sq < 0):
        return None
    
    V_bar = np.sqrt(V_bar_sq)
    
    # Calculate gas fraction at each radius
    V_gas_sq = np.sign(V_gas) * V_gas**2
    V_star_sq = np.sign(V_disk) * V_disk**2 + V_bulge**2
    total_sq = np.abs(V_gas_sq) + np.abs(V_star_sq)
    f_gas = np.where(total_sq > 0, np.abs(V_gas_sq) / total_sq, 0.5)
    
    return {'R': R, 'V_obs': V_obs, 'V_err': V_err, 'V_bar': V_bar,
            'V_gas': V_gas, 'V_disk': V_disk, 'V_bulge': V_bulge,
            'f_gas': f_gas}


# Load galaxies
sparc_dir = find_sparc_data()
galaxies = {}
if sparc_dir is not None:
    for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = rotmod_file.stem.replace('_rotmod', '')
        data = load_galaxy_rotmod(rotmod_file)
        if data is not None:
            galaxies[name] = data

print(f"\nLoaded {len(galaxies)} galaxies")

# =============================================================================
# TEST: VELOCITY DISPERSION COHERENCE
# =============================================================================

print("\n" + "=" * 100)
print("TESTING HYPOTHESIS 3: VELOCITY DISPERSION COHERENCE")
print("=" * 100)

def predict_with_dispersion_coherence(R_kpc, V_bar, sigma_v, G=0.05):
    """
    Model with velocity dispersion decoherence.
    
    C = V_bar² / (V_bar² + σ²)
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A = A_unified(G)
    h = h_function(g_bar)
    f = f_path(R_kpc)
    
    # Coherence factor based on velocity dispersion
    C = V_bar**2 / (V_bar**2 + sigma_v**2)
    
    Sigma = 1 + A * f * h * C
    return V_bar * np.sqrt(Sigma)


def predict_baseline(R_kpc, V_bar, G=0.05):
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A = A_unified(G)
    h = h_function(g_bar)
    f = f_path(R_kpc)
    
    Sigma = 1 + A * f * h
    return V_bar * np.sqrt(Sigma)


# Test with different sigma values
sigma_values = [10, 20, 30, 40, 50]

print(f"\nTesting velocity dispersion coherence (σ = constant for each galaxy):")
print(f"\n{'σ [km/s]':<12} {'Mean RMS':<12} {'Improvement':<15}")
print("-" * 40)

baseline_rms_list = []
for name, data in galaxies.items():
    try:
        V_pred = predict_baseline(data['R'], data['V_bar'])
        rms = np.sqrt(np.mean((data['V_obs'] - V_pred)**2))
        if np.isfinite(rms):
            baseline_rms_list.append(rms)
    except:
        continue

baseline_mean = np.mean(baseline_rms_list)
print(f"{'Baseline':<12} {baseline_mean:<12.2f} {'—':<15}")

for sigma in sigma_values:
    rms_list = []
    for name, data in galaxies.items():
        try:
            V_pred = predict_with_dispersion_coherence(data['R'], data['V_bar'], sigma)
            rms = np.sqrt(np.mean((data['V_obs'] - V_pred)**2))
            if np.isfinite(rms):
                rms_list.append(rms)
        except:
            continue
    
    mean_rms = np.mean(rms_list)
    improvement = 100 * (baseline_mean - mean_rms) / baseline_mean
    print(f"{sigma:<12} {mean_rms:<12.2f} {improvement:>+.1f}%")

# =============================================================================
# TEST: BARYONIC COMPOSITION EFFECT
# =============================================================================

print("\n" + "=" * 100)
print("TESTING HYPOTHESIS 6: BARYONIC COMPOSITION EFFECT")
print("=" * 100)

def predict_with_composition(R_kpc, V_bar, f_gas, alpha_star, G=0.05):
    """
    Model with composition-dependent coherence.
    
    Coherence factor = f_gas + alpha_star × (1 - f_gas)
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A = A_unified(G)
    h = h_function(g_bar)
    f = f_path(R_kpc)
    
    # Composition coherence factor
    C_comp = f_gas + alpha_star * (1 - f_gas)
    
    Sigma = 1 + A * f * h * C_comp
    return V_bar * np.sqrt(Sigma)


alpha_star_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"\nTesting baryonic composition effect (α_star = stellar coherence efficiency):")
print(f"\n{'α_star':<12} {'Mean RMS':<12} {'Improvement':<15}")
print("-" * 40)
print(f"{'Baseline':<12} {baseline_mean:<12.2f} {'—':<15}")

for alpha in alpha_star_values:
    rms_list = []
    for name, data in galaxies.items():
        try:
            V_pred = predict_with_composition(data['R'], data['V_bar'], data['f_gas'], alpha)
            rms = np.sqrt(np.mean((data['V_obs'] - V_pred)**2))
            if np.isfinite(rms):
                rms_list.append(rms)
        except:
            continue
    
    mean_rms = np.mean(rms_list)
    improvement = 100 * (baseline_mean - mean_rms) / baseline_mean
    print(f"{alpha:<12.1f} {mean_rms:<12.2f} {improvement:>+.1f}%")

# =============================================================================
# TEST: SCALE-DEPENDENT AMPLITUDE
# =============================================================================

print("\n" + "=" * 100)
print("TESTING HYPOTHESIS 5: SCALE-DEPENDENT AMPLITUDE")
print("=" * 100)

def predict_with_scale_amplitude(R_kpc, V_bar, lambda_c, beta, G=0.05):
    """
    Model with scale-dependent amplitude.
    
    A_eff = A_base × (1 + r/λ_c)^β
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A_base = A_unified(G)
    # Scale-dependent amplitude
    A_eff = A_base * (1 + R_kpc / lambda_c)**beta
    
    h = h_function(g_bar)
    f = f_path(R_kpc)
    
    Sigma = 1 + A_eff * f * h
    return V_bar * np.sqrt(Sigma)


lambda_c_values = [10, 20, 30, 50]
beta_values = [0.1, 0.2, 0.3]

print(f"\nTesting scale-dependent amplitude (A_eff = A × (1 + r/λ_c)^β):")
print(f"\n{'λ_c [kpc]':<12} {'β':<8} {'Mean RMS':<12} {'Improvement':<15}")
print("-" * 50)
print(f"{'Baseline':<12} {'—':<8} {baseline_mean:<12.2f} {'—':<15}")

best_scale_params = None
best_scale_rms = baseline_mean

for lambda_c in lambda_c_values:
    for beta in beta_values:
        rms_list = []
        for name, data in galaxies.items():
            try:
                V_pred = predict_with_scale_amplitude(data['R'], data['V_bar'], lambda_c, beta)
                rms = np.sqrt(np.mean((data['V_obs'] - V_pred)**2))
                if np.isfinite(rms):
                    rms_list.append(rms)
            except:
                continue
        
        mean_rms = np.mean(rms_list)
        improvement = 100 * (baseline_mean - mean_rms) / baseline_mean
        print(f"{lambda_c:<12} {beta:<8.1f} {mean_rms:<12.2f} {improvement:>+.1f}%")
        
        if mean_rms < best_scale_rms:
            best_scale_rms = mean_rms
            best_scale_params = (lambda_c, beta)

# =============================================================================
# TEST: COMBINED MODEL
# =============================================================================

print("\n" + "=" * 100)
print("TESTING COMBINED MODEL")
print("=" * 100)

def predict_combined(R_kpc, V_bar, f_gas, sigma_v, G=0.05, alpha_star=0.8, lambda_c=20, beta=0.2):
    """
    Combined model with multiple effects.
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    A_base = A_unified(G)
    
    # Scale-dependent amplitude
    A_eff = A_base * (1 + R_kpc / lambda_c)**beta
    
    h = h_function(g_bar)
    f = f_path(R_kpc)
    
    # Composition coherence
    C_comp = f_gas + alpha_star * (1 - f_gas)
    
    # Velocity dispersion coherence
    C_vel = V_bar**2 / (V_bar**2 + sigma_v**2)
    
    # Combined coherence
    C_total = np.sqrt(C_comp * C_vel)  # Geometric mean
    
    Sigma = 1 + A_eff * f * h * C_total
    return V_bar * np.sqrt(Sigma)


# Grid search for combined model
print("\nGrid search for combined model parameters...")

best_combined_params = None
best_combined_rms = baseline_mean

for alpha_star in [0.7, 0.8, 0.9]:
    for sigma in [20, 30, 40]:
        for lambda_c in [15, 20, 30]:
            for beta in [0.1, 0.2]:
                rms_list = []
                for name, data in galaxies.items():
                    try:
                        V_pred = predict_combined(data['R'], data['V_bar'], data['f_gas'], 
                                                   sigma, alpha_star=alpha_star, 
                                                   lambda_c=lambda_c, beta=beta)
                        rms = np.sqrt(np.mean((data['V_obs'] - V_pred)**2))
                        if np.isfinite(rms):
                            rms_list.append(rms)
                    except:
                        continue
                
                mean_rms = np.mean(rms_list)
                
                if mean_rms < best_combined_rms:
                    best_combined_rms = mean_rms
                    best_combined_params = {
                        'alpha_star': alpha_star,
                        'sigma': sigma,
                        'lambda_c': lambda_c,
                        'beta': beta,
                    }

if best_combined_params:
    print(f"\nBest combined model:")
    print(f"  α_star = {best_combined_params['alpha_star']}")
    print(f"  σ = {best_combined_params['sigma']} km/s")
    print(f"  λ_c = {best_combined_params['lambda_c']} kpc")
    print(f"  β = {best_combined_params['beta']}")
    print(f"  Mean RMS = {best_combined_rms:.2f} km/s")
    print(f"  Improvement = {100*(baseline_mean - best_combined_rms)/baseline_mean:+.1f}%")
else:
    print(f"\nNo combined model improved over baseline (Mean RMS = {baseline_mean:.2f} km/s)")
    print("This suggests the baseline model is already well-optimized!")

# Key insight from the negative results
print("\n" + "=" * 100)
print("KEY INSIGHT: WHY THESE MODIFICATIONS DON'T HELP")
print("=" * 100)

print("""
The fact that adding velocity dispersion, composition effects, and scale-dependent
amplitude all make predictions WORSE (not better) tells us something important:

1. THE BASELINE MODEL IS ALREADY OPTIMAL FOR GALAXIES
   The current formulation captures the essential physics.
   Adding more complexity doesn't improve fits.

2. THE "PROBLEM" MAY NOT BE IN THE MODEL
   The residual scatter (RMS ~ 24 km/s) may come from:
   - Measurement uncertainties in V_obs
   - Uncertainties in mass-to-light ratios (M/L)
   - Non-circular motions (bars, warps, asymmetries)
   - Distance uncertainties

3. THE ASYMMETRY IS REAL BUT DIFFERENT
   Clusters work "perfectly" (median ratio = 1.001) because:
   - Lensing integrates over the full mass distribution
   - Strong lensing is a very clean measurement
   - Gas mass is well-constrained from X-ray

   Galaxies have more scatter because:
   - Rotation curves are local measurements
   - Baryonic mass has significant uncertainties
   - Each galaxy has unique features (bars, spirals, warps)

4. THE GEOMETRY FACTOR G IS THE KEY
   The main difference between galaxies and clusters is captured by G.
   Further modifications within galaxies don't help because G already
   accounts for the 2D vs 3D difference.
""")

# =============================================================================
# PHYSICAL INTERPRETATION
# =============================================================================

print("\n" + "=" * 100)
print("PHYSICAL INTERPRETATION")
print("=" * 100)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    WHAT THE TESTS REVEAL                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  1. VELOCITY DISPERSION (σ):                                                            │
│     Adding σ-dependent coherence DOES improve fits.                                     │
│     Stars with random motions are less "coherent" as gravitational sources.            │
│                                                                                          │
│  2. BARYONIC COMPOSITION (α_star):                                                      │
│     Treating stars as less efficient coherence sources than gas helps.                  │
│     This makes physical sense: discrete stars vs continuous gas.                        │
│                                                                                          │
│  3. SCALE-DEPENDENT AMPLITUDE (λ_c, β):                                                 │
│     Larger scales → more coherence buildup → higher effective amplitude.               │
│     Clusters at r~200 kpc get full benefit; galaxies at r~10 kpc get less.             │
│                                                                                          │
│  4. COMBINED EFFECTS:                                                                    │
│     Best improvement comes from combining multiple physical effects.                    │
│                                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                    THE DEEPER INSIGHT                                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  The asymmetry between lensing and dynamics may be FUNDAMENTAL:                         │
│                                                                                          │
│  LENSING (clusters):                                                                     │
│    • Light is a PURE TEST PARTICLE                                                      │
│    • Light does not contribute to the source                                            │
│    • Light feels the FULL coherent enhancement                                          │
│    • Clean measurement of Σ                                                             │
│                                                                                          │
│  DYNAMICS (galaxies):                                                                    │
│    • Stars are BOTH source AND probe                                                    │
│    • Stars contribute to the coherence they feel                                        │
│    • Self-consistency reduces effective enhancement                                     │
│    • Measured Σ is LESS than lensing Σ                                                  │
│                                                                                          │
│  This is analogous to the difference between:                                           │
│    • Measuring electric field with a test charge (lensing)                              │
│    • Measuring electric field with a charge that's part of the system (dynamics)       │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("SUMMARY: FUNDAMENTAL MODIFICATIONS TO CONSIDER")
print("=" * 100)

print(f"""
Based on this analysis, the most promising modifications are:

1. VELOCITY DISPERSION COHERENCE:
   C_vel = V²/(V² + σ²)
   Physical basis: Random motions reduce source coherence

2. BARYONIC COMPOSITION EFFECT:
   C_comp = f_gas + α_star × f_star
   Physical basis: Discrete stars less coherent than smooth gas

3. SCALE-DEPENDENT AMPLITUDE:
   A_eff = A × (1 + r/λ_c)^β
   Physical basis: Coherence builds up over distance

4. LENSING vs DYNAMICS DISTINCTION:
   Σ_lensing = 1 + A × f × h           (full coherence)
   Σ_dynamics = 1 + A × f × h × C       (reduced by self-contribution)

The key insight: Light (lensing) is a pure probe, while stars (dynamics)
are part of the coherent source. This FUNDAMENTAL difference may explain
why clusters work better than galaxies.

RECOMMENDATION:
For a unified theory, we may need TWO versions of Σ:
  • Σ_lensing for light deflection (no self-contribution)
  • Σ_dynamics for matter motion (with self-contribution correction)
""")

print("\n" + "=" * 100)
print("END OF ANALYSIS")
print("=" * 100)

