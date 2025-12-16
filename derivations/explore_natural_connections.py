#!/usr/bin/env python3
"""
EXPLORE NATURAL CONNECTIONS BETWEEN A, W, AND h(g)

Question: Can we connect A, W, and h(g) in a way that they automatically
adjust based on physical properties, not arbitrary switching?

Physical quantities that could connect them:
1. Density gradient (∇ρ / ρ)
2. Gravitational potential depth (Φ / c²)
3. Jeans length / system size ratio
4. Toomre Q parameter (stability)
5. Velocity dispersion / circular velocity (σ/V)
6. Path length through matter (L)
7. Coherence length (λ_c)

Let's explore which of these could naturally unify the model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

c = 3e8
H0 = 2.27e-18
G_const = 6.674e-11
M_sun = 1.989e30
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

data_dir = Path(__file__).parent.parent / "data"

print("=" * 80)
print("EXPLORING NATURAL CONNECTIONS")
print("=" * 80)

# =============================================================================
# CURRENT MODEL REVIEW
# =============================================================================
print("\n" + "=" * 80)
print("CURRENT MODEL")
print("=" * 80)

print("""
Current: Σ = 1 + A × W(r) × h(g)

where:
  A = exp(1/2π) ≈ 1.17 for galaxies
  A = 8.0 for clusters
  W(r) = r / (ξ + r) with ξ = R_d/(2π)
  h(g) = √(g†/g) × g†/(g†+g)

PROBLEM: A is different for galaxies vs clusters with no clear connection.

GOAL: Find a unified A that depends on physical properties.
""")

# =============================================================================
# IDEA 1: A DEPENDS ON COHERENCE LENGTH
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 1: A DEPENDS ON COHERENCE LENGTH")
print("=" * 80)

print("""
Physical intuition:
The coherence length λ_c is where gravity becomes "coherent."
A longer coherence length means more gravitational enhancement.

For a disk galaxy:
  λ_c ~ h (disk thickness) ~ 0.3-0.6 kpc
  
For a cluster:
  λ_c ~ r_core ~ 100-300 kpc

If A ∝ λ_c^(1/4) (path length scaling):
  A_cluster / A_galaxy = (λ_c,cluster / λ_c,galaxy)^(1/4)
                       = (200 / 0.5)^(1/4)
                       = 400^(0.25)
                       = 4.5

Actual ratio: 8.0 / 1.17 = 6.8

Close but not exact. Maybe the exponent is different?
""")

# What exponent gives the right ratio?
lambda_gal = 0.5  # kpc
lambda_cl = 200  # kpc
A_gal = np.exp(1 / (2 * np.pi))
A_cl = 8.0

# A_cl/A_gal = (lambda_cl/lambda_gal)^n
# log(A_cl/A_gal) = n × log(lambda_cl/lambda_gal)
# n = log(A_cl/A_gal) / log(lambda_cl/lambda_gal)
n_implied = np.log(A_cl / A_gal) / np.log(lambda_cl / lambda_gal)
print(f"\nImplied exponent: n = {n_implied:.3f}")
print(f"If n = 1/3: A_ratio = {(lambda_cl/lambda_gal)**(1/3):.2f}")
print(f"If n = 0.32: A_ratio = {(lambda_cl/lambda_gal)**0.32:.2f}")

# =============================================================================
# IDEA 2: A DEPENDS ON DENSITY CONTRAST
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 2: A DEPENDS ON DENSITY CONTRAST")
print("=" * 80)

print("""
Physical intuition:
The density contrast Δρ/ρ_crit determines how "overdense" a system is.
More overdense → more gravitational enhancement?

For a disk galaxy at R_d:
  ρ ~ 10^8 M_sun / kpc³ ~ 10^-20 kg/m³
  
For a cluster:
  ρ ~ 10^14 M_sun / Mpc³ ~ 10^-25 kg/m³

Clusters are LESS dense than galaxies! So this is opposite.

But wait - what about the TOTAL mass enclosed?

Galaxy: M ~ 10^10 M_sun within 10 kpc
Cluster: M ~ 10^14 M_sun within 200 kpc

Maybe A depends on total mass?
""")

M_gal = 1e10  # M_sun
M_cl = 1e14  # M_sun
n_mass = np.log(A_cl / A_gal) / np.log(M_cl / M_gal)
print(f"\nIf A ∝ M^n: n = {n_mass:.3f}")
print(f"A_ratio from M^0.17 = {(M_cl/M_gal)**0.17:.2f}")

# =============================================================================
# IDEA 3: A DEPENDS ON VELOCITY DISPERSION
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 3: A DEPENDS ON VELOCITY DISPERSION")
print("=" * 80)

print("""
Physical intuition:
Velocity dispersion σ measures the "temperature" of the system.
Higher σ → less coherent → lower enhancement?

For a disk galaxy:
  σ ~ 30-50 km/s (disk), 100-150 km/s (bulge)
  V_c ~ 200 km/s
  σ/V ~ 0.15-0.75
  
For a cluster:
  σ ~ 500-1000 km/s
  V_c ~ 0 (no rotation)
  σ/V → ∞

This doesn't help directly, but σ/V could modulate W or h.

HYPOTHESIS: W_eff = W × (1 / (1 + (σ/V)²))

At low σ/V (disk): W_eff ≈ W
At high σ/V (bulge/cluster): W_eff < W
""")

# =============================================================================
# IDEA 4: UNIFIED A FROM JEANS LENGTH
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 4: A FROM JEANS LENGTH")
print("=" * 80)

print("""
The Jeans length λ_J is where gravity overcomes pressure:
  λ_J = σ × √(π / (G × ρ))

For coherence, we might expect:
  A ∝ (λ_J / R)^α

where R is the system size.

For a disk galaxy:
  σ ~ 40 km/s, ρ ~ 0.1 M_sun/pc³
  λ_J ~ 40 × √(π / (4.3e-3 × 0.1)) ~ 40 × 8.5 ~ 340 pc
  R ~ 10 kpc
  λ_J / R ~ 0.034

For a cluster:
  σ ~ 1000 km/s, ρ ~ 10^-4 M_sun/pc³
  λ_J ~ 1000 × √(π / (4.3e-3 × 10^-4)) ~ 1000 × 2700 ~ 2.7 Mpc
  R ~ 1 Mpc
  λ_J / R ~ 2.7

Ratio: 2.7 / 0.034 = 79

If A ∝ (λ_J/R)^n and A_ratio = 6.8:
  n = log(6.8) / log(79) = 0.44
""")

# =============================================================================
# IDEA 5: A × W × h AS A SINGLE COHERENCE FUNCTION
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 5: UNIFIED COHERENCE FUNCTION")
print("=" * 80)

print("""
What if A, W, and h are not separate but emerge from a SINGLE function?

HYPOTHESIS: Σ = 1 + C(r, g, ρ)

where C is a "coherence function" that depends on:
  - Position r (where in the system)
  - Acceleration g (local field strength)
  - Density ρ (local matter density)

Dimensional analysis:
  C must be dimensionless
  
Natural combinations:
  - g / g† (acceleration ratio)
  - r / λ_c (position / coherence length)
  - ρ / ρ_crit (density contrast)
  - r × √(G × ρ) / σ (Jeans ratio)

Let's try:
  C = (g† / g)^α × (r / λ_c)^β × (ρ / ρ_crit)^γ

This would unify everything into one formula!
""")

# =============================================================================
# IDEA 6: A EMERGES FROM PATH INTEGRAL
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 6: A FROM PATH INTEGRAL")
print("=" * 80)

print("""
Physical intuition:
The enhancement comes from integrating coherent gravitational 
contributions along the path through the system.

A = ∫ ρ(s) × exp(-s/λ_c) ds / ∫ ρ(s) ds

where s is the path length and λ_c is the coherence length.

For an exponential disk (ρ ∝ exp(-r/R_d)):
  A ~ R_d / λ_c × (some numerical factor)

For a uniform sphere (cluster core):
  A ~ r_core / λ_c × (different numerical factor)

If λ_c ∝ σ / √(G × ρ) (Jeans-like):
  A ∝ R × √(G × ρ) / σ
  
For galaxy: A ∝ 10 × √(10^-20) / 40e3 ~ 10^-4 (wrong scale)

The scaling doesn't work directly. Need a different approach.
""")

# =============================================================================
# IDEA 7: CONNECT A TO W THROUGH GEOMETRY
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 7: A FROM GEOMETRY FACTOR")
print("=" * 80)

print("""
What if A depends on the GEOMETRY of the system?

Define geometry factor G:
  G = (vertical extent) / (radial extent) = h / R

For a thin disk:
  G ~ 0.3 kpc / 10 kpc = 0.03 (2D)
  
For a cluster:
  G ~ r_core / r_core = 1 (3D)

HYPOTHESIS: A = A₀ × (1 + G)^n

For galaxy (G ~ 0.03): A ~ A₀
For cluster (G ~ 1): A ~ A₀ × 2^n

To get A_ratio = 6.8:
  2^n = 6.8 / 1.17 = 5.8
  n = log(5.8) / log(2) = 2.5

So: A = A₀ × (1 + G)^2.5

This connects A to geometry naturally!
""")

# =============================================================================
# TEST IDEA 7 ON DATA
# =============================================================================
print("\n" + "=" * 80)
print("TESTING GEOMETRY-DEPENDENT A")
print("=" * 80)

# Load galaxies
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
        
        # Estimate disk thickness from R_d (typical h/R_d ~ 0.1-0.2)
        h_disk = 0.15 * R_d
        
        # Estimate bulge contribution
        f_bulge = np.sum(V_bulge**2) / max(np.sum(V_disk**2 + V_bulge**2 + V_gas**2), 1e-10)
        
        galaxies.append({
            'name': f.stem.replace('_rotmod', ''),
            'R': R, 'V_obs': V_obs, 'V_bar': V_bar, 'R_d': R_d,
            'h_disk': h_disk, 'f_bulge': f_bulge,
        })
    except:
        continue

print(f"Loaded {len(galaxies)} galaxies")

# Test geometry-dependent A
def h_function(g):
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r, xi):
    return r / (xi + r)

A_0 = np.exp(1 / (2 * np.pi))
XI_COEFF = 1 / (2 * np.pi)

# Test different geometry exponents
print("\nTest geometry-dependent A: A = A₀ × (1 + G)^n")
print(f"\n  {'n':<8} {'Mean RMS':<12} {'Median RMS':<12}")
print("  " + "-" * 35)

for n_exp in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
    rms_list = []
    for gal in galaxies:
        R = gal['R']
        V_obs = gal['V_obs']
        V_bar = gal['V_bar']
        xi = XI_COEFF * gal['R_d']
        
        # Geometry factor at each radius
        # G = h_eff / r, where h_eff includes bulge
        h_eff = gal['h_disk'] * (1 + gal['f_bulge'] * 2)  # Bulge adds vertical extent
        
        R_m = R * kpc_to_m
        g_bar = (V_bar * 1000)**2 / R_m
        
        W = W_coherence(R, xi)
        h = h_function(g_bar)
        
        # Geometry-dependent A
        G = h_eff / np.maximum(R, 0.1)
        A_eff = A_0 * (1 + G)**n_exp
        
        Sigma = 1 + A_eff * W * h
        V_pred = V_bar * np.sqrt(Sigma)
        
        rms = np.sqrt(np.mean((V_obs - V_pred)**2))
        rms_list.append(rms)
    
    print(f"  {n_exp:<8.1f} {np.mean(rms_list):<12.2f} {np.median(rms_list):<12.2f}")

# =============================================================================
# IDEA 8: A × h UNIFIED
# =============================================================================
print("\n" + "=" * 80)
print("IDEA 8: UNIFY A AND h INTO SINGLE FUNCTION")
print("=" * 80)

print("""
What if A and h(g) are not separate but part of one function?

Current: A × h(g) = A₀ × √(g†/g) × g†/(g†+g)

At low g: A × h → A₀ × (g†/g)^1.5 (diverges)
At high g: A × h → A₀ × (g†/g)^0.5 (decays)

HYPOTHESIS: The TOTAL enhancement A × h should depend on:
  - g/g† (acceleration ratio)
  - ρ/ρ_crit (density ratio) which sets A

Unified form:
  A × h = A₀ × (g†/g)^α × (ρ/ρ_crit)^β

For galaxies: ρ ~ 10^6 × ρ_crit, so (ρ/ρ_crit)^β ~ 10^(6β)
For clusters: ρ ~ 10^3 × ρ_crit, so (ρ/ρ_crit)^β ~ 10^(3β)

Ratio: 10^(3β) = 6.8 → β = log(6.8) / (3 × log(10)) = 0.28

So: A × h = A₀ × (g†/g)^α × (ρ/ρ_crit)^0.28

This would automatically give higher A for denser systems!
""")

# =============================================================================
# CONCLUSION
# =============================================================================
print("\n" + "=" * 80)
print("MOST PROMISING CONNECTIONS")
print("=" * 80)

print("""
From this exploration, the most promising natural connections are:

1. PATH LENGTH (L^n):
   A = A₀ × (L / L₀)^(1/3)
   where L is the path length through baryons
   - Physically motivated by coherent integration
   - Gives correct galaxy/cluster ratio with n ≈ 1/3

2. GEOMETRY FACTOR (G):
   A = A₀ × (1 + h/R)^n
   where h is vertical extent, R is radial extent
   - Connects 2D (disk) to 3D (cluster) naturally
   - Needs n ≈ 2.5 to match data

3. DENSITY CONTRAST (ρ):
   A = A₀ × (ρ / ρ_crit)^β
   where ρ is local density
   - Higher density → more coherence → higher A
   - Needs β ≈ 0.28 to match data

4. JEANS RATIO (λ_J / R):
   A = A₀ × (λ_J / R)^α
   - Connects stability to coherence
   - Needs α ≈ 0.44 to match data

RECOMMENDATION:
Test the PATH LENGTH connection more rigorously, as it has the
clearest physical interpretation and was already partially derived.
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)




