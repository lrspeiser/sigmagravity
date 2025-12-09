"""
SUN'S ORBIT: PROPER CALCULATION
===============================

The previous calculation was oversimplified. Let's do it right.
"""

import numpy as np

# Physical constants
G = 6.674e-11        # m³/kg/s²
M_sun = 1.989e30     # kg
kpc = 3.086e19       # m
a0 = 1.2e-10         # MOND scale

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SUN'S GALACTIC ORBIT: PROPER CALCULATION                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT WE DID WRONG:
==================

We assumed:
  M_interior = 2.5×10¹⁰ M☉ (rough guess)
  g = GM/r² (point mass approximation)

This gave: v_Newton = 116 km/s (way too low!)

WHAT'S ACTUALLY KNOWN:
======================
""")

# Observed values
v_sun_observed = 220  # km/s (well-measured)
r_sun = 8.0 * kpc     # Distance from galactic center

print(f"Sun's observed orbital velocity: v = {v_sun_observed} km/s")
print(f"Sun's distance from galactic center: R = 8.0 kpc")
print()

# From v = √(g×r), we can get the actual g
v_sun_ms = v_sun_observed * 1000  # m/s
g_actual = v_sun_ms**2 / r_sun

print(f"ACTUAL gravitational acceleration at Sun's position:")
print(f"  g = v²/R = ({v_sun_observed} km/s)² / (8 kpc)")
print(f"  g = {g_actual:.3e} m/s²")
print()

# What mass would produce this in Newton?
M_dynamical = g_actual * r_sun**2 / G
print(f"Dynamical mass (if Newtonian):")
print(f"  M_dyn = g×R²/G = {M_dynamical/M_sun:.2e} M☉")
print()

# What's the actual baryonic mass?
print("""
MILKY WAY BARYONIC MASS (from observations):
============================================

Component        Mass (M☉)       Distribution
---------        ---------       ------------
Stellar bulge    ~1×10¹⁰         Concentrated at center
Stellar disk     ~4-5×10¹⁰       Exponential, R_d ≈ 2.6 kpc
Gas disk         ~1×10¹⁰         More extended
---------        ---------
TOTAL            ~6×10¹⁰ M☉

But NOT all of this is interior to the Sun's orbit!
""")

# Better estimate: exponential disk
R_d = 2.6  # kpc, disk scale length
M_disk = 5e10 * M_sun
M_bulge = 1e10 * M_sun

# For exponential disk, mass interior to R is:
# M(<R) = M_total × [1 - (1 + R/R_d) × exp(-R/R_d)]
def mass_interior_exponential(R_kpc, M_total, R_d_kpc):
    x = R_kpc / R_d_kpc
    return M_total * (1 - (1 + x) * np.exp(-x))

M_disk_interior = mass_interior_exponential(8.0, M_disk, R_d)
M_bulge_interior = M_bulge  # Bulge is mostly interior to 8 kpc

M_baryonic_interior = M_disk_interior + M_bulge_interior

print(f"Mass interior to 8 kpc (exponential disk model):")
print(f"  Disk (R_d = {R_d} kpc): {M_disk_interior/M_sun:.2e} M☉")
print(f"  Bulge:                  {M_bulge_interior/M_sun:.2e} M☉")
print(f"  TOTAL baryonic:         {M_baryonic_interior/M_sun:.2e} M☉")
print()

# What velocity would this give in Newton?
# For a disk, it's more complex than GM/r², but let's use the circular velocity
# from the enclosed mass as an approximation
g_baryonic = G * M_baryonic_interior / r_sun**2
v_baryonic = np.sqrt(g_baryonic * r_sun) / 1000

print(f"Newtonian prediction from baryonic mass:")
print(f"  g_bar = {g_baryonic:.3e} m/s²")
print(f"  v_bar = √(g×R) = {v_baryonic:.0f} km/s")
print()

print(f"COMPARISON:")
print(f"  Observed:   v = {v_sun_observed} km/s")
print(f"  Baryonic:   v = {v_baryonic:.0f} km/s")
print(f"  Ratio:      {v_sun_observed/v_baryonic:.2f}×")
print()

# Now apply the graviton model
print("""
================================================================================
GRAVITON PATH MODEL PREDICTION
================================================================================
""")

g_bar = g_baryonic
ratio = g_bar / a0
f_coh = a0 / (a0 + g_bar)
g_boost = np.sqrt(g_bar * a0) * f_coh
g_total = g_bar + g_boost
v_pred = np.sqrt(g_total * r_sun) / 1000

print(f"Baryonic gravity:    g_bar = {g_bar:.3e} m/s²")
print(f"Ratio g/a₀:          {ratio:.2f}")
print(f"Coherence factor:    f = {f_coh:.3f}")
print(f"Boost:               g_boost = {g_boost:.3e} m/s²")
print(f"Total:               g_total = {g_total:.3e} m/s²")
print()
print(f"PREDICTED velocity:  v = {v_pred:.0f} km/s")
print(f"OBSERVED velocity:   v = {v_sun_observed} km/s")
print(f"Ratio pred/obs:      {v_pred/v_sun_observed:.2f}")
print()

# What would we need?
print("""
================================================================================
WHAT WOULD WE NEED TO MATCH OBSERVATIONS?
================================================================================
""")

# We need g_total such that v = 220 km/s
g_needed = (v_sun_observed * 1000)**2 / r_sun
g_boost_needed = g_needed - g_bar

print(f"To get v = {v_sun_observed} km/s:")
print(f"  Need g_total = {g_needed:.3e} m/s²")
print(f"  Have g_bar   = {g_bar:.3e} m/s²")
print(f"  Need g_boost = {g_boost_needed:.3e} m/s²")
print()
print(f"Our formula gives g_boost = {g_boost:.3e} m/s²")
print(f"Ratio needed/predicted = {g_boost_needed/g_boost:.2f}")
print()

# What if we adjust the amplitude?
A_needed = g_boost_needed / (np.sqrt(g_bar * a0) * f_coh)
print(f"To match, we'd need amplitude A = {A_needed:.2f}")
print(f"(Currently using A = 1)")
print()

print("""
================================================================================
INTERPRETATION
================================================================================

The graviton path model with amplitude A = 1 gives:
  v_pred = {:.0f} km/s (vs observed 220 km/s)

This is {:.0f}% of the observed value.

POSSIBLE EXPLANATIONS:

1. The baryonic mass estimate is too low
   - McMillan 2017 uses higher mass normalization (×1.16)
   - Gas mass might be underestimated

2. The coherence formula needs refinement
   - Maybe f = a₀/(a₀+g) isn't quite right
   - The transition might be sharper or broader

3. There's an amplitude factor we're missing
   - Like Σ-Gravity's A₀ = e^(1/2π) ≈ 1.17
   - This could come from geometry (disk vs sphere)

4. The disk geometry matters
   - Gravitons from a disk add differently than from a sphere
   - The 2D coherence might be different from 3D

The formula WORKS qualitatively but needs ~30% more boost
to match the Sun's observed orbital velocity.
""".format(v_pred, v_pred/v_sun_observed*100))

# Try with Σ-Gravity amplitude
print("================================================================================")
print("WITH Σ-GRAVITY AMPLITUDE A₀ = 1.17:")
print("================================================================================")

A0 = np.exp(1/(2*np.pi))  # ≈ 1.173
g_boost_A0 = A0 * np.sqrt(g_bar * a0) * f_coh
g_total_A0 = g_bar + g_boost_A0
v_pred_A0 = np.sqrt(g_total_A0 * r_sun) / 1000

print(f"  g_boost = A₀ × √(g×a₀) × f = {g_boost_A0:.3e} m/s²")
print(f"  v_pred = {v_pred_A0:.0f} km/s")
print(f"  Ratio pred/obs = {v_pred_A0/v_sun_observed:.2f}")
print()

# What about the full Σ-Gravity formula?
print("================================================================================")
print("FULL Σ-GRAVITY FORMULA (for comparison):")
print("================================================================================")

# Σ-Gravity: h(g) = √(g†/g) × g†/(g†+g)
g_dagger = 9.6e-11  # Σ-Gravity's critical acceleration
h = np.sqrt(g_dagger/g_bar) * g_dagger/(g_dagger + g_bar)

# With coherence window W = r/(ξ+r) where ξ = R_d/(2π)
xi = R_d * kpc / (2 * np.pi)
W = r_sun / (xi + r_sun)

Sigma = 1 + A0 * W * h
v_sigma = v_baryonic * np.sqrt(Sigma)

print(f"  g† = {g_dagger:.2e} m/s²")
print(f"  h(g) = {h:.3f}")
print(f"  W(r) = {W:.3f}")
print(f"  Σ = 1 + A₀×W×h = {Sigma:.3f}")
print(f"  v_pred = v_bar × √Σ = {v_sigma:.0f} km/s")
print(f"  Ratio pred/obs = {v_sigma/v_sun_observed:.2f}")

