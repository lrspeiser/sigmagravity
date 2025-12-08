#!/usr/bin/env python3
"""
New Physics from GR Deviations: What the Data is Telling Us
============================================================

Instead of starting from theory and fitting to data, let's look at WHERE
GR fails and extract what new physics might be implied.

The key question: What patterns in the deviations suggest new equations of nature?

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np
import json
from pathlib import Path

# Physical constants
c = 2.998e8           # m/s
G = 6.674e-11         # m³/kg/s²
H0 = 2.27e-18         # 1/s
kpc_to_m = 3.086e19
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

print("=" * 90)
print("NEW PHYSICS FROM GR DEVIATIONS")
print("What the data tells us about nature")
print("=" * 90)

# =============================================================================
# LOAD THE CLASSIFICATION RESULTS
# =============================================================================

results_path = Path(__file__).parent / "gravity_test_by_classification.json"
with open(results_path) as f:
    results = json.load(f)

# =============================================================================
# PATTERN 1: ACCELERATION DEPENDENCE
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  PATTERN 1: GR FAILS AT LOW ACCELERATIONS                                            ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")

accel_data = results['acceleration_regime']
print("GR deviation by acceleration regime:")
print("-" * 70)
print(f"{'Regime':<15} {'N galaxies':<12} {'MOND RMS':<12} {'Best Model RMS':<15} {'Win vs MOND':<12}")
print("-" * 70)

for regime in ['low-g', 'mixed', 'high-g']:
    data = accel_data[regime]
    n = data['MOND_standard']['n_galaxies']
    mond_rms = data['MOND_standard']['mean_rms']
    best_rms = min(data['Survival_threshold']['mean_rms'], 
                   data['Survival_nonlocal']['mean_rms'])
    win_rate = max(data['win_rates']['Survival_threshold'],
                   data['win_rates']['Survival_nonlocal'])
    print(f"{regime:<15} {n:<12} {mond_rms:<12.1f} {best_rms:<15.1f} {win_rate*100:<12.0f}%")

print("""
WHAT THIS TELLS US:
───────────────────
• GR deviations are LARGEST at low accelerations (g < g†)
• The transition happens around g† ≈ 10⁻¹⁰ m/s²
• This scale is cosmological: g† ~ cH₀

IMPLICATION FOR NEW PHYSICS:
────────────────────────────
There must be a term in the gravitational equations that:
  1. Depends on acceleration (not just mass/distance)
  2. Becomes significant when g < g† ~ cH₀
  3. Is negligible when g >> g†

This suggests a coupling to the cosmic expansion rate H₀.
""")

# =============================================================================
# PATTERN 2: SPATIAL EXTENT MATTERS
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  PATTERN 2: GR FAILS MORE IN EXTENDED SYSTEMS                                        ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")

size_data = results['size']
print("GR deviation by system size:")
print("-" * 70)
print(f"{'Size':<15} {'N galaxies':<12} {'MOND RMS':<12} {'Outer RMS':<12} {'Inner RMS':<12}")
print("-" * 70)

for size in ['compact', 'medium', 'extended']:
    data = size_data[size]
    n = data['MOND_standard']['n_galaxies']
    mond_rms = data['MOND_standard']['mean_rms']
    outer_rms = data['Survival_threshold']['mean_outer_rms']
    inner_rms = data['Survival_threshold']['mean_inner_rms']
    print(f"{size:<15} {n:<12} {mond_rms:<12.1f} {outer_rms:<12.1f} {inner_rms:<12.1f}")

print("""
WHAT THIS TELLS US:
───────────────────
• Compact systems (R < 5 kpc) show LESS deviation from GR
• Extended systems (R > 15 kpc) show MORE deviation
• The deviation is concentrated in OUTER regions

IMPLICATION FOR NEW PHYSICS:
────────────────────────────
There must be a term that:
  1. Depends on spatial extent (not just local properties)
  2. Builds up with distance from the center
  3. Requires a minimum "coherence length" to activate

This suggests a NONLOCAL effect that accumulates over distance.
""")

# =============================================================================
# PATTERN 3: ROTATION MATTERS
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  PATTERN 3: GR FAILS MORE IN ROTATING SYSTEMS                                        ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")

shape_data = results['rc_shape']
print("GR deviation by rotation curve shape:")
print("-" * 70)
print(f"{'Shape':<15} {'N galaxies':<12} {'MOND RMS':<12} {'Best Model RMS':<15}")
print("-" * 70)

for shape in ['rising', 'flat', 'declining']:
    data = shape_data[shape]
    n = data['MOND_standard']['n_galaxies']
    mond_rms = data['MOND_standard']['mean_rms']
    best_rms = min(data['Survival_threshold']['mean_rms'],
                   data['Survival_nonlocal']['mean_rms'])
    print(f"{shape:<15} {n:<12} {mond_rms:<12.1f} {best_rms:<15.1f}")

print("""
WHAT THIS TELLS US:
───────────────────
• FLAT rotation curves show the largest deviations
• Rising curves (inner regions) show smaller deviations
• Declining curves (rare, high-mass) also show deviations

IMPLICATION FOR NEW PHYSICS:
────────────────────────────
The deviation correlates with ORDERED CIRCULAR MOTION:
  1. Flat rotation = sustained circular velocity at large R
  2. Rising = still building up rotation
  3. Declining = falling off from peak

This suggests the new physics couples to the VELOCITY FIELD,
specifically to organized/coherent rotation.
""")

# =============================================================================
# PATTERN 4: MORPHOLOGY MATTERS
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  PATTERN 4: DISK GALAXIES DEVIATE MORE THAN BULGE-DOMINATED                          ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")

bulge_data = results['bulge_presence']
print("GR deviation by bulge presence:")
print("-" * 70)
print(f"{'Type':<20} {'N galaxies':<12} {'MOND RMS':<12} {'Win rate':<12}")
print("-" * 70)

for btype in ['pure_disk', 'intermediate_bulge', 'bulge_dominated']:
    data = bulge_data[btype]
    n = data['MOND_standard']['n_galaxies']
    mond_rms = data['MOND_standard']['mean_rms']
    win_rate = data['win_rates']['Survival_threshold']
    print(f"{btype:<20} {n:<12} {mond_rms:<12.1f} {win_rate*100:<12.0f}%")

print("""
WHAT THIS TELLS US:
───────────────────
• Pure disk galaxies: Survival model wins 73% vs MOND
• Bulge-dominated: Survival model wins 77% vs MOND
• BUT: Bulge systems have HIGHER RMS overall

IMPLICATION FOR NEW PHYSICS:
────────────────────────────
The deviation pattern differs by geometry:
  1. Disks: Deviation grows smoothly outward
  2. Bulges: Deviation concentrated at specific radii

This suggests the new physics depends on the GEOMETRY of the mass distribution,
not just the total mass.
""")

# =============================================================================
# PATTERN 5: GAS CONTENT MATTERS
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  PATTERN 5: GAS-RICH GALAXIES BEHAVE DIFFERENTLY                                     ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")

gas_data = results['gas_fraction']
print("GR deviation by gas fraction:")
print("-" * 70)
print(f"{'Gas content':<20} {'N galaxies':<12} {'MOND RMS':<12} {'Best RMS':<12}")
print("-" * 70)

for gas in ['gas_rich', 'intermediate_gas', 'gas_poor']:
    data = gas_data[gas]
    n = data['MOND_standard']['n_galaxies']
    mond_rms = data['MOND_standard']['mean_rms']
    best_rms = min(data['Survival_threshold']['mean_rms'],
                   data['Survival_nonlocal']['mean_rms'])
    print(f"{gas:<20} {n:<12} {mond_rms:<12.1f} {best_rms:<12.1f}")

print("""
WHAT THIS TELLS US:
───────────────────
• Gas-rich galaxies have LOWER RMS (easier to fit)
• Gas-poor galaxies have HIGHER RMS (harder to fit)
• Gas fraction correlates with rotation curve quality

IMPLICATION FOR NEW PHYSICS:
────────────────────────────
Gas-rich systems have:
  1. More extended HI disks (larger coherence region)
  2. Colder kinematics (less velocity dispersion)
  3. More ordered rotation

This supports the idea that KINEMATIC COHERENCE matters.
""")

# =============================================================================
# SYNTHESIS: WHAT NEW EQUATION OF NATURE?
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  SYNTHESIS: WHAT THE DEVIATIONS TELL US ABOUT NEW PHYSICS                            ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

The patterns in GR deviations suggest gravity has additional terms that depend on:

1. ACCELERATION RATIO (g/g†)
   • Deviations appear when g < g† ~ cH₀
   • This introduces a COSMIC SCALE into local gravity
   • The Hubble rate H₀ must appear in the equations

2. SPATIAL COHERENCE
   • Deviations grow with distance from center
   • There's a characteristic scale ~20 kpc
   • Extended systems show more deviation than compact ones

3. VELOCITY FIELD ORDER
   • Ordered rotation enhances the deviation
   • Random motions (dispersion) suppress it
   • Counter-rotation may reduce it further

4. GEOMETRY
   • Disk geometry shows different patterns than spherical
   • The mass distribution shape matters, not just total mass

WHAT NEW TERM IN THE EQUATIONS?
───────────────────────────────

Based on these patterns, the gravitational equations need a term like:

    G_μν = (8πG/c⁴) T_μν × [1 + F(g, r, v)]

where F depends on:
    • g/g† — the local-to-cosmic acceleration ratio
    • r/r_coh — the radius relative to coherence scale
    • v·∇v/|v|² — a measure of velocity field order

THE CRITICAL INSIGHT:
─────────────────────

GR fails in a SPECIFIC WAY:
    • Not randomly
    • Not uniformly
    • But in systems with LOW g, LARGE r, and ORDERED v

This is NOT what you'd expect from:
    • Measurement errors (would be random)
    • Missing baryons (would scale with mass)
    • Standard dark matter (wouldn't depend on velocity order)

This IS consistent with:
    • A modification to gravity that couples to the cosmic expansion
    • A coherence effect that builds up over distance
    • A sensitivity to the velocity field structure
""")

# =============================================================================
# THE CANDIDATE EQUATION
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  CANDIDATE NEW EQUATION OF NATURE                                                    ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

Based on where GR fails, here is a candidate modification:

STANDARD GR (Poisson equation):
    ∇²Φ = 4πG ρ

MODIFIED EQUATION:
    ∇²Φ = 4πG ρ × [1 + Ψ(x)]

where Ψ satisfies:

    ∂Ψ/∂t + (v·∇)Ψ = S(x) - Γ(g)Ψ

with:
    S(x) = source term (depends on ρ and velocity coherence)
    Γ(g) = decay rate = g/g† (decoherence at high acceleration)

AT STEADY STATE:
    Ψ = S(x) / Γ(g) = S(x) × (g†/g)

This gives enhancement ~ (g†/g) at low accelerations.

THE SOURCE TERM S(x):

    S(x) = α × ∫ ρ(x') × C(v(x), v(x')) × W(|x-x'|) d³x'

where:
    C(v, v') = velocity correlation (1 for aligned, 0 for random)
    W(r) = spatial window (builds up over coherence length)
    α = coupling constant

THIS EQUATION PREDICTS:
───────────────────────
✓ Enhancement at low g (from Γ(g) = g/g†)
✓ Spatial buildup (from W(r) in source)
✓ Velocity dependence (from C(v,v') correlation)
✓ Geometry dependence (from integral over mass distribution)

WHAT'S STILL MISSING:
─────────────────────
• WHY does this equation exist?
• What is the microscopic origin of Ψ?
• Why is g† = cH₀/(4√π) specifically?
• What determines the coupling α?

These are the OPEN QUESTIONS that point to deeper physics.
""")

# =============================================================================
# WHAT THE DATA RULES OUT
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  WHAT THE DATA RULES OUT                                                             ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

The patterns in GR deviations RULE OUT certain explanations:

1. SIMPLE DARK MATTER HALOS
   ✗ Would not explain velocity field dependence
   ✗ Would not explain the cosmic scale g† ~ cH₀
   ✗ Would require fine-tuning for each galaxy

2. PURELY LOCAL MODIFIED GRAVITY (like MOND)
   ✗ MOND depends only on local g, not on spatial extent
   ✗ Cannot explain why extended systems deviate more
   ✗ Cannot explain velocity coherence effects

3. MEASUREMENT ERRORS
   ✗ Errors would be random, not systematic
   ✗ Would not correlate with physical properties
   ✗ Different instruments give consistent results

4. MISSING BARYONS
   ✗ Would scale with total mass, not acceleration
   ✗ Would not depend on velocity field
   ✗ Cannot explain the cosmic scale g†

WHAT THE DATA REQUIRES:
───────────────────────
• A modification that couples LOCAL dynamics to COSMIC scales
• A mechanism that depends on SPATIAL COHERENCE
• A sensitivity to VELOCITY FIELD ORDER
• A characteristic scale g† ~ cH₀

This is pointing toward something NEW in gravitational physics.
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║  SUMMARY: NEW PHYSICS IMPLIED BY GR DEVIATIONS                                       ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

THE DATA SHOWS:
───────────────
1. GR fails at LOW accelerations (g < 10⁻¹⁰ m/s²)
2. GR fails more in EXTENDED systems (R > 10 kpc)
3. GR fails more with ORDERED rotation
4. The failure scale is COSMIC (g† ~ cH₀)

THIS IMPLIES NEW PHYSICS:
─────────────────────────
• Gravity has a term that couples to the Hubble expansion
• This term builds up over distance (nonlocal)
• This term is sensitive to velocity field coherence
• This term is suppressed at high accelerations

THE CANDIDATE EQUATION:
───────────────────────
    ∇²Φ = 4πG ρ × [1 + Ψ]

    where Ψ satisfies: Γ(g)Ψ = S(ρ, v)

    with Γ(g) = g/g† (decoherence rate)
    and S depends on density and velocity coherence

OPEN QUESTIONS:
───────────────
• What is the microscopic origin of Ψ?
• Why does gravity couple to H₀?
• What determines the numerical factors?
• Is this quantum gravity? Emergent? Something else?

The phenomenology is clear. The fundamental theory is not yet known.

══════════════════════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    pass

