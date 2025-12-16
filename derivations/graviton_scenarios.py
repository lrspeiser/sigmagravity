"""
GRAVITON PATH MODEL: SPECIFIC SCENARIOS
=======================================

Tracing exactly what happens with gravitons in:
1. Sun → Earth
2. Sun → Distant Milky Way star  
3. Bulge star → Earth
"""

import numpy as np
from pathlib import Path
import json

# Physical constants
c = 2.998e8          # m/s
G = 6.674e-11        # m³/kg/s²
M_sun = 1.989e30     # kg
AU = 1.496e11        # m
kpc = 3.086e19       # m
pc = 3.086e16        # m
year = 3.156e7       # seconds

a0 = 1.2e-10         # MOND acceleration scale

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         GRAVITON SCENARIOS: STEP BY STEP                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
SCENARIO A: SUN → EARTH
================================================================================

The Setup:
- Source: Sun (M = 2×10³⁰ kg)
- Target: Earth (at 1 AU = 1.5×10¹¹ m)
- Single source, single target

""")

# Sun → Earth
r_earth = 1 * AU
g_earth = G * M_sun / r_earth**2
L_coh_earth = c**2 / g_earth

print(f"STEP 1: Sun emits gravitons")
print(f"  - Gravitons are virtual particles (like virtual photons in QED)")
print(f"  - They don't drain the Sun's mass")
print(f"  - They exist in superposition of ALL possible paths")
print()

print(f"STEP 2: Gravitons propagate (superposition of paths)")
print(f"  - Travel time: {r_earth/c:.0f} seconds ({r_earth/c/60:.1f} minutes)")
print(f"  - Newtonian field strength: g = {g_earth:.3e} m/s²")
print(f"  - Coherence length: L_coh = c²/g = {L_coh_earth:.3e} m = {L_coh_earth/AU:.1f} AU")
print()

print(f"  KEY: The coherence length ({L_coh_earth/AU:.1f} AU) is MUCH SMALLER than")
print(f"       the Sun-Earth distance (1 AU)? NO! It's {L_coh_earth/AU:.0f}× LARGER!")
print()
print(f"  Wait, that seems backwards. Let me reconsider...")
print()

print(f"  The coherence length L_coh = c²/g = {L_coh_earth:.2e} m")
print(f"  This is {L_coh_earth/r_earth:.0f}× the Sun-Earth distance")
print()

print(f"  But the RATIO g/a₀ = {g_earth/a0:.2e}")
print(f"  This is what matters for the MOND transition!")
print()

print(f"STEP 3: Gravitons hit Earth")
print(f"  - The superposition collapses")
print(f"  - Earth receives momentum toward the Sun")
print(f"  - Net acceleration: g = {g_earth:.3e} m/s²")
print()

print(f"STEP 4: Why no enhancement?")
print(f"  - Single source → all paths originate from same point")
print(f"  - Paths are automatically coherent (same origin)")
print(f"  - No 'extra' interference from distributed sources")
print(f"  - g/a₀ = {g_earth/a0:.0e} >> 1 → decoherence dominates")
print()

f_coh = a0 / (a0 + g_earth)
boost_ratio = np.sqrt(a0/g_earth) * f_coh
print(f"  Coherence factor: f = a₀/(a₀+g) = {f_coh:.2e}")
print(f"  Boost ratio: √(a₀/g) × f = {boost_ratio:.2e}")
print(f"  This is {boost_ratio*100:.2e}% - completely negligible!")
print()

print("""
================================================================================
SCENARIO B: SUN → DISTANT STAR (at 30 kpc from galactic center)
================================================================================

The Setup:
- Source: Sun (in the disk at 8 kpc from center)
- Target: A star at 30 kpc from galactic center
- Distance: ~22 kpc between them

But wait - the distant star doesn't just feel the Sun's gravity.
It feels gravity from the ENTIRE GALAXY.

Let me reframe: What does the distant star experience?

""")

# Galaxy edge star
r_edge = 30 * kpc
M_galaxy_baryonic = 5e10 * M_sun
g_edge = G * M_galaxy_baryonic / r_edge**2

print(f"The star at 30 kpc feels gravitons from:")
print(f"  - The bulge (~10¹⁰ M☉)")
print(f"  - The disk (~4×10¹⁰ M☉)")  
print(f"  - Including our Sun (tiny contribution)")
print()

print(f"STEP 1: ALL galaxy mass emits gravitons")
print(f"  - Each mass element dm emits gravitons")
print(f"  - Gravitons from different locations take different paths")
print(f"  - They arrive at the distant star from many directions")
print()

print(f"STEP 2: Gravitons propagate through the galaxy")
print(f"  - From bulge: ~22 kpc travel")
print(f"  - From nearby disk: varies")
print(f"  - From far side of disk: ~60 kpc travel")
print()

print(f"STEP 3: What is the local field strength?")
print(f"  - Total enclosed baryonic mass: ~{M_galaxy_baryonic/M_sun:.0e} M☉")
print(f"  - Distance from center: {r_edge/kpc:.0f} kpc")
print(f"  - Newtonian gravity: g_N = {g_edge:.3e} m/s²")
print(f"  - Ratio g/a₀ = {g_edge/a0:.2f}")
print()

print(f"  THIS IS IN THE MOND REGIME! g < a₀")
print()

print(f"STEP 4: Path interference")
print(f"  - Coherence factor: f = a₀/(a₀+g) = {a0/(a0+g_edge):.3f}")
print(f"  - Many sources → many different paths")
print(f"  - At low g, these paths can interfere CONSTRUCTIVELY")
print()

boost = np.sqrt(g_edge * a0) * a0/(a0 + g_edge)
g_total = g_edge + boost
v_newton = np.sqrt(g_edge * r_edge) / 1000
v_total = np.sqrt(g_total * r_edge) / 1000

print(f"STEP 5: The gravitons collapse at the star")
print(f"  - They arrive from many directions (distributed sources)")
print(f"  - Constructive interference adds extra momentum")
print(f"  - g_boost = √(g_N × a₀) × f = {boost:.3e} m/s²")
print(f"  - g_total = g_N + g_boost = {g_total:.3e} m/s²")
print()

print(f"RESULT:")
print(f"  - Newtonian prediction: v = {v_newton:.0f} km/s")
print(f"  - With interference:    v = {v_total:.0f} km/s")
print(f"  - Enhancement factor:   {g_total/g_edge:.2f}×")
print()

print("""
================================================================================
SCENARIO C: BULGE STAR → EARTH
================================================================================

The Setup:
- Source: A star in the galactic bulge (8 kpc from us)
- Target: Earth

But again, Earth doesn't just feel that one star.
Earth feels gravity from the ENTIRE GALAXY.

What matters is: what is the gravitational field at Earth's location
due to the galaxy as a whole?

""")

# Earth's position in the galaxy
r_sun_from_center = 8 * kpc
M_interior = 5e10 * M_sun * 0.5  # Rough estimate of mass interior to Sun's orbit
g_galactic_at_sun = G * M_interior / r_sun_from_center**2

print(f"Earth's galactic environment:")
print(f"  - Distance from galactic center: {r_sun_from_center/kpc:.0f} kpc")
print(f"  - Mass interior to our orbit: ~{M_interior/M_sun:.0e} M☉")
print(f"  - Galactic gravity at Sun's position: g = {g_galactic_at_sun:.3e} m/s²")
print(f"  - Ratio g/a₀ = {g_galactic_at_sun/a0:.2f}")
print()

print(f"STEP 1: The bulge emits gravitons toward us")
print(f"  - Travel time: ~26,000 years")
print(f"  - But so does the rest of the galaxy!")
print()

print(f"STEP 2: What matters is the LOCAL field")
print(f"  - At Earth's galactic position, g ≈ {g_galactic_at_sun:.2e} m/s²")
print(f"  - This is {g_galactic_at_sun/a0:.1f}× a₀")
print(f"  - We're in the TRANSITION region!")
print()

f_coh_gal = a0 / (a0 + g_galactic_at_sun)
boost_gal = np.sqrt(g_galactic_at_sun * a0) * f_coh_gal
g_total_gal = g_galactic_at_sun + boost_gal
v_sun_newton = np.sqrt(g_galactic_at_sun * r_sun_from_center) / 1000
v_sun_total = np.sqrt(g_total_gal * r_sun_from_center) / 1000

print(f"STEP 3: Path interference at our location")
print(f"  - Coherence factor: f = {f_coh_gal:.3f}")
print(f"  - g_boost = {boost_gal:.3e} m/s²")
print(f"  - g_total = {g_total_gal:.3e} m/s²")
print()

print(f"RESULT:")
print(f"  - Newtonian prediction for Sun's orbit: v = {v_sun_newton:.0f} km/s")
print(f"  - With interference:                    v = {v_sun_total:.0f} km/s")
print(f"  - Observed Sun orbital velocity:        v ≈ 220 km/s")
print()

print(f"  The model predicts {v_sun_total:.0f} km/s, observed is ~220 km/s")
print(f"  This is in the right ballpark!")
print()

print("""
================================================================================
THE KEY INSIGHT: IT'S ABOUT THE LOCAL FIELD, NOT THE PATH
================================================================================

The graviton path model says:

1. Gravitons from ALL sources arrive at a test mass
2. They interfere constructively or destructively
3. The DEGREE of interference depends on the LOCAL field strength
4. When g << a₀: high coherence → constructive interference → boost
5. When g >> a₀: low coherence → no extra interference → Newton

The "path" through space matters less than:
- How many sources contribute (geometry)
- What the local field strength is (determines coherence)

This is why:
- Solar System: g >> a₀, no boost (even though gravitons travel through space)
- Galaxy edge: g ~ a₀, big boost (many sources, high coherence)
- Our position: g ~ a₀, moderate boost (transition region)

================================================================================
THE ANGLE QUESTION
================================================================================

You asked about gravitons hitting matter "at a specific angle."

In this model:

1. Gravitons arrive from the direction of their source
2. The momentum transfer is along that direction
3. For a SINGLE source (Sun), all gravitons come from one direction
   → Net force points toward Sun
   
4. For MANY sources (galaxy), gravitons come from many directions
   → They add vectorially
   → In the coherent regime, they can add more efficiently
   
The "angle" is simply the direction from source to target.
The "collapse" is the quantum measurement when the graviton interacts.

The enhancement at low g isn't about angles per se - it's about
how many different source-paths can contribute coherently.

""")

# Summary table
print("================================================================================")
print("SUMMARY TABLE")
print("================================================================================")
print()
print(f"{'Location':<20} {'g [m/s²]':<12} {'g/a₀':<10} {'f_coh':<10} {'boost/g':<10} {'v [km/s]':<10}")
print("-" * 75)

scenarios = [
    ("Earth (from Sun)", g_earth, r_earth),
    ("Sun orbit (galaxy)", g_galactic_at_sun, r_sun_from_center),
    ("Galaxy edge", g_edge, r_edge),
]

for name, g, r in scenarios:
    f = a0 / (a0 + g)
    boost_ratio = np.sqrt(a0/g) * f if g > 0 else 0
    v = np.sqrt((g + np.sqrt(g*a0)*f) * r) / 1000
    print(f"{name:<20} {g:<12.2e} {g/a0:<10.2f} {f:<10.4f} {boost_ratio:<10.4f} {v:<10.0f}")

print()
print("The pattern is clear:")
print("  - High g/a₀ → low coherence → standard Newton")
print("  - Low g/a₀ → high coherence → enhanced gravity")

# Save results
output = {
    'scenarios': {
        'sun_earth': {
            'g': float(g_earth),
            'g_over_a0': float(g_earth/a0),
            'f_coherence': float(a0/(a0+g_earth)),
            'boost_negligible': True
        },
        'sun_galactic_orbit': {
            'g': float(g_galactic_at_sun),
            'g_over_a0': float(g_galactic_at_sun/a0),
            'f_coherence': float(f_coh_gal),
            'v_newton_kms': float(v_sun_newton),
            'v_total_kms': float(v_sun_total)
        },
        'galaxy_edge': {
            'g': float(g_edge),
            'g_over_a0': float(g_edge/a0),
            'f_coherence': float(a0/(a0+g_edge)),
            'v_newton_kms': float(v_newton),
            'v_total_kms': float(v_total)
        }
    },
    'key_insight': 'Enhancement depends on LOCAL field strength, not path through space'
}

output_file = Path(__file__).parent / "graviton_scenarios_results.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_file}")




