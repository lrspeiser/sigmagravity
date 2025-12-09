"""
GRAVITY-ENERGY CONVERSION: STEP-BY-STEP TIMELINE
=================================================

Walking through EXACTLY what happens as gravity propagates from source to observer.

No equations first - just the physical story.
"""

import numpy as np
import json
from pathlib import Path

# Physical constants
c = 2.998e8          # Speed of light [m/s]
G = 6.674e-11        # Gravitational constant
M_sun = 1.989e30     # Solar mass [kg]
AU = 1.496e11        # Astronomical unit [m]
kpc = 3.086e19       # Kiloparsec [m]
year = 3.156e7       # Seconds per year

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║            GRAVITY-ENERGY CONVERSION: THE PHYSICAL STORY                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Let's trace EXACTLY what happens, step by step.

================================================================================
SCENARIO 1: SUN → EARTH (1 AU, ~8 light-minutes)
================================================================================

THE STANDARD PICTURE (General Relativity):
------------------------------------------
- The Sun curves spacetime around it
- This curvature extends outward at the speed of light
- Earth "falls" along the curved spacetime
- The gravitational field at Earth is g = GM/r² = 5.9×10⁻³ m/s²
- This is STATIC - no energy is being transferred (in equilibrium)

OUR NEW PICTURE (Gravity-Energy Conversion):
--------------------------------------------

TIME = 0: At the Sun's surface
    
    The Sun's mass creates a gravitational field.
    
    QUESTION: Does this field "radiate" energy?
    
    In standard GR: NO (static fields don't radiate)
    In our model: We're proposing YES, continuously
    
    But wait - if the Sun radiates gravitational energy, it would lose mass!
    The Sun would evaporate. This is a problem.

TIME = 500 seconds: Halfway to Earth

    The "gravity energy" (whatever it is) has traveled 0.5 AU.
    
    QUESTION: What is this energy? Is it:
    a) Gravitational waves? (No - those require acceleration)
    b) Virtual gravitons? (Quantum picture)
    c) Some new form of energy?
    
    QUESTION: Does it interact with anything along the way?
    
    Between Sun and Earth there is:
    - Solar wind plasma (~5 protons/cm³)
    - Interplanetary dust
    - Magnetic fields
    
    If our energy interacts with this matter, some would be absorbed.
    If it doesn't interact, it just passes through.

TIME = 500 seconds: At Earth

    The "gravity energy" arrives at Earth.
    
    QUESTION: What happens when it encounters Earth's mass?
    
    Option A: It gets absorbed and converted to "extra" gravity
    Option B: It passes through (Earth is transparent)
    Option C: Some fraction is absorbed, some passes through
    
    If Option A or C: Earth would feel MORE gravity than GM/r²
    But we DON'T observe this! Earth's orbit is perfectly Keplerian.
    
    So either:
    1. The Sun doesn't radiate gravity energy, OR
    2. The conversion efficiency is essentially zero at g = 5.9×10⁻³ m/s²

""")

# Calculate actual numbers for Sun-Earth
r_earth = 1 * AU
g_earth = G * M_sun / r_earth**2
print(f"ACTUAL NUMBERS for Sun-Earth:")
print(f"  Distance: {r_earth/AU:.1f} AU = {r_earth:.3e} m")
print(f"  Light travel time: {r_earth/c:.0f} seconds = {r_earth/c/60:.1f} minutes")
print(f"  Newtonian gravity at Earth: g = {g_earth:.3e} m/s²")
print(f"  Earth's orbital velocity: v = {np.sqrt(g_earth * r_earth)/1000:.1f} km/s")
print()

# MOND scale for comparison
a0 = 1.2e-10
print(f"  MOND scale a₀ = {a0:.1e} m/s²")
print(f"  Ratio g/a₀ = {g_earth/a0:.1e}")
print(f"  This is {g_earth/a0:.0e} times LARGER than a₀")
print()

print("""
CONCLUSION FOR SUN-EARTH:
    
    The gravitational field at Earth (g ≈ 6×10⁻³ m/s²) is about
    50 MILLION times stronger than the MOND scale (a₀ ≈ 10⁻¹⁰ m/s²).
    
    If there's any "gravity energy conversion" happening, it must be
    COMPLETELY SUPPRESSED at these field strengths.
    
    This is why we need η → 0 when g >> a₀.
    
    But this feels like we're just DEFINING the suppression to match
    observations, not DERIVING it from physics.

================================================================================
SCENARIO 2: SUN → VOYAGER 1 (160 AU, ~22 light-hours)
================================================================================
""")

r_voyager = 160 * AU
g_voyager = G * M_sun / r_voyager**2
print(f"ACTUAL NUMBERS for Sun-Voyager:")
print(f"  Distance: {r_voyager/AU:.0f} AU = {r_voyager:.3e} m")
print(f"  Light travel time: {r_voyager/c/3600:.1f} hours")
print(f"  Newtonian gravity: g = {g_voyager:.3e} m/s²")
print(f"  Ratio g/a₀ = {g_voyager/a0:.1f}")
print()

print("""
    At Voyager's distance, g is still ~2000× larger than a₀.
    
    The "Pioneer anomaly" was once thought to show extra acceleration,
    but it was explained by thermal radiation pressure.
    
    No anomalous gravity has been detected in the outer Solar System.
    
    So the suppression must still be nearly complete at g ~ 2000 a₀.

================================================================================
SCENARIO 3: GALACTIC BULGE STAR → EARTH (8 kpc, ~26,000 light-years)
================================================================================
""")

r_bulge = 8 * kpc  # Distance to galactic center
M_bulge = 1e10 * M_sun  # Rough bulge mass
g_from_bulge = G * M_bulge / r_bulge**2

print(f"ACTUAL NUMBERS for Bulge → Earth:")
print(f"  Distance to galactic center: 8 kpc = {r_bulge:.3e} m")
print(f"  Light travel time: {r_bulge/c/year:.0f} years")
print(f"  Approximate bulge mass: {M_bulge/M_sun:.0e} M☉")
print(f"  Newtonian gravity from bulge: g = {g_from_bulge:.3e} m/s²")
print(f"  Ratio g/a₀ = {g_from_bulge/a0:.2f}")
print()

print("""
THE JOURNEY FROM BULGE TO EARTH:
--------------------------------

TIME = 0: A photon/graviton leaves a star in the galactic bulge

TIME = 0 to 26,000 years: Traveling through the galaxy

    The "gravity energy" must travel through:
    - ~8 kpc of interstellar medium
    - Average density: ~1 hydrogen atom per cm³ = 1.7×10⁻²¹ kg/m³
    - Stars, gas clouds, dust
    - The galactic disk (we're in it)
    
    Total column density along the path:
    Σ = ρ × L ≈ 1.7×10⁻²¹ kg/m³ × 2.5×10²⁰ m ≈ 0.4 kg/m²
    
    (This is actually quite low - space is very empty!)

TIME = 26,000 years: Arrives at Earth

    QUESTION: How much "gravity energy" was absorbed along the way?
    
    If cross-section σ ~ 10⁻⁶ m²/kg:
        Optical depth τ = σ × Σ = 10⁻⁶ × 0.4 = 4×10⁻⁷
        Survival fraction = e^(-τ) ≈ 0.9999996
        
    Almost NOTHING is absorbed! The galaxy is too sparse.

THE PROBLEM:
------------

If matter absorbs gravity energy and converts it to gravity...
And the galaxy is almost transparent...
Then WHERE does the "dark matter" effect come from?

The optical depth model says: energy accumulates in empty space,
matter depletes it by absorbing it.

But if almost nothing is absorbed (τ << 1), then:
- Energy just keeps accumulating forever
- Or there's no accumulation mechanism at all

This suggests the "matter absorption" model might be WRONG.

================================================================================
SCENARIO 4: STAR AT GALAXY EDGE → US (30 kpc from center)
================================================================================
""")

r_edge = 30 * kpc  # Distance from galactic center
M_galaxy = 5e10 * M_sun  # Baryonic mass of Milky Way
g_edge = G * M_galaxy / r_edge**2

print(f"ACTUAL NUMBERS for Galaxy Edge:")
print(f"  Distance from center: 30 kpc = {r_edge:.3e} m")
print(f"  Galaxy baryonic mass: {M_galaxy/M_sun:.0e} M☉")
print(f"  Newtonian gravity: g = {g_edge:.3e} m/s²")
print(f"  Ratio g/a₀ = {g_edge/a0:.2f}")
print()

print("""
    At the galaxy edge, g ≈ 0.08 a₀
    
    This is in the MOND regime where we expect significant "extra" gravity.
    
    Observed rotation velocity: ~200 km/s (flat)
    Newtonian prediction: ~70 km/s (declining)
    
    The "missing gravity" is a factor of ~3 in acceleration,
    or a factor of ~9 in "mass".

================================================================================
THE FUNDAMENTAL QUESTION
================================================================================

We've been assuming:
    
    1. Gravity "radiates" some form of energy
    2. This energy propagates outward
    3. It accumulates or converts under certain conditions
    4. This creates "extra" gravity
    
But we haven't answered:

    Q1: WHAT is this energy? 
        - It can't be gravitational waves (static sources don't radiate)
        - It can't drain the source's mass (Sun would evaporate)
        
    Q2: WHY does it accumulate in weak fields but not strong fields?
        - We've been DEFINING η = a₀/(a₀+g) to match observations
        - But what PHYSICAL mechanism causes this?
        
    Q3: HOW does it convert back to gravity?
        - If matter absorbs it → galaxy is too transparent
        - If distance triggers it → why distance?
        - If field strength triggers it → circular reasoning

================================================================================
A DIFFERENT PERSPECTIVE
================================================================================

Maybe we're thinking about this wrong.

What if the "extra gravity" isn't from energy conversion at all?

What if it's a MODIFICATION of how gravity works at low accelerations?

The MOND formula g_total = g_N × ν(g_N/a₀) says:
    - Gravity is enhanced when g < a₀
    - The enhancement factor ν depends only on local field strength
    - No "energy" needs to propagate or convert

This is simpler, but feels less physical.

OR...

What if the "energy" picture is about COHERENCE, not propagation?

Gravity from distributed mass (galaxy) vs point mass (Sun):
    - Sun: All field lines from one point → coherent
    - Galaxy: Field lines from many sources → can interfere
    
At low accelerations (large distances), the field from many sources
might "add coherently" in a way that enhances the total.

This would explain:
    - Why Solar System is Newtonian (one dominant source)
    - Why galaxies show enhancement (many distributed sources)
    - Why clusters show even more (even more distributed)

But this is the Σ-Gravity coherence model, not energy conversion!

================================================================================
HONEST ASSESSMENT
================================================================================

The "gravity energy conversion" picture has problems:

1. SOURCE PROBLEM: Static masses don't radiate energy in GR
   - If they did, they'd lose mass
   - No mechanism for continuous "gravity energy" emission

2. PROPAGATION PROBLEM: What carries the energy?
   - Not gravitational waves (need acceleration)
   - Not virtual gravitons (those ARE the static field)
   
3. CONVERSION PROBLEM: What triggers conversion?
   - Matter absorption → galaxy too transparent
   - Distance → why?
   - Field strength → circular

4. SUPPRESSION PROBLEM: Why is conversion suppressed at high g?
   - We DEFINE η = a₀/(a₀+g) to match data
   - No physical derivation

The formula g_boost = √(g×a₀) × a₀/(a₀+g) WORKS numerically,
but we haven't explained WHY it should be this way.

Maybe the answer is simpler: gravity just BEHAVES differently
at low accelerations, and a₀ ≈ cH₀ tells us the universe's
expansion rate sets this scale.

""")

# Save the analysis
output = {
    'scenarios': {
        'sun_earth': {
            'distance_AU': 1,
            'distance_m': float(r_earth),
            'g_newton': float(g_earth),
            'g_over_a0': float(g_earth/a0),
            'light_time_minutes': float(r_earth/c/60)
        },
        'sun_voyager': {
            'distance_AU': 160,
            'distance_m': float(r_voyager),
            'g_newton': float(g_voyager),
            'g_over_a0': float(g_voyager/a0)
        },
        'bulge_earth': {
            'distance_kpc': 8,
            'distance_m': float(r_bulge),
            'g_newton': float(g_from_bulge),
            'g_over_a0': float(g_from_bulge/a0),
            'light_time_years': float(r_bulge/c/year)
        },
        'galaxy_edge': {
            'distance_kpc': 30,
            'distance_m': float(r_edge),
            'g_newton': float(g_edge),
            'g_over_a0': float(g_edge/a0)
        }
    },
    'problems_identified': [
        'Static masses do not radiate energy in GR',
        'No physical mechanism for gravity energy propagation',
        'Galaxy is too transparent for matter absorption model',
        'Suppression factor is defined, not derived'
    ],
    'a0': float(a0)
}

output_file = Path(__file__).parent / "gravity_energy_timeline_results.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_file}")

