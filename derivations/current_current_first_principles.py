#!/usr/bin/env python3
"""
First Principles Investigation: What Causes Current-Current Correlations?
=========================================================================

The fundamental question: WHY does the current-current correlator
<j(x)·j(x')>_c affect gravity?

This script investigates the PHYSICAL ORIGIN of the correlator, not just
its phenomenological effects.

Key Questions:
1. What is the microscopic origin of the connected correlator?
2. Why does velocity alignment matter for gravity?
3. What is the relationship to quantum coherence vs classical correlation?
4. How does this connect to the stress-energy tensor structure?

Author: Sigma Gravity Research
Date: December 2025
"""

import numpy as np
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
hbar = 1.055e-34  # J·s
G = 6.674e-11  # m³/(kg·s²)
H0 = 2.27e-18  # 1/s

print("=" * 100)
print("FIRST PRINCIPLES: WHAT CAUSES CURRENT-CURRENT CORRELATIONS?")
print("=" * 100)

# =============================================================================
# PART 1: THE STRESS-ENERGY TENSOR STRUCTURE
# =============================================================================

print("""
================================================================================
PART 1: STRESS-ENERGY TENSOR STRUCTURE
================================================================================

The stress-energy tensor for a perfect fluid is:

    T^μν = (ρ + P/c²) u^μ u^ν + P g^μν

where:
    ρ = energy density
    P = pressure
    u^μ = 4-velocity

In the non-relativistic limit (v << c):

    T^00 ≈ ρc²           (energy density)
    T^0i ≈ ρc v^i        (momentum density = mass current)
    T^ij ≈ ρ v^i v^j + P δ^ij  (stress tensor)

The CURRENT j^i = ρ v^i appears in T^0i, the momentum density.

KEY INSIGHT: The current-current correlator <j·j'> is really the
correlator of the MOMENTUM DENSITY components of T^μν:

    <T^0i(x) T^0j(x')> / c² = <j^i(x) j^j(x')>

This is NOT just a kinematic quantity - it's the off-diagonal part
of the stress-energy tensor that couples to the gravitational field.
""")

# =============================================================================
# PART 2: WHY CONNECTED CORRELATORS MATTER
# =============================================================================

print("""
================================================================================
PART 2: WHY CONNECTED CORRELATORS MATTER
================================================================================

The CONNECTED correlator is:

    <T_μν(x) T_ρσ(x')>_c = <T_μν(x) T_ρσ(x')> - <T_μν(x)><T_ρσ(x')>

The subtraction removes the "incoherent" contribution where the two
points are statistically independent.

PHYSICAL MEANING:
-----------------
If matter at x and x' is UNCORRELATED (like two independent dust clouds):
    <j(x)·j(x')> = <j(x)>·<j(x')>
    → Connected correlator = 0
    → Standard GR/Poisson applies

If matter at x and x' is CORRELATED (like a coherent rotating disk):
    <j(x)·j(x')> ≠ <j(x)>·<j(x')>
    → Connected correlator ≠ 0
    → Additional gravitational effect

The connected correlator measures HOW MUCH the matter distribution
deviates from statistical independence.

ANALOGY: Laser vs Lightbulb
---------------------------
- Lightbulb: photons emitted independently → <E(x)E(x')>_c ≈ 0
- Laser: photons phase-correlated → <E(x)E(x')>_c >> 0

For gravity:
- Random dust: mass elements independent → <j·j'>_c ≈ 0
- Rotating disk: mass elements phase-correlated → <j·j'>_c >> 0
""")

# =============================================================================
# PART 3: MICROSCOPIC ORIGIN - WHAT CREATES THE CORRELATION?
# =============================================================================

print("""
================================================================================
PART 3: MICROSCOPIC ORIGIN OF CURRENT-CURRENT CORRELATIONS
================================================================================

What PHYSICALLY creates the correlation <j(x)·j(x')>_c ≠ 0?

MECHANISM 1: Gravitational Self-Organization
---------------------------------------------
A rotating disk is NOT a random collection of particles. It's a
SELF-ORGANIZED structure where:

    1. Gravitational collapse creates angular momentum conservation
    2. Dissipation (gas cooling) creates a thin disk
    3. Circular orbits mean v(x) || v(x') for nearby points

The correlation arises because GRAVITY ITSELF organized the matter
into a coherent rotating structure!

This is a BOOTSTRAP: gravity creates coherence, coherence affects gravity.


MECHANISM 2: Dynamical Phase Locking
------------------------------------
Consider two mass elements at radii r₁ and r₂ in a disk:

    Ω(r₁) = V(r₁)/r₁   (angular velocity at r₁)
    Ω(r₂) = V(r₂)/r₂   (angular velocity at r₂)

If the rotation curve is FLAT (V = const):
    Ω(r) ∝ 1/r  (differential rotation)
    
But if there's a coherence mechanism:
    Ω(r₁) ≈ Ω(r₂)  over some scale ξ
    → Velocities stay aligned → <j·j'>_c large

The COHERENCE SCALE ξ determines how far this phase-locking extends.


MECHANISM 3: Collective Modes
-----------------------------
A disk supports collective oscillations (density waves, spiral arms).
These are COHERENT excitations where many particles move together.

    ρ(x,t) = ρ₀(x) + δρ(x) e^{i(k·x - ωt)}
    v(x,t) = v₀(x) + δv(x) e^{i(k·x - ωt)}

In a collective mode:
    δv(x) and δv(x') are PHASE-CORRELATED
    → <j(x)·j(x')>_c ≠ 0 even for |x-x'| >> mean free path

The disk acts like a COHERENT MEDIUM, not a collection of particles.
""")

# =============================================================================
# PART 4: THE DECOHERENCE MECHANISMS
# =============================================================================

print("""
================================================================================
PART 4: WHAT DESTROYS CURRENT-CURRENT CORRELATIONS?
================================================================================

If correlations ENHANCE gravity, what DESTROYS them?

DECOHERENCE MECHANISM 1: Velocity Dispersion (σ)
-------------------------------------------------
Random thermal motions scramble the phase relationship:

    v_total = v_circular + v_random

If σ >> v_circular:
    <j(x)·j(x')> ≈ <ρ(x)><ρ(x')> × <v_random·v'_random>
                 ≈ 0  (random vectors don't correlate)

This explains why:
    - Hot bulges (high σ) show less enhancement
    - Cold disks (low σ) show more enhancement
    - Clusters (very high σ) need different treatment


DECOHERENCE MECHANISM 2: Counter-Rotation
------------------------------------------
If half the mass rotates one way and half the other:

    j_total = j_co + j_counter = ρv_co + ρ(-v_co) ≈ 0

More precisely:
    <j·j'>_c = f_co² × <j_co·j_co'> + f_counter² × <j_counter·j_counter'>
             + 2 f_co f_counter × <j_co·j_counter'>
             
The cross-term is NEGATIVE (anti-aligned velocities), reducing total coherence.


DECOHERENCE MECHANISM 3: Spatial Separation
--------------------------------------------
Correlations decay with distance:

    <j(x)·j(x')>_c ∝ W(|x-x'|/ξ) → 0 as |x-x'| → ∞

The coherence scale ξ sets the range of correlation.
Beyond ξ, the system looks "incoherent" and standard GR applies.


DECOHERENCE MECHANISM 4: High Acceleration (g >> g†)
-----------------------------------------------------
At high acceleration, orbital periods are SHORT:

    T_orbit = 2π/Ω ∝ 1/√g

Fast orbital evolution → rapid phase mixing → decoherence

This is why the enhancement function h(g) → 0 as g → ∞.
""")

# =============================================================================
# PART 5: QUANTUM VS CLASSICAL COHERENCE
# =============================================================================

print("""
================================================================================
PART 5: QUANTUM VS CLASSICAL - WHAT KIND OF COHERENCE?
================================================================================

Is this QUANTUM coherence or CLASSICAL correlation?

ARGUMENT FOR CLASSICAL:
-----------------------
The correlations we observe are in MACROSCOPIC quantities:
    - Stellar velocities (measured by Doppler)
    - Gas rotation curves (measured by 21cm)
    - Velocity dispersions (measured by line widths)

These are all CLASSICAL observables. The correlations arise from:
    - Gravitational dynamics (Newtonian/GR)
    - Hydrodynamics (gas flows)
    - Collisionless dynamics (stellar orbits)

No quantum mechanics required to explain the correlations themselves.


ARGUMENT FOR QUANTUM (OR QUANTUM-LIKE):
----------------------------------------
The EFFECT of correlations on gravity might be quantum-like:

1. SUPERPOSITION: In quantum mechanics, correlated states interfere.
   Similarly, correlated mass currents might "interfere" gravitationally.

2. MEASUREMENT: The gravitational field "measures" the stress-energy.
   Correlated sources might couple differently than uncorrelated ones.

3. PATH INTEGRAL: In quantum gravity, you sum over paths.
   Correlated sources might have different path weights.


THE KEY QUESTION:
-----------------
Does GR (or its quantum extension) treat correlated sources differently
from uncorrelated sources?

STANDARD GR: No. T_μν is a classical field. Correlations don't matter.
             The field equation is LINEAR in T_μν.

MODIFIED GR: Possibly. If there's a NON-LINEAR coupling to T_μν,
             correlations could matter.

Σ-GRAVITY HYPOTHESIS:
    The effective gravitational coupling depends on the COHERENCE STATE
    of the source, measured by the connected correlator <T T>_c.

This is NOT standard GR, but it's also NOT necessarily quantum.
It's a statement about how gravity responds to ORGANIZED vs RANDOM matter.
""")

# =============================================================================
# PART 6: THE DEEP QUESTION - WHY WOULD GRAVITY CARE?
# =============================================================================

print("""
================================================================================
PART 6: THE DEEP QUESTION - WHY WOULD GRAVITY CARE ABOUT COHERENCE?
================================================================================

The fundamental puzzle: In standard GR, gravity couples to T_μν locally.
Why would CORRELATIONS between different points matter?

POSSIBILITY 1: Non-Local Gravity
---------------------------------
Maybe gravity is fundamentally non-local:

    G_μν(x) = 8πG ∫ K(x,x') T_μν(x') d⁴x'

where K(x,x') is a non-local kernel. If K depends on the CORRELATION
structure of T, you naturally get coherence effects.

This is like how the refractive index of a medium depends on the
COLLECTIVE response of atoms, not just individual atoms.


POSSIBILITY 2: Emergent Gravity
-------------------------------
If gravity is EMERGENT (like thermodynamics from statistical mechanics),
then:
    - "Temperature" = some measure of gravitational field
    - "Entropy" = some measure of matter organization
    
Coherent matter has LOWER entropy than random matter.
Lower entropy → different thermodynamic response → different gravity.

This connects to Verlinde's entropic gravity and Jacobson's thermodynamic
derivation of Einstein's equations.


POSSIBILITY 3: Graviton Self-Interaction
-----------------------------------------
In quantum gravity, gravitons interact with each other.
If the source is coherent, it might emit gravitons that:
    - Are phase-correlated
    - Interfere constructively
    - Produce stronger effective field

This is analogous to stimulated emission in lasers.


POSSIBILITY 4: Modified Stress-Energy Coupling
----------------------------------------------
Maybe the coupling isn't G_μν = 8πG T_μν but:

    G_μν = 8πG T_μν + f(<T T>_c)

where f is some functional of the connected correlator.

This is what Σ-Gravity effectively proposes:
    g_eff = g_N × Σ
    Σ = 1 + A × h(g) × C_j

where C_j encodes the current-current correlation.
""")

# =============================================================================
# PART 7: TESTABLE CONSEQUENCES
# =============================================================================

print("""
================================================================================
PART 7: TESTABLE CONSEQUENCES OF THE CORRELATOR PICTURE
================================================================================

If gravity really responds to current-current correlations, we predict:

1. VELOCITY STRUCTURE MATTERS
   Two galaxies with same mass profile but different velocity structure
   should have different gravitational fields.
   
   TEST: Compare f_DM for:
   - Rotating disk vs pressure-supported spheroid (same M, R)
   - Co-rotating vs counter-rotating disks
   - Cold vs hot stellar populations

2. COHERENCE SCALE MATTERS
   The range of correlation ξ should affect the enhancement.
   
   TEST: Compare f_DM for:
   - Compact vs extended disks (same M)
   - Galaxies with different R_d/R_max ratios

3. DYNAMICAL STATE MATTERS
   Disturbed/merging systems should have lower coherence.
   
   TEST: Compare f_DM for:
   - Isolated vs interacting galaxies
   - Relaxed vs disturbed systems
   - Pre-merger vs post-merger

4. COLLECTIVE MODES MATTER
   Galaxies with strong spiral structure (collective modes) might
   show different coherence than smooth disks.
   
   TEST: Compare f_DM for:
   - Grand-design spirals vs flocculent spirals
   - Barred vs unbarred galaxies

5. REDSHIFT EVOLUTION
   High-z galaxies are more turbulent (higher σ/V) → lower coherence.
   
   TEST: f_DM(z) should decrease with z (observed in KMOS3D!)
""")

# =============================================================================
# PART 8: MATHEMATICAL FORMULATION
# =============================================================================

print("""
================================================================================
PART 8: MATHEMATICAL FORMULATION
================================================================================

Let's write down the correlator explicitly.

DEFINITION:
-----------
The current density is:
    j^i(x) = ρ(x) v^i(x)

The connected current-current correlator is:
    C^{ij}(x,x') = <j^i(x) j^j(x')> - <j^i(x)><j^j(x')>

For a statistically homogeneous system, this depends only on separation:
    C^{ij}(x,x') = C^{ij}(|x-x'|)

DECOMPOSITION:
--------------
We can decompose into density and velocity correlations:

    <j^i j'^j> = <ρ v^i ρ' v'^j>
               = <ρρ'><v^i v'^j> + <ρρ'>_c <v^i><v'^j> + <ρ><ρ'><v^i v'^j>_c
                 + <ρρ'>_c <v^i v'^j>_c + ...

For a rotating disk with small density fluctuations:
    <ρρ'>_c << <ρ><ρ'>
    
So the dominant term is:
    C^{ij} ≈ <ρ><ρ'> × <v^i v'^j>_c

The VELOCITY CORRELATION is the key quantity.

VELOCITY CORRELATION IN A DISK:
-------------------------------
For circular rotation with velocity V(r) at radius r:
    v = V(r) ê_φ  (azimuthal unit vector)

The velocity correlation between points at (r,φ) and (r',φ') is:
    <v·v'> = V(r) V(r') cos(φ - φ')

If the disk is axisymmetric, averaging over φ:
    <v·v'>_φ = V(r) V(r') × (1/2π) ∫ cos(φ-φ') dφ'
             = 0  (for different rings)

But LOCALLY (same ring, nearby points):
    <v·v'> = V² cos(Δφ) ≈ V² (1 - Δφ²/2)

The correlation is POSITIVE for nearby points on the same orbit.

COHERENCE SCALE:
----------------
The coherence scale ξ determines how far the correlation extends:
    <v(r)·v(r')>_c ∝ exp(-|r-r'|²/2ξ²)

For a disk:
    ξ ~ R_d / (2π)  (one azimuthal wavelength)

This is the scale over which velocities remain correlated.
""")

# =============================================================================
# PART 9: CONNECTION TO FIELD THEORY
# =============================================================================

print("""
================================================================================
PART 9: CONNECTION TO FIELD THEORY
================================================================================

In quantum field theory, connected correlators appear naturally.

TWO-POINT FUNCTION:
-------------------
The propagator (two-point function) is:
    G(x,x') = <φ(x) φ(x')>

For a free field:
    G(x,x') = <0|φ(x) φ(x')|0>  (vacuum expectation)

For an interacting field or finite-temperature system:
    G(x,x') = <φ(x) φ(x')>_β  (thermal average)

The CONNECTED correlator removes the disconnected part:
    G_c(x,x') = G(x,x') - <φ(x)><φ(x')>

STRESS-ENERGY CORRELATOR:
-------------------------
In quantum gravity (or semiclassical gravity), the stress-energy
tensor has quantum fluctuations:

    <T_μν(x) T_ρσ(x')>_c ≠ 0

These fluctuations can affect the gravitational field through:
    1. Backreaction on the metric
    2. Noise in the Einstein equations
    3. Modified dispersion relations

ANALOGY TO CASIMIR EFFECT:
--------------------------
The Casimir effect arises from CORRELATED vacuum fluctuations:
    <E(x) E(x')>_c ≠ 0 between conducting plates

This produces a FORCE between the plates.

Similarly, correlated stress-energy fluctuations might produce
ADDITIONAL gravitational effects beyond the mean-field <T_μν>.

THE Σ-GRAVITY INTERPRETATION:
-----------------------------
Σ-Gravity can be viewed as a MEAN-FIELD + FLUCTUATION theory:

    g_eff = g_N × (1 + δg/g_N)

where δg/g_N depends on the connected correlator:
    δg/g_N ∝ A × h(g) × <j·j'>_c / <j>²

This is analogous to how the dielectric constant of a medium
depends on the polarization correlations of atoms.
""")

# =============================================================================
# PART 10: SUMMARY AND OPEN QUESTIONS
# =============================================================================

print("""
================================================================================
PART 10: SUMMARY AND OPEN QUESTIONS
================================================================================

WHAT WE UNDERSTAND:
-------------------
1. The current-current correlator <j·j'>_c measures velocity alignment
2. It's positive for co-rotating, ordered systems
3. It's suppressed by dispersion, counter-rotation, and spatial separation
4. It correctly predicts the observed f_DM suppression in CR galaxies

WHAT WE DON'T FULLY UNDERSTAND:
-------------------------------
1. WHY does gravity respond to correlations?
   - Is it non-local gravity?
   - Emergent gravity?
   - Quantum gravity effect?
   - Modified stress-energy coupling?

2. What is the EXACT form of the coupling?
   - Is Σ = 1 + A × h(g) × C_j the right formula?
   - What determines A, h(g), and ξ?

3. How does this connect to fundamental physics?
   - Teleparallel gravity?
   - f(T) modifications?
   - Quantum corrections to GR?

NEXT STEPS FOR INVESTIGATION:
-----------------------------
1. Derive the correlator coupling from a Lagrangian
2. Connect to teleparallel gravity framework
3. Compute quantum corrections to stress-energy correlator
4. Test predictions for velocity-structure dependence
5. Look for signatures in gravitational lensing (different from dynamics?)

THE BOTTOM LINE:
----------------
The current-current correlator provides a PREDICTIVE framework that:
- Explains counter-rotation suppression (confirmed)
- Explains v/σ dependence (consistent with data)
- Makes unique predictions testable against ΛCDM and MOND

Whether this is "fundamental" or "emergent" remains to be understood,
but the PHENOMENOLOGY is working.
""")

print("=" * 100)
print("END OF FIRST PRINCIPLES INVESTIGATION")
print("=" * 100)

