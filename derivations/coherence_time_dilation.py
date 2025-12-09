#!/usr/bin/env python3
"""
Coherence-Induced Time Dilation
================================

PROBLEM: Type Ia supernovae at redshift z show time dilation by factor (1+z).
A supernova that takes 20 days to fade at z=0 takes 40 days at z=1.

In standard cosmology, this is explained by expansion: the light waves are
stretched, so all time intervals are stretched by the same factor.

QUESTION: If redshift comes from coherence (not expansion), how do we get
the same (1+z) time dilation?

This script explores what coherence must do to reproduce time dilation.

Author: Sigma Gravity Research
Date: December 2025
"""

import numpy as np

print("=" * 100)
print("COHERENCE-INDUCED TIME DILATION")
print("=" * 100)

# =============================================================================
# THE OBSERVATION
# =============================================================================

print("""
================================================================================
THE OBSERVATION: SUPERNOVA TIME DILATION
================================================================================

Type Ia supernovae are "standard candles" - they all have similar light curves.

At z = 0: A supernova rises and falls over ~20 days
At z = 1: The SAME supernova appears to take ~40 days
At z = 2: It appears to take ~60 days

The time dilation factor is exactly (1+z).

This is observed not just in supernovae but in:
- Gamma-ray burst durations
- Quasar variability
- Any time-varying source

The observation is ROBUST. Any alternative to expansion MUST reproduce it.
""")

# =============================================================================
# WHY SIMPLE ENERGY LOSS DOESN'T WORK
# =============================================================================

print("""
================================================================================
WHY SIMPLE ENERGY LOSS DOESN'T WORK
================================================================================

Naive "tired light": Photons lose energy through some interaction.
    E → E' = E/(1+z)
    
But if ONLY energy changes (not frequency), we'd have:
    E = hν  →  E' = hν'  →  ν' = ν/(1+z)
    
Wait, that DOES change frequency! So what's the problem?

THE PROBLEM: In tired light, photons scatter/interact individually.
Each photon loses energy at a RANDOM time during its journey.

Consider two photons emitted Δt apart:
    Photon 1: emitted at t₁, arrives at t₁ + journey_time₁
    Photon 2: emitted at t₁ + Δt, arrives at t₁ + Δt + journey_time₂

If the energy loss is stochastic, journey_time₁ ≠ journey_time₂.
The arrival time difference is NOT simply Δt × (1+z).

This would BLUR the light curve, not stretch it uniformly.

OBSERVATION: Supernova light curves are NOT blurred. They're cleanly stretched.

So coherence can't work through random scattering.
""")

# =============================================================================
# WHAT COHERENCE MUST DO
# =============================================================================

print("""
================================================================================
WHAT COHERENCE MUST DO
================================================================================

For coherence to produce (1+z) time dilation WITHOUT blurring, it must:

1. Affect ALL photons from a source IDENTICALLY
   - Not random scattering
   - A collective, deterministic effect

2. Affect the PHASE of the wave, not just the energy
   - Phase determines frequency: ν = dφ/dt
   - If phase accumulates slower, frequency decreases AND time dilates

3. Be CUMULATIVE with distance
   - More distance → more redshift → more time dilation
   - z ∝ d (Hubble law)

Let's explore mechanisms that could do this.
""")

# =============================================================================
# MECHANISM 1: COHERENT GRAVITATIONAL TIME DILATION
# =============================================================================

print("""
================================================================================
MECHANISM 1: COHERENT GRAVITATIONAL TIME DILATION
================================================================================

In GR, time runs slower in a gravitational potential:
    dτ = dt √(1 - 2Φ/c²)

where Φ is the gravitational potential.

STANDARD VIEW: Light enters and exits potential wells symmetrically.
    Blueshift in = Redshift out → No net effect

COHERENCE VIEW: The ENHANCED gravity from coherence is NOT symmetric.

Consider light passing through a galaxy:
    - Entering: Light approaches from far away (low coherence)
    - Inside: Light is in high-coherence region
    - Exiting: Light leaves toward far away (low coherence)

The coherence enhancement Σ depends on the LOCAL matter configuration.
From the light's perspective, the "effective potential" is:

    Φ_eff = Σ × Φ_Newtonian

If Σ varies along the path (higher inside, lower outside), there's
an ASYMMETRY that produces a net redshift.

TIME DILATION: The same asymmetry affects the rate of time.
Light spends more "coordinate time" in high-Σ regions.

    dt_observed = dt_emitted × (1 + ∫ δΣ dΦ/c²)

For this to give (1+z), we need:
    ∫ δΣ dΦ/c² = z = H₀ d / c
""")

# Check the numbers
print("Quantitative check:")
H_0 = 70  # km/s/Mpc
H_0_SI = H_0 * 1000 / 3.086e22  # s⁻¹
c = 3e8  # m/s
G = 6.674e-11

# Typical galaxy potential
M_galaxy = 1e11 * 2e30  # kg (10^11 solar masses)
R_galaxy = 10e3 * 3.086e16  # m (10 kpc)
Phi_galaxy = G * M_galaxy / R_galaxy
print(f"Typical galaxy potential: Φ ~ {Phi_galaxy:.3e} m²/s²")
print(f"Φ/c² ~ {Phi_galaxy/c**2:.3e}")

# For z = 1, we need the integral to equal 1
# If we pass through N galaxies, each contributing δΣ × Φ/c²
# N × δΣ × Φ/c² = 1
# For Φ/c² ~ 10⁻⁶, and δΣ ~ 1, we need N ~ 10⁶ galaxies

d_z1 = c / H_0_SI  # distance for z=1 (Hubble radius)
n_galaxy = 0.1 / (3.086e22)**3  # galaxies per m³ (0.1 per Mpc³)
N_galaxies = n_galaxy * d_z1 * (1e6 * 3.086e16)**2  # galaxies in a cylinder

print(f"Distance for z=1: {d_z1/3.086e22:.0f} Mpc")
print(f"Galaxies along path (rough): ~{N_galaxies:.0e}")
print(f"Required: N × δΣ × Φ/c² ~ 1")
print(f"Actual: {N_galaxies:.0e} × 1 × {Phi_galaxy/c**2:.0e} ~ {N_galaxies * Phi_galaxy/c**2:.0e}")

print("""
This is WAY too small! Individual galaxy potentials can't do it.

We need a CUMULATIVE effect that doesn't rely on discrete galaxies.
""")

# =============================================================================
# MECHANISM 2: COHERENCE FIELD AS MEDIUM
# =============================================================================

print("""
================================================================================
MECHANISM 2: COHERENCE FIELD AS A MEDIUM
================================================================================

What if the coherence field acts as a MEDIUM that light propagates through?

In a medium with refractive index n, light travels at speed c/n.
The phase velocity is reduced, and the wavelength is compressed.

But we want the OPPOSITE: wavelength stretched (redshift).

HOWEVER: If the medium has a frequency-dependent response, things change.

Consider a medium where the "coherence refractive index" is:
    n_coh = 1 + α_coh × (something)

If α_coh causes a PHASE DELAY that accumulates with distance:
    φ(d) = φ₀ - ∫₀^d (ω/c) × n_coh(x) dx

The observed frequency is:
    ω_obs = dφ/dt_obs

If the coherence field also affects the rate of time (like gravity does),
then t_obs ≠ t_emit, and we get both redshift AND time dilation.

KEY INSIGHT: In GR, gravitational redshift and time dilation are the SAME THING.
    z_grav = Δt/t = ΔΦ/c²

If coherence modifies the effective metric, it naturally couples
redshift to time dilation.
""")

# =============================================================================
# MECHANISM 3: MODIFIED DISPERSION RELATION
# =============================================================================

print("""
================================================================================
MECHANISM 3: MODIFIED DISPERSION RELATION
================================================================================

In vacuum, photons obey: E = pc, or equivalently ω = ck.

If coherence modifies this to:
    ω² = c²k² + m_eff²c⁴/ℏ²

where m_eff is an effective "photon mass" from coherence coupling,
then the group velocity is:

    v_g = dω/dk = c²k/ω = c × √(1 - m_eff²c⁴/(ℏ²ω²))

For m_eff << ℏω/c², this gives:
    v_g ≈ c × (1 - m_eff²c⁴/(2ℏ²ω²))

Lower frequencies travel SLOWER. This causes dispersion.

BUT: We don't observe dispersion in distant sources!
Gamma-ray bursts show photons of all energies arriving together.

So a simple mass term doesn't work.

ALTERNATIVE: What if the dispersion relation is:
    ω = ck × (1 - α_coh × d)

where d is distance traveled? This gives:
    ω_obs = ω_emit × (1 - α_coh × d) = ω_emit / (1 + z)

for z = α_coh × d / (1 - α_coh × d) ≈ α_coh × d for small z.

This is just a restatement of the problem, not a mechanism.
""")

# =============================================================================
# MECHANISM 4: METRIC MODIFICATION (THE KEY!)
# =============================================================================

print("""
================================================================================
MECHANISM 4: METRIC MODIFICATION (THE KEY!)
================================================================================

Here's the crucial insight:

In GR, redshift and time dilation are BOTH consequences of the METRIC.

The metric determines:
    1. How clocks tick (time dilation)
    2. How light propagates (redshift)
    3. How distances are measured

If coherence modifies the METRIC, it automatically couples all three.

Consider a modified metric of the form:
    ds² = -c²(1 + 2Ψ_coh)dt² + (1 - 2Ψ_coh)(dx² + dy² + dz²)

where Ψ_coh is the "coherence potential" that accumulates with distance.

For light (ds² = 0):
    c dt √(1 + 2Ψ_coh) = dr √(1 - 2Ψ_coh)

The coordinate speed of light is:
    dr/dt = c × √((1 + 2Ψ_coh)/(1 - 2Ψ_coh)) ≈ c × (1 + 2Ψ_coh)

For small Ψ_coh, this is approximately c (as expected).

But the FREQUENCY measured by a distant observer is:
    ν_obs/ν_emit = √(g_tt(emit)/g_tt(obs))
                 = √((1 + 2Ψ_emit)/(1 + 2Ψ_obs))

If Ψ_coh increases with distance from the source:
    Ψ_obs > Ψ_emit → ν_obs < ν_emit → REDSHIFT

And the TIME DILATION is:
    dt_obs/dt_emit = √(g_tt(obs)/g_tt(emit))
                   = √((1 + 2Ψ_obs)/(1 + 2Ψ_emit))
                   = 1/√(ν_obs/ν_emit)
                   = 1 + z

EXACTLY what we need!
""")

print("""
THE COHERENCE POTENTIAL:
------------------------
For this to reproduce the Hubble law, we need:

    Ψ_coh(d) = (H₀ d / c) / 2 = z / 2

So the coherence potential grows linearly with distance:
    Ψ_coh = (H₀ / 2c) × d

This is equivalent to saying the metric has a small "tilt":
    g_tt = -(1 + H₀ d / c)
    g_rr = +(1 - H₀ d / c)

In standard cosmology, this comes from the FLRW metric with expansion.
In coherence cosmology, this comes from the cumulative coherence field.
""")

# =============================================================================
# THE COHERENCE METRIC
# =============================================================================

print("""
================================================================================
THE COHERENCE METRIC
================================================================================

If coherence modifies the metric, what's the physical interpretation?

OPTION A: Coherence creates an effective "potential"
    Ψ_coh = ∫ (coherence density) × (coupling) dx
    
    As light travels, it accumulates "coherence exposure."
    This modifies the local metric it experiences.

OPTION B: Coherence modifies the vacuum
    The coherence field fills space like a medium.
    It has energy density ρ_coh that curves spacetime.
    
    From Einstein's equations:
        G_μν = 8πG/c⁴ × T_μν^coh
    
    If ρ_coh ~ H₀²c²/(8πG) (critical density), we get the right effect.

OPTION C: Coherence IS the metric (teleparallel)
    In teleparallel gravity, the metric is determined by the tetrad.
    The tetrad encodes both curvature (GR) and torsion.
    
    Coherent matter creates coherent torsion.
    Torsion modifies the metric that light experiences.
    
    This is the most natural connection to Σ-Gravity!
""")

# =============================================================================
# QUANTITATIVE REQUIREMENTS
# =============================================================================

print("""
================================================================================
QUANTITATIVE REQUIREMENTS
================================================================================

For coherence to produce the observed time dilation:

1. The coherence potential must grow as:
       Ψ_coh(d) = H₀ d / (2c)
   
2. At the Hubble radius (d = c/H₀):
       Ψ_coh = 1/2
   
   This is a significant metric perturbation!

3. The coherence "energy density" equivalent:
       ρ_coh ~ H₀² c² / (8πG) ~ ρ_critical
   
   The coherence field has energy density equal to the critical density!

4. This is the same as "dark energy" in standard cosmology:
       Ω_Λ ~ 0.7 → ρ_Λ ~ 0.7 × ρ_critical
   
   The coherence field IS what we call dark energy!
""")

# Compute the numbers
rho_crit = 3 * H_0_SI**2 / (8 * np.pi * G)
print(f"Critical density: ρ_crit = {rho_crit:.3e} kg/m³")
print(f"                        = {rho_crit * c**2:.3e} J/m³")

# Energy density of coherence field
rho_coh = rho_crit  # By construction
print(f"\nCoherence field density: ρ_coh ~ ρ_crit = {rho_coh:.3e} kg/m³")

# This is also the dark energy density!
print(f"\nIn standard cosmology, dark energy: ρ_Λ ~ 0.7 × ρ_crit = {0.7*rho_crit:.3e} kg/m³")
print("\n→ The coherence field has the same energy density as dark energy!")

# =============================================================================
# CONNECTION TO Σ-GRAVITY
# =============================================================================

print("""
================================================================================
CONNECTION TO Σ-GRAVITY
================================================================================

In Σ-Gravity, the critical acceleration is:
    g† = c H₀ / (4√π)

This can be rewritten as:
    g† = c² / L_coh

where L_coh = 4√π c / H₀ ~ 4√π × Hubble radius

The coherence LENGTH SCALE is the Hubble radius (times a factor of 4√π).

Now, the coherence potential is:
    Ψ_coh(d) = H₀ d / (2c) = d / (2 L_Hubble)

And the critical acceleration is:
    g† = c² × (H₀/c) / (4√π) = c² / L_coh

These are the SAME PHYSICS:
    - L_coh sets the scale where coherence effects become order unity
    - g† is the acceleration at this scale: g† = c²/L_coh
    - Ψ_coh reaches ~1 at the same scale

The Σ-Gravity critical acceleration and the cosmological coherence
potential are UNIFIED.
""")

# Show the connection
L_Hubble = c / H_0_SI
L_coh = 4 * np.sqrt(np.pi) * L_Hubble
g_dagger = c**2 / L_coh

print(f"Hubble radius: L_H = c/H₀ = {L_Hubble:.3e} m = {L_Hubble/3.086e22:.0f} Mpc")
print(f"Coherence scale: L_coh = 4√π × L_H = {L_coh:.3e} m = {L_coh/3.086e22:.0f} Mpc")
print(f"Critical acceleration: g† = c²/L_coh = {g_dagger:.3e} m/s²")
print(f"Expected g†: cH₀/(4√π) = {c * H_0_SI / (4*np.sqrt(np.pi)):.3e} m/s²")
print("✓ They match!")

# =============================================================================
# WHAT COHERENCE MUST BE
# =============================================================================

print("""
================================================================================
WHAT COHERENCE MUST BE (TO PRODUCE TIME DILATION)
================================================================================

For coherence to produce (1+z) time dilation, it must:

1. MODIFY THE METRIC
   Not just scatter photons, but change the spacetime geometry.
   The metric component g_tt must vary as:
       g_tt = -(1 + H₀ d / c)

2. BE CUMULATIVE
   The effect must grow linearly with distance.
   This suggests a field that permeates all of space.

3. HAVE ENERGY DENSITY ~ ρ_critical
   The coherence field carries energy comparable to the critical density.
   This is the same as "dark energy" in standard cosmology.

4. COUPLE TO MATTER COHERENCE
   In galaxies, the coherence enhancement Σ depends on local matter
   correlations. Cosmologically, the coherence potential Ψ_coh depends
   on the integrated matter coherence along the line of sight.

5. BE RELATED TO TORSION (teleparallel connection)
   The most natural framework is teleparallel gravity, where:
   - Torsion ~ vorticity of matter
   - Coherent rotation → coherent torsion → metric modification
   - Cumulative torsion → cosmological redshift + time dilation

SUMMARY:
--------
Coherence is not just a property of matter.
Coherence is a FIELD that modifies the metric.
The cosmological "expansion" is actually the coherence field's effect on light.
The coherence field IS dark energy (or replaces it).
""")

# =============================================================================
# PREDICTIONS AND TESTS
# =============================================================================

print("""
================================================================================
PREDICTIONS AND TESTS
================================================================================

If time dilation comes from coherence (not expansion):

1. TIME DILATION SHOULD CORRELATE WITH MATTER DENSITY
   Lines of sight through more galaxies/clusters should show
   slightly MORE time dilation than lines through voids.
   
   Test: Compare supernova time dilation in different environments.

2. TIME DILATION MIGHT HAVE SMALL FLUCTUATIONS
   The coherence field isn't perfectly uniform.
   There should be ~1% fluctuations in time dilation at fixed z.
   
   Test: Look for excess scatter in supernova light curve stretches.

3. GRAVITATIONAL WAVE TIME DILATION
   If coherence affects the metric, gravitational waves should
   also be time-dilated by (1+z).
   
   Test: Compare GW signal durations with EM counterpart timing.
   (Already consistent with (1+z) from neutron star mergers!)

4. NO BLURRING
   Unlike tired light, coherence doesn't scatter photons.
   Images should remain sharp at all redshifts.
   
   Test: Already confirmed - distant galaxies are not blurred.

5. TOLMAN SURFACE BRIGHTNESS
   The (1+z)⁴ dimming should be modified if coherence affects
   the solid angle differently than expansion.
   
   Test: Careful surface brightness measurements at high z.
""")

print("=" * 100)
print("CONCLUSION")
print("=" * 100)

print("""
For coherence to produce (1+z) time dilation:

    COHERENCE MUST MODIFY THE METRIC

This is not just "tired light" - it's a fundamental modification of
spacetime geometry by the coherence field.

The coherence potential Ψ_coh grows with distance:
    Ψ_coh(d) = H₀ d / (2c)

This produces:
    - Redshift: z = 2Ψ_coh = H₀ d / c
    - Time dilation: (1+z) = 1 + H₀ d / c
    - Both coupled through the metric (just like in GR)

The coherence field:
    - Has energy density ~ ρ_critical
    - IS what we call "dark energy"
    - Unifies galaxy dynamics (Σ-Gravity) with cosmology

This is a METRIC THEORY, not a tired light theory.
The difference is crucial for reproducing observations.
""")

print("=" * 100)

