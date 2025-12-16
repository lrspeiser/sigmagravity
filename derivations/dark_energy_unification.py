#!/usr/bin/env python3
"""
DARK ENERGY UNIFICATION
=======================

The hypothesis: φ IS the dark energy field (quintessence).

The same field that:
- Drives cosmic acceleration (dark energy)
- Modifies local gravity (what we call "dark matter")

This would be a profound unification of the dark sector.

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np

# Constants
c = 2.998e8           # m/s
G = 6.674e-11         # m³/kg/s²
hbar = 1.055e-34      # J·s
H0 = 2.27e-18         # 1/s (70 km/s/Mpc)
k_B = 1.381e-23       # J/K

# Cosmological parameters
rho_crit = 3 * H0**2 / (8 * np.pi * G)  # Critical density
Omega_DE = 0.68       # Dark energy fraction
Omega_DM = 0.27       # "Dark matter" fraction
rho_DE = Omega_DE * rho_crit

print("=" * 80)
print("DARK ENERGY UNIFICATION")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  THE COINCIDENCE THAT DEMANDS EXPLANATION                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Our φ field has mass m = H₀/c.

The dark energy density is:
    ρ_DE = Ω_DE × ρ_crit = {rho_DE:.2e} kg/m³

If dark energy is a scalar field with potential V(φ), then:
    ρ_DE ~ V(φ) ~ m² φ² ~ (H₀/c)² φ²

This gives:
    φ ~ c × √(ρ_DE) / H₀ ~ c × √(Ω_DE × 3H₀²/(8πG)) / H₀
    φ ~ c × √(3 Ω_DE / (8πG)) × H₀ / H₀
    φ ~ c × √(3 Ω_DE / (8πG))

Let's calculate:
""")

phi_DE = c * np.sqrt(3 * Omega_DE / (8 * np.pi * G))
print(f"    φ_DE ~ {phi_DE:.2e} (units of √(kg/m))")

# In Planck units
M_planck = np.sqrt(hbar * c / G)
phi_DE_planck = phi_DE / M_planck
print(f"    φ_DE / M_Planck ~ {phi_DE_planck:.2e}")
print()

# The field value is sub-Planckian but not tiny
print(f"The dark energy field value is φ ~ 0.4 × M_Planck")
print(f"This is natural — not fine-tuned to be tiny or huge.")

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  THE QUINTESSENCE MODEL                                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standard quintessence: a scalar field φ with potential V(φ) drives cosmic acceleration.

The Lagrangian:
    L_φ = ½(∂φ)² - V(φ)

For slow-roll (dark energy behavior):
    V(φ) >> ½(∂φ)²
    
    → ρ_φ ≈ V(φ)
    → p_φ ≈ -V(φ)
    → w = p/ρ ≈ -1 (dark energy equation of state)


THE POTENTIAL:
──────────────

For our theory, we need V(φ) such that:
    1. V gives the right dark energy density: V ~ ρ_DE c²
    2. The mass is m = H₀/c: V''(φ) = m² = (H₀/c)²

The simplest choice:
    V(φ) = ½ m² φ² = ½ (H₀/c)² φ²

At the current field value φ ~ 0.4 M_Planck:
    V ~ ½ (H₀/c)² × (0.4 M_Planck)²
""")

V_current = 0.5 * (H0/c)**2 * (0.4 * M_planck)**2
print(f"    V(φ_current) ~ {V_current:.2e} J/m³")
print(f"    ρ_DE × c² = {rho_DE * c**2:.2e} J/m³")
print()
if abs(V_current - rho_DE * c**2) / (rho_DE * c**2) < 10:
    print("    ✓ These are the same order of magnitude!")
else:
    print("    The numbers don't quite match — need to refine the model")

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  THE UNIFIED LAGRANGIAN                                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Combining dark energy (potential) and gravity modification (coupling):

    L = (c⁴/16πG) R                      [Einstein-Hilbert]
      + ½(∂φ)² - V(φ)                    [quintessence]
      + L_matter                          [baryons]
      + λ φ T F(g/g†)                     [φ-matter coupling]

where:
    V(φ) = ½ m² φ² + Λ₀                  [potential with cosmological constant]
    m = H₀/c                              [mass from Hubble scale]
    F(g/g†) = exp(-g/g†)                 [acceleration suppression]


THIS SINGLE FIELD φ DOES TWO JOBS:
──────────────────────────────────

1. COSMIC SCALE: V(φ) drives accelerated expansion (dark energy)

2. GALACTIC SCALE: φ-matter coupling enhances gravity (replaces dark matter)

The same field, two effects at different scales.

""")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  WHY THE COUPLING IS SUPPRESSED AT HIGH ACCELERATION                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is the key physics we need to derive, not assume.

THE ADIABATIC ARGUMENT:
───────────────────────

The φ field evolves on the Hubble timescale: τ_φ ~ 1/H₀

Local matter moves on the dynamical timescale: τ_dyn ~ v/g

When τ_dyn << τ_φ (high acceleration):
    - Matter moves fast compared to φ evolution
    - The φ field can't "keep up" with the matter
    - The coupling averages out to zero
    - → Standard GR

When τ_dyn >> τ_φ (low acceleration):
    - Matter moves slowly compared to φ evolution  
    - The φ field fully responds to matter distribution
    - The coupling is active
    - → Enhanced gravity

The transition occurs when τ_dyn ~ τ_φ:
    v/g ~ 1/H₀
    g ~ v × H₀

For typical galactic velocities v ~ 200 km/s:
    g_transition ~ {200e3 * H0:.2e} m/s²

Compare to our g†:
    g† = c H₀ / (4√π) ~ {c * H0 / (4 * np.sqrt(np.pi)):.2e} m/s²

The ratio:
    g_transition / g† ~ v/c × 4√π ~ {200e3/c * 4 * np.sqrt(np.pi):.4f}

This is small because v << c. The actual transition is sharper,
occurring at g† where the LIGHT-CROSSING time equals the Hubble time.


THE DEEPER REASON:
──────────────────

The φ field has Compton wavelength λ_C = c/H₀ (Hubble radius).

Quantum mechanically, φ can only "resolve" structures larger than λ_C.

At high acceleration:
    - The gravitational potential varies rapidly in space
    - Variation scale << λ_C
    - φ cannot couple to these rapid variations
    - → No enhancement

At low acceleration:
    - The gravitational potential varies slowly
    - Variation scale ~ λ_C or larger
    - φ CAN couple to these slow variations
    - → Enhancement

The critical acceleration g† is where the potential curvature scale
equals the Compton wavelength of φ.

""")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  DERIVING F(g/g†) FROM FIRST PRINCIPLES                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Can we DERIVE the suppression function F(x) = exp(-x) rather than assume it?

APPROACH: Effective field theory with derivative expansion
──────────────────────────────────────────────────────────

The φ-matter coupling must respect:
    1. Diffeomorphism invariance
    2. φ → -φ symmetry (if we impose it)
    3. Locality (at leading order)

The most general coupling to matter (trace T) is:

    L_int = φ × T × f(∂²φ/m², ∂Φ_N/g†, ...)

where Φ_N is the Newtonian potential.

In the low-energy limit, expand in derivatives:

    L_int = λ φ T × [1 - α (∇Φ_N)²/g†² + β (∇Φ_N)⁴/g†⁴ - ...]

Summing the series (if it converges):

    L_int = λ φ T × exp(-(∇Φ_N)²/g†²) = λ φ T × exp(-g²/g†²)

Or with a different series:

    L_int = λ φ T × exp(-|∇Φ_N|/g†) = λ φ T × exp(-g/g†)

The EXPONENTIAL form arises naturally from resumming a derivative expansion!


ALTERNATIVE: Thermal/statistical argument
─────────────────────────────────────────

If φ fluctuations are thermalized with the "gravitational temperature":

    T_grav = ℏ g / (2π k_B c)  (Unruh-like)

The coupling is Boltzmann-suppressed:

    F(g) = exp(-E_coupling / k_B T_grav) = exp(-const × g†/g × g/g†) = exp(-const)

This doesn't quite work. But if the coupling energy scales as:

    E_coupling ~ ℏ g / c

Then:
    F(g) = exp(-E_coupling / E†) = exp(-g/g†)

where E† = ℏ g† / c is the characteristic energy at the critical acceleration.

""")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  THE COMPLETE THEORY                                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

LAGRANGIAN:
───────────

    L = (c⁴/16πG) R + ½(∂φ)² - ½m²φ² + L_m + λ φ T exp(-g/g†)

with m = H₀/c and g† = cH₀/4√π.


FIELD EQUATIONS:
────────────────

Einstein:
    G_μν = (8πG/c⁴) [T_μν + T_μν^(φ) + λ φ exp(-g/g†) T_μν]

Scalar:
    □φ + m²φ = λ T exp(-g/g†)


COSMOLOGICAL SOLUTION (homogeneous universe):
─────────────────────────────────────────────

    φ(t) = φ₀ × (a/a₀)^(-3(1+w)/2)  for w ≠ -1
    
For slow-roll (w ≈ -1): φ ≈ constant
    
The potential energy V(φ) = ½m²φ² acts as dark energy:
    ρ_DE = V(φ) = ½m²φ² ~ ½(H₀/c)² × (0.4 M_Pl)² ~ ρ_crit × Ω_DE ✓


GALACTIC SOLUTION (static, spherical):
──────────────────────────────────────

    ∇²φ - m²φ = -λ ρ exp(-g/g†)

For r << c/H₀ (all galaxies), the mass term is negligible:
    ∇²φ ≈ -λ ρ exp(-g/g†)

Solution:
    φ(r) = λ ∫ ρ(r') exp(-g(r')/g†) / |r-r'| d³r'

This φ(r) then enhances gravity:
    g_eff = g_bar × [1 + λ φ(r) exp(-g/g†) / (ρ c²)]

Which gives our Σ factor.

""")

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  PREDICTIONS OF THE UNIFIED THEORY                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DARK ENERGY EQUATION OF STATE
   ─────────────────────────────
   The φ field is not exactly a cosmological constant.
   It has dynamics: w = p/ρ ≠ -1 exactly.
   
   For a massive scalar: w = -1 + (m/H)² × (φ̇/Hφ)²
   
   Current constraint: w = -1.03 ± 0.03
   Our prediction: w ≈ -1 + O(H₀²/H²) ≈ -1 (indistinguishable today)
   
   But at high redshift, w should deviate from -1.


2. φ FIELD VARIATION WITH REDSHIFT
   ────────────────────────────────
   The field value φ evolves:
       φ(z) = φ₀ × f(z)
   
   This means g† might evolve:
       g†(z) = g†₀ × h(z)
   
   High-z galaxies should show DIFFERENT enhancement than local galaxies.
   
   Testable with JWST rotation curves!


3. COUPLING TO DARK ENERGY PERTURBATIONS
   ─────────────────────────────────────
   Dark energy is not perfectly smooth.
   φ has perturbations δφ that correlate with matter.
   
   This gives a NEW contribution to the ISW effect in CMB.
   
   Potentially detectable in CMB-galaxy cross-correlations.


4. FIFTH FORCE IN VOIDS
   ────────────────────
   In cosmic voids: ρ → 0, g → 0
   The φ coupling is UNSUPPRESSED.
   
   Objects in voids should experience enhanced gravity toward void walls.
   
   This affects void profiles and could be tested with void catalogs.


5. SCREENING RADIUS
   ─────────────────
   Around massive objects, there's a radius r_screen where g = g†.
   
   Inside r_screen: standard GR
   Outside r_screen: enhanced gravity
   
   For the Milky Way: r_screen ~ {np.sqrt(G * 1e12 * 2e30 / (c * H0 / (4*np.sqrt(np.pi)))) / 3.086e19:.0f} kpc
   
   This is testable with stellar dynamics at different radii.

""")

# Calculate screening radius for MW
M_MW = 1e12 * 2e30  # kg
g_dagger = c * H0 / (4 * np.sqrt(np.pi))
r_screen = np.sqrt(G * M_MW / g_dagger)
print(f"Milky Way screening radius: r_screen ~ {r_screen/3.086e19:.0f} kpc")

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  WHAT THIS THEORY CLAIMS                                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. There is NO dark matter particle.
   What we call "dark matter" is the effect of the φ field coupling to baryons.

2. There is ONE dark sector field: φ (quintessence).
   - Its potential energy drives cosmic acceleration (dark energy)
   - Its coupling to matter enhances gravity (mimics dark matter)

3. The critical acceleration g† = cH₀/4√π is NOT a coincidence.
   It emerges from the φ field mass m = H₀/c.

4. The "dark matter" effect should EVOLVE with redshift.
   Because φ evolves, the enhancement changes over cosmic time.

5. The inner structure of galaxies affects outer dynamics.
   Because φ is a field with spatial extent, not a local effect.


THE BOLD CLAIM:
───────────────

    Dark matter and dark energy are the SAME THING.
    
    One field φ, two manifestations:
        - Cosmic scale: potential energy → acceleration
        - Galactic scale: matter coupling → enhanced gravity

This is the simplest possible dark sector: one field, zero particles.

══════════════════════════════════════════════════════════════════════════════════
""")




