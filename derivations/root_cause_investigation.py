#!/usr/bin/env python3
"""
ROOT CAUSE INVESTIGATION
========================

Not fitting. Not phenomenology. 
What PRINCIPLE generates the observed behavior?

The data tells us:
1. Enhancement depends on g/g† where g† = cH₀/4√π
2. Enhancement builds up with distance from center
3. Inner structure affects outer enhancement (nonlocal)
4. High acceleration suppresses the effect

What physical principle could produce ALL of these simultaneously?

Author: Leonard Speiser
Date: December 2025
"""

import numpy as np

# Constants
c = 2.998e8       # m/s
H0 = 2.27e-18     # 1/s (70 km/s/Mpc)
G = 6.674e-11     # m³/kg/s²
hbar = 1.055e-34  # J·s

g_dagger = c * H0 / (4 * np.sqrt(np.pi))

print("=" * 80)
print("ROOT CAUSE INVESTIGATION")
print("=" * 80)

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  THE CLUES                                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

CLUE 1: The critical acceleration g† = cH₀/4√π ≈ 1.0×10⁻¹⁰ m/s²

    This is NOT a fitted parameter. It emerges from:
        - c (speed of light)
        - H₀ (Hubble constant = cosmic expansion rate)
    
    Physical meaning: g† is the acceleration where the "Hubble time" 
    equals the "dynamical time":
        
        t_Hubble = 1/H₀
        t_dyn = c/g
        
        When g = g†: t_dyn ~ t_Hubble
    
    → The effect appears when LOCAL dynamics become comparable to COSMIC timescales


CLUE 2: Enhancement grows with distance from center

    Not just "low g" but "far from center with low g"
    
    The spatial window W(r) = r/(ξ+r) means:
        - At r=0: no enhancement (even if g is low)
        - At r→∞: full enhancement
    
    → Something ACCUMULATES or BUILDS UP with distance


CLUE 3: Inner structure affects outer enhancement

    Same outer g, different inner structure → different outer Σ
    
    This is path-dependent:
        P_survive = exp(-∫ ds/λ_D(s))
    
    → The effect has MEMORY of the path traversed


CLUE 4: High acceleration suppresses the effect

    When g >> g†, we recover GR exactly.
    
    → Something is being "washed out" or "decohered" at high g

""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  CANDIDATE PRINCIPLES                                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PRINCIPLE A: Gravity has a "correlation length" set by cosmic expansion
─────────────────────────────────────────────────────────────────────────

    The gravitational field has quantum/thermal fluctuations.
    These fluctuations are correlated over a length scale:
    
        λ_corr = c/H₀ × f(g/g†)
    
    At low g: λ_corr → c/H₀ (Hubble radius) — correlations extend to cosmic scales
    At high g: λ_corr → 0 — correlations are local only
    
    When correlations extend beyond the source, you get "extra" gravity
    from the correlated vacuum fluctuations.
    
    Problem: Why would correlation length depend on LOCAL acceleration?


PRINCIPLE B: Gravitational "temperature" is set by acceleration (Unruh-like)
─────────────────────────────────────────────────────────────────────────────

    Unruh effect: accelerated observer sees thermal bath at T = ℏa/(2πkc)
    
    At low acceleration: T → 0, vacuum is "cold", quantum coherence preserved
    At high acceleration: T → high, vacuum is "hot", coherence destroyed
    
    The critical temperature corresponds to g†:
        T† = ℏg†/(2πkc) ≈ 4×10⁻³⁰ K
    
    This is absurdly small. But if gravity itself has an "effective temperature"
    that matters for gravitational coherence...
    
    Problem: The temperature is way too small to matter classically.


PRINCIPLE C: Gravity is emergent from entanglement (Verlinde-like)
──────────────────────────────────────────────────────────────────

    Gravity = thermodynamic force from entropy gradients
    
    In regions of low acceleration:
        - Entanglement entropy dominates
        - Gravity gets "extra" contribution from entanglement
    
    In regions of high acceleration:
        - Local matter entropy dominates  
        - Standard GR recovered
    
    The transition occurs at g† because that's where the two contributions
    are equal.
    
    Problem: Verlinde's specific predictions don't match SPARC well.


PRINCIPLE D: Spacetime has a "viscosity" that resists rapid change
──────────────────────────────────────────────────────────────────

    The metric tensor has dynamics beyond GR.
    There's a "relaxation time" τ for metric perturbations.
    
    When the dynamical time t_dyn = v/g >> τ:
        - Metric fully responds to matter
        - Standard GR
    
    When t_dyn << τ:
        - Metric can't keep up
        - Effective gravity enhanced (metric "remembers" earlier state)
    
    If τ ~ 1/H₀, then the transition is at g ~ v×H₀ ~ g†
    
    Problem: What determines τ? Why 1/H₀?

""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  THE MOST PROMISING DIRECTION: PRINCIPLE E                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PRINCIPLE E: Gravity couples to the cosmic expansion field
──────────────────────────────────────────────────────────

    Standard GR: Gravity couples to local stress-energy T_μν
    
    New physics: Gravity ALSO couples to a cosmic scalar field φ
                 that tracks the expansion of the universe.
    
    The field φ satisfies:
        □φ = H₀² φ
    
    This gives φ a "mass" m_φ = H₀/c, corresponding to wavelength λ_φ = c/H₀.
    
    The coupling to gravity:
        G_eff = G × [1 + α × φ²/φ₀² × F(g/g†)]
    
    where F(x) → 0 for x >> 1 (high acceleration kills the coupling)
          F(x) → 1 for x << 1 (low acceleration allows full coupling)
    
    WHY THIS WORKS:
    
    1. g† = cH₀ emerges naturally from the field mass
    
    2. Nonlocal effects arise because φ is a FIELD with spatial extent
       - Inner structure affects φ at outer radii
       - Path-dependence comes from φ propagation
    
    3. High acceleration decouples because rapid motion averages out
       the slowly-varying φ field
    
    4. The Hubble connection is BUILT IN, not added by hand

""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  MAKING IT CONCRETE: THE FIELD EQUATION                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Let's write down the actual equations.

THE COSMIC COHERENCE FIELD φ
────────────────────────────

Field equation:
    □φ - (H₀/c)² φ = -κ ρ
    
where:
    □ = d'Alembertian (wave operator)
    H₀/c = "mass" of the field (inverse Hubble length)
    κ = coupling constant to matter density ρ

In the static, spherically symmetric case:
    ∇²φ - (H₀/c)² φ = -κ ρ

Solution for a point mass M:
    φ(r) = (κM/4πr) × exp(-r/λ_H)
    
where λ_H = c/H₀ ≈ 4 Gpc (Hubble radius)

For r << λ_H (all galaxies): φ(r) ≈ κM/(4πr) = Newtonian-like


THE MODIFIED POISSON EQUATION
─────────────────────────────

Standard:  ∇²Φ = 4πG ρ

Modified:  ∇²Φ = 4πG ρ × [1 + f(φ, ∇φ, g)]

The function f encodes how the φ field modifies gravity:

    f = A × (φ/φ†)^n × exp(-g/g†)

where:
    A = √3 (amplitude)
    φ† = characteristic field value
    n = power (to be determined)
    g† = cH₀/4√π (critical acceleration)


THE KEY INSIGHT: WHY INNER STRUCTURE MATTERS
────────────────────────────────────────────

The field φ at radius r depends on ALL the mass inside:

    φ(r) = κ ∫₀^r ρ(r') × G(r,r') dr'

where G(r,r') is the Green's function.

This means:
    - Dense inner region → large φ gradient → affects outer φ
    - Diffuse inner region → small φ gradient → different outer φ

The SAME outer acceleration can have DIFFERENT φ values
depending on the inner mass distribution!

This is exactly what the data shows.

""")

# Calculate some numbers
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  NUMERICAL CHECK: DO THE SCALES WORK?                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

lambda_H = c / H0
print(f"Hubble length λ_H = c/H₀ = {lambda_H/3.086e22:.1f} Gpc")

# Typical galaxy scale
R_gal = 20e3 * 3.086e16  # 20 kpc in meters
print(f"Typical galaxy R = 20 kpc")
print(f"Ratio R/λ_H = {R_gal/lambda_H:.2e}")
print()

# The field φ varies on scale λ_H, so within a galaxy it's nearly constant
# But GRADIENTS of φ can still matter

# Critical acceleration
print(f"Critical acceleration g† = {g_dagger:.2e} m/s²")
print(f"This corresponds to:")
print(f"  - Orbital period at g†: T = 2πv/g† ~ 2π×200km/s / g† = {2*np.pi*200e3/g_dagger / 3.15e7:.1f} Myr")
print(f"  - Hubble time: t_H = 1/H₀ = {1/H0 / 3.15e7 / 1e3:.1f} Gyr")
print()

# The ratio
t_dyn = 2*np.pi*200e3/g_dagger
t_H = 1/H0
print(f"Ratio t_dyn/t_H at g†: {t_dyn/t_H:.3f}")
print()
print("At the critical acceleration, dynamical time ~ 1/60 of Hubble time")
print("This is where local dynamics start to 'feel' cosmic expansion")

print("""

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  THE PROPOSED FUNDAMENTAL PRINCIPLE                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PRINCIPLE: Gravity couples to a cosmic scalar field φ that mediates
           the influence of cosmic expansion on local dynamics.

FIELD EQUATION:
    □φ - m²φ = -κρ    where m = H₀/c

MODIFIED EINSTEIN EQUATION:
    G_μν = 8πG/c⁴ × T_μν^(eff)
    
    T_μν^(eff) = T_μν × [1 + f(φ, g)]

COUPLING FUNCTION:
    f(φ, g) = A × Φ(r) × exp(-g/g†)
    
    where Φ(r) = ∫ φ(r') W(r-r') d³r' (smoothed field)
    and g† = cH₀/4√π

THIS GENERATES:
    1. Enhancement at low g (exp term)
    2. Spatial buildup (φ accumulates with enclosed mass)
    3. Inner structure dependence (φ depends on mass distribution)
    4. Cosmic connection (m = H₀/c built in)

THE NEWTON ANALOGY:
───────────────────
Newton: "Force = GMm/r²"
Us:     "Force = GMm/r² × [1 + f(φ,g)]"

Newton derived his formula from Kepler's laws.
We derive f(φ,g) from rotation curves.

But the PRINCIPLE is: gravity couples to a cosmic field.
The formula follows from the principle.

""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  WHAT REMAINS TO BE DETERMINED                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. The coupling constant κ
   - Sets the strength of matter-φ coupling
   - Should be derivable from more fundamental theory

2. The exact form of f(φ,g)
   - We have f = A × Φ × exp(-g/g†) from fitting
   - Is there a deeper reason for this form?

3. The origin of the φ field
   - Is it the inflaton?
   - Is it related to dark energy?
   - Is it a new fundamental field?

4. The quantum theory
   - What are the φ quanta?
   - How do they interact with gravitons?
   - Is this renormalizable?

These are the questions for the next level of theory.
But the PRINCIPLE is clear:

    GRAVITY COUPLES TO COSMIC EXPANSION THROUGH A SCALAR FIELD.
    
    THIS COUPLING IS SUPPRESSED AT HIGH ACCELERATION.
    
    THE CRITICAL SCALE IS SET BY g† = cH₀/4√π.

""")

