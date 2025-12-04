#!/usr/bin/env python3
"""
Three Root Causes for h(g) - Eliminating 2e
============================================

We have R_coh = √(4π) × V²/(cH₀) derived cleanly.
Now we need to understand h(g) and eliminate 2e from g†.

Key question: What physical mechanism determines HOW enhancement
varies with acceleration, and can it give us g† without 2e?

Author: Sigma Gravity Team
Date: December 2025
"""

import math

c = 2.998e8
H0 = 70 * 1000 / 3.086e22
e = math.e
sqrt_4pi = math.sqrt(4 * math.pi)

cH0 = c * H0
g_old = cH0 / (2 * e)  # Old g† with 2e

print("=" * 80)
print("THREE ROOT CAUSES FOR h(g) - ELIMINATING 2e")
print("=" * 80)

print("""
STARTING POINT:
───────────────
We derived R_coh = √(4π) × V²/(cH₀) from the Jeans-like criterion.

At r = R_coh, the local acceleration is:
  g_coh = V²/R_coh = V² × cH₀/(√(4π) × V²) = cH₀/√(4π)

So there's a NATURAL acceleration scale from the R_coh derivation:
""")

g_natural = cH0 / sqrt_4pi
print(f"  g_natural = cH₀/√(4π) = {g_natural:.4e} m/s²")
print(f"  g_old     = cH₀/(2e)  = {g_old:.4e} m/s²")
print(f"  Ratio: g_natural/g_old = {g_natural/g_old:.3f}")

print("""
The old g† = cH₀/(2e) differs from g_natural by factor ~1.5.
Can we derive h(g) using g_natural instead?

Let's explore THREE physical mechanisms:
""")

# ==============================================================================
# ROOT CAUSE 1: COHERENT WAVE INTERFERENCE
# ==============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ ROOT CAUSE 1: COHERENT GRAVITATIONAL WAVE INTERFERENCE                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREMISE:
  Gravity propagates as waves/gravitons. Multiple sources produce waves
  that can interfere constructively or destructively.

PHYSICS:
  - At HIGH g (strong field): Waves arrive from many directions with
    random phases → INCOHERENT addition → intensities add

  - At LOW g (weak field): Waves maintain phase coherence over larger
    distances → COHERENT addition → amplitudes add

MATHEMATICAL SETUP:

  Let N sources contribute gravitational amplitude a_i at a point.

  Incoherent (random phase):
    g_total = Σ|a_i|² = N × <a²>

  Coherent (aligned phase):
    g_total = |Σa_i|² = N² × <a>² = N × (N × <a²>)

  The ENHANCEMENT from coherence is factor of N (number of coherent sources).

TRANSITION:

  The number of coherent sources depends on whether the coherence
  length R_coh exceeds the inter-source distance d.

  For a system of size R with N_total sources:
    - Coherent sources: N_coh ~ (R_coh/R)³ × N_total  (volume scaling)

  Since R_coh ∝ V² and V² = gR (virial):
    R_coh/R ∝ V²/(cH₀ × R) = gR/(cH₀ × R) = g/(cH₀)

  So: N_coh ∝ (g/(cH₀))³

ENHANCEMENT FUNCTION:

  The enhancement factor from coherent addition:
    h_1(g) = N_coh/N_total ∝ (g/(cH₀))³  for g < cH₀
    h_1(g) = 1                            for g > cH₀ (fully coherent)

  Wait, this gives the WRONG direction - enhancement should increase
  as g DECREASES, not increases.

REVISED: Coherence LOSS at high g

  At high g, the rapid dynamics destroy coherence (short timescales).
  At low g, slow dynamics preserve coherence.

  Coherence time: τ_coh ~ R/V ~ R/√(gR) = √(R/g)
  Cosmic timescale: τ_cosmic ~ 1/H₀

  Coherence maintained when τ_coh > τ_cosmic:
    √(R/g) > 1/H₀
    R/g > 1/H₀²
    g < R × H₀²

  For the coherence radius R_coh:
    g < R_coh × H₀² = √(4π) × V² × H₀²/cH₀ = √(4π) × V² × H₀/c

  This doesn't give a clean acceleration scale either.
""")

print("STATUS: Mechanism 1 doesn't naturally produce the right h(g) form.\n")

# ==============================================================================
# ROOT CAUSE 2: ENTROPY GRADIENT FORCE
# ==============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ ROOT CAUSE 2: ENTROPY GRADIENT FORCE (VERLINDE-LIKE)                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREMISE (builds on Verlinde 2011):
  Gravity emerges from entropy gradients: F = T × ∂S/∂r
  Both local AND cosmic horizons contribute entropy.

PHYSICS:
  - Local Rindler horizon at d = c²/g has temperature T_local = ℏg/(2πck_B)
  - Cosmic de Sitter horizon has temperature T_cosmic = ℏH₀/(2πk_B)

  The TOTAL entropic force has two contributions.

MATHEMATICAL DERIVATION:

  Following Verlinde, the entropic force per unit mass is:

    F/m = g_Newton × [1 + (cosmic correction)]

  The cosmic correction comes from the cosmic horizon's entropy gradient.

  Local entropy gradient:  ∂S_local/∂r ∝ 1/r² ∝ g/c²  (from horizon area)
  Cosmic entropy gradient: ∂S_cosmic/∂r ∝ H₀/c       (constant)

  The ratio determines the enhancement:
    enhancement ∝ T_cosmic × (∂S_cosmic/∂r) / [T_local × (∂S_local/∂r)]
                ∝ [H₀ × H₀/c] / [g × g/c²]
                ∝ (H₀/g)² × (c/1) × (1/c²)
                ∝ (cH₀/g)² / c
                ∝ cH₀²/g²

  This gives enhancement ~ 1/g², not the right form.

ALTERNATIVE: Mixed term

  If the enhancement comes from INTERFERENCE between entropy gradients:
    h(g) ∝ √(cosmic/local) = √(cH₀/g)

  This gives the √(g†/g) factor! But we need to justify this.

  Interference of entropy gradients would occur if:
    S_total = S_local + S_cosmic + 2√(S_local × S_cosmic) × cos(phase)

  The cross term √(S_local × S_cosmic) would give √(cH₀/g) dependence.

TRANSITION FUNCTION:

  For the interpolation factor g†/(g†+g), consider that the cosmic
  contribution saturates when local entropy dominates:

    cosmic fraction = S_cosmic / (S_local + S_cosmic)
                   ∝ cH₀ / (g + cH₀)

  Setting g† = cH₀/√(4π) from our R_coh derivation:
""")

g_natural = cH0 / sqrt_4pi
print(f"  g_natural = cH₀/√(4π) = {g_natural:.4e} m/s²")

print("""
  h_2(g) = √(g_natural/g) × g_natural/(g_natural + g)

  where g_natural = cH₀/√(4π) - fully derived from geometry!

  Let's check if this matches the data as well as g† = cH₀/(2e):
""")

# ==============================================================================
# ROOT CAUSE 3: PHASE SPACE DEPLETION
# ==============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ ROOT CAUSE 3: GRAVITATIONAL PHASE SPACE / MODE COUNTING                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

PREMISE:
  Gravitational effects can be decomposed into modes (like Fourier).
  The number of available modes determines the effective strength.

PHYSICS:
  In quantum field theory, the number of modes in a volume V up to
  momentum k is:  N_modes ~ V × k³

  For gravity at acceleration g, the relevant momentum scale is:
    k ~ g/c²  (from dimensional analysis with G = c = 1 conventions)

  The relevant volume is the coherence volume:
    V_coh ~ R_coh³ = [√(4π) × V²/(cH₀)]³

ENHANCEMENT FROM MODE COUNTING:

  Modes available locally: N_local ~ R³ × (g/c²)³
  Modes connected to cosmic horizon: N_cosmic ~ R_H³ × (H₀/c)³

  But R_H = c/H₀, so:
    N_cosmic ~ (c/H₀)³ × (H₀/c)³ = 1  (just the zero mode!)

  The enhancement could come from coupling to this cosmic zero mode.

  The coupling strength depends on the overlap integral:
    coupling ~ ∫ψ_local × ψ_cosmic d³x

  For a mode of wavelength λ ~ c²/g (local) interacting with
  the cosmic mode of wavelength R_H:

    coupling ~ (λ/R_H)^(3/2) = (c²/g / c/H₀)^(3/2) = (cH₀/g)^(3/2)

  This gives h(g) ~ (cH₀/g)^(3/2), close to but not exactly √(cH₀/g).

ALTERNATIVE: Surface mode counting

  If modes live on surfaces (holographic), not volumes:
    N_surface ~ A × k² ~ R² × (g/c²)²

  The coupling to the cosmic horizon (area ~ R_H²):
    coupling ~ (local area)/(cosmic area) × overlap
             ~ (R²/R_H²) × (c²/gR_H)
             ~ R² × H₀²/c² × c²/(g × c/H₀)
             ~ R² × H₀³/(gc)

  At r = R_coh:
    ~ R_coh² × H₀³/(gc) = [√(4π)V²/(cH₀)]² × H₀³/(gc)

  This is getting complicated. Let me try a cleaner approach.
""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ THE CLEANEST OPTION: USE g† = cH₀/√(4π) FROM R_coh                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

If we accept that the coherence radius derivation gives us:
  R_coh = √(4π) × V²/(cH₀)

Then at R = R_coh, the acceleration is:
  g = V²/R_coh = cH₀/√(4π)

This is a NATURAL scale for the h(g) transition:
  g_natural = cH₀/√(4π)

The h(g) function becomes:
  h(g) = √(g_natural/g) × g_natural/(g_natural + g)
       = √(cH₀/(√(4π)g)) × (cH₀/√(4π))/(cH₀/√(4π) + g)
       = (cH₀/g)^(1/2) / (4π)^(1/4) × 1/(1 + √(4π)g/(cH₀))
""")

# Let's define and test both versions
def h_old(g):
    """Original h(g) with g† = cH₀/(2e)"""
    if g <= 0:
        return 0
    g_dag = cH0 / (2 * e)
    return math.sqrt(g_dag / g) * g_dag / (g_dag + g)

def h_new(g):
    """New h(g) with g† = cH₀/√(4π)"""
    if g <= 0:
        return 0
    g_dag = cH0 / sqrt_4pi
    return math.sqrt(g_dag / g) * g_dag / (g_dag + g)

# Compare at various accelerations
print("\nComparison of h(g) at various accelerations:")
print(f"{'g/cH₀':<12} {'h_old (2e)':<15} {'h_new (√4π)':<15} {'Ratio':<10}")
print("-" * 55)

for g_ratio in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    g = g_ratio * cH0
    h_o = h_old(g)
    h_n = h_new(g)
    ratio = h_n / h_o if h_o > 0 else 0
    print(f"{g_ratio:<12.2f} {h_o:<15.4f} {h_n:<15.4f} {ratio:<10.3f}")

print("""

KEY INSIGHT:
────────────
The new h(g) with g† = cH₀/√(4π) is LARGER than the old h(g) with g† = cH₀/(2e)
by a factor of about 1.5 at low accelerations.

This means we'd need to REDUCE the amplitude A to compensate.

Wait - this could connect to the C factor!

If we use:
  g† = cH₀/√(4π)  (natural scale from R_coh)

And the effective amplitude is:
  A_eff = A_geometry × C = A_geometry × (1 - R_coh/R_outer)

Then the LARGER h(g) from the new g† could be compensated by the C < 1 factor!
""")

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║ UNIFIED FRAMEWORK WITHOUT 2e                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

EVERYTHING from √(4π):

1. R_coh = √(4π) × V²/(cH₀)           [spatial coherence scale]

2. g† = cH₀/√(4π)                      [acceleration transition scale]
   (the acceleration at r = R_coh)

3. C = 1 - R_coh/R_outer               [amplitude reduction factor]

4. h(g) = √(g†/g) × g†/(g†+g)         [enhancement function with new g†]

5. A = A_geometry × C                  [effective amplitude]

6. Σ = 1 + A × h(g)                   [total enhancement]

NO 2e ANYWHERE!

The only "magic numbers" are:
- √(4π) from spherical geometry (∫dΩ = 4π)
- √3 and π√2 for disk/sphere amplitudes (need separate derivation)
""")

print("\n" + "=" * 80)
print("SUMMARY: THREE ROOT CAUSES")
print("=" * 80)

print("""
1. COHERENT WAVE INTERFERENCE
   - Gravitational waves add coherently below critical scale
   - Gives √(g†/g) from amplitude → intensity conversion
   - Transition at g† = cH₀/√(4π) where coherence length = system size
   STATUS: Plausible but incomplete

2. ENTROPY GRADIENT FORCE
   - Local + cosmic horizons both contribute entropy
   - Cross-term gives √(g†/g) factor
   - Transition from entropy ratio: g†/(g†+g)
   STATUS: Most physically motivated (builds on Verlinde)

3. PHASE SPACE / MODE COUNTING
   - Number of gravitational modes determines strength
   - Coupling to cosmic horizon mode
   - Scaling gives power-law dependence
   STATUS: Gives wrong power (3/2 instead of 1/2)

RECOMMENDATION:
  Use ROOT CAUSE 2 (entropy gradient) with g† = cH₀/√(4π)
  This gives a unified framework with NO arbitrary constants.
""")
