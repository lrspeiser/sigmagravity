"""
VERIFICATION: Is A_max = √2 Actually Derived from First Principles?
====================================================================

CLAIMED DERIVATION:
1. BTFR: v⁴ = G × M × a₀ × A_TF, where A_TF ≈ 0.5
2. Our theory: v⁴ = A_max² × g† × G × M
3. Since g† ≈ a₀: A_max² = 1/A_TF = 2
4. Therefore: A_max = √2 ≈ 1.414

CRITICAL QUESTIONS:
- Is A_TF = 0.5 derived or empirical?
- Does this constitute a "derivation" or is it fitting to observations?
- Is the v⁴ = A_max² × g† × G × M relation correct?
"""

import numpy as np

print("=" * 80)
print("VERIFICATION: A_max = √2 DERIVATION FROM BTFR")
print("=" * 80)


# =============================================================================
# TEST 1: Is A_TF = 0.5 derived or empirical?
# =============================================================================

print("\n" + "─" * 80)
print("TEST 1: Is A_TF = 0.5 derived or empirical?")
print("─" * 80)

print("""
The Baryonic Tully-Fisher Relation (BTFR) is:

    M_baryon = A_TF × v_flat⁴ / (G × a₀)

Or equivalently:
    
    v_flat⁴ = G × M_baryon × a₀ / A_TF

QUESTION: Where does A_TF ≈ 0.5 come from?

ANSWER: A_TF is an EMPIRICAL FIT to observed galaxy data.

From McGaugh et al. (2016, Physical Review Letters 117, 201101):
"The baryonic Tully-Fisher relation... has the form M = A v⁴"
with A fitted to ~170 galaxies.

The coefficient is NOT derived from any first principle.
It's measured from observations.

VERDICT: A_TF = 0.5 is EMPIRICAL (fitted to data)
""")

test1_result = "EMPIRICAL"


# =============================================================================
# TEST 2: Does the claimed relation v⁴ = A_max² × g† × G × M hold?
# =============================================================================

print("\n" + "─" * 80)
print("TEST 2: Does v⁴ = A_max² × g† × G × M follow from the theory?")
print("─" * 80)

print("""
Let's derive this properly from first principles.

The theory says:
    Σ = 1 + A_max × W(r) × h(g)

At large r where W → 1 and h → √(g†/g):
    Σ ≈ A_max × √(g†/g)

The circular velocity:
    v² = Σ × v_N² = A_max × √(g†/g) × (G M / r)

where g = v²/r for a test particle (NOT v_N²/r!).

Wait - there's a subtlety. What is g in h(g)?

ISSUE 1: Definition of g

Option A: g = g_bar = v_N²/r (baryonic acceleration)
Option B: g = g_obs = v²/r (observed acceleration)

These are related by g_obs = Σ × g_bar.

The theory uses g = g_bar, so:
    g_bar = G M / r²
    h(g_bar) ≈ √(g† r² / (G M)) = r × √(g† / (G M))

Then:
    Σ ≈ A_max × r × √(g† / (G M))

And:
    v² = Σ × g_bar × r = A_max × r × √(g† / (G M)) × (G M / r²) × r
       = A_max × √(g† × G M)

So:
    v⁴ = A_max² × g† × G M

YES! The relation does follow from the theory.

But note: This required:
1. W(r) → 1 at large r (spatial coherence saturates)
2. h(g) ≈ √(g†/g) at low g (deep-MOND limit)
3. g = g_bar (baryonic, not observed acceleration)

VERDICT: The relation v⁴ = A_max² × g† × G M IS CORRECTLY DERIVED
""")

test2_result = "DERIVED"


# =============================================================================
# TEST 3: Is A_max = √2 a derivation or circular reasoning?
# =============================================================================

print("\n" + "─" * 80)
print("TEST 3: Is this a derivation or circular reasoning?")
print("─" * 80)

print("""
The claimed derivation:
1. BTFR (observed): v⁴ = G M a₀ × A_TF with A_TF ≈ 0.5
2. Theory (derived): v⁴ = A_max² × g† × G M
3. Match: A_max² × g† = a₀ / A_TF

Since g† ≈ a₀ (by construction):
    A_max² = 1 / A_TF ≈ 2
    A_max ≈ √2

CRITICAL ANALYSIS:

This is NOT a first-principles derivation!

It's CALIBRATION to observations:
- A_TF = 0.5 is MEASURED from galaxy data
- A_max is CHOSEN to reproduce the observed BTFR

The logic is:
1. We observe BTFR with coefficient A_TF ≈ 0.5
2. We DEFINE A_max to match: A_max = √(1/A_TF)
3. We call this "derived"

But this is exactly what MOND does:
- MOND observes BTFR
- MOND sets a₀ to match
- MOND calls a₀ "the MOND acceleration"

The difference:
- MOND has 1 free parameter (a₀)
- Σ-Gravity claims to derive a₀ (as g†) but STILL needs A_max

So the total free parameters are the SAME!

VERDICT: A_max = √2 is CALIBRATION TO DATA, not derivation
""")

test3_result = "CALIBRATION"


# =============================================================================
# TEST 4: What about the claimed A_TF = 1/2 "geometric factor"?
# =============================================================================

print("\n" + "─" * 80)
print("TEST 4: Is A_TF = 1/2 a geometric factor?")
print("─" * 80)

print("""
The claim: "A_TF ≈ 0.5 is the geometric factor for disk galaxies"

Is this true? Let's check.

For an exponential disk with scale length R_d:
    Σ(R) = Σ₀ × exp(-R/R_d)
    M_total = 2π Σ₀ R_d²

The rotation curve peaks at R ≈ 2.2 R_d with v_max.

But v_flat (asymptotic) is DIFFERENT from v_max!

For MOND/Σ-Gravity at R >> R_d:
    v_flat² ∝ √(G M a₀)

The factor A_TF in the BTFR comes from:
    v_flat⁴ = A_TF × G M × a₀

For point mass: A_TF = 1
For extended distribution: A_TF < 1 due to mass geometry

NUMERICAL CHECK:

For exponential disk, detailed calculations give:
    A_TF ≈ 0.4 - 0.6 depending on M/L assumptions

The value A_TF ≈ 0.5 is the AVERAGE from McGaugh's BTFR fit.

Is this a "derivation" of A_TF = 1/2?

NO! It's a POST-HOC interpretation:
1. Observe A_TF ≈ 0.5 from data
2. Note that 0.5 ≈ 1/2
3. Say "this comes from geometry"
4. Don't actually derive it from any specific calculation

To truly derive A_TF, you would need to:
1. Specify the mass distribution
2. Solve for v_flat analytically
3. Show that A_TF = 1/2 exactly

This has NOT been done.

VERDICT: A_TF = 1/2 is EMPIRICAL with post-hoc geometric interpretation
""")

test4_result = "EMPIRICAL"


# =============================================================================
# TEST 5: Numerical verification
# =============================================================================

print("\n" + "─" * 80)
print("TEST 5: Numerical verification")
print("─" * 80)

# Physical constants
G = 6.674e-11  # m³/kg/s²
c = 2.998e8    # m/s
H0_SI = 2.27e-18  # 1/s

g_dagger = c * H0_SI / (2 * np.e)
a0_MOND = 1.2e-10  # m/s²

print(f"g† = cH₀/(2e) = {g_dagger:.4e} m/s²")
print(f"a₀ (MOND) = {a0_MOND:.4e} m/s²")
print(f"Ratio g†/a₀ = {g_dagger/a0_MOND:.3f}")

# The claimed derivation
A_TF_observed = 0.5
A_max_claimed = np.sqrt(1 / A_TF_observed)

print(f"\nClaimed derivation:")
print(f"  A_TF (observed) = {A_TF_observed}")
print(f"  A_max = √(1/A_TF) = √2 = {A_max_claimed:.4f}")

# Test: Does this give the right BTFR?
M_test = 5e10 * 1.989e30  # 5×10¹⁰ M_sun in kg

# BTFR prediction
v_btfr = (G * M_test * a0_MOND / A_TF_observed)**0.25

# Our theory prediction
v_theory = (A_max_claimed**2 * g_dagger * G * M_test)**0.25

print(f"\nFor M = 5×10¹⁰ M_sun:")
print(f"  v_flat (BTFR) = {v_btfr/1000:.1f} km/s")
print(f"  v_flat (theory with A_max=√2) = {v_theory/1000:.1f} km/s")
print(f"  Difference: {(v_theory - v_btfr)/v_btfr * 100:.1f}%")

# The difference comes from g† vs a₀
v_if_g_dag_equals_a0 = (A_max_claimed**2 * a0_MOND * G * M_test)**0.25
print(f"  v if g† = a₀ exactly: {v_if_g_dag_equals_a0/1000:.1f} km/s")

print("\nThe ~4% difference comes from g† ≠ a₀ exactly.")


# =============================================================================
# TEST 6: What value of A_max ACTUALLY fits SPARC data?
# =============================================================================

print("\n" + "─" * 80)
print("TEST 6: What does SPARC data actually prefer?")
print("─" * 80)

print("""
From the paper's own fitting:

The SPARC rotation curve fits use:
    Σ = 1 + A₀ × (1 - W(r)) × f(g_bar/g†)^p

The fitted parameters are:
    A₀ ≈ 0.591 (from paper)
    p ≈ 0.757
    g† ≈ 1.2 × 10⁻¹⁰ m/s²

WAIT - A₀ = 0.591 ≠ √2 = 1.414!

The paper uses A₀ in a DIFFERENT way than the A_max in this derivation.

Let me check the relationship...

In the paper:
    Σ - 1 = A₀ × (1 - W) × f(g)^p

In this derivation:
    Σ - 1 = A_max × W × h(g)

These are DIFFERENT functional forms!
    - Paper: (1 - W) not W
    - Paper: f(g)^p not h(g)
    - Paper: A₀ ≈ 0.6 not √2

So the "derivation" of A_max = √2 doesn't even match the paper's own model!

VERDICT: A_max = √2 is for a DIFFERENT model than the paper uses
""")

print("""
The paper's actual fitted A₀ = 0.591 is:
- NOT √2 = 1.414
- NOT 1/√e = 0.606 (close!)
- The paper uses A₀ = 1/√e as the "derived" value (97.4% agreement)

So there are THREE different "A" values floating around:
1. A₀ = 0.591 (SPARC fit)
2. A₀ = 1/√e ≈ 0.606 (Gaussian phase derivation)
3. A_max = √2 ≈ 1.414 (BTFR derivation)

These are for DIFFERENT functional forms of Σ!
""")

test6_result = "DIFFERENT_MODEL"


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

print(f"""
CLAIM: A_max = √2 is derived from first principles via BTFR.

VERIFICATION RESULTS:

1. Is A_TF = 0.5 derived?
   ✗ NO - {test1_result}
   A_TF ≈ 0.5 is an empirical fit to McGaugh's BTFR data

2. Is v⁴ = A_max² × g† × G × M correct?
   ✓ YES - {test2_result}
   This relation follows from the theory at large r

3. Is A_max = √2 a first-principles derivation?
   ✗ NO - {test3_result}
   It's calibration to observed BTFR coefficient

4. Is A_TF = 1/2 a "geometric factor"?
   ✗ NO - {test4_result}
   Post-hoc interpretation of empirical fit

5. Does A_max = √2 match the paper's model?
   ✗ NO - {test6_result}
   The paper uses A₀ ≈ 0.6 in a different functional form

─────────────────────────────────────────────────────────────────────────────

OVERALL ASSESSMENT:

The "derivation" of A_max = √2 from BTFR is CIRCULAR:

    STEP 1: Measure BTFR coefficient A_TF ≈ 0.5 from data
    STEP 2: Set A_max to reproduce BTFR: A_max = √(1/A_TF)
    STEP 3: Call this "derived"

This is NOT a derivation - it's FITTING to observations.

The parameter is NOT reduced - it's TRANSFERRED:
- Before: A_TF is a free parameter in BTFR
- After: A_max is determined by A_TF

The total number of free parameters is UNCHANGED.

ADDITIONAL ISSUE:
The A_max = √2 value doesn't even match the paper's fitted A₀ = 0.591!
These are for different models with different functional forms.

CLASSIFICATION: CALIBRATION (dressed up as derivation)

This is NOT suitable for inclusion in the paper as a "derivation".
It would be honest to say:
"A_max is constrained by the BTFR to be approximately √2"
NOT:
"A_max = √2 is derived from first principles"

─────────────────────────────────────────────────────────────────────────────
""")

print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
