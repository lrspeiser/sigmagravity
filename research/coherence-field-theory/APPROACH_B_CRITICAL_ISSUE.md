# Approach B: Critical Issue with Real Galaxies

## Problem

**Approach B (Wave Amplification) fails on real SPARC galaxies because it requires disk instability (Q < Q_c), but real galaxies are stable.**

### Test Results

Tested on 4 SPARC galaxies:
- **DDO154**: Q range 0.00 - 5769, only 3/12 points have Q < 1.5
- **DDO170**: Similar (very high Q)
- **NGC2403**: Gain range 0.0000 - 0.0428 kpc⁻² (negligible)
- **NGC6503**: Gain ~0

**Result**: g(r) ≈ 0 everywhere → No field → No fits

### Root Cause

The gain function has **THREE gates** that all must be satisfied:

```
g(r) = g₀ · S_Q(r) · S_σ(r) · S_res(r)
```

**Gate 1: Toomre Q (coldness)**
```
S_Q = [1 + tanh((Q_c - Q)/ΔQ)] / 2
```
- Requires Q < Q_c (~1.5) for amplification
- **Real galaxies**: Q > 2-10 (gravitationally stable!)
- **Result**: S_Q ≈ 0 for most radii

**Gate 2: Dispersion**
```
S_σ = exp[-(σ_v/σ_c)²]
```
- Dwarfs: σ_v ~ 15 km/s, σ_c = 30 km/s → S_σ = 0.78 (OK)

**Gate 3: Resonance**
```
S_res = Σ_m exp[-(2πr/λ_φ - m)²/(2σ_m²)]
```
- Peaks at r ~ mλ_φ/(2π)
- For λ_φ = 8 kpc: peaks at r ~ 1.3, 2.6 kpc
- **Real galaxies**: Data coverage varies, but some points align (OK)

### Why Q is High

Toomre Q = (κ σ_v) / (π G Σ_b)

For DDO154 (dwarf):
- κ ~ Ω√2 ~ 30 km/s/kpc (typical)
- σ_v ~ 15 km/s (cold)
- **Σ_b ~ 10² - 10⁴ M☉/kpc²** (from derivative of v_disk²)

**Problem**: We're computing Σ_b from gradient of v², which is extremely noisy for sparse data (12 points), giving artificially low Σ → artificially high Q.

Even with correct Σ_b from SPARC master table (Σ₀ ~ 100 M☉/pc² = 10⁷ M☉/kpc²), real galaxies are typically Q > 1.5 (stable!).

---

## Why Approach B Worked in Test

The **synthetic test galaxy** in `test_resonant_halo_solver.py` had:
- **Manually constructed** cold, unstable disk:
  - Σ_b ~ 10⁸ M☉/kpc² (very high!)
  - σ_v = 20 km/s (cold)
  - Result: Q < 1.5 in inner disk → S_Q ~ 1
- **Smooth** profiles (300 points, no noise)
- **Tuned** λ_φ to match disk scale

Real galaxies don't satisfy these conditions.

---

## Comparison to Phenomenological Σ-Gravity

Your original phenomenological model (K(R) formalism) did **NOT** require disk instability:

```python
# From many_path_model/ and GravityWaveTest/
K(R) ~ function of (Σ_b, v_c, R_d, morphology, ...)
```

It worked by:
1. Directly fitting K(R) or parameters to rotation curves
2. Using morphology/Q/dispersion as **tuning knobs**, not gates
3. No requirement that Q < Q_c everywhere

**Approach B** is more restrictive: it demands physical amplification conditions that real galaxies don't meet.

---

## Options Forward

### Option 1: Abandon Approach B for galaxies
- Keep it as a theoretical curiosity (works in principle)
- Acknowledge it requires artificially cold/unstable disks
- Move back to phenomenological Σ-Gravity or Approach C

### Option 2: Modify gates to work on stable disks
Instead of requiring Q < Q_c, make gain depend on:
```python
S_Q_mod = exp[-(Q - Q_target)²/(2ΔQ²)]  # Gaussian centered at Q~1-2
```
This allows gain even for stable disks (Q > 1).

**But this loses physical motivation**: why would stable disks amplify?

### Option 3: Use Σ₀ from SPARC master table
- Load proper Σ_b(r) = Σ₀ exp(-r/R_d) from SPARC
- Recalculate Q with correct surface densities
- May still give Q > 1.5 (real disks are stable!)

### Option 4: Redefine "gain" as coupling strength
Don't require tachyonic amplification. Instead:
```python
g(r) = coupling(baryons, geometry, dynamics)
μ²(r) = m₀² (always positive, no tachyon)
```
Then solve:
```
∇²φ - m₀²φ - λφ³ = β g(r) ρ_b(r)
```
where g(r) modulates the **source**, not the mass.

---

## Recommendation

**Short term**: Document that Approach B requires idealized conditions not met by real galaxies, and it's not viable for SPARC fits.

**Medium term**: If you want a field theory backbone for your phenomenological Σ-Gravity, Approach C (symmetron/Landau-Ginzburg) is more flexible:
- Can have ρφ ≪ ρm (no CC problem if parameters tuned)
- Doesn't require Q < Q_c
- Connects to screening mechanisms

**Long term**: The phenomenological K(R) or many-path kernel may be the most robust—it doesn't depend on unrealistic microphysical assumptions.

---

## Files

**Test results**:
- `outputs/sparc_resonant_fits/` (all failed, plots show g≈0)
- `outputs/gain_diagnostics/DDO154_gain_diagnostic.png` (shows S_Q≈0)

**Code**:
- `galaxies/test_resonant_on_sparc.py` (SPARC test script)
- `galaxies/diagnose_gain.py` (gate diagnostics)
- `galaxies/resonant_halo_solver.py` (Approach B solver)

**Theory**:
- `APPROACH_B_IMPLEMENTED.md` (implementation details)
- `NUMERICS_FIXED_READY_FOR_SPARC.md` (numerical fixes)
- `derivations/FIRST_PRINCIPLES_SUMMARY.md` (all three approaches)
