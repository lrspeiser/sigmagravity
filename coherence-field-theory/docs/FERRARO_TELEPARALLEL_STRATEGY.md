# Ferraro Teleparallel Strategy

## Background

Prof. Rafael Ferraro (IAFE, Buenos Aires) provided the following insight on f(T) teleparallelism:

> "When a f(T) function is chosen, necessarily a constant with units of square length must be introduced. For instance, the function f(T)=a⁻¹ Exp[a T] needs a constant 'a' to have a dimensionless argument in the exponential function. **This constant fixes the scale at which the modified gravity deviates from standard (GR) gravity.**"

This is a crucial structural observation. In our phenomenological model, we have scale-dependent effects that activate based on baryonic geometry. The question is: **can this be formalized as an f(T) theory with field-dependent 'a'?**

---

## The Dimensional Puzzle

### Our Model's Scales

| Scale | Formula | Typical Value | Origin |
|-------|---------|---------------|--------|
| ℓ_toomre | σ_v / √(GΣ_b) | ~1 kpc | Disk stability |
| R_disk | (empirical) | ~3 kpc | Baryonic geometry |
| ℓ_fitted | ℓ₀ × (R_disk/2)^0.5 | ~2 kpc | Phenomenological |
| L_H | c/H₀ | ~4×10⁶ kpc | Cosmological |

### The Critical Acceleration

The MOND-like critical acceleration g† ~ cH₀ ~ 1.2×10⁻¹⁰ m/s² emerges in our model.

In f(T) language, if the modification scale is set by 'a' [length²], then:
```
g* ~ c² / √a
```

For g* ~ g† ~ cH₀:
```
√a ~ c² / (cH₀) = c/H₀ = L_H
a ~ L_H² ~ (4×10⁶ kpc)² ~ 10¹³ kpc²
```

But our phenomenology has ℓ² ~ (1-3 kpc)² ~ 1-10 kpc².

**This is a ratio of 10¹² — the scales don't match naively!**

---

## Resolution: Field-Dependent 'a'

The resolution is that 'a' in f(T) need not be a universal constant. It can be a **functional of the matter distribution**.

### Proposed Form

```
a = ℓ_local² × f(ℓ_local / L_H)
```

where:
- ℓ_local = σ_v / √(GΣ_b) is the local Toomre scale
- f(x) is a function that brings in the cosmological scale

### Candidate f(x) Forms

1. **Linear mixing**: f(x) = 1/x → a = ℓ_local × L_H
2. **Quadratic mixing**: f(x) = 1/x² → a = L_H²
3. **Power law**: f(x) = x^(-n) for some n

The linear mixing gives:
```
a = ℓ_local × L_H ~ 1 kpc × 4×10⁶ kpc ~ 4×10⁶ kpc²
g* = c² / √a ~ c² / (2×10³ kpc) ~ 4×10⁷ (km/s)²/kpc
```

This is still ~2000× larger than g†. So we need:
```
a ~ L_H² × (ℓ_local / L_H)^n for n ~ 0.1-0.2
```

Or equivalently:
```
a ~ ℓ_local^n × L_H^(2-n)
```

---

## Tests to Determine the Correct Form

### Test 1: Scale Ratio Universality

**Hypothesis**: ℓ_fitted / ℓ_toomre is universal across galaxies.

**Method**:
1. For each SPARC galaxy, compute:
   - ℓ_toomre = σ_v / √(GΣ_b)
   - ℓ_fitted from best-fit coherence length
2. Check if ratio is constant (scatter < 30%)

**Prediction**:
- If universal → 'a' has form a = const × ℓ_toomre²
- If varies systematically → 'a' depends on additional physics

### Test 2: Acceleration-Scale Correlation

**Hypothesis**: The acceleration where modification activates correlates with baryonic properties.

**Method**:
1. Define g_transition as acceleration where (g_obs - g_bar)/g_bar = 0.5
2. Plot g_transition vs. (σ_v, Σ_b, R_disk)
3. Check if g_transition ~ g† × f(baryonic observables)

**Prediction**:
- If g_transition ~ cH₀ universally → pure cosmological 'a'
- If g_transition varies → field-dependent 'a'

### Test 3: f(T) Form Discrimination

**Hypothesis**: Different f(T) forms make different predictions.

**Method**: For each candidate f(T):
1. f(T) = T + α T²/T* (quadratic)
2. f(T) = T × (1 - e^(-T/T*)) (exponential)
3. f(T) = T × √(1 + T*/T) (square-root)

Compute:
- Predicted rotation curve
- Predicted RAR scatter
- Predicted cluster lensing

**Discriminator**: Which f(T) form best matches all three?

### Test 4: Solar System Consistency

**Hypothesis**: Field-dependent 'a' naturally suppresses modification in Solar System.

**Method**:
1. Compute 'a' for Solar System using:
   - σ_v ~ 0.01 km/s (planetary orbits very circular)
   - Σ_b ~ 0 (point mass, no surface density)
2. Check if this gives negligible modification

**Prediction**:
- ℓ_local → 0 when Σ_b → 0 (point mass)
- Therefore a → 0 and T* → ∞
- Modification vanishes automatically

---

## Proposed f(T) Ansatz

Based on our phenomenology, we propose:

```
f(T) = T + α × T² / T*(ρ_b, σ_v)
```

where:
```
T* = (g†)² / ℓ_b²
ℓ_b = σ_v / √(2πG Σ_b)
g† = c H₀
```

This gives:
```
T/T* = (g/g†)² × (ℓ_b/ℓ)²
```

At r ~ ℓ_b (characteristic baryonic scale):
```
T/T* ~ (g/g†)²
```

**Properties**:
1. Modification activates when g < g† (low acceleration)
2. Modification suppressed when Σ_b → 0 (point mass)
3. Modification suppressed when σ_v → ∞ (hot system)
4. Scale 'a' = ℓ_b² is field-dependent

---

## Connection to Born-Infeld and Lorentz Structure

Ferraro has worked extensively on:
1. Born-Infeld teleparallelism
2. Remnant Lorentz structures in f(T)

**Question for follow-up**: Does a field-dependent T* preserve the key properties of these frameworks?

In Born-Infeld f(T):
```
f(T) = λ × [√(1 + 2T/λ) - 1]
```

If λ is field-dependent: λ = λ(ρ_b, σ_v), this might naturally incorporate our phenomenology while maintaining the determinantal structure.

---

## Next Steps

1. **Run Test 1** with SPARC data (requires data setup)
2. **Derive field equations** for f(T) with field-dependent T*
3. **Check Lorentz invariance** — does field-dependent T* break it?
4. **Compare to Bekenstein's RAQUAL** — similar field-dependent modification
5. **Draft follow-up to Ferraro** with specific f(T) proposal

---

## Code Reference

The tests are implemented in:
```
coherence-field-theory/tests/ferraro_scale_tests.py
```

Run with:
```bash
python coherence-field-theory/tests/ferraro_scale_tests.py
```

---

## Summary

Ferraro's insight points to the structural requirement: f(T) needs a dimensional constant 'a' [length²].

Our phenomenology suggests 'a' is **not universal** but depends on:
```
a = (σ_v / √(GΣ_b))² × cosmological_factor
```

This is similar to other field-dependent modified gravity theories (RAQUAL, emergent gravity, superfluid DM).

The key test is whether the ratio ℓ_fitted / ℓ_toomre is universal across the SPARC sample. If so, we have a candidate for the teleparallel formalization.

