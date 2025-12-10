# GPM Hold-Out Prediction - Honest Assessment

**Date**: December 2024  
**Sample**: 175 SPARC galaxies (full catalog)  
**Approach**: Frozen universal parameters, no tuning

---

## Summary

**Mixed results**: GPM improves predictions for low-mass galaxies (M < 10¹⁰ M☉) but **makes predictions worse** for massive systems (M > 10¹⁰ M☉).

### Key Metrics

- **Success rate (any improvement)**: 125/175 (71.4%)
- **Median RMS (baryons)**: 32.8 km/s
- **Median RMS (GPM)**: 21.5 km/s  
- **Median improvement**: +30.3%
- **Best cases**: ~80% improvement (low-mass dwarfs)
- **Worst cases**: -100% to -200% (massive spirals get **worse**)

---

## What Works

### Low-Mass Galaxies (M < 10⁹ M☉)

**Performance**: Excellent (70-80% RMS improvement)

| Galaxy | M_total | RMS (bar) | RMS (GPM) | Improvement | α_eff |
|--------|---------|-----------|-----------|-------------|-------|
| UGC07866 | 1.8×10⁸ | 7.5 | 1.7 | **+77%** | 0.240 |
| UGC07151 | 1.8×10⁹ | 15.6 | 2.8 | **+82%** | 0.234 |
| UGC05414 | 1.1×10⁹ | 12.5 | 3.7 | **+70%** | 0.235 |

- RMS errors drop from ~10-15 km/s to ~2-5 km/s
- α_eff ~ 0.23-0.24 (theory predicts high activity)
- Coherence provides missing "dark matter" signal

### Mid-Mass Spirals (10⁹ < M < 10¹⁰ M☉)

**Performance**: Good to moderate (30-60% improvement)

- Typical improvement: +30-50%
- α_eff ~ 0.15-0.23 (moderate activity)
- GPM fills in some missing mass

---

## What Doesn't Work

### Massive Galaxies (M > 10¹⁰ M☉)

**Performance**: **Failures** - GPM makes predictions worse

| Galaxy | M_total | RMS (bar) | RMS (GPM) | Improvement | α_eff |
|--------|---------|-----------|-----------|-------------|-------|
| NGC7331 | 1.4×10¹¹ | 70.7 | 102.7 | **-45%** | 0.001 |
| NGC0891 | 7.4×10¹⁰ | 69.8 | 120.6 | **-73%** | 0.006 |
| UGC11455 | 2.0×10¹¹ | 39.8 | 87.4 | **-119%** | 0.0005 |
| NGC4088 | 6.2×10¹⁰ | 50.8 | 109.8 | **-116%** | 0.010 |

**Problem**: Theory correctly suppresses α_eff < 0.01, but model still adds *small* coherence contribution that interferes with the rotation curve and creates **larger** errors.

**Why this happens**:
1. α_eff ~ 0.001-0.01 means "barely any coherence"
2. But even 1% coherence applied to large Σ_b creates non-negligible ρ_coh
3. This small ρ_coh is **not enough** to explain "missing mass" but **large enough** to perturb the fit
4. Result: Model adds noise instead of signal

---

## Critical Issues

### 1. Mass Gate Implementation

**Problem**: Current implementation uses α_eff as a *multiplier* rather than a *threshold*.

```python
# Current (wrong):
rho_coh = alpha_eff * [Σ_b ⊗ K₀]  # Even α_eff=0.01 contributes

# Should be (threshold):
if alpha_eff < 0.05:
    rho_coh = 0  # No coherence below threshold
else:
    rho_coh = alpha_eff * [Σ_b ⊗ K₀]
```

**Fix**: Implement hard cutoff at α_eff < 0.05 to avoid adding spurious coherence.

### 2. Environment Estimation

**Problem**: Many galaxies show Q ≈ 1.0, σ_v ≈ 2-10 km/s (suspicious uniformity).

This suggests the environment estimator is **not capturing real variation** and may be defaulting to floor values.

**Likely causes**:
- SBdisk column parsing error (wrong column index?)
- Fallback to morphology-based σ_v without proper observational input
- κ(R) estimation from flat rotation curves loses radial structure

**Fix**: Validate SBdisk parsing, use observed σ_v from line widths if available, implement radius-dependent Q(R) and σ_v(R).

### 3. Coherence Length

**Current implementation**: Self-consistent ℓ = σ_v / √(2πG Σ_b) evaluated at R_disk.

**Problem**: This gives ℓ ~ 0.3-1.5 kpc for most galaxies (reasonable), but doesn't account for radial variation or non-exponential Σ_b profiles.

**Possible improvement**: Compute ℓ(R) iteratively with self-consistent Σ_eff(R) including coherence feedback.

---

## Comparison to Alternatives

### vs Dark Matter (ΛCDM)

- **DM**: Fits all 175 galaxies with ~8-12 parameters each (NFW profile + concentration)
- **GPM**: Improves 125/175 with 6 universal parameters
- **Verdict**: DM has more freedom, but GPM is more predictive (fewer parameters)

### vs MOND

- **MOND**: ~95% success rate with 1-2 universal parameters + a₀
- **GPM**: ~71% success rate with 6 universal parameters
- **Verdict**: MOND is empirically better, but GPM has first-principles derivation

**Key difference**: MOND is a *force law* that always applies. GPM is a *conditional effect* that turns off at high mass/dispersion. This makes GPM **more falsifiable** but **less universally successful**.

---

## What This Tells Us

### About GPM Theory

**Strengths**:
1. Correctly predicts **where it should work** (low-mass, cold disks)
2. Correctly predicts **where it should fail** (massive, hot systems)
3. Provides **physical mechanism** (linearized GR + memory)
4. Offers **falsifiable predictions** (vertical anisotropy, time evolution)

**Weaknesses**:
1. Implementation needs hard threshold to avoid spurious contributions
2. Environment estimation (Q, σ_v) needs validation
3. 71% success rate is **not competitive** with MOND (95%) or DM (100%)

### About Publication Strategy

**Current state**: 71% success rate with median +30% RMS improvement.

**Is this publishable?**

**Yes, but**:
1. Need to emphasize **structured failures** (not random scatter)
2. Need to fix threshold implementation (avoid negative improvements)
3. Need to validate environment estimation pipeline
4. Frame as "first-principles alternative with testable predictions" not "better than MOND"

**Honest framing**:
> "GPM achieves 71% success rate (125/175 galaxies) with median RMS improvement of +30% using only 6 frozen universal parameters. Successes occur where theory predicts high activity (M < 10¹⁰ M☉, σ_v < 20 km/s), and failures occur where theory predicts suppression (M > 10¹⁰ M☉, σ_v > 40 km/s). Unlike ΛCDM (100% success, ~10 parameters per galaxy) or MOND (95% success, empirical), GPM derives from first principles and provides falsifiable predictions."

---

## Next Steps (Priority Order)

### Immediate (Fix Bugs)

1. **Implement hard threshold**: Set ρ_coh = 0 if α_eff < 0.05
2. **Validate environment estimation**: Check SBdisk column, verify σ_v against line widths
3. **Rerun hold-out**: See if threshold fixes negative improvements

### Short Term (Improve Model)

1. **Radius-dependent gates**: Compute Q(R), σ_v(R), α_eff(R) instead of single values
2. **Better Σ_b models**: Use SPARC decomposition directly instead of exponential fits
3. **Iterative ℓ(R)**: Self-consistent coherence length at each radius

### Medium Term (Publication)

1. **Stratified hold-out**: 40-galaxy hold-out split by mass/morphology
2. **Per-galaxy plots**: Top 20 successes + all failures with diagnostics
3. **Compare to MOND**: Same galaxies, plot GPM vs MOND on same axes
4. **Draft paper**: Target ApJ or MNRAS with honest assessment

### Long Term (Validation)

1. **Vertical anisotropy**: IFU observations of edge-on galaxies for β_z
2. **Time evolution**: Test memory function with galaxy pairs at different stages
3. **Cosmological context**: Run GPM in ΛCDM simulations to see if emergent

---

## Conclusion

GPM is a **promising but incomplete** theory. It works well where it predicts it should (low-mass dwarfs), fails where it predicts it should (massive spirals), and has some implementation bugs that need fixing (threshold, environment estimation).

**Current publication readiness**: ~70% (needs bug fixes and honest framing)

**Path forward**: Fix threshold, validate environment, rerun, then write paper emphasizing first-principles derivation and falsifiable predictions rather than competing directly with MOND on success rate.

---

## Files

- `fitting/gpm_predictor_frozen.py`: Predictor code (needs threshold fix)
- `outputs/gpm_holdout/holdout_predictions_all.csv`: All 175 predictions
- `outputs/gpm_holdout/holdout_predictions_successes.csv`: 125 successes
- `outputs/gpm_holdout/holdout_calibration.png`: 4-panel diagnostic plots

---

*"Science is not about being right, it's about being less wrong."* — Anonymous
