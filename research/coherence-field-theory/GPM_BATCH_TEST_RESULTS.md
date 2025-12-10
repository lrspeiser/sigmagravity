# GPM Batch Test Results - CRITICAL FAILURE

**Date**: 2024
**Test**: Gravitational Polarization with Memory (GPM) on 10 SPARC galaxies
**Status**: ❌ FAILED - 9/10 galaxies show catastrophic χ² degradation

## Executive Summary

GPM with global parameters (α₀=0.9, ℓ=2.0 kpc) **adds far too much coherence mass** for most galaxies, worsening χ² by 2000-427000%. Only DDO154 (smallest, coldest dwarf) improved (+81.5%).

**Problem**: Environmental gating (Q, σ_v) reduces α from 0.9 → 0.2-0.6, but this is still ~10-100× too high for spirals and larger dwarfs.

## Test Results

### Global Parameters
- α₀ = 0.9 (base susceptibility)
- ℓ₀ = 2.0 kpc (coherence length)
- Q* = 2.0, σ* = 25 km/s (gate thresholds)
- nQ = 2.0, nσ = 2.0 (gate exponents)

### Success Rate
- **Improved**: 1/10 (10.0%)
- **Degraded**: 9/10 (90.0%)
- **Mean Δχ²**: -63,926% ❌
- **Median Δχ²**: -24,334% ❌

### Per-Galaxy Results

| Galaxy   | N  | M_total [M☉] | R_d [kpc] | Q   | σ_v [km/s] | α_eff | ℓ [kpc] | χ²_bar | χ²_gpm  | Δχ² [%]    |
|----------|----|--------------|-----------| --- |------------|-------|---------|--------|---------|------------|
| DDO154   | 12 | 3.0×10⁷      | 0.44      | 1.5 | 1.8        | 0.574 | 2.0     | 10,893 | 2,014   | **+81.5** ✅ |
| DDO170   | 8  | 2.6×10⁸      | 1.77      | 1.5 | 2.9        | 0.571 | 2.0     | 1,234  | 35,873  | -2,808 ❌   |
| IC2574   | 34 | 6.1×10⁸      | 3.04      | 1.5 | 2.7        | 0.572 | 2.0     | 724    | 108,526 | -14,886 ❌  |
| NGC2403  | 73 | 3.8×10⁹      | 0.99      | 2.0 | 12.6       | 0.399 | 2.0     | 5,188  | 235,270 | -4,435 ❌   |
| NGC6503  | 31 | 7.3×10⁹      | 1.19      | 2.0 | 12.2       | 0.402 | 2.0     | 924    | 413,247 | -44,609 ❌  |
| NGC3198  | 43 | 9.7×10⁹      | 1.79      | 2.0 | 15.7       | 0.376 | 2.0     | 1,063  | 360,319 | -33,783 ❌  |
| NGC2841  | 50 | 8.1×10¹⁰     | 4.52      | 2.0 | 28.3       | 0.274 | 2.0     | 846    | 3.6M    | -427,423 ❌ |
| UGC00128 | 22 | 3.4×10⁹      | 3.00      | 1.5 | 5.6        | 0.558 | 2.0     | 8,880  | 3.8M    | -42,180 ❌  |
| UGC02259 | 8  | 9.2×10⁸      | 1.72      | 1.5 | 4.1        | 0.566 | 2.0     | 853    | 55,351  | -6,389 ❌   |
| NGC0801  | 13 | 6.1×10¹⁰     | 1.92      | 2.0 | 33.5       | 0.237 | 2.0     | 419    | 263,401 | -62,830 ❌  |

### Correlations
- **α vs Q**: -0.921 ✅ (strong anticorrelation, as expected)
- **α vs σ_v**: -0.972 ✅ (very strong anticorrelation, as expected)
- **ℓ vs R_disk**: nan (ℓ fixed at 2.0 kpc for all galaxies)

**Environmental gating IS working** (strong correlations), but α values remain far too high in absolute terms.

## Root Cause Analysis

### Why DDO154 Succeeded
- **Extremely low mass**: M_total = 3×10⁷ M☉ (smallest in sample by 10×)
- **Very cold**: σ_v = 1.8 km/s (coldest in sample)
- **Compact**: R_d = 0.44 kpc
- **High α**: 0.574 adds coherence density that dominates over tiny baryon baseline
- **Result**: GPM amplifies baryons ~10× → flat rotation curve → good fit

### Why Others Failed
All other galaxies have:
- **10-1000× more baryons** than DDO154
- **Higher σ_v** (2.7-33 km/s) → modest α reduction (0.57 → 0.24)
- **Result**: Even α=0.2-0.6 adds massive coherence halos → χ² explodes

**Example (NGC2403)**:
- M_total = 3.8×10⁹ M☉ (130× more than DDO154)
- α = 0.399 (30% lower than DDO154)
- But α × M_total is still ~50× larger → coherence density dominates → overpredicts v(r)

## Physical Interpretation

**GPM adds coherence density**: ρ_coh(r) = α ∫ exp(-|r-s|/ℓ) ρ_b(s) d³s / (4π ℓ² |r-s|)

For Yukawa kernel with ℓ=2 kpc:
- Baryons "smear out" over ~ℓ scale
- Total coherence mass: M_coh ≈ α × M_baryon
- **α=0.5 means coherence halo has half the baryonic mass**

**In phenomenological Σ-Gravity**:
- User's K(R) function adds ~10-30% extra mass at outer radii
- GPM with α=0.2-0.6 adds 20-60% everywhere → far too much

## What Worked vs Failed

### What Worked ✅
1. **Physics framework**: Yukawa convolution numerically stable
2. **Environmental gating**: Correlations α vs (Q, σ_v) correct (-0.92, -0.97)
3. **DDO154 fit**: +81% improvement shows GPM can work for cold dwarfs

### What Failed ❌
1. **Absolute α scale**: 0.2-0.6 is 10-100× too high for most galaxies
2. **No mass-dependent gating**: α same for 10⁷ M☉ and 10¹⁰ M☉ galaxies
3. **Fixed ℓ**: 2 kpc too large for small dwarfs, too small for massive spirals
4. **Lack of diversity**: 9/10 failures means parameters don't span viable range

## Proposed Fixes

### Option A: Drastically Reduce α₀
- **Change**: α₀ = 0.9 → 0.05-0.1
- **Rationale**: Need α_eff ~ 0.01-0.05 for spirals, 0.1-0.2 for dwarfs
- **Issue**: Environmental gating already reduces α by 2-4×, so α₀=0.1 → α_eff=0.025-0.05 (might still overshoot)

### Option B: Add Mass-Dependent Gating
- **Physics**: Coherence effects weaker in massive galaxies (more internal pressure, disruption)
- **Implementation**: α → α / (1 + (M_total/M*)^nM)
- **Example**: M* = 10⁹ M☉, nM = 1.0 → α(10⁷ M☉) = 0.9, α(10¹⁰ M☉) = 0.09

### Option C: Scale ℓ with R_disk
- **Currently**: ℓ fixed at 2 kpc for all
- **Proposed**: ℓ = ℓ₀ × (R_disk / R*)^p with R* = 2 kpc, p = 0.5-1.0
- **Rationale**: Coherence length should track disk size
- **Effect**: Small dwarfs (R_d=0.5 kpc) → ℓ=1 kpc; spirals (R_d=3 kpc) → ℓ=2.5 kpc

### Option D: Fit User's K(R) → Extract Effective α(R)
- **Strategy**: Reverse-engineer from successful fits in `many_path_model/`
- **Process**:
  1. Load user's best-fit K(R) for 175 SPARC galaxies
  2. For each galaxy, solve: ρ_coh(r) = ρ_K(r) = α(r) ∫ G_ℓ ρ_b d³s
  3. Extract α(r), ℓ that reproduce ρ_K(r)
  4. Regress α vs (M_total, Q, σ_v, R_disk) to find global formula
- **Advantage**: Guaranteed to match phenomenology by construction
- **Disadvantage**: More complex (spatially varying α)

## Next Steps (User Decision Required)

### Immediate Actions
1. **Validate baryon masses**: Check if M_total estimates from SBdisk are accurate (NGC2841 = 8×10¹⁰ M☉ seems high)
2. **Test Option A**: Re-run with α₀ = 0.05, see if spirals improve without losing DDO154
3. **Test Option B**: Add mass-dependent gating, scan (M*, nM) parameter space
4. **Implement Option D**: Extract α_eff(galaxy) from user's existing fits → reverse-engineer microphysics

### Questions for User
1. **Accept failure and pivot to Option D** (reverse-engineer from successful K(R) fits)?
2. **Tune global parameters** (try α₀ << 1, add mass gating, scale ℓ)?
3. **Change physics model** (e.g., ρ_coh ∝ ρ_b² instead of ρ_coh ∝ ρ_b)?

## Files Generated
- `coherence-field-theory/examples/batch_gpm_test.py` (381 lines)
- `coherence-field-theory/outputs/gpm_tests/batch_gpm_results.csv`
- `coherence-field-theory/examples/test_gpm_ddo154.py` (working reference)
- `coherence-field-theory/GPM_IMPLEMENTATION_STATUS.md` (outdated - needs update)

## Conclusion

**GPM framework is solid** (Yukawa convolution, environmental gating work), but **parameter choices catastrophically wrong** for 90% of SPARC sample. The model needs either:
- **Drastically smaller α₀** (~0.01-0.1 instead of 0.9), OR
- **Mass-dependent gating** to suppress α in massive galaxies, OR
- **Reverse-engineering from user's successful phenomenological fits**

**Recommendation**: Implement Option D (reverse-engineer α from K(R)) to guarantee consistency with user's 175-galaxy fits, then identify microphysical scaling laws from the extracted α(M, Q, σ_v, R_d) dataset.
