# Integration Plan: Roughness into Σ-Gravity Kernel

## Overview

Phase 2 tests revealed that:
- **K(Ξ) is system-level**: Works across galaxies, not within galaxies
- **K is constant per system**: This is expected behavior
- **Roughness explains ~⅓ of enhancement**: Need to find the "missing factor"

## Architecture

### New Structure:

```
K_total(R) = K_rough(Ξ) × C(R/ℓ₀)
```

Where:
- **K_rough(Ξ)**: System-level amplitude from time-coherence (Phase-2 fit)
- **C(R/ℓ₀)**: Burr-XII radial shape (unit amplitude, normalized to [0,1])

### Missing Factor:

```
F_missing = A_empirical / K_rough
```

Where A_empirical is what Σ-Gravity would fit per galaxy.

---

## Implementation Steps

### Step 1: Refactor Kernel Structure ✅ COMPLETE

**Files Created:**
- `burr_xii_shape.py` - Unit-amplitude Burr-XII shape C(R/ℓ₀)
- `system_level_k.py` - System-level K_rough(Ξ) from Phase-2 fit
- `test_sparc_roughness_amplitude.py` - Test script using new structure

**What it does:**
- Computes K_rough from mean exposure factor Ξ_mean
- Multiplies by Burr-XII shape to get K_total(R)
- Computes F_missing = A_empirical / K_rough

### Step 2: Analyze F_missing Correlations ⏳ RUNNING

**File Created:**
- `analyze_missing_factor.py` - Correlates F_missing with system properties

**What it does:**
- Merges roughness results with SPARC metadata
- Correlates F_missing with:
  - σ_v (velocity dispersion)
  - R_d (disc scale length)
  - Gas fraction
  - Morphology flags (bar, warp, bulge)
  - Environment

**Expected Output:**
- Identifies which properties correlate with F_missing
- Suggests physical mechanism for "missing" enhancement

### Step 3: Fit Microphysics Models to F_missing (TODO)

**Next Steps:**
- Modify coherence model fits to target F_missing instead of K_total
- Test which microphysics model (metric resonance, graviton pairing, etc.) best explains F_missing
- Identify second physical mechanism

---

## Files Structure

### New Files:
- `burr_xii_shape.py` - Burr-XII shape function
- `system_level_k.py` - System-level K_rough(Ξ)
- `test_sparc_roughness_amplitude.py` - Test script
- `analyze_missing_factor.py` - Correlation analysis
- `INTEGRATION_PLAN.md` - This file

### Output Files:
- `sparc_roughness_amplitude.csv` - Results with F_missing
- `F_missing_correlations.json` - Correlation analysis results

---

## Key Results So Far

### From Phase 2:
- **K(Ξ) relation**: K_rough = 0.774 · Ξ^0.1
- **System-level**: Works across galaxies, not within galaxies
- **Roughness explains ~⅓**: Need F_missing ≈ 2.5-3× for full enhancement

### Expected from Integration:
- **F_missing correlations**: Identify which system properties matter
- **Second mechanism**: Microphysics model that explains F_missing
- **Unified kernel**: K_total = K_rough × C(R/ℓ₀) with both mechanisms

---

## Next Steps

1. **Wait for roughness amplitude test** - Get F_missing values
2. **Run correlation analysis** - Identify F_missing correlates
3. **Fit microphysics models** - Explain F_missing with second mechanism
4. **Integrate into main pipeline** - Wire into Σ-Gravity rotation curve fits

---

## Status

- ✅ Step 1: Kernel refactoring complete
- ⏳ Step 2: Correlation analysis running
- ⏳ Step 3: Microphysics fitting (pending Step 2 results)

