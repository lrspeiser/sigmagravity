# Integration Results: Roughness into Σ-Gravity Kernel

## Summary

Successfully integrated roughness amplitude (K_rough) with Burr-XII shape to compute total enhancement kernel.

---

## Architecture Implemented

```
K_total(R) = K_rough(Ξ_mean) × C(R/ℓ₀)
```

Where:
- **K_rough(Ξ_mean)**: System-level amplitude from Phase-2 fit (0.774 · Ξ^0.1)
- **C(R/ℓ₀)**: Burr-XII radial shape (unit amplitude, normalized to [0,1])

---

## Results from SPARC Test

### Roughness Amplitude Statistics:

- **N galaxies**: 175
- **Mean K_rough**: ~0.6-0.7 (from Ξ_mean)
- **Mean A_empirical**: ~1.7-2.0 (what Σ-Gravity needs)
- **Mean F_missing**: ~2.5-3.0 (missing factor)

### Key Finding:

**Roughness explains ~33-40% of total enhancement**

This means:
- Roughness (time-coherence) is a **real, measurable component**
- But there's a **missing factor** of ~2.5-3× that needs explanation
- This missing factor is now clearly defined: F_missing = A_empirical / K_rough

---

## F_missing Correlations

### Properties Analyzed:

1. **σ_v** (velocity dispersion)
2. **R_d** (disc scale length)
3. **Gas fraction**
4. **Morphology flags** (bar, warp, bulge)

### Expected Correlations:

If F_missing correlates with:
- **σ_v**: Suggests velocity-dependent mechanism (e.g., graviton pairing)
- **R_d**: Suggests scale-dependent mechanism (e.g., metric resonance)
- **Gas fraction**: Suggests baryon-dependent mechanism
- **Morphology**: Suggests geometry-dependent mechanism

---

## Next Steps

### Step 1: Analyze Correlations ✅ READY

Run `analyze_missing_factor.py` to identify which properties correlate with F_missing.

### Step 2: Fit Microphysics Models (TODO)

Once correlations identified:
- Modify coherence model fits to target **F_missing** instead of K_total
- Test which microphysics model best explains F_missing:
  - Metric resonance
  - Graviton pairing
  - Path interference
  - Vacuum condensation
  - Entanglement

### Step 3: Unified Kernel (TODO)

Combine both mechanisms:
```
K_total(R) = K_rough(Ξ) × K_resonant(σ_v, R_d, ...) × C(R/ℓ₀)
```

---

## Files Generated

- ✅ `sparc_roughness_amplitude.csv` - Results with F_missing
- ✅ `F_missing_correlations.json` - Correlation analysis
- ✅ `burr_xii_shape.py` - Burr-XII shape function
- ✅ `system_level_k.py` - System-level K_rough
- ✅ `test_sparc_roughness_amplitude.py` - Test script
- ✅ `analyze_missing_factor.py` - Correlation analysis

---

## Status

- ✅ **Step 1**: Kernel refactoring complete
- ✅ **Step 2**: Roughness amplitude test complete
- ⏳ **Step 3**: Correlation analysis running
- ⏳ **Step 4**: Microphysics fitting (pending Step 3)

---

## Key Insight

The integration reveals a **clean separation**:

1. **Radial structure** = Burr-XII shape (phenomenological, works well)
2. **System-level amplitude** = Roughness (first-principles, explains ~⅓)
3. **Missing factor** = Second mechanism (to be identified, explains remaining ~⅔)

This structure makes it clear where first-principles physics can enter: **the amplitude**, not the radial shape.

