# Flow Coherence: Hybrid Approach Implementation

## Answer to Your Question

**Is the 6D data version not possible with SPARC because:**
1. SPARC data doesn't have 6D information, OR
2. It doesn't predict SPARC properly?

**Answer: SPARC data doesn't have 6D information.**

### SPARC Data Limitations
- **Available:** Only rotation curves (R, V_obs, V_gas, V_disk, V_bulge)
- **Missing:** Individual star positions/velocities (6D phase-space)
- **Result:** Can only use axisymmetric approximation (ω = V/R, s = |dV/dR|)

### Gaia Data Advantages
- **Available:** 6D phase-space for individual stars (R, z, φ, v_R, v_φ, v_z)
- **Enables:** Direct computation of flow topology from velocity gradients
- **Current limitation:** `test_gaia()` still uses rotation curve approach, not individual star data

## Hybrid Implementation

**Solution:** Use flow topology only where we have 6D data:

1. **SPARC:** Force baseline C (even when `--coherence=flow` is set)
   - SPARC doesn't have 6D data → use baseline C = v²/(v²+σ²)
   - This preserves the proven baseline performance

2. **Gaia/MW:** Use flow coherence when `--coherence=flow` is set
   - Gaia has 6D data → can use flow topology
   - Currently uses rotation curve approximation, but could be extended to use individual star data

## Current Status

- ✅ SPARC: Uses baseline C (even in FLOW mode) - preserves RMS=17.42 km/s
- ⚠️ Gaia: Uses flow coherence but still via rotation curve (not individual star data)
- 🔄 Future: Extend `test_gaia()` to compute flow topology from individual star positions/velocities

## Why This Makes Sense

1. **Respects data limitations:** SPARC can't provide what it doesn't have
2. **Leverages available data:** Gaia 6D data enables flow topology
3. **Preserves proven performance:** SPARC baseline remains intact
4. **Enables future improvement:** Can extend Gaia to use true 6D flow topology

## Next Steps

1. **Extend `test_gaia()`** to compute flow topology from individual star data:
   - Use `export_gaia_pointwise_features.py` approach
   - Compute vorticity, shear from actual star positions/velocities
   - This would be the "true" 6D flow topology approach

2. **Test if 6D flow topology improves Gaia predictions:**
   - Compare rotation-curve flow vs. individual-star flow
   - See if vorticity being #1 driver translates to better predictions

3. **Keep hybrid approach:**
   - SPARC: baseline C (no 6D data)
   - Gaia: flow topology (6D data available)


