# Next Steps Completion Status

## Checklist Review

### ✅ 1. Tune parameters (especially α - shear weight)
**Status:** ⚠️ **PARTIALLY DONE - Not meaningful for SPARC**

**Findings:**
- Sweep completed: All α, β, γ values produce identical results (RMS=17.42 km/s)
- **Reason:** SPARC uses baseline C (hybrid approach), so flow parameters don't affect SPARC
- **Action needed:** Tune parameters for **Gaia** instead (where flow topology is actually used)

### ⚠️ 2. Refine unit normalization between vorticity and shear
**Status:** ⚠️ **BASIC VERSION DONE, MAY NEED REFINEMENT**

**Current implementation:**
- `omega2_normalized = omega2 * (kpc_to_m / 1000.0)**2` converts (rad/s)² to (km/s/kpc)²
- This makes omega² and shear² comparable units
- **Action needed:** Test if this normalization is optimal, or if a different scaling factor is needed

### ✅ 3. Hybrid approach: use flow topology for Gaia (6D data) but baseline C for SPARC (rotation curves only)
**Status:** ✅ **COMPLETE**

**Implementation:**
- SPARC: Uses baseline C (even when `--coherence=flow` is set)
- Gaia: Uses flow coherence when `--coherence=flow` is set
- **Result:** SPARC preserves RMS=17.42 km/s, Gaia can use flow topology

### ✅ 4. Export and compare C_flow vs C_baseline values to understand the difference
**Status:** ✅ **COMPLETE**

**Findings:**
- Comparison shows C_flow = C_baseline for SPARC (correlation = 1.0000)
- **Expected:** SPARC uses baseline C in both modes (hybrid approach)
- **Key insight:** Flow parameters don't affect SPARC because it uses baseline C

## Key Discovery

**The comparison reveals:** C_flow and C_baseline are **identical** for SPARC because:
1. SPARC uses baseline C (no 6D data available)
2. Both "baseline" and "flow" modes use baseline C for SPARC
3. Flow parameters (α, β, γ) don't affect SPARC results

**This means:**
- Parameter tuning for SPARC is not meaningful (it uses baseline C)
- Parameter tuning should focus on **Gaia** (where flow topology is actually used)
- The unit normalization may be fine, but we can't test it on SPARC since it uses baseline C

## Next Actions

1. **Tune flow parameters for Gaia:**
   - Create a Gaia-specific sweep that tests different α, β, γ values
   - Focus on improving Gaia RMS (currently 33.2 km/s vs 29.8 baseline)

2. **Test unit normalization:**
   - Since SPARC uses baseline C, we can't test normalization on SPARC
   - Need to test on Gaia or create synthetic test cases

3. **Extend Gaia to use true 6D flow topology:**
   - Currently `test_gaia()` uses rotation curve approach
   - Could extend to use individual star positions/velocities for true 6D flow topology

## Summary

**Completed:**
- ✅ Hybrid approach (SPARC: baseline, Gaia: flow)
- ✅ Coherence comparison export
- ✅ Parameter sweep (shows no effect on SPARC, as expected)

**Remaining:**
- ⚠️ Tune parameters for **Gaia** (not SPARC)
- ⚠️ Test unit normalization on Gaia or synthetic cases
- 🔄 Extend Gaia to use true 6D flow topology from individual stars



