# Next Steps Status

## Checklist

### ✅ 1. Tune parameters (especially α - shear weight)
**Status:** ⚠️ **PARTIALLY DONE**
- Created `sweep_flow_coherence.py` script
- **Issue:** Sweep needs to be re-run now that hybrid approach is working
- **Action needed:** Run sweep to find optimal α, β, γ values

### ⚠️ 2. Refine unit normalization between vorticity and shear
**Status:** ⚠️ **BASIC VERSION DONE, MAY NEED REFINEMENT**
- Implemented normalization: `omega2_normalized = omega2 * (kpc_to_m / 1000.0)**2`
- This converts (rad/s)² to (km/s/kpc)² for comparison with shear²
- **Action needed:** Test if this normalization is optimal, or if a different scaling factor is needed

### ✅ 3. Hybrid approach: use flow topology for Gaia (6D data) but baseline C for SPARC (rotation curves only)
**Status:** ✅ **COMPLETE**
- SPARC: Uses baseline C (even when `--coherence=flow` is set)
- Gaia: Uses flow coherence when `--coherence=flow` is set
- **Result:** SPARC preserves RMS=17.42 km/s, Gaia can use flow topology

### ✅ 4. Export and compare C_flow vs C_baseline values to understand the difference
**Status:** ✅ **COMPLETE**
- Added `C_baseline` to FLOW mode diagnostics
- Added `C_flow` to baseline mode diagnostics (when FLOW is globally enabled)
- Created `compare_flow_vs_baseline.py` script
- **Result:** Comparison CSV generated showing C_baseline vs C_flow

## Summary

**Completed:**
- ✅ Hybrid approach implemented
- ✅ Coherence comparison export working

**In Progress:**
- ⚠️ Parameter tuning (sweep script ready, needs re-run)
- ⚠️ Unit normalization (basic version done, may need refinement)

**Next Actions:**
1. Run `sweep_flow_coherence.py` to tune α, β, γ parameters
2. Analyze `coherence_comparison.csv` to understand C_flow vs C_baseline differences
3. Refine unit normalization if needed based on comparison results



