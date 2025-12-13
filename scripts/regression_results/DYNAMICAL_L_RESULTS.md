# Dynamical Path-Length (L_eff) Results

## Concept

Instead of hard-coding `L = L_0 = 0.4 kpc` for disk galaxies, compute effective coherence depth from local dynamical time:

**L_eff(r) = v_coh / Ω(r) = v_coh × r / V_pred(r)**

This is based on the idea that coherence is a phase-ordering process that propagates at universal speed `v_coh`. In one dynamical time `τ_dyn = 1/Ω`, the ordering can correlate over distance `L_eff`.

## Implementation

- Added `--ldyn` flag to enable dynamical path-length mode
- Added `--vcoh-kms` parameter to set coherence transport speed (default: 20 km/s)
- Modified `predict_velocity()` to compute `L_eff(r)` inside the fixed-point loop
- Uses existing `unified_amplitude(L_eff)` to compute radius-dependent `A(r)`

## Tuning Results

Tested **14 values** of `v_coh` from 5 to 100 km/s:

| v_coh (km/s) | SPARC RMS (km/s) | vs Baseline | Cluster Ratio | Status |
|--------------|------------------|-------------|---------------|--------|
| **Baseline** | **17.42** | - | 0.987 | ✓ Best |
| 5 | 19.09 | +9.6% | 0.987 | ✗ Worse |
| 8 | 20.60 | +18.3% | 0.987 | ✗ Worse |
| 10 | 21.57 | +23.8% | 0.987 | ✗ Worse |
| 15 | 23.77 | +36.5% | 0.987 | ✗ Worse |
| 20 | 25.67 | +47.4% | 0.987 | ✗ Worse |
| 30 | 28.80 | +65.3% | 0.987 | ✗ Worse |
| 50 | 33.49 | +92.3% | 0.987 | ✗ Worse |
| 100 | 41.11 | +136% | 0.987 | ✗ Worse |

## Key Findings

1. **Very small v_coh recovers baseline** - v_coh ≤ 0.5 km/s gives RMS=17.42 km/s (identical to baseline) because minimum constraint `L_eff ≥ L_0` dominates
2. **Larger v_coh makes SPARC worse** - Best is v_coh=5 km/s with RMS=18.87 km/s (vs baseline 17.42 km/s, +8.3% worse)
3. **Clusters unaffected** - All values give same cluster ratio (0.987), suggesting dynamic L doesn't affect cluster predictions
4. **No improvement found** - The dynamical path-length approach, as currently implemented, does not improve SPARC predictions

## Minimum Constraint Effect

With `L_eff ≥ L_0` constraint:
- **Very small v_coh** (≤0.5 km/s): `L_eff = L_0` everywhere → `A = A_0` → baseline behavior
- **Larger v_coh** (≥5 km/s): `L_eff > L_0` in outer disk → `A > A_0` → stronger enhancement, but worsens SPARC

## Possible Issues

1. **Feedback instability**: `L_eff` depends on `V_pred`, which depends on `A(r)`, which depends on `L_eff` - this creates a complex feedback loop that may be unstable
2. **Baseline L_0 is correct**: The fixed `L = L_0 = 0.4 kpc` for disk galaxies may actually be the right value
3. **Wrong scaling**: The formula `L_eff = v_coh × r / V_pred` may not be the correct relationship
4. **Missing constraint**: Maybe `L_eff` should have a minimum value (e.g., `L_eff ≥ L_0`) to prevent it from getting too small

## Next Steps

1. **Test with minimum constraint**: `L_eff = max(v_coh × r / V_pred, L_0)`
2. **Test different formula**: Maybe `L_eff = v_coh / Ω` should use `V_bar` instead of `V_pred` to break feedback loop
3. **Test with sigma-components**: See if component-mixed σ profiles help
4. **Analyze why it fails**: Check if `L_eff` values are reasonable (not too small/large)

## Files

- Tuning results: `regression_results/dyn_l_tuning_results.json`
- Script: `scripts/tune_dyn_l.py`
- Implementation: `scripts/run_regression_experimental.py` (with `--ldyn` flag)

