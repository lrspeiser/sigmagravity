# SRC2 and SC Coherence Models - Results

## Implementation

Implemented two new coherence modes based on the insight that **coherence must be a source-current order parameter, not a test-particle-speed order parameter**:

### SRC2 (Fixed Source-Current)
- Computes coherence from **component speeds** (V_gas, V_disk) only, not total V
- Adds **coherent-fraction gating**: `f_coh = V_coh^2 / V_bar^2`
- Applies Σ only to disk+gas (bulge stays Newtonian)
- Prevents bulge from "leaking" into coherence by inflating total V

### SC (Field-Level Coherence)
- Same coherence calculation as SRC2 (component speeds + gating)
- But applies Σ to **total V_bar** (field-level coherence interpretation)
- Represents the idea that coherence is a field state that amplifies whatever mass is inside it

## Test Results

| Model | Overall RMS | Bulge RMS | Disk RMS | vs Baseline |
|-------|-------------|-----------|----------|-------------|
| **C (baseline)** | **17.42** | **28.93** | **16.06** | - |
| **SRC (legacy)** | 18.22 | 33.66 | 16.40 | +0.80 km/s |
| **SRC2** | 19.51 | 41.63 | 16.91 | +2.09 km/s |
| **SC** | 18.48 | 34.60 | 16.59 | +1.06 km/s |
| **SRC2 + σ-components** | 19.51 | 41.63 | 16.91 | +2.09 km/s |

## Key Finding: Coherent-Fraction Gating Works, But Makes Things Worse

For a high-bulge galaxy (NGC4217, f_bulge=0.832):
- At R=0.87 kpc: V_bar=129.4, V_gas=3.1, V_disk=15.9, V_bulge=128.4
- `f_coh = 0.016` (very small, as intended)
- The gate is working: bulge-dominated regions get low coherence

**However**, this makes predictions **worse**, not better:
- Bulge RMS: 28.93 → 41.63 km/s (+44% worse)
- Overall RMS: 17.42 → 19.51 km/s (+12% worse)

## Why It Makes Things Worse

1. **Bulge component gets no enhancement** (as intended in SRC2)
2. **Disk+gas component gets very little enhancement** because `f_coh` is so small
3. **Total prediction is too low** because the dominant bulge component doesn't get enhanced

This suggests that:
- The coherent-fraction gating is working correctly (suppressing coherence when bulge dominates)
- But the **fundamental assumption** that "bulge should not get enhancement" may be wrong
- Or the problem is that **bulge galaxies need a different coherence mechanism entirely**

## Comparison with A-Suppression Approach

| Approach | Overall RMS | Bulge RMS | Change |
|----------|-------------|-----------|--------|
| Baseline | 17.42 | 28.93 | - |
| A-suppression (best) | 17.41 | 28.87 | -0.01 km/s |
| SRC2 | 19.51 | 41.63 | +2.09 km/s |

**Conclusion**: Neither A-suppression nor SRC2/SC solve the bulge problem. The gap persists and in some cases widens.

## Next Steps

1. **Investigate if bulge should get some enhancement** (maybe different mechanism)
2. **Try radius-dependent coherence** (bulge more important at small radii)
3. **Consider that the problem might be in the coherence calculation itself** (not just which components participate)
4. **Test if the issue is with how bulge fraction is calculated** (maybe f_bulge is not the right metric)

## Implementation Details

- Added `SRC2` and `SC` modes to coherence model selection
- Updated SRC-family precompute to include all three modes
- Component-speed coherence calculation for SRC2/SC
- Coherent-fraction gating: `f_coh = V_coh^2 / V_bar^2`
- SC mode applies Σ to total V_bar instead of just disk+gas



