# Flow Coherence Implementation Status

## Summary

Flow-based coherence mode (`--coherence=flow`) has been successfully implemented based on the residual discovery findings that **vorticity is the #1 driver** in Gaia residuals.

## Implementation Complete

✅ **Flow coherence functions:**
- `flow_coherence_proxy()` - computes ω², s², θ from rotation curves
- `flow_coherence_from_proxy()` - computes C_flow = ω²/(ω² + α·s² + β·θ² + γ·T²)

✅ **Integration:**
- Added to `predict_velocity()` fixed-point loop
- CLI flags: `--coherence=flow`, `--flow-alpha`, `--flow-beta`, `--flow-gamma`
- Diagnostics: exports `omega2`, `shear2`, `C_term` (flow coherence)

✅ **Parameter sweep script:**
- `sweep_flow_coherence.py` created
- Tests different α, β, γ values

## Current Results

**Baseline (C = v²/(v²+σ²)):**
- SPARC RMS: 17.42 km/s
- Bulge RMS: 28.93 km/s
- Disk RMS: 16.06 km/s

**Flow mode (α=1.0, β=0.1, γ=0.5):**
- SPARC RMS: 25.38 km/s ⚠️ **Worse**
- Bulge RMS: 41.49 km/s ⚠️ **Much worse**
- Disk RMS: 23.49 km/s ⚠️ **Worse**

## Analysis

The flow mode is producing **worse results** than baseline, which suggests:

1. **Flow coherence values may be too low** - The C_flow values might be suppressing enhancement too much
2. **Unit normalization may need refinement** - The conversion from (rad/s)² to (km/s/kpc)² might not be optimal
3. **Parameter tuning needed** - The default α=1.0, β=0.1, γ=0.5 may not be optimal
4. **Implementation refinement** - The flow proxies (especially shear) might need better handling

## Test Results

Manual testing shows that α does affect C_flow values:
- α=0.1: C_flow ≈ 0.99 (very high, similar to baseline C)
- α=1.0: C_flow ≈ 0.91-0.96 (moderate)
- α=10.0: C_flow ≈ 0.49-0.86 (lower, more suppression)

This confirms the implementation is working, but the parameters need tuning.

## Key Insight from Discovery

The residual discovery showed:
- **Gaia**: `log10_omega` (vorticity) is #1 feature
- **SPARC**: Velocity/acceleration gradients dominate

This suggests flow topology **is** the missing variable, but:
- The **axisymmetric approximation** (ω = V/R, s = |dV/dR|) may not capture the full 3D flow structure
- The **unit normalization** between vorticity and shear may need refinement
- The **tidal proxy** (γ term) may need better implementation

## Next Steps

1. **Refine unit normalization** - Ensure ω² and s² are properly normalized for comparison
2. **Tune parameters systematically** - Run a focused sweep on α (shear weight) since it has the most impact
3. **Compare C_flow vs C_baseline** - Export and compare coherence values to understand why flow mode is worse
4. **Test on specific galaxies** - Focus on bulge galaxies to see if flow topology helps there
5. **Consider hybrid approach** - Maybe combine flow topology with baseline C: `C_hybrid = f * C_flow + (1-f) * C_baseline`

## Hypothesis

The flow topology hypothesis is still valid (vorticity is #1 in Gaia), but the **axisymmetric approximation** for SPARC rotation curves may not be sufficient. The real 3D flow structure (which we can measure in Gaia) may be fundamentally different from what we can infer from 1D rotation curves.

**Possible solution:** Use flow topology for Gaia/MW test (where we have 6D data), but keep baseline C for SPARC (where we only have rotation curves).



