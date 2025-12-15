# Flow-Based Coherence Implementation

## Overview

Implemented `--coherence=flow` mode based on the residual discovery findings that **vorticity is the #1 driver** in Gaia residuals.

## Key Finding

The residual discovery pipeline revealed:
- **Gaia**: `log10_omega` (vorticity) is the #1 feature predicting residuals
- **SPARC**: Velocity/acceleration gradients dominate residuals
- **Interpretation**: Flow topology (vorticity vs shear) is the missing variable, not just scalar speed

## Implementation

### Flow Coherence Formula

```
C_flow = ω²/(ω² + α·s² + β·θ² + γ·T²)
```

Where:
- **ω²**: Vorticity magnitude squared (from rotation curve: ω = V/R)
- **s²**: Shear magnitude squared (from velocity gradient: s = |dV/dR|)
- **θ²**: Divergence squared (typically ~0 for axisymmetric rotation)
- **T²**: Tidal/dephasing proxy (from acceleration gradient)

### Functions Added

1. **`flow_coherence_proxy(R_kpc, V_kms, R_d_kpc)`**
   - Computes ω², s², θ from rotation curve
   - For axisymmetric systems: ω = V/R, s = |dV/dR|

2. **`flow_coherence_from_proxy(omega2, shear2, theta, tidal_proxy)`**
   - Computes C_flow using the formula above
   - Handles unit normalization (rad²/s² → (km/s/kpc)²)

### Integration

- Added to `predict_velocity()` fixed-point loop
- CLI flag: `--coherence=flow` (also accepts `vorticity`, `vort`, `shear`, `topology`)
- Parameters: `--flow-alpha`, `--flow-beta`, `--flow-gamma`
- Diagnostics: exports `omega2`, `shear2`, `C_term` (flow coherence)

## Current Status

**Initial test results:**
- Flow mode runs successfully
- Same RMS as baseline (17.42 km/s) - needs tuning

**Next steps:**
1. Tune `FLOW_ALPHA`, `FLOW_BETA`, `FLOW_GAMMA` parameters
2. Test on bulge galaxies specifically
3. Compare with baseline to see if flow topology improves bulge predictions

## Hypothesis

Flow-based coherence should:
- **Disks**: High vorticity, low shear → high C_flow → enhancement on
- **Bulges**: Low vorticity, high shear → low C_flow → enhancement off/screened
- **Solar System**: Negligible vorticity → C_flow ~ 0 → no modification (safe)

This aligns with the discovery that bulges have different flow topology than disks.


