# Gaia Bulge Regression Test: Design Document

## Objective

Use Gaia bulge as calibration lab for the **covariant coherence scalar**:

\[
\mathcal{C}_{\rm cov} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}
\]

Then translate back to SPARC via proxies.

## Test Framework

### 1. Data Selection

**Bulge Star Selection:**
- Spatial cuts: `R ≤ 3.0 kpc`, `|z| ≥ 0.5 kpc`
- Minimum stars per bin: 50 (for stable gradients)
- Full 6D phase-space information required

**Binning:**
- Radial bins: `R ∈ [0, 3.5] kpc` in 0.5 kpc steps
- Vertical bins: `z ∈ [-2.0, 2.0] kpc` in 0.5 kpc steps
- Only keep bins with ≥50 stars

### 2. Flow Invariants Computation

**From Binned Velocity Field:**
- **ω²**: Compute from mean v_φ and R (ω ≈ v_φ/R for axisymmetric)
- **θ²**: Expansion scalar (≈ 0 for steady-state bulge)
- **ρ**: Evaluate from Milky Way baryonic model (McMillan 2017)

**Key:** Use baryonic density model, NOT fitted parameters (avoid leakage)

### 3. Covariant Coherence

**Implementation:**
```python
C_cov = C_covariant_coherence(omega2, rho_kg_m3, theta2)
```

**Formula:**
\[
\mathcal{C}_{\rm cov} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}
\]

All terms in consistent units: (km/s/kpc)²

### 4. Prediction

**Enhancement:**
\[
\Sigma = 1 + A(L) \cdot \mathcal{C}_{\rm cov} \cdot h(g_N)
\]

**Circular Speed:**
\[
V_{\rm pred} = V_{\rm bar} \sqrt{\Sigma}
\]

### 5. Validation

**Metrics:**
- RMS residual: `V_obs - V_pred`
- Comparison to baseline: `C = v²/(v²+σ²)`
- Improvement threshold: **>1 km/s** required to pass

**Observable:**
- Mean rotation velocity from binned v_φ
- Corrected for asymmetric drift if needed

## Success Criteria

### Pass/Fail Threshold
- **Pass**: Improvement > 1.0 km/s over baseline
- **Fail**: Improvement ≤ 1.0 km/s

### Reporting
- Report bulge-only pass/fail alongside existing suite
- Include: RMS, baseline RMS, improvement, n_stars, n_bins
- Details: mean C_cov, mean ω², mean ρ, mean θ²

## Integration into Regression Suite

### Test Function Signature
```python
def test_gaia_bulge_covariant(
    gaia_df: Optional[pd.DataFrame],
    use_covariant: bool = True,
) -> TestResult:
```

### CLI Flag
```bash
--test-gaia-bulge  # Enable Gaia bulge covariant test
```

### Output
- Separate test result: "Gaia Bulge (Covariant)"
- Included in core/extended suite as appropriate
- JSON report includes bulge-specific metrics

## Implementation Status

### ✓ Completed
- `C_covariant_coherence()` function implemented
- Test framework designed (`test_gaia_bulge_covariant.py`)
- Binning and selection logic outlined

### ⏳ Pending
- Validate bulge star selection criteria
- Test binning and gradient computation on real data
- Integrate into `run_regression_experimental.py`
- Validate baryonic density model computation
- Set final success thresholds

## Next Steps

1. **Validate Selection**: Test bulge star selection on Gaia catalog
2. **Test Binning**: Ensure stable gradients with ≥50 stars/bin
3. **Validate Density**: Compare computed ρ to literature values
4. **Run Initial Test**: Execute on existing Gaia data
5. **Set Thresholds**: Determine realistic improvement targets
6. **Integrate**: Add to regression suite

## Expected Outcomes

**If Covariant C Works:**
- Significant improvement (>1 km/s) over baseline
- Better capture of bulge kinematics
- Validates theory direction

**If Not:**
- Identifies where theory needs refinement
- Guides next steps (different invariants, different density model, etc.)

## Files

- `test_gaia_bulge_covariant.py`: Test implementation
- `GAIA_BULGE_TEST_DESIGN.md`: This document
- Integration into `run_regression_experimental.py`: Pending

