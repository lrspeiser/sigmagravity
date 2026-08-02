# P0671 spent RX J2129 absolute 3D field results

## Frozen result: pass

All 22 structural gates pass on the same P0670 physical source and boundary:

- scalar/tensor normalized residuals: `8.773e-6 / 8.808e-6`;
- nonlinear iterations: `10 / 10`;
- maximum boundary mismatch: `0`;
- minimum tensor constitutive eigenvalue: `0.06053`;
- scalar/tensor median strong-lens physical deflection:
  `3.066 / 3.064 arcsec` before `Dds/Ds`;
- tensor/scalar deflection RMS ratio: `0.998730`;
- tensor-minus-scalar relative RMS: `0.0016773`; and
- scalar/tensor normalized deflection curl: `4.07e-16 / 4.48e-16`.

The tensor effect is therefore physically nonzero in the solved equation but
small: it changes the strong-lens deflection field by about `0.168%`. No photon
amplitude, slip, or per-object gravity parameter was fitted.

## Interpretation

This result separates two issues that earlier empirical lens corrections mixed
together. The absolute baryonic simple-AQUAL sector supplies roughly three
arcseconds of physical deflection on the audited annulus. The P0669 geometry
then redistributes that solution only slightly. A raw topology failure can now
be diagnosed as insufficient scalar strength, insufficient tensor leverage,
incorrect angular structure, or loss of critical curves, instead of merely as
a bad residual from an arbitrary additive correction.

P0671 is still not a lensing fit. The next spent-data protocol must freeze all
nuisance parameters and thresholds before calculating roots, parity,
multiplicity, critical curves, and image residuals.

## Reproduction

```powershell
python scripts/run_p0671_spent_rxj2129_absolute_3d_field_solve.py
python -m pytest tests/test_p0671_spent_rxj2129_absolute_3d_field_solve.py -q
```
