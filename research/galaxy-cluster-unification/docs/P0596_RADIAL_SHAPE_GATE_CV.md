# P0596: radial-shape gate cross-validation

P0595 showed that redistribution success was associated most strongly with the
dimensionless baryonic ratio `C = R50/R80`. P0596 adds one logistic gate,

`H(C) = 1 / (1 + exp(-(C-C0)/w))`,

to the routed fraction. This quantity can be calculated from either a galaxy
radial mass profile or a cluster member-baryon map; it does not require a
galaxy type, gas label, or bulge classification. Larger `R50/R80` is described
only as a radial-shape ratio because its interpretation as concentration
depends on the profile family.

The same five whole-galaxy folds compare the complete shape-gated family, the
same search with `H=1`, and fixed RAR. Nuisance values remain fixed from the
parent analysis, so this is formula cross-validation rather than a nested
nuisance refit.

Run:

```powershell
python scripts/run_p0596_radial_shape_gate_cv.py
python -m pytest tests/test_conservative_diffusion.py tests/test_p0596_radial_shape_gate_cv_results.py -q
```
