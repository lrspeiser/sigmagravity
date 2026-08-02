# P0662 physical tidal-length tensor audit

## Correction tested

P0662 removed the inherited one-pixel floor and 48-pixel cap from the coherence
length. It used

\[
\ell=\min\left({|g_N|\over\|\nabla g_N\|_F},
d_{\rm boundary}(\hat g_N)\right)
\]

in physical units. The only cap is the forward distance to the observed map
boundary. The quadratic accumulation kernel and all other universal settings
were unchanged.

## Frozen result

The corrected estimator passes controlled tests but P0662 still fails the
registered-map resolution gate:

- physical scale-covariance error: `1.78e-15`;
- synthetic median resolution change: `2.03%`;
- galaxy nominal median weighted `sigma`: `0.00121138`;
- cluster nominal median weighted `sigma`: `0.0730985`;
- cluster/galaxy ratio: `60.341x`;
- cluster median registered resolution change: `11.76%`;
- galaxy median registered resolution change: `50.64%`; and
- frozen resolution limit: `35%`.

All other gates pass. The candidate does not advance and no outcome is opened.

## What remains unstable

The close agreement on smooth synthetic maps demonstrates that the physical
tidal estimator itself converges. The same dwarf galaxies remain outliers as in
P0661. When the 65-cell maps are directly sampled onto 33 cells, both the local
component mismatch and derivative scale change. The current resampler performs
linear point sampling without an anti-alias filter. It therefore folds
unresolved fine structure into the coarse map instead of comparing the two
grids at a common physical resolution.

The next numerical question is whether a conservative, anti-aliased
common-resolution comparison removes that artifact. This cannot rescue P0662;
it requires a new frozen audit. If it does not, the quadratic coefficient is
genuinely too dependent on unresolved dwarf morphology.

## Claim boundary

P0662 is still a structural coefficient test, not evidence for rotation curves
or lensing. The 10 kpc scale is not derived, and finite field of view remains an
explicit limitation.

## Reproduction

```powershell
python scripts/run_p0662_physical_tidal_length_tensor_audit.py
python -m pytest tests/test_physical_tensor_activation.py tests/test_p0662_physical_tidal_length_tensor_audit.py -q
```
