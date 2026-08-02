# P0663 common-resolution tensor audit

## Measurement correction

P0661 and P0662 failed because directly point-sampling 65-cell dwarf maps onto
33 cells changed the proposed quadratic activation by about 51%. P0663 did not
change the gravity formula or any primary score. It corrected only the
resolution comparison:

1. convolve each native component map to an effective Gaussian width of one
   half coarse pixel;
2. evaluate the filtered map on its native grid;
3. conservatively sample that same filtered map onto the coarse grid; and
4. compare the two predictions at their now-common physical resolution.

The added native Gaussian width is fixed by the grid ratio,
`0.5 sqrt(r^2-1)` pixels. This is a measurement operator, not an adjustable
gravity parameter.

## Frozen result

All 15 gates pass:

- P0662 primary scores reproduce bitwise when its CSV is read with round-trip
  float precision;
- maximum component-map mass-conservation error: `4.44e-16`;
- galaxy median common-resolution change: `30.115%`;
- cluster median common-resolution change: `11.880%`;
- frozen maximum allowed change: `35%`;
- unchanged primary cluster/galaxy activation ratio: `60.3433x`; and
- no new universal or per-object gravity parameter.

The squared physical tidal-length tensor is authorized for outcome-blind
real-map field solves. Galaxy velocities and lens constraints remain sealed.

## Interpretation

The earlier registered-map failure was dominated by aliasing, not by the field
coefficient's mathematical grid convergence. This does not mean all
small-scale structure is irrelevant. It means comparisons between data sets
must carry their measured beam or point-spread function into the prediction.
Future public simulator requests should therefore include a map-resolution or
PSF descriptor as observational metadata.

## Claim boundary

P0663 establishes neither rotation-curve accuracy nor cluster lensing accuracy.
The candidate remains projected and nonrelativistic, and its 10 kpc coherence
length remains phenomenological.

## Reproduction

```powershell
python scripts/run_p0663_common_resolution_tensor_audit.py
python -m pytest tests/test_observational_resampling.py tests/test_p0663_common_resolution_tensor_audit.py -q
```
