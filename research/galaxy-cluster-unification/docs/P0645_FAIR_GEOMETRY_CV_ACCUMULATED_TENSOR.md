# P0645 fair geometry-refit cross-validation

## Why P0645 was necessary

P0644 froze lens geometry at values optimized for `lambda=0`.  That gave the
radial baseline an optimization advantage.  P0645 removes it by refitting the
same six conventional center, ellipticity, orientation, and external-shear
parameters for every tensor strength in every fold.

Both F160W and X-ray maps are convolved to a common 25 kpc resolution before
the component-cancellation field is calculated.  The smoothing scale, 10 kpc
coherence length, mass fractions, folds, and lambda grid were frozen before
the run.

## Cross-validation design

The 15 P0601 training images are split into five deterministic folds with
2--4 validation images per fold.  Assignment is stratified by source family:
every validation image leaves at least one other image of its family in the fit
so a source position can be inferred.  Across the five folds, all 15 images are
predicted once for each lambda.

This is stricter than reusing the seven P0601 spent holdouts to choose lambda,
but it is not leave-family-out validation.  Every RX J2129 image is already
spent.

## Result

The fair comparison rejects the current tensor:

| lambda | pooled CV RMS | complete CV roots |
|---:|---:|---:|
| 0 | 2.7603" | 15/15 |
| 0.5 | undefined | 13/15 |
| 1 | 2.9808" | 15/15 |
| 2 | 2.8540" | 15/15 |
| 3.5 | 3.0813" | 15/15 |
| 5 | 2.9835" | 15/15 |

The training-internal selection is `lambda=0`.  No positive, root-complete
candidate beats it, so the required one-percent CV improvement and nonzero
strength gates fail.  The complete progression audit passes 6 of 9 gates.

The final `lambda=0` full refit retains all 15 training and seven spent-heldout
roots, stays inside all ordinary parameter bounds, and does not materially
worsen P0599.  Those checks validate the comparison machinery, not the tensor.

## Scientific conclusion

P0643 and P0645 answer different questions:

- P0643: **yes**, finite path accumulation creates a real, baryon-only
  galaxy/cluster domain lever.
- P0645: **no**, the present choice
  `div[A h h grad(psi_carrier)]` does not place that response where the spent
  RX J2129 raw images require it.

The coherence-length idea is therefore not disproved, but this tensor closure
is rejected.  It must not be frozen for the four untouched RELICS clusters.

## Most informative next alternatives

The next formula should change *where* accumulated response lands, not merely
increase lambda:

1. Use the gas--star **difference direction** rather than the path-mean total
   field direction in the rank-one tensor.
2. Replace the rank-one tensor with a traceless tidal projector, comparing it
   against a matched generic quadrupole.
3. Introduce a time/relaxation equation so present gas--star offsets retain
   merger history rather than being treated as a static local coefficient.
4. Use a signed divergence/curl-free Helmholtz projection of the unsummed
   component flux rather than multiplying the already-summed carrier.

Each alternative should first be scored on the same five frozen folds.  The
four P0640 clusters remain sealed until one alternative clears the existing
root, CV, Solar, resolution, and conventional-control gates.

## Reproduction

```powershell
python scripts/run_p0645_fair_geometry_cv_accumulated_tensor.py
python -m pytest tests/test_p0645_fair_geometry_cv_accumulated_tensor.py -q
```
