# Sigma V19AV signed-flux stack plan

## Why this is a different test

V19AU required at least 80% of individual candidate exposures to have positive
flux and failed at 59.74%. Lowering that threshold after the result would be
invalid. V19AV instead tests the observable appropriate to repeated imaging: a
single multi-epoch flux likelihood that includes positive and negative
measurements.

## Frozen stack

Every exposure is transformed to a common characterization zeropoint of 30:

\[
F_{30}=F\,10^{0.4(30-ZP)}.
\]

The uncertainty receives the same scale factor. An inverse-variance mean is
iteratively reweighted with a Huber threshold of 2.5 standardized residuals,
for at most ten iterations. The final uncertainty is the larger of the formal
reweighted error and the robust exposure-to-exposure scatter divided by
square-root N.

No exposure is dropped for negative flux. A candidate band is called detected
only when the stacked signal-to-noise is at least three.

## Anchor consistency

The same stack is applied to the 670 development and 362 validation
measurements. A Bessel-to-stacked-DECam color transform is fitted only on the
unchanged ten development anchors. The already-open five validation anchors
must again meet the 0.25-mag, 3/5 rank-one and 0.65 mean-reciprocal-rank gates.

This is a consistency check for a changed flux-combination observable, not a
claim of a second untouched validation.

## Candidate source-sufficiency gates

- exactly 2,840 candidate/filter stacks and 75 anchor/filter stacks;
- at least 90% of 568 candidates detected at S/N at least three in every
  `griz` band;
- every one of 57 members has at least one such candidate;
- no candidate Bessel-color, positional, membership, mass, lensing, halo or
  gravity score.

Passing authorizes a separately frozen joint candidate likelihood. It does not
choose a counterpart or infer a mass map.
