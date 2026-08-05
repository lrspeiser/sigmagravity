# Sigma V19AD fixed-aperture color commissioning plan

## Question

V19AB's catalog-total magnitudes predicted held-out offsets well, but color-only
retrieval failed.  V19AD tests whether that failure was caused by inconsistent
aperture/segmentation rather than an unusable Bessel-to-DECam color relation.

No ambiguous candidate is opened.  The ten development and five validation
singleton IDs are unchanged.

## Frozen measurement

The primary measurement is the median `c4d` magnitude in the same
4-arcsec-diameter aperture for each of `g/r/i/z`.  Only individual measurements
with SExtractor flag zero, finite magnitude below 90, and finite reported
error between zero and one magnitude enter the median.  The rule does not
select on seeing or agreement with the published Bessel value.

Two- and eight-arcsecond apertures are computed as sensitivity diagnostics.
They cannot rescue a failure of the frozen four-arcsecond primary.

The model predicts only the three DECam colors `g-r`, `r-i`, and `i-z` from
the published Bessel colors `B-R` and `R-I`.  Absolute magnitude never enters
the validation score.

## Gates

The primary aperture must have complete `griz` for all 15 rows, all optimizers
must succeed, and each held-out median absolute color error must be at most
0.25 mag.  Blind retrieval among the five validation detections must reproduce
at least 3/5 provisional pairs at rank one with mean reciprocal rank at least
0.65—the same retrieval thresholds that V19AB missed.

A pass only authorizes a separately frozen application of the color likelihood
to ambiguous candidates.  A failure sends source reconstruction to synthetic
SED/filter curves, forced photometry on the original images, or better source
coordinates.  Neither outcome is a gravity result.
