# Sigma v19AA member-counterpart association plan

## Purpose

V19Y and V19Z retained every HSC and NSC candidate around the 141 frozen
spectroscopic members.  V19AA converts those candidates into auditable
association probabilities.  It does not infer stellar mass, construct a
mass-current map, open a lensing target, or modify gravity physics.

The Bullet coordinates are too coarsely rounded for nearest-neighbor matching.
Their published right ascensions are quantized to a whole time-second, which
corresponds to a half-bin of about 4.2 arcsec on the sky at declination
`-56 deg`, while declination has a half-bin of 0.5 arcsec.  Abell 2146 uses the
smaller rounding footprint implied by its decimal coordinates.  The primary
likelihood is therefore the exact rectangular rounding kernel convolved with
the catalog astrometric uncertainty, not a circular distance rank.

## Frozen measurement model

HSC and NSC detections are first deduplicated by their stable catalog IDs.
Cross-survey rows are merged only when they are reciprocal nearest neighbors
within 0.5 arcsec.  HSC `DSigma` is interpreted in milliarcseconds and NSC
`raerr/decerr` in arcseconds, following their catalog metadata.

For one coordinate offset `d`, rounding half-width `w`, and catalog error
`sigma`, the positional density is

\[
 p(d)={1\over2w}\left[
 \Phi\!\left({w-d\over\sigma}\right)
 -\Phi\!\left({-w-d\over\sigma}\right)
 \right].
\]

The two coordinate densities are multiplied and divided by a conservative
cluster-level background density.  That density subtracts at most one possible
counterpart from each nonempty cone and adds a Jeffreys half-count.  It is not
fit from a gravity or lensing residual.

The counterpart prior is not treated as known.  Posteriors are reported for
`Q = 0.80, 0.90, 0.95, 0.99`.  A hard counterpart is called secure only when
the same candidate has posterior at least 0.90 at every `Q`, exceeds the second
candidate likelihood by a factor of ten, agrees with a global one-to-one
assignment, has dual-survey or repeated-detection support, and is not jointly
flagged as a high-significance proper-motion point source.  Every other member
remains ambiguous or unmatched, with its complete posterior retained.

## Why photometry is not used yet

Published Bessel B/R/I, NSC griz, and HST ACS magnitudes are different passband
systems.  V19AA attaches their availability but does not invent a color
equivalence or tune a transformation while resolving ambiguous identities.
The next protocol may calibrate a filter/SED likelihood using only secure
associations and must propagate that calibration uncertainty before stellar
mass inference.

## Decision

V19AA advances if every input hash verifies, all 141 members receive a
normalized candidate-plus-null posterior, and the global MAP assignment is
one-to-one.  Scientific sufficiency is separate: the report must state how
many secure members each cluster supports.  Ambiguous members are never
silently converted to nearest neighbors.

The HSC field definitions and artifact warning are documented by
[STScI](https://catalogs.mast.stsci.edu/hsc/detailed-fields.html).  The NSC DR2
catalog description and astrometric context are documented by
[NOIRLab](https://datalab.noirlab.edu/data/nsc).
