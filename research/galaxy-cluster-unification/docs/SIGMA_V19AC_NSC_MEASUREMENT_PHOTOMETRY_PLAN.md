# Sigma V19AC NSC per-measurement photometry acquisition plan

## Purpose

V19AB failed because color-only retrieval was insufficient even though total
magnitudes retrieved all five provisional pairs.  Total magnitudes are
aperture- and deblending-sensitive.  V19AC acquires the lower-level NSC
measurements needed to compare every source through the same fixed aperture.

This is acquisition only.  It cannot choose an aperture, reject an exposure,
combine measurements, score a color, rank a candidate, or select a
counterpart.

## Metadata finding before freeze

NSC DR2 exposes a `meas` table containing `MAG_AUTO` and fixed 1-, 2-, 4-, and
8-arcsec diameter aperture measurements for each exposure.  Its `exposure`
table records the instrument, filter, seeing, and zero-point uncertainty.

A preflight used only the 15 singleton IDs already opened by V19AB.  It found
1,032 measurements, all from `c4d` (DECam): 196 `g`, 348 `r`, 208 `i`, 152
`z`, and 128 `Y`.  No ambiguous candidate was queried before this freeze.
Thus the immediate Bullet comparison does not suffer from mixed-camera filter
curves, although exposure and aperture effects still require measurement.

## Frozen acquisition

The 226 unique nonblank NSC IDs in the V19AA unified-candidate table are sorted
and queried in deterministic batches of 25.  Each TAP query joins
`nsc_dr2.meas` to `nsc_dr2.exposure`, requests the exact frozen schema, and is
ordered by object, filter, MJD, and measurement ID.

Every returned row, exact ADQL query, and encoded request form is retained and
hashed.  No quality flag, aperture, band, instrument, morphology, or candidate
identity is filtered.

## Gate and downstream boundary

The stage passes only if all 226 requested objects return at least one
measurement, every batch is HTTP 200 with the exact schema, no unrequested ID
appears, and all payloads and requests are preserved.

A pass authorizes a separately frozen fixed-aperture aggregation and the same
10/5 color-only validation split used by V19AB.  It does not authorize opening
ambiguous-member photometric scores until that new validation passes.
