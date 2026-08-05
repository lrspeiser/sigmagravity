# Sigma V19Z NSC member-photometry acquisition plan

## Purpose

V19Z repairs the source-coverage limitation found by V19Y without choosing a
convenient counterpart.  It acquires all NOIRLab Source Catalog DR2 candidates
inside the same frozen member-coordinate cones and losslessly extracts the
Bullet paper's reported Bessel `B`, `B-R` and `B-I` values.

The output remains an association problem, not a matched catalog.  Every NSC
row, exact ADQL statement and exact request URL is retained and hashed.

## Why NSC DR2

A preflight used only each cluster center, not any member coordinate.  Inside
a 0.02-degree cone, NSC DR2 returned 222 objects at the Bullet center and 104
at the Abell 2146 center.  DES DR2 returned zero Bullet-center objects.  The
Astro Data Lab metadata also confirmed the `nsc_dr2.object` schema and the
public `q3c_radial_query` interface.

NSC DR2 supplies precise positions, position errors, proper-motion estimates,
multi-epoch counts, `u/g/r/i/z/Y/VR` photometry, shape, star/galaxy and quality
metadata.  V19Z requests these measurements but applies no quality or
morphology cut and does not reinterpret catalog sentinel values.

## Frozen inputs

- all 78 Bullet and 63 Abell 2146 spectroscopic coordinates;
- the unchanged 6-arcsec Bullet and 1-arcsec Abell 2146 cone radii;
- the NSC DR2 `object` table and exact requested column list;
- deterministic `ORDER BY id` for raw-response reproducibility; and
- the already-acquired primary source `astro-ph/0202323`, Table 1.

The Bullet source paper reports FORS1 imaging through Bessel `B`, `R` and `I`
filters on a 6-by-6 arcmin field.  Seventy-two of its 78 spectroscopic members
have all three table entries.  The derived columns

\[
R=B-(B-R),\qquad I=B-(B-I)
\]

are exact algebraic reconstructions, not a transformation to NSC or HST
filters.

## What V19Z cannot do

V19Z cannot select the nearest object, reject a star, apply an NSC flag cut,
transform Bessel colors into NSC passbands, fit a spectral energy distribution,
infer stellar mass, construct a current map, inspect lensing targets or fit a
gravity parameter.

The later association protocol must freeze:

- the coordinate-quantization likelihood for each paper catalog;
- NSC astrometric uncertainty and a local-background/null-match model;
- proper-motion and morphology likelihoods rather than retrospective cuts;
- Bessel-to-NSC color prediction with calibration uncertainty;
- one-to-one/global assignment behavior for overlapping Bullet cones; and
- posterior ambiguity thresholds and marginalization before seeing any
  gravity or lensing outcome.

## Relationship to the physics

Mass-weighted member currents are needed to distinguish ordinary GR frame
dragging from any enhanced current response and from the source-generated
long-wave metric hypothesis.  The added mode is permitted to vary negligibly
inside a stellar system because its wavelength is much larger than that
system, but its galaxy/cluster phase, amplitude and tensor orientation must be
predicted from baryonic stress-energy.  A wrong member association would
rotate or displace that source tensor, so resolving source uncertainty is part
of the physics test rather than clerical cleanup.

## Reproduction after freeze

```powershell
python scripts/download_sigma_v19z_nsc_member_photometry.py
python -m pytest -q tests/test_sigma_v19z_nsc_member_photometry.py
```
