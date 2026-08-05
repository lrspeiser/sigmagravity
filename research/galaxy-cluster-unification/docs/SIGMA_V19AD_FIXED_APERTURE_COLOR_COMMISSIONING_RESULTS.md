# Sigma V19AD fixed-aperture color commissioning results

## Decision

V19AD failed its frozen data-completeness gate before fitting a color model.
No held-out color score, validation retrieval, ambiguous-member likelihood,
counterpart, mass, current, lensing, halo, or gravity calculation was made.

This is not the same failure as V19AB.  V19AB had complete catalog-total
photometry but insufficient color-only retrieval.  V19AD attempted cleaner
same-aperture colors and found that one predetermined validation anchor lacks
accepted measurements in two bands.

## Exact failure

Bullet member 57, NSC object `179969_8549`, has the following accepted
four-arcsecond-aperture measurements under the frozen rule:

| Band | Accepted measurements |
|---|---:|
| `g` | 1 |
| `r` | 0 |
| `i` | 0 |
| `z` | 4 |

The underlying acquisition contains 25 `r` and 14 `i` measurements, but every
one has SExtractor flag 3.  The preregistered rule required flag zero, so none
can enter.  The two- and eight-arcsecond sensitivity apertures have the same
missing-band problem and cannot repair the primary result.

The first runner invocation stopped at this expected gate without serializing
a report.  A post-freeze implementation-only correction made the runner write
the failure and missing-band evidence.  It did not change the aperture, flag,
error, split, model, threshold, or authorization, and the model still was not
fit.

## Interpretation

The obstacle is now deblending rather than filter transformation.  Catalog
`MAG_AUTO` values exist for member 57, but using them would return to the
aperture-sensitive evidence that V19AB refused to authorize.  Simply allowing
flag 3 after seeing this row would be a retrospective quality-cut change.

The clean next routes are:

1. independent deblended photometry from a catalog such as DELVE or DECaPS;
2. forced photometry on the DECam images with one frozen segmentation/aperture
   model; or
3. higher-precision source coordinates or original FORS1 object identifiers.

A synthetic SED alone cannot replace the missing observed `r/i` colors; it can
only forward-model colors once usable measurements exist.

## Subsequent independent-catalog preflight

A read-only NOIRLab Data Lab crossmatch was run only for the same 15
already-opened V19AB singleton anchors.  No V19AA ambiguous-candidate cone was
queried, and no association, mass, lensing, halo, or gravity result was read.

- `nsc_dr2.x1p5__object__delve_dr2__objects` returned one DELVE DR2 match for
  every anchor, with separations from 0.0021 to 0.0618 arcsec.
- `nsc_dr2.x1p5__object__decaps_dr2__object` returned no match for any anchor.
- All 15 DELVE matches have catalog `MAG_AUTO` values in `g/r/i/z`.
- Thirteen anchors have DELVE SExtractor flag zero in every band.  Member 71
  has flag 3 in `i`.  Problem member 57 has flag 3 in **all four bands**, with
  3, 5, 2 and 2 catalog epochs in `g/r/i/z`, respectively.

Thus DELVE confirms the object and supplies multiband values, but it does not
supply the clean independent deblending required to rescue the frozen V19AD
gate.  Accepting member 57 after observing the flag pattern would be a
retrospective quality-rule change.  DECaPS cannot help because it has no
coverage.  The remaining defensible routes are image-level forced/profile
photometry under a preregistered model or recovery of the original FORS1
identifiers/segmentation.

## Reproducibility

- Frozen/corrected protocol: `configs/sigma_v19ad_fixed_aperture_color_commissioning.json`
- Runner: `scripts/run_sigma_v19ad_fixed_aperture_color_commissioning.py`
- Aggregated aperture inventory: `data/derived/sigma_v19ad_fixed_aperture_color_commissioning/aggregated_sample.csv`
- Machine-readable report: `results/sigma_v19ad_fixed_aperture_color_commissioning/report.json`
