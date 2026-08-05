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

## Reproducibility

- Frozen/corrected protocol: `configs/sigma_v19ad_fixed_aperture_color_commissioning.json`
- Runner: `scripts/run_sigma_v19ad_fixed_aperture_color_commissioning.py`
- Aggregated aperture inventory: `data/derived/sigma_v19ad_fixed_aperture_color_commissioning/aggregated_sample.csv`
- Machine-readable report: `results/sigma_v19ad_fixed_aperture_color_commissioning/report.json`
