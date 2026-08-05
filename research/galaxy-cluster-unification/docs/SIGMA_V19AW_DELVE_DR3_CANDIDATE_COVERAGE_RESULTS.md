# Sigma V19AW DELVE DR3 candidate-coverage results

## Decision

V19AW **failed closed**. Only 56/568 candidate positions (9.86%) have a DELVE
DR3 catalog match within the frozen 0.5-arcsecond radius with measurable signed
flux in all four `griz` bands. The frozen requirement was at least 90%.

Only 38/57 spectroscopic members have at least one complete candidate, so the
second source-sufficiency gate also failed. No candidate was selected or
ranked.

## Frozen result

| Gate | Requirement | Result | Pass? |
|---|---:|---:|---:|
| DELVE field rows | exactly 2,351 | 2,351 | yes |
| Candidates evaluated | exactly 568 | 568 | yes |
| Candidates with any 0.5-arcsec match | diagnostic | 108 (19.01%) | -- |
| Complete signed-flux `griz` candidates | at least 90% | **56 (9.86%)** | **no** |
| Members with at least one complete candidate | 57/57 | **38/57** | **no** |
| Multiple matches retained | all | 0/0 | yes |
| Null matches retained | all | 460/460 | yes |
| Candidate association score | forbidden | none | yes |

Among the 108 matched objects, valid signed-flux coverage exists for 65 in
`g`, 98 in `r`, 104 in `i` and 83 in `z`. The `g` band is the largest
photometric completeness loss, but it is not the main failure: 460 candidates
have no catalog object inside the astrometrically meaningful match radius.

## What the failure identifies

The broad candidate list is dominated by HSC identities inherited from the
earlier source-cone construction:

| Candidate provenance | Candidates | DELVE matches | Complete `griz` matches |
|---|---:|---:|---:|
| HSC | 529 | 85 (16.1%) | 36 (6.8%) |
| NSC | 39 | 23 (59.0%) | 20 (51.3%) |

The nearest DELVE-object separation has a median of 2.52 arcsec. There are only
135 nearest objects inside 1.0 arcsec, so even doubling the radius would remain
far below the 90% gate while creating a serious false-association risk in this
crowded cluster field. The correct response is not to enlarge the radius.

DELVE DR3 is deep enough to contain 2,351 detections in the small field, but a
catalog is still a detection and deblending product. It does not guarantee a
catalog row at every externally supplied HSC coordinate. Therefore this result
separates two issues that had previously been mixed together:

1. the DELVE catalog identity does not reproduce most of the HSC candidate
   identities at sub-arcsecond precision; and
2. even among the exact matches, only about half have complete signed `griz`
   flux under the frozen definition.

This is stronger than saying only that the images are shallow. It shows that a
second catalog crossmatch cannot construct the required baryonic map from this
broad candidate set.

## Consequence for the gravity investigation

V19AW does not test the long-wavelength gravity equation. It tests whether one
of its required inputs—a defensible baryonic density/current map—can be made
from catalog crossmatches. This route cannot.

The next defensible route is forced photometry on homogeneous **coadd images**
at the already-frozen HSC/NSC positions, preferably with a simultaneous crowded
field or prior-based deblending model. That keeps the source coordinates fixed
and asks the pixels for a flux instead of requiring DELVE to have independently
created the same catalog object. It must be preregistered separately and
validated on the unchanged 10/5 anchors before candidate fluxes are interpreted.

If coadd-image forced photometry also fails source sufficiency, the mass map
must carry explicit null/low-information states; it cannot be made complete by
looser positional matching.

## Claim boundary

The result establishes catalog coverage and matching failure only. It makes no
stellar-mass, mass-current, lensing, halo or gravity claim, and it does not
change any parameter of the long-wavelength candidate.

Reproducibility:

- `configs/sigma_v19aw_delve_candidate_coverage.json`
- `scripts/run_sigma_v19aw_delve_candidate_coverage.py`
- `results/sigma_v19aw_delve_candidate_coverage/report.json`
- `data/raw/sigma_v19aw_delve_dr3_candidate_coverage/`
- `data/derived/sigma_v19aw_delve_candidate_coverage/`
