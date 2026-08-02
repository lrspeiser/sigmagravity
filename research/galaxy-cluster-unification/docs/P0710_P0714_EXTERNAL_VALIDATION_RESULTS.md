# P0710-P0714 external validation results and next stage

## Bottom line

The frozen two-potential candidate transferred well to 13 new dwarf-galaxy
rotation curves, but it did not establish galaxy/cluster unification.

- On published 1D circular speeds, the candidate scores **10.735 km/s** versus
  **12.403 km/s** for the best frozen full-field MOND comparator, with no target
  refits and one universal setting.
- On resolved 2D velocity pixels, it scores **32.270 km/s** versus **32.827
  km/s** for the best frozen MOND field. Newtonian's lower **24.962 km/s** pixel
  residual shows that fixed geometry and non-circular gas motions dominate this
  particular metric; it does not erase Newtonian's poor 1D circular-speed score.
- The four-cluster preregistered lens test is **not validly evaluable**: only two
  selected clusters pass the frozen raw-constraint readiness requirements.
- On those two ready clusters, an explicitly exploratory raw-root diagnostic
  finds only **one image root per source family** for the candidate, including
  after repairing a discovered sky-axis convention error. The observations
  contain two to nine images per family.

The honest status is therefore: promising universal dwarf-galaxy transfer,
but no defensible cluster-lensing success and no unified theory yet.

## What was opened

P0710 acquired all 51 products frozen by P0709: 26 LITTLE THINGS moment maps,
the Iorio source package, and 24 RELICS compact-halo map products. They total
1,004,204,557 bytes and are content-addressed. The official Iorio online-table
supplement was resolved separately after the arXiv source proved not to contain
the ring tables. Its SHA-256 is
`967110269d59357ee3a94d1d6e46c2402aef38da3f674180d42044ceaf094173`.

Opening those outcomes permanently spends P0633. No equation or parameter
change after P0710 can be described as validation on this sample.

## Galaxy results

### Published circular speeds (P0711)

| Model | Equal-galaxy RMSE (km/s) |
|---|---:|
| P0707 time potential | 10.735 |
| QUMOND simple-nu 3D | 12.403 |
| AQUAL simple-mu 3D | 12.422 |
| Newtonian 3D | 22.070 |

The candidate/best-MOND ratio is `0.8655`; the frozen gate was `<=1.05`. Its
worst predefined morphology-bin ratio is `0.9347`; the gate was `<=1.25`.
All 13 galaxies are valid. Distance, inclination, PA, stellar mass-to-light,
and gravity parameters were not refitted.

### Resolved velocity fields (P0712)

| Model | Equal-galaxy weighted pixel RMSE (km/s) |
|---|---:|
| Newtonian 3D | 24.962 |
| P0707 time potential | 32.270 |
| QUMOND simple-nu 3D | 32.827 |
| AQUAL simple-mu 3D | 32.892 |

The candidate/best-MOND ratio is `0.9830`, so it passes the declared MOND
comparison. The measured receding-side handedness is shared by all models and
was not selected by score. The radio beam is measured from each FITS product.

This metric should not be overinterpreted. A circular model cannot reproduce
feedback, streaming, warps, shells, and other non-circular gas velocities. The
large common residual and Newtonian ordering show that the pixel score is more
sensitive to those effects and frozen photometric geometry than the 1D
asymmetric-drift-corrected circular-speed product.

## Cluster readiness (P0713)

The preregistration required at least three secure families, one spectroscopic
family, eight images, and all four selected clusters ready.

| Cluster | Secure families | Spectroscopic families | Secure images | Ready |
|---|---:|---:|---:|:---:|
| AS295 | 6 | 4 | 18 | yes |
| MACS0025 | 2 | 1 | 7 | no |
| MACS0159 | 4 | 0 | 10 | no |
| PLCKG287 | 12 | 12 | 47 | yes |

Only two of the required four are ready. The selected failures cannot be
silently replaced after unsealing. This is a catalog-readiness failure, not a
formula score, so the four-cluster claim is neither passed nor failed by the
candidate.

## Descriptive ready-subset lensing (P0714)

The ready-subset run is explicitly exploratory. It uses one mean source
position per family, the deterministic frozen family split, Planck18 distance
ratios, global root search, minimum-cost image/root assignment, and the same
glafic v2 comparator method for both clusters. No center, shear, ellipticity,
mass sheet, radial scale, or gravity parameter is fitted.

The candidate, baryon-only GR, AQUAL, and QUMOND fields each yield one root per
family. Consequently only one of every two to nine observed images can be
matched, and image-plane RMS is incomplete rather than a finite bad number.
AS295 glafic reconstructs all six heldout images at **0.718 arcsec RMS** with
correct heldout multiplicity. The frozen 2019 PLCK glafic map finds roots for
all 12 heldout images but scores **27.893 arcsec RMS**, driven by later 2024
families—especially family 40—that were not constraints of that older map.
This makes it a poor numerical dark-matter benchmark for those new families,
even though it remains the method frozen before unsealing.

The catalogs omit arc orientation/parity, so a critical curve is not an
independent observable. P0714 reports model-to-model curve distances but marks
the gate `not_observable` rather than treating a published lens map as data.

## Coordinate audit

P0641 sky maps are stored as image row `north`, image column `east`. P0708 fed
those arrays to a Cartesian solver whose component 0 was labeled `x/east` and
component 1 `y/north`. P0714 therefore reports two versions:

1. the exact frozen P0708 axis contract; and
2. a post-unseal axis-repaired diagnostic that samples `[north,east]` and swaps
   the returned components back to `[east,north]`.

Both versions produce one root per family. The bug matters for map orientation
and invalidates any claim based on the repaired field as prospective
validation, but it is not the reason the candidate lacks multiple-image
topology.

## What this teaches us about the equation

The two-potential split remains conceptually useful: massive tracers respond to
the time potential `Psi`, while photons respond to the Weyl combination
`(Psi+Phi)/2`. P0711 supports the fixed RAR-like `Psi` response in new dwarfs.
P0714 shows that making the Weyl deflection large is not enough.

Strong lensing is controlled by spatial derivatives of the deflection—the
local Hessian/Jacobian that creates folds and caustics. The current routed
correction is broad and smooth. It can supply tens of arcseconds of deflection
without placing the derivative structure around the observed arcs. The next
equation must predict that anisotropic local structure from baryons; another
global amplitude or radial multiplier will not address the observed failure.

## Concrete next stage and rethink points

### A. Repair and lock the measurement engine

Outcome: point-mass and asymmetric-map tests prove that east/north sampling,
components, WCS signs, distance ratios, Jacobians, and global roots are correct.
The source-plane scatter of a comparator on the constraints it actually fit
must reproduce its published scale.

Rethink if: any convention can be changed without failing a hash or regression
test, or root counts depend materially on grid/start density.

### B. Develop topology-producing physics only on spent data

Outcome: with one universal setting, a baryon-derived Weyl/Hessian law must
create at least the observed root multiplicity for every AS295 family before
image RMS is considered. PLCK's later families are a secondary stress test,
not a clean comparator fit.

Rethink if: after three distinct, physically derived tensor/Hessian mechanisms,
the field still produces one root per family, or success requires per-cluster
orientation/amplitude/shear. At that point the current routing premise should
be retired rather than given more parameters.

### C. Build a genuinely prospective cluster sample

Outcome: before computing predictions, select at least four clusters that each
already have three secure families, one spectroscopic family, eight images,
modern raw catalogs, baryonic maps, and a comparator fit version matched to
that catalog. Freeze source-redshift rules, root search, topology threshold,
arc parity/orientation availability, formula, and parameters before unsealing.

Required progression numbers remain: 100% heldout root convergence, candidate
image RMS no worse than `1.25x` the common compact-halo comparator, at least 75%
of the baryon-to-halo gap closed, correct heldout topology, and no cluster more
than 5% worse than baryon-only. Galaxy and Solar gates pass independently.

### D. Host the simulator for external researchers

The researcher API should be developed from these same immutable manifests and
runners. The deployment sequence, safe formula language, asynchronous worker
architecture, reproducibility requirements, and public-launch checks are in
[`PUBLIC_SIMULATOR_API_PLAN.md`](PUBLIC_SIMULATOR_API_PLAN.md). The draft
machine-readable contract is [`../api/openapi.yaml`](../api/openapi.yaml).

Vercel should host the browser and short control-plane calls. Container workers
on Modal, Cloud Run Jobs, AWS Batch, or Kubernetes should run the Python field
solvers beside versioned S3/R2 data. Researchers will be able to call real
systems, generate seeded observation-matched galaxies, validate a safe formula,
queue a full test, and retrieve content-addressed artifacts.

## Reproduction

```powershell
python scripts/run_p0711_external_galaxy_rotation_validation.py
python scripts/run_p0712_external_galaxy_velocity_field_validation.py
python scripts/run_p0713_external_cluster_readiness_audit.py
python scripts/run_p0714_ready_subset_raw_lensing.py
python -m pytest tests/test_p0709_external_unlock_manifest.py tests/test_p0710_p0714_external_results.py
```
