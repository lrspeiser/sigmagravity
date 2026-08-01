# R0 observable and identifiability audit

Status: provenance complete for the columns currently scored; the raw/likelihood
acquisition gate fails, so R1 and R2 are not yet authorized.

## Result

The current data cannot identify a single dynamical-plus-lensing response, nor can
they establish gravitational slip. The same-object pilot has **0 eligible systems
out of 10 required**.

| Sample | Systems | What is locally scored | Ready radial dynamics | Ready radial lensing | Eligible |
|---|---:|---|---:|---:|---:|
| SPARC | resolved galaxy sample | rotation curves plus baryonic component curves | yes, with geometry nuisances | 0 | not a same-object lensing sample |
| CLASH | 20 | 84 GR+NFW-deprojected summaries, 3-5 per cluster | 0 | 0 | 0 |
| SPIDERS-MaNGA BCG | 34 | 11 one-radius Jeans summaries and 23 calibrated proxies | 0 resolved likelihoods | 0 | 0 |

The distinction between “3-5 scored CLASH points” and “0 forward-model-ready
lensing points” is deliberate. Tian et al. (2020) calculate 3D total acceleration
from the joint NFW posterior supplied by Umetsu et al. (2016). That is a valuable
empirical reconstruction, but it is not a metric-independent lensing observable.
Tian et al. also ignored radial covariance in the RAR fit. Umetsu et al. describe a
total covariance containing statistical, systematic, uncorrelated-LSS and
intrinsic-profile terms, but those numerical per-cluster covariance products were
not located in the audited public release.

Likewise, the Tian et al. (2024) MaNGA BCG values are one-radius summaries produced
by a spherical, isotropic Jeans/Abel inversion of fitted velocity-dispersion
profiles. They are not the underlying radial IFS likelihood. The other 23 bridge
objects are calibrated mass proxies and cannot be used to reconstruct a potential.

## Artifacts and reproducibility

- `configs/r0_observable_audit.json` freezes the required columns, provenance and
  the 10-system/3+3 radial-point gate.
- `data/derived/r0_observable_provenance.csv` is the one-row-per-scored-column
  provenance matrix.
- `data/derived/r0_scored_observable_instance_provenance.csv` expands that lineage
  to every scored scalar: dataset, system, radial row, value and unit, exact local
  score-input file and SHA-256, raw-observable description, publication,
  transformation, metric/dynamics assumptions, covariance disposition and
  alternative-theory forward-model status.
- `data/derived/r0_same_object_coverage.csv` records coverage and the shortfall for
  every CLASH and frozen BCG system.
- `results/r0_observable_audit/report.json` is the machine-readable gate decision.
- `data/raw/clash_likelihood_audit/` records the public-product search.

Regenerate them with:

```powershell
python scripts/audit_r0_observables.py
```

## Concrete next-stage acquisition target

R1 can begin only after one of these two routes succeeds:

1. Obtain the 20 Umetsu per-cluster projected surface-density profiles and total
   covariance (preferably the shear, magnification and strong-lensing likelihood),
   then add resolved same-object BCG/galaxy kinematics and baryonic profiles.
2. Freeze a replacement sample of at least 10 systems selected solely by coverage,
   each with at least three overlapping radial constraints in stellar/gas dynamics,
   three in lensing, a measured baryonic profile and usable covariance.

The practical searches are public strong-lens galaxy samples with IFS kinematics,
cluster-member/BCG kinematic data overlapping public weak/strong lens models, and
author/data-center requests for the Umetsu likelihood products. A sample is not
accepted because its residual looks promising.

## Decision rule

No theory-free potential reconstruction and no new covariant action is run on the
current mismatch. When the coverage gate reaches 10 systems, reconstruct the
dynamical potential and Weyl potential separately, propagate covariance, and only
then compare one-latent versus two-potential held-out predictions.
