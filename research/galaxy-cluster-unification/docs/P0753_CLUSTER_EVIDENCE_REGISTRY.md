# P0753: role-safe resolved cluster evidence registry

## Outcome

The simulator now publishes a deterministic inventory of the four P0633
RELICS clusters through `GET /api/v1/cluster-evidence`. It answers three
different questions without letting one kind of evidence impersonate another:

1. What can source a forward gravity solve?
2. What can help discover a candidate baryon-to-halo response?
3. What can score a frozen prediction against raw observations?

The numerical answer is concise:

| Readiness class | Systems | Meaning |
|---|---:|---|
| Registered projected baryonic maps | 4 | Ready for local 2D field input, with declared stellar and gas uncertainty brackets |
| Two-method model-derived lens targets | 4 | Ready for inverse hypothesis generation, not theory validation |
| Raw catalog passes the frozen family/image/redshift gate | 2 | AS295 and PLCKG287 only |
| Raw forward score ready now | 0 | Registered per-image positional uncertainties are missing |
| Prospective blind holdouts | 0 | The P0633 sample was opened and is permanently spent |

Registry SHA-256:
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.

## What the four baryonic maps contain

Each 1025 by 1025 map combines actual member-galaxy F160W morphology with
reprojected Chandra gas morphology. A shared near-infrared mass-to-light rule,
published gas-mass normalization, and explicit uncertainty variants turn those
proxies into projected mass maps. No gravity parameter is selected per
cluster. The maps preserve measured offsets between stars and gas, ranging
from roughly 50 to 125 kpc in this sample.

This is materially better than member-light-only routing, but it is not a
unique three-dimensional baryon reconstruction. The principal limits are
single-band stellar population calibration, approximate X-ray emissivity and
line-of-sight depth, and absent uniformly calibrated intracluster light.

## What the inferred lens maps can do

Every system has GLAFIC and Zitrin LTM-Gauss convergence and deflection maps
with source URLs and component hashes. They can be used as explicitly labeled
`model_derived_discovery_target` arrays to ask whether one shared compact
operator maps baryons toward the effective mass distribution inferred by two
independent modeling methods.

That inverse is useful only as a hypothesis generator. A stable response that
beats radial-angle, phase, system-permutation, target-angle, and missing-baryon
nulls may deserve compression into an analytic law. It does not establish
that gravity physically traveled through the reconstructed halo locations.

## What blocks a defensible forward lensing verdict

The raw P0633 catalogs were opened after the frozen candidate lock, making all
four clusters spent. Only AS295 and PLCKG287 passed the preregistered minimum of
three secure families, one spectroscopic family, and eight images. Those two
contain 65 images across 18 families and round-trip through the raw-image
adapter, but their registered catalogs do not include defensible per-image
positional uncertainties. The adapter therefore refuses to invent a
likelihood.

MACS0025 has too few secure families and images. MACS0159 has no secure
spectroscopic family. The failed four-cluster readiness gate is a data result,
not evidence for or against a gravity formula.

## Minimum useful next build

The shortest path to a meaningful Sigma Gravity or inverse-halo test is:

1. Select at least four new clusters before opening their outcomes.
2. Require complete raw multiple-image positions, family membership,
   spectroscopic source redshifts, positional covariance, and a frozen image
   detectability policy.
3. Build uncertainty ensembles for stellar mass, hot gas, intracluster light,
   line-of-sight structure, WCS, PSF, and masks independently of the theory.
4. Learn only on spent model-derived maps, require all nulls to pass, and
   compress the result to a small universal law.
5. Freeze the equation, constants, metric/photon coupling, solver, and
   nuisance policy. Remove every halo target.
6. Predict raw held-out image positions and topology from baryons alone, then
   compare with baryons-only GR, MOND-like relativistic comparators, and
   disclosed per-cluster dark-matter fits with parameter counts visible.
7. Require the same constants to pass galaxy velocity fields and derived
   Solar-System limits in one conjunctive report.

## Reproduction

```powershell
python scripts/build_p0753_cluster_evidence_registry.py
python -m pytest tests/test_p0753_cluster_evidence_registry.py -q
cd hosted-simulator
npm.cmd test
npm.cmd run build
```

The builder reads existing public provenance and result metadata. It does not
open or parse the original sealed payload files. The hosted JSON contains
metadata, source URLs, hashes, scientific roles, and blockers—not the large
FITS/NPZ products themselves.
