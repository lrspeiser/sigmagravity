# Sigma v18A collisionless-stress readiness

## Result

The currently verified public AS295 spectroscopy does **not** satisfy the
already-frozen stage-B collisionless-member-stress gate.

Ruel et al. (2014) provides 39 AS295 spectra and reports 30 members. Bayliss et
al. (2016) provides 38 spectra and reports 29 members. Those numbers cannot be
added: all 38 Bayliss objects have a one-to-one Ruel counterpart within the
fixed 1-arcsec match radius. The combined release therefore contains 39 unique
spectra, not 77. A common transparent `+/-4500 km/s` cluster-rest-frame window
contains 30 unique candidates, leaving a 20-member shortfall against the
preregistered minimum of 50. The count also applies the frozen 1.8-Mpc
projected aperture using the registered AS295 center and Planck18 angular
scale.

| Evidence | Ruel 2014 | Bayliss 2016 | Deduplicated union |
|---|---:|---:|---:|
| Spectra | 39 | 38 | 39 |
| Published members | 30 | 29 | not additive |
| Fixed-window members | 30 | 29 | 30 |

The raw VizieR query responses are stored and hashed under
`data/raw/sigma_v18a_collisionless_stress/AS295`. The executable audit is
`scripts/audit_sigma_v18a_collisionless_stress_readiness.py`; its authoritative
result is `results/sigma_v18a_collisionless_stress_readiness/report.json`.

The obvious independent ACT releases do not fill the gap. An exact
`J0245-5302` query returns zero rows from both the Sifon et al. (2013) cluster
table and its 961-member object table. A 0.15-degree (9-arcmin) cone around
AS295 likewise returns zero rows from the 9,203-object Sifon et al. (2016)
catalog. That cone is wider than the frozen 1.8-Mpc aperture. These negative
query products are retained alongside the SPT tables so catalog coverage can
be rechecked if VizieR changes.

## What this changes

It closes a data-accounting loophole. The two citations used by later papers
do not supply two independent AS295 phase-space samples. The PLCKG287 catalog
is already adequate, but selecting a collisionless-stress formula on that one
cluster would violate the frozen symmetric-transfer design.

This is not a failure of collisionless stress as physics. It is a failure of
the current data package to test it at the declared resolution. We therefore
keep the following boundaries:

- do not lower the 50-member threshold after seeing the shortfall;
- do not replace missing velocities with photometric redshifts;
- do not run or rank a PLCKG287-only stress feature;
- do not count duplicated spectra twice; and
- do not open any holdout or interpret catalog readiness as theory evidence.

If v17E thermal stress fails, the defensible next options are to locate a truly
independent AS295 velocity release beyond the checked SPT/ACT catalogs that
supplies at least 20 new secure members,
freeze a different matched spent pair before constructing its stress field, or
move to the materially different causal-state branch already allowed by v18.

## Provenance boundary

The aggregate Ruel and Bayliss sample sizes and an exploratory near-complete
overlap were known before this audit file was written. This is consequently a
reproducible readiness audit, not a blinded protocol. It reads no lensing target
and selects no formula, kernel, amplitude, length, shear, or orientation.
