# Sigma V19X3 full regional spectral-production preflight

## The missing executable stage is now prepared

V19X2 commissions two integrated spectra and one target-blind high-count
region in each cluster.  A pass authorizes—but does not itself execute—the
remaining regional workload.  V19X3 now supplies the checkpointed executor
and mechanical freezer for all 494 accepted regions: 366 in the Bullet Cluster
and 128 in Abell 2146, using all 5,082 response cells exactly once.

V19X3 is not frozen and cannot run yet.  Its freezer requires a terminal
passing V19X2 report, which in turn requires the still-running V19W production,
the V19W4 recovery/audit and the V19X2 commissioning run.  No spectrum,
temperature, gas source state, lensing target or gravity result was opened by
this preflight.

## Exact inherited calculation

The future freezer copies the following sections from the passing V19X2
configuration without changing them:

- direct `combine_spectra` summation and ASCA background scaling;
- grouping to 25 counts per channel group;
- the `xstbabs * xsapec` model over 0.5--7.0 keV;
- fixed HI4PI absorption and cluster redshift;
- cluster-wide abundance fixed to the passing integrated V19X2 fit;
- the optimizer fallback sequence and 68% profile-likelihood interval; and
- response, count-conservation, fit-statistic, interval and parameter-bound
  gates.

Each regional result includes separate 68% profile-likelihood intervals for
temperature and APEC normalization. The normalization profile is required
because it controls emission measure and therefore the gas surface-density
posterior. A finite best fit without an ordered normalization interval is
retained but does not count toward the individual-quality minimum.

Every manifest cell is grouped by its already-frozen cluster and `bin_id`.
Duplicate task keys, changed cluster/region counts or a cell not represented
exactly once fail before combination.

## Crash-safe and outcome-safe checkpoints

Each region has two independently written checkpoints:

1. A combination checkpoint records an input digest made from the ordered cell
   identities, source-PHA hashes, PHA count totals and fixed abundance.  Reuse
   requires all four frozen spectrum/response products to remain byte exact,
   the PHA counts to remain conserved and the FITS links to remain correct.
2. A fit checkpoint records the same digest and the complete fit or retained
   exception.  A failed or low-quality fit is not silently rerun until it gives
   a preferred temperature.

This lets a long production run resume after a process interruption without
recombining completed regions or selecting outcomes.

## Correct regional decision rule

The earlier V19BJ wording incorrectly suggested that all 494 individual
quality gates must pass.  That was corrected before any regional result
existed.  The inherited V19H rule is:

- all 494 regions must be attempted;
- every region must have a finite temperature, normalization and fixed
  abundance best fit;
- every response cell must participate in exactly one region;
- every combination must conserve counts and response links; and
- at least 12 regions per cluster must pass the complete individual
  uncertainty/statistic/bound gate.

A finite fit that misses the individual quality cut remains in the posterior
with its uncertainty.  It cannot be dropped, merged, split or selectively
refit.  This matches the prior V17 thermal-map precedent and the frozen V19H
source protocol.

## What V19X3 still will not provide

Regional APEC temperature and normalization are inputs to, not substitutes
for, a gas-state posterior.  After V19X3 passes, a separate frozen stage must
derive emission measure, gas surface density, pressure, entropy, shock state,
line-of-sight depth/projection uncertainty and their covariance.  Only then
can V19BJ score source invariants.  Lensing remains sealed throughout.

## Verification

Nine synthetic tests currently prove region grouping, interval parsing, cell accounting,
finite-fit versus quality-gate behavior, checkpoint reuse, V19X2 authorization
and mechanical inheritance of the 494-region/12-quality-pass rules.

```powershell
python -m pytest tests/test_sigma_v19x3_full_regional_spectral_production.py tests/test_freeze_sigma_v19x3_full_regional_spectral_production.py -q
```
