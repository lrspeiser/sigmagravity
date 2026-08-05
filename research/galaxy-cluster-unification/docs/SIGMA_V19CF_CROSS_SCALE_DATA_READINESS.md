# Sigma V19CF cross-scale data readiness

## Decision

The source universes are broad enough to build the preregistered galaxy and
cluster tests, but the prospective core is **not execution-ready**. This is a
useful distinction: having hundreds of catalog entries or several lens maps is
not the same as possessing an admitted, sealed, raw-observation holdout.

V19CF reads only existing source and metadata products. It opens no new
rotation speed, velocity field, lensing coordinate, convergence map, shear
map, gravity residual or Solar-System target. It selects no galaxy, cluster,
source invariant, action, formula or universal constant.

## Galaxy readiness

The local target-blind WALLABY source frame contains 592 unique canonical H I
sources distributed across many source-shape and observing-condition cells.
Of those, 103 have kinematic availability under every registered source-row
policy and another six under some policies, for 109 under at least one policy.
That exceeds the future requirement of 32 WALLABY members in a 48-galaxy
holdout.

The important unresolved issue is identity, not sample size. The release-side
inventory contains 711 source rows and 18,550 possible optical candidates.
V19CE carries all candidates through four foreground treatments and four H I
smoothing kernels. No counterpart, treatment or kernel is selected. A later
photometric and kinematic likelihood must marginalize over this mixture
without allowing the velocity result to change the source weights.

Accordingly:

- source universe: ready;
- final morphology-balanced sample: not selected;
- new velocity targets opened: zero;
- prospective galaxy execution: not ready.

## Cluster readiness

The metadata-only shortlist has eight systems: four on the relaxed side and
four on the disturbed side. This is enough to seek six systems satisfying the
frozen balance requirements, but none is admitted yet. Admission still needs
source-complete stars, gas, BCG, intracluster light and member-galaxy
uncertainties, plus constraint-count and positional-error metadata, while raw
image coordinates remain sealed.

The four locally resolved RELICS systems do not fill this gap. They are spent
development systems. All four have projected baryonic maps and two published
lens-model methods, but zero are prospective holdouts and zero are currently
raw-score-ready with registered per-image positional uncertainties. Their halo
or convergence maps can generate hypotheses or act as disclosed comparators;
they cannot validate Sigma Gravity.

Accordingly:

- balanced source-side candidate universe: ready;
- admitted prospective clusters: zero of six;
- selected final clusters: zero;
- prospective raw-lensing execution: not ready.

## Ordered work

1. Complete the protected V19W5 through V19BQ source chain and run the frozen
   V19BS disposition.
2. Admit six source-complete cluster holdouts without opening their image
   coordinates.
3. Admit 48 morphology-balanced galaxies without opening their velocity
   targets.
4. If V19BS authorizes it, derive and freeze the least-field-content healthy
   one-metric action and no more than five universal constants.
5. Open the galaxy and cluster targets once, score every core gate, then apply
   the unchanged metric to weak lensing, merger transfer and the broader
   dark-matter-attributed phenomena.

Solar-System optimization remains intentionally later. Local consistency is
still a mandatory exclusion gate after a candidate has earned galaxy and
cluster testing.

## Reproduction

```powershell
python scripts/audit_sigma_v19cf_cross_scale_data_readiness.py
python -m pytest tests/test_sigma_v19cf_cross_scale_data_readiness.py -q
```

The machine-readable result is
`results/sigma_v19cf_cross_scale_data_readiness/report.json`.
