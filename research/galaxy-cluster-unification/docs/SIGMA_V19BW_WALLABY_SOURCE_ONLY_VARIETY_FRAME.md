# Sigma V19BW WALLABY source-only variety frame

## Purpose

V19BW makes the galaxy-first priority testable before any rotation speed is
opened. It converts the complete 592-name V19BV source universe into a
coverage frame and separately describes the 109 names for which the public
archive says a successful kinematic product exists.

That flag is only target availability. The kinematic table, velocity fields,
rotation curves, inclinations, position angles, residuals and halo products
remain sealed.

## Five independent coverage axes

The availability lane is divided into quartiles using only source-side H I
measurements:

1. H I mass;
2. a relative H I compactness proxy combining mass, angular ellipse and
   distance;
3. source ellipse axis ratio;
4. distance; and
5. source extent in the catalogue's declared pixel units.

The compactness value is intentionally a relative proxy. It is not presented
as a calibrated physical surface density. These quartiles diagnose coverage;
they are not galaxy labels and they are not allowed into the gravity formula.

Together they expose failures that an aggregate rotation-curve average can
hide: low versus high gas mass, diffuse versus compact H I, flattened versus
round projected sources, nearby versus distant systems, and smaller versus
larger source masks. Release field is retained separately so Hydra, Norma
and NGC 4636 environments cannot be silently conflated.

## Coverage result

The 109-name kinematic-availability lane has 27 or 28 objects in every
quartile of every axis. Its release-field counts are 35 Hydra, 31 Norma and 43
NGC 4636. The five axes form 95 distinct occupied cells; the largest contains
only three names. This is good source-side variety, not proof of a representative
final physics sample.

Of all 592 source names, 103 retain the successful-model flag under every
prespecified release-row policy, six retain it under only some policies and
483 under none. Seventy-eight names change at least one source-metric quartile
under the alternate release choice. Those variations must be propagated when
the baryonic reconstructions are eventually scored.

## Release-row uncertainty

V19BV found 92 names whose preferred Hydra TR1/TR2 row changes under at least
one reasonable source-quality ordering. V19BW carries all five prespecified
choices through the availability and quartile calculations. Each name records
whether the kinematic-availability state or any source-metric quartile changes.

This is essential for a universal-gravity test: a result that depends on which
archive release happened to be preferred is a baryonic-source systematic, not
evidence for the force law.

## What this does not yet cover

H I source metadata alone cannot establish stellar surface brightness, bulge
fraction, bar/arm structure, stellar mass-to-light uncertainty, environment
membership, warp, or full three-dimensional gas geometry. Those source-side
measurements must be joined under a new blind protocol before the final galaxy
split is frozen.

No development, validation or holdout identifier is selected here. That split
must wait until the action and universal constants are frozen and the complete
baryonic/nuisance eligibility rules are fixed.

## Reproduction

```powershell
python scripts/build_sigma_v19bw_wallaby_source_only_variety_frame.py
python -m pytest tests/test_sigma_v19bw_wallaby_source_only_variety_frame.py -q
```

The machine-readable report is
`results/sigma_v19bw_wallaby_source_only_variety_frame/report.json`.

## Source context

WALLABY PDR1 contains source products for nearly 600 H I detections and
kinematic products for 109 spatially resolved galaxies. Primary sources:

- <https://arxiv.org/abs/2211.07094>
- <https://wallaby-survey.org/data/data-pilot-survey-dr1/>
