# Sigma V19BX SkyMapper source-only optical candidates

## Frozen acquisition boundary

V19BX will query the SkyMapper DR4 master catalog around all 592 V19BW H I
centroids. Every query uses the same 60-arcsecond radius and exact source-only
projection. Every returned object is retained. No nearest object or apparently
galaxy-like object is declared to be the optical counterpart.

This order matters. If the association rule were chosen after seeing which
optical object improved a gravity residual, an apparent morphology effect could
be manufactured by counterpart selection.

## Why SkyMapper DR4

SkyMapper DR4 covers the Southern sky through declination +16 degrees in all
six optical bands, which includes the Hydra, Norma and NGC 4636 WALLABY fields.
Its master table provides positions, optical photometry, Petrosian radius,
quality flags and a stellarity diagnostic through a public TAP service.

SkyMapper also states that its extended-source photometry has not been
optimized. V19BX therefore uses the catalog only to measure candidate density,
quality and source-side coverage. Definitive morphology, bulge fraction and
stellar mass will require image-level forward modeling and independent
cross-checks.

## Prespecified diagnostics

For each H I source the future report will count:

- all optical objects inside 60 arcseconds;
- objects with usable r-band catalog photometry; and
- extended candidates having usable r-band photometry, a Petrosian radius and
  `class_star <= 0.5`.

These are counts, not associations. Crowding—especially in the Norma field—is
a scientific source-systematic that must remain visible.

## Sealed information

The query contains no WALLABY frequency, line width, velocity, rotation curve,
inclination, kinematic angle, surface-density model, residual, halo result,
Sigma score or MOND score. It cannot select an evidence split, alter a gravity
action or tune the Solar-System limit.

## Reproduction after the contract commit

```powershell
python scripts/acquire_sigma_v19bx_skymapper_source_only_candidates.py
python -m pytest tests/test_sigma_v19bx_skymapper_source_only_candidates.py -q
```

Primary source: <https://skymapper.anu.edu.au/data-release/dr4/>.
