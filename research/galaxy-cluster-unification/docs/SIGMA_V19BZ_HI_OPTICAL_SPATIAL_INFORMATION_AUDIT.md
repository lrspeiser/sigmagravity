# Sigma V19BZ H I/optical spatial-information audit

## Why this audit exists

The blind WALLABY lane has 17,094 SkyMapper objects around 592 H I sources and
711 release-specific moment-zero maps. A convenient nearest-neighbor match
would silently turn foreground stars and crowded-field objects into baryonic
galaxies, especially in Norma. V19BZ asks a narrower question: does the H I
image itself contain enough positional information to distinguish the optical
candidates?

This is explicitly an exploratory source audit. The source pixels were
inspected before the protocol was written. No velocity, rotation curve,
lensing result, halo map, gravity residual or holdout result was inspected, so
the work cannot validate or tune a gravity law; it also cannot be described as
prospectively preregistered source evidence.

## Source-only calculation

For each release map, negative moment-zero noise is clipped to zero. The
positive image is evaluated without extra smoothing and after smoothing by
0.5, 1 and 2 geometric-mean beam FWHM. Each branch is normalized over the full
FITS cutout and sampled at every SkyMapper candidate. Dividing by the uniform
per-pixel density produces a dimensionless spatial likelihood ratio.

The audit reports every score, every ranking, the top-to-second margin and
whether the top identity survives all four kernel widths. SkyMapper
extendedness is carried as a diagnostic but receives zero weight. There is no
counterpart prior, null probability, posterior, candidate removal or hard
assignment.

## Interpretation boundary

A large, kernel-stable margin would show that H I morphology can inform a
future association model. A small margin means that optical images,
foreground-star masks, deblending uncertainty and a probabilistic mixture are
needed. Neither outcome tests Sigma Gravity.

The descriptive 3:1 margin and 90% coverage checks are exploratory because the
source data had already been inspected. Their role is to prevent an
unjustified hard match, not to select galaxies.

## Reproduction

```powershell
python scripts/run_sigma_v19bz_hi_optical_spatial_information_audit.py
python -m pytest tests/test_sigma_v19bz_hi_optical_spatial_information_audit.py -q
```
