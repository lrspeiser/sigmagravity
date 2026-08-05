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

## Result

All seven execution gates passed and all 18,550 candidate/release pairs were
retained. The primary one-beam branch has a median top-to-second margin of only
1.059. Although 24 of 711 maps reach 3:1 in that single branch, only 3 maps
(0.42%) retain both the same top identity and a margin of at least 3:1 across
all four kernel widths. Those three are in NGC 4636; Hydra and Norma have zero.

The field contrast is severe:

| Field | Release maps | Median candidates | Median one-beam margin | Robust 3:1 |
|---|---:|---:|---:|---:|
| Hydra | 420 | 12 | 1.115 | 0 |
| NGC 4636 | 147 | 8 | 1.064 | 3 |
| Norma | 144 | 82 | 1.019 | 0 |

Only 82 of the 119 duplicated Hydra names retain one top object across both
releases and all four kernels. The conclusion is therefore not merely that a
larger matching radius is noisy. The candidate identity is sensitive to the H
I reconstruction and smoothing scale, while crowded Norma sightlines contain
too many nearly tied optical objects.

V19BZ consequently rejects a hard H I-overlap counterpart. The next legitimate
source step is uniform optical imaging plus star masks and deblending, followed
by a mixture that retains ambiguity. This result does not score or modify a
gravity theory.

## Reproduction

```powershell
python scripts/run_sigma_v19bz_hi_optical_spatial_information_audit.py
python -m pytest tests/test_sigma_v19bz_hi_optical_spatial_information_audit.py -q
```
