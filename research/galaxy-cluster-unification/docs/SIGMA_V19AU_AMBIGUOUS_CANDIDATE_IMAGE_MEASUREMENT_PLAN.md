# Sigma V19AU ambiguous-candidate image-measurement plan

## Purpose

V19AT showed that one fixed image rule can recover internally consistent
colors for five sealed singleton galaxies, including the catalog-crowded member
57. V19AU now applies that same measurement to every candidate for the
remaining Bullet members with published Bessel `BRI`.

It deliberately stops before association. Candidate colors, Bessel colors,
positional posteriors, masses and gravity outcomes cannot select or rank an
image measurement.

## Frozen metadata plan

- Eligible spectroscopic members: **57**
- Member–candidate hypotheses: **640**
- Unique sky candidates: **568**
- Detector groups containing candidates: **123**
- Candidate–exposure measurements: **40,812**
- Geometric coverage: **all 568 candidates in `grizY`**

The exact measurement counts are 7,553 `g`, 13,757 `r`, 8,133 `i`, 6,175 `z`
and 5,194 `Y`. Each candidate has at least 11, 22, 11, 9 and 7 geometrically
available exposures in those bands, respectively.

The WCS plan was built from headers only. No candidate science pixel was
interpreted before the plan and implementation were frozen.

## Frozen measurement

V19AU reuses the exact V19AS/V19AT rule:

- four-arcsecond circular aperture;
- locally clipped affine background over 8--13 arcseconds;
- 2.5-sigma PSF-smoothed watershed detection;
- mask every non-target segment;
- scale the clean target sum by total-to-clean aperture area;
- no exposure or candidate selection.

Every planned row is retained. Positive fluxes receive characterization
magnitudes and uncertainties. Non-positive fluxes remain signed flux
measurements, and processing failures remain explicit rows. This is important:
a faint candidate must not disappear merely because it is inconvenient for a
later color likelihood.

## Source-sufficiency gates

The stage requires:

1. all 40,812 plan rows retained;
2. at least 80% positive-flux measurements overall;
3. at least 75% of candidates with at least one positive measurement in every
   `griz` band; and
4. no Bessel, positional, association, mass, lensing, halo or gravity score.

These gates test whether the images support a later likelihood. They do not
declare candidates with missing bands to be non-members.

## Claim boundary

A pass authorizes a separately frozen joint color/position likelihood with
explicit null and ambiguous states. It does not choose a counterpart, provide
final native-band AB photometry, infer stellar mass, construct a mass-current
map, or test the long-wave Sigma field.
