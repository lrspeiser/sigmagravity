# Sigma V19DF MACS J0018 component-current audit

## Terminal result

The public component-current source is **not admitted**. The exact public
catalog reconstruction succeeds, but only two of five member-gradient gates
pass:

| Diagnostic | Result | Gate |
|---|---:|---:|
| Final member rows | 156 | exactly 156: pass |
| OLS positive-velocity direction | 289.18 degrees east-counterclockwise | reported |
| Huber direction | 285.86 degrees | OLS/Huber axial difference 3.32 degrees: pass |
| OLS peak-to-peak plane amplitude | 1,505.84 km/s | reported |
| Velocity-shuffle null | p = 0.1313 | p <= 0.05: fail |
| Literature/Keck axial difference | 54.58 degrees | <= 30 degrees: fail |
| Bootstrap 95% direction displacement | -74.25 to +66.93 degrees | half-width <= 45 degrees: fail |

The visible northwest--southeast tendency is therefore not a stable global
directional source under the frozen reconstruction. The raw gas branch also
remains unavailable. V19DF does not calibrate a current coupling and does not
authorize an action derivation.

## Scope

V19DF asks whether public data for MACS J0018.5+1626 can support the next
Sigma source decision without using a lensing map, halo reconstruction or
gravity residual. It is deliberately narrower than a gravity test: it tests
whether signed gas and galaxy motion are independently observable well enough
to enter a later covariant action.

The system is development-only. The published velocity-map outcome was known,
and the paper's TeX rows were exposed while checking public availability.

## Public-data boundary

The exact arXiv v2 source contains the literature and Keck member-redshift
tables. It does **not** contain analysis-grade 140 and 270 GHz FITS maps, their
transfer functions, the X-ray temperature map used for the SZ separation, or
the 1,000 correlated kSZ noise realizations described in the paper. The
CaltechAUTHORS record attaches the accepted and published PDFs, not those map
products.

Consequently:

- the galaxy-member branch can be reconstructed;
- the gas-current branch cannot be executed independently;
- figure digitization is forbidden;
- the published 135-degree component offset is a diagnostic, not a scored
  measurement in this project.

## Frozen galaxy reconstruction

The two TeX tables contain 168 literature rows and 117 Keck rows. Removing rows
explicitly marked as duplicates and applying the paper's 6.7-by-6.7 arcmin box
leaves 161 rows. Exactly five literature rows are within one arcsecond of Keck
rows. Retaining the Keck measurement for those matches reproduces the paper's
stated sample of 156 galaxies: 98 literature and 58 Keck.

This one-arcsecond cross-table rule was inferred after seeing the development
tables and reported sample count. It is reproducible, but it is not prospective
validation.

For each retained galaxy the frozen velocity is

\[
v_{\rm los}=c\frac{z-0.546}{1+0.546}-100\ {\rm km\,s^{-1}}.
\]

No redshift uncertainty is supplied in the public tables, so none is invented.
No stellar mass or luminosity is supplied, so this is a number-sampled velocity
field rather than a physical mass-current map.

## Registered decision

The primary diagnostic fits one plane to velocity as a function of east and
north position. Mandatory checks use Huber robust regression, literature-only
and Keck-only fits, 4,096 bootstrap resamples and a 4,096-draw velocity-shuffle
null. A directional source is admitted only if all frozen stability and null
gates pass. Even then, a component-resolved source would remain closed until
the independent gas map and covariance existed.

The terminal report is
`results/sigma_v19df_macsj0018_component_current/report.json`.

## Interpretation boundary

The published gas/galaxy misalignment has one immediate architectural lesson:
a candidate should not collapse gas and collisionless-galaxy motion into one
scalar coherence value before nonlinear response. That would erase observed
component information. It does not tell us the coefficient or action term that
should couple to either component, and it cannot calibrate Sigma Gravity.
