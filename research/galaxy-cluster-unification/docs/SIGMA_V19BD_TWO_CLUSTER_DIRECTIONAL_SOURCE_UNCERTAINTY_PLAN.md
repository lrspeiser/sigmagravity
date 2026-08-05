# Sigma V19BD two-cluster directional-source uncertainty plan

## Question

The Bullet and Abell ensemble-mean maps both show displaced velocity moments
and approximately 50-arcsec separation between positive and negative
line-of-sight current centroids. V19BD asks whether those features survive the
complete positional, counterpart, missing-photometry, and redshift uncertainty
in each frozen member ensemble.

The comparison is source-only. It does not use gas maps, lensing, halo models,
or a gravitational response.

## Scale-free construction

For each draw, let `ell_i` be the measured relative luminosity and
`beta_i=v_parallel/c`. Compute the luminosity centroid and covariance,

\[
 \boldsymbol r_L={\sum_i\ell_i\boldsymbol r_i\over\sum_i\ell_i},
 \qquad
 C_L={\sum_i\ell_i(\boldsymbol r_i-\boldsymbol r_L)
 (\boldsymbol r_i-\boldsymbol r_L)^T\over\sum_i\ell_i},
\]

and use

\[
 R_L=\sqrt{\operatorname{tr}C_L}
\]

as that draw's own light-distribution scale. The second-moment centroid is
weighted by `ell_i beta_i^2`; the two signed-current centroids are weighted by
`ell_i max(beta_i,0)` and `ell_i max(-beta_i,0)`.

The primary distances are

\[
 d_\Pi={|\boldsymbol r_\Pi-\boldsymbol r_L|\over R_L},
 \qquad
 d_j={|\boldsymbol r_+-\boldsymbol r_-|\over R_L}.
\]

The current separation axis is also compared with the luminosity major axis
using `cos(2 Delta theta)`. Positive one means major-axis alignment; negative
one means minor-axis alignment. This spin-2 statistic is unchanged if an axis
is reversed.

## Uncertainty discipline

All 8,192 draws in each frozen ensemble are used. Abell rows with missing F814W
remain in member accounting but do not receive invented weights. The two
photometric bands are never compared in amplitude. Equal sample IDs are paired
only to produce a deterministic Monte Carlo distribution of the difference
between two independent posteriors.

The report will give 2.5, 16, 50, 84, and 97.5 percentiles. These are source
uncertainty intervals, not frequentist detection significances or a
two-cluster population inference.

## Why this precedes the field equation

A long-wavelength mode can only be predictive if its source orientation and
coherence are determined by baryonic measurements. If the dimensionless
directional statistics are dominated by catalog uncertainty, they should not
be elevated into a field source. If they remain narrow but differ strongly
between the two clusters, a universal operator must respond to the actual
tensor state rather than assume one merger template. If they are both stable
and comparable, the result supports carrying the Reynolds-stress construction
into the gas-combined action study.

No outcome in V19BD selects a wavelength, coupling, phase, polarization, or
gravity formula. V19W/V19X remain mandatory before the source-state operator is
frozen and exposed to sealed lensing data.
