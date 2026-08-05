# Sigma V19BB Abell 2146 luminosity-current ensemble plan

## Why this stage is needed

The directional long-wavelength investigation now has a complete
candidate-marginalized luminosity/current ensemble for the Bullet Cluster, but
not for the second causal-development system, Abell 2146.  V19I could construct
only a number-current field there.  That is enough to study projected flow
geometry but not enough to compare physical luminosity-weighted collisionless
source moments under one rule in both clusters.

Abell has unusually favorable imaging coverage: 59 of 63 member cones contain
a frozen candidate and 60 of the 64 unified candidates have finite HST F814W.
However, the V19AA position-only model called only 13 associations secure.  It
would be invalid to replace that result with nearest-neighbor matching merely
because the field looks sparse.

The source inventory supplies an independent test.  V19AA measured a
background density of `0.027789 arcsec^-2`.  Across 63 one-arcsecond cones this
predicts about 5.5 unrelated candidates, while 64 candidate hypotheses were
observed.  V19BB asks whether one additional publication-to-HST centroid
scatter, shared by the complete Abell table, accounts for that excess and
predicts held-out member candidate sets.

## Source-only calibration

For member `m` and candidate `i`, retain the exact V19AA rounding rectangle and
candidate astrometric error but add one global catalog nuisance width `s`:

\[
 \sigma_i(s)=\sqrt{\sigma_{i,\rm ast}^2+(0.1\ {\rm arcsec})^2+s^2}.
\]

Let `f_mi(s)` be the V19AA quantized-coordinate density at the measured
candidate offset and `rho_bg` the already-frozen background density.  At
counterpart prior `q`, the member evidence and exact local state probabilities
are

\[
 Z_m(s;q)=(1-q)+q\sum_i {f_{mi}(s)\over\rho_{\rm bg}},
\]

\[
 p_{mi}={q f_{mi}/\rho_{\rm bg}\over Z_m},
 \qquad p_{m0}={1-q\over Z_m}.
\]

The primary prior remains the V19AA value `q=0.9`; `0.8`, `0.95`, and `0.99`
are mandatory sensitivities.  The width is selected from the frozen grid
`0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20 arcsec` by summed
source-only evidence.  Magnitudes do not enter selection.

Seven deterministic member folds provide the predictive check.  Each fold
selects the width on the other six and compares held-out log evidence with the
unchanged `s=0` control.  The result fails if the full-sample width is an
endpoint, fewer than five folds improve, or the selected grid index moves by
more than three positions across the prior sensitivities.

This width is a measurement nuisance for one published coordinate table.  It
is not `L_Sigma`, a cluster smoothing length, a halo scale, or a per-galaxy
parameter.

## Luminosity and current ensemble

Every one of 8,192 deterministic draws contains all 63 spectroscopic members.
Each member draws a candidate or its private null state from the exact selected
posterior.  Candidate positions are fixed catalog measurements; null positions
are sampled inside the original decimal-degree rounding rectangle.  Because
no Abell candidate is shared between members, the joint posterior factorizes
exactly and no one-to-one approximation is needed.

When the drawn candidate has finite HSC `A_F814W`, define

\[
 \ell_i=10^{-0.4(A_{F814W,i}-20)}.
\]

Quoted redshift errors are redrawn, the cluster median is recomputed, and the
rest-frame line-of-sight velocity supplies `ell_i v_parallel`.  A null state or
candidate without finite F814W remains explicitly missing.  No luminosity is
inferred from the `R_c<21` target limit, another member, or the subcluster
label.

The source is considered usable only if the exact posterior expects at least
50 measured-F814W members per draw.  This is intentionally comparable to the
Bullet ensemble's explicit completeness accounting; it is not permission to
fill the missing rows.

## Claim boundary and next step

A pass creates a matched Abell luminosity/current uncertainty object.  It does
not identify hard counterparts, infer stellar masses or transverse velocities,
or support a gravity equation.  The immediate successor would rasterize the
ensemble on the frozen V19H Abell grid using the already-tested V19BA moment
definitions.  Only after the V19W/V19X gas products exist can the two clusters'
collisionless and thermodynamic states be combined into a causal long-wave
source and tested against sealed lensing targets.
