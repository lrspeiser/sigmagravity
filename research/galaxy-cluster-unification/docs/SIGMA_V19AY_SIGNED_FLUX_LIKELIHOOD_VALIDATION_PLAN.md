# Sigma V19AY signed-flux likelihood validation plan

## Why this is different from another completeness cut

V19AU through V19AX repeatedly showed that requiring a formal detection in
every band discards too much information. V19AV nevertheless produced a signed
flux and uncertainty in each `grizY` band for every candidate. A negative flux
is not a negative galaxy; it is an ordinary noisy measurement whose likelihood
can still be calculated.

V19AY tests that likelihood on the five unchanged validation anchors before
any ambiguous candidate is scored.

## Model

The unchanged ten-anchor V19AV fit predicts three colors from published Bessel
`B/R/I`:

\[
\mathbf c=(g-r,\ r-i,\ i-z).
\]

Those colors define a relative flux template with arbitrary `r`-band amplitude:

\[
\mathbf t(\mathbf c)=
\left(10^{-0.4(g-r)},\ 1,\ 10^{0.4(r-i)},\
10^{0.4[(r-i)+(i-z)]}\right).
\]

For observed signed fluxes `F_b` and frozen uncertainties `s_b`, each template
profiles one nonnegative brightness `A`:

\[
\hat A=\max\!\left[0,
{\sum_b F_b t_b/s_b^2\over\sum_b t_b^2/s_b^2}\right],
\qquad
\chi^2=\sum_b{(F_b-\hat A t_b)^2\over s_b^2}.
\]

The V19AV predictive color uncertainties are integrated with five-point
Gauss-Hermite quadrature in each of the three colors, giving 125 deterministic
templates per score. The final score is the weighted average of
`exp(-chi2/2)`.

This construction has three useful properties:

- brightness and aperture normalization are not identity evidence;
- a weak or negative band is retained rather than converted to a missing
  magnitude; and
- a low-information source supplies little discrimination instead of being
  forced into a detection/no-detection category.

## Frozen validation

For each of the five validation BRI rows, the likelihood ranks all five
validation signed-flux vectors. The unchanged gates are at least 3/5 true pairs
ranked first and mean reciprocal rank at least 0.65. All 25 scores must be
finite.

Failure closes this likelihood unchanged. A pass authorizes a separate frozen
application to the 57 ambiguous member cones. That later application may only
redistribute candidate probability conditional on a catalog counterpart; it
must preserve each member's positional null probability and output every
candidate posterior rather than a hard identity.

V19AY is a source-measurement validation. It does not test the long-wavelength
gravity equation or infer a stellar mass.
