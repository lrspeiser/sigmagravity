# Sigma V19AW DELVE DR3 candidate-coverage plan

## Question

Can a single homogeneous calibrated coadd catalog provide measurable signed
`griz` fluxes for at least 90% of the unchanged 568 Bullet candidate sources,
while leaving all ambiguous and null matches intact?

V19AU and V19AV showed that the current single-exposure DECam measurements are
not sufficiently complete. V19AW changes the data product, not the candidate
list or the acceptance threshold.

## Why DELVE DR3

[NOIRLab's DELVE release page](https://datalab.noirlab.edu/data/delve) identifies
`delve_dr3.coadd_objects` as the newest and deepest DELVE coadd catalog. It
combines `griz` measurements from one reduction pipeline and exposes signed
four-arcsecond Gaussian-aperture fluxes, their uncertainties, contributing
epoch counts, quality flags and several morphology estimators.

The four-arcsecond Gaussian-aperture observable is close to the aperture scale
validated in V19AS/V19AT. Unlike a magnitude-only requirement, it retains zero
and negative flux measurements and therefore separates missing information
from an ordinary noisy non-detection.

## Frozen acquisition

Before opening the full field, the public schema and only the total row count
were preflighted. A radius of 0.07 degrees around
`(104.6247543743987, -55.94659781854907)` contains exactly 2,351 DR3 objects and
covers all 568 frozen candidate coordinates. The production query will acquire
that field once and then perform all matching offline.

Each candidate retains every catalog object within 0.5 arcsec. Zero matches is
an explicit null state and more than one match is an explicit ambiguous state.
No nearest-neighbor choice, quality cut, morphology cut, color score or BRI
comparison is authorized.

Complete coverage requires, independently in `g`, `r`, `i` and `z`:

- a finite Gaussian-aperture flux, including a negative or zero value;
- a finite, positive flux uncertainty; and
- at least one good contributing epoch.

The gate passes only if at least 90% of the 568 candidates have one or more
complete matches and each of the 57 spectroscopic members has at least one
complete candidate. The exact 2,351-row response is also an integrity gate.

## Relationship to the gravity-wave hypothesis

This is a prerequisite data test, not a gravity result. The long-wavelength
candidate requires baryonic density, current, stress and morphology to source a
field such as

\[
(1-L_\Sigma^2\Box)X_{\mu\nu}
=S_{\mu\nu}[T,j,\Pi].
\]

If the source catalog passes, a separately frozen probabilistic association
can turn the preserved flux and morphology information into baryonic source
maps with uncertainties. Those maps can then test whether a source-directed
mode is nearly uniform across a star system but changes across galaxies and
clusters. V19AW cannot test the wavelength, amplitude or field equation by
itself.

## Decision rule

A pass authorizes a new, separately frozen color-position-morphology
likelihood with explicit null and ambiguous states. A failure closes this
catalog route without increasing the match radius, dropping a band or lowering
the 90% gate.
