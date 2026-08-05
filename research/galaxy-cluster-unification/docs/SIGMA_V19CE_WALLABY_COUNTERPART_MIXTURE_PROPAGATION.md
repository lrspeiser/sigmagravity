# Sigma V19CE WALLABY counterpart-mixture propagation

## Decision

V19BZ and V19CB show that the available catalog positions, H I morphology and
Gaia foreground astrometry do not identify one optical counterpart reliably.
Choosing the highest-ranked object would silently make the later baryonic
model conditional on a mostly arbitrary association.  More catalog weighting
is closed as a productive route.

V19CE therefore keeps all 18,550 candidates for all 711 WALLABY release maps.
It carries four foreground-treatment scenarios and four H I smoothing kernels
as sixteen separate source-systematic branches.  It selects no counterpart,
foreground treatment, smoothing kernel, galaxy sample or gravity law.

## Mixture construction

For candidate (i), treatment branch (b), and spatial kernel (k), define

\[
q_{ibk}=\max(0,L_{ik})f_{ib},
\qquad
p_{ibk}=\frac{q_{ibk}}{\sum_j q_{jbk}},
\]

where (L_{ik}) is the already measured source-only spatial likelihood-ratio
score and (f_{ib}) is the already declared V19CB foreground weight.  This
introduces no temperature, exponent, cutoff or other tunable hyperparameter.

If a diagnostic hard-mask branch assigns zero weight to every candidate, the
scenario is marked undefined and all normalized weights remain blank.  A
uniform fallback would hide the fact that the assumed mask erased the source
information, so none is invented.

The normalized values are uncertainty-scenario weights, not validated
counterpart probabilities.  A high weight does not establish that an object
is the galaxy.

## How later galaxy tests must use this

After the gravity equation and constants are frozen, a release likelihood is
marginalized over counterpart identity:

\[
\mathcal L_{bk}(\theta)=
\sum_i p_{ibk}\,
\mathcal L(D_{\rm release}\mid i,\theta).
\]

Every one of the sixteen source scenarios is reported separately.  The target
likelihood may not update these source weights, choose a favorable candidate,
or select the treatment/kernel branch that gives the smallest gravity
residual.  Independent optical pixels, bitmasks and deblending may replace the
mixture only under a new source-only protocol frozen before kinematic access.

## Result

The output carries all 711 releases, all 18,550 candidates and all 11,376
release/scenario combinations.  Every defined scenario is normalized to one
to numerical precision.  Undefined hard-mask scenarios remain explicit.  No
WALLABY velocity, rotation curve, gravity residual, halo result, lensing target
or holdout label was read.

This does not make the blind galaxy sample ready by itself: candidate-specific
stellar mass and geometry still need either trustworthy optical pixels or an
equally frozen photometric likelihood.  It does prevent counterpart ambiguity
from being hidden or optimized after the gravity result is known.

## Reproduction

```powershell
python scripts/run_sigma_v19ce_wallaby_counterpart_mixture_propagation.py
python -m pytest tests/test_sigma_v19ce_wallaby_counterpart_mixture_propagation.py -q
```
