# P0608 source-redshift tomography of a gravity route

## Outcome

RX J2129 cannot presently tell us whether the projected route occupies one
lens plane or follows a hidden off-plane arc. The data have genuine redshift
leverage, but the angular correction allowed by P0607 is too small: changing
the route's distance-ratio exponent from 0 to 3 changes fixed-geometry training
RMS by only 0.000155 arcsec and spent held-out RMS by 0.000165 arcsec.

The corrected random-start audit reaches the same conclusion. Across 48
two-start repeats, the median gamma=0 versus gamma=1 training difference is
0.000063 arcsec, smaller than the pooled 16--84% geometry-basin span of
0.000074 arcsec. Rare lower-training basins reverse the held-out ordering.
Gamma is not identified, and no hidden arc height is measured.

## Why redshift scaling is the right next observable

The inverse transport map can pair a baryonic launch point $\mathbf x_i$ with
an apparent-excess arrival point $\mathbf y_j$, but a two-dimensional lens map
cannot distinguish a straight projected chord from an arc that leaves the
lens plane. Source-redshift tomography adds another dimension.

For ordinary matter localized to a single plane,

$$
\beta(z_s)={D_{ls}\over D_s}, \qquad
\delta\boldsymbol\alpha(z_s)\propto\beta(z_s).
$$

P0608 tests the operational generalization

$$
\delta\boldsymbol\alpha(z_s)=s_\theta\,
\delta\boldsymbol\alpha_{\rm ref}
\left[{\beta(z_s)\over\beta(z_{\rm ref})}\right]^\gamma,
$$

with the P0607 gas direction and $s_\theta=0.0025$ locked before inspecting
gamma. The standard one-plane value is $\gamma=1$; $\gamma\ne1$ would be a
motivation for a multi-plane or nonlocal derivation, not proof of one.

Seven image redshifts span 0.6786 to 3.427. Their distance ratios range from
0.748 to 1.051 of the reference value, so the input catalog does contain a
roughly 30% geometric lever arm. The failure is therefore not zero redshift
coverage. It is that only 0.25% of the unit angular template survives the raw
component-direction test.

## Numerical response

| Effective route strength | Training RMS span across gamma | Held-out RMS span |
|---:|---:|---:|
| 0.0025 (locked primary) | 0.000155 arcsec | 0.000165 arcsec |
| 0.010 (leverage diagnostic) | 0.000605 | 0.000580 |
| 0.050 (leverage diagnostic) | 0.002614 | 0.002742 |

The approximately linear growth with route strength confirms that tomography
could become informative if an independently validated component carried a
larger angular correction. P0607 rejects that premise in RX J2129 training.

At fixed geometry, training chooses $\gamma=0$, while $\gamma=1$ differs by
only 0.000063 arcsec. Eight-start refits land in different structural minima,
so P0608C repeated each gamma with a fixed start plus one independent random
start. The median gamma difference remains below basin width, and the best
training minima predict worse held-out images. This is classic
non-identifiability, not evidence for redshift-independent gravity.

## What would identify an arc

A defensible test now needs at least one observable that reacts to the path,
not just its two projected endpoints:

1. multiple source families with precise spectroscopic redshifts and a route
   amplitude large enough to affect them;
2. time delays, which constrain the lensing potential rather than deflection
   positions alone;
3. weak shear or magnification outside the strong-image annulus, preventing six
   central geometry parameters from absorbing the correction; and
4. a different cluster whose component direction, amplitude, and gamma are all
   frozen before its raw data are scored.

If the same baryonic route predicts image positions, time delays, and outer
shear with the standard $\beta$ law, there is no observational need for an
off-plane arc. If a single nonstandard scaling transfers across clusters and
observables, it becomes a real target for a new covariant field equation.

## Reproduction

```powershell
python scripts/run_p0608_route_redshift_tomography.py
python scripts/run_p0608c_tomography_random_start_robustness.py
python -m pytest tests/test_p0608_route_redshift_tomography_results.py -q
```

P0608B is retained and marked superseded because its nominal independent
one-start repeats reused the explicit initial geometry deterministically.
