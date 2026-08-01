# Spherical spacetime and matter-cavity test

## Result

The literal forms of the proposal do not provide the missing galaxy or cluster
gravity.  Two mathematically distinct readings were tested:

1. gravity spreading through a positively curved three-dimensional space; and
2. gravity behaving like potential flow around an impermeable spherical hole.

The first gives a radius-dependent amplification but cannot use one curvature
scale across galaxies and clusters.  The second redirects a field locally but
has no first-order isotropic amplification, falls too rapidly with distance,
and conflicts with the requirement that gravity also enter and act on the body.

This rejects the formulas tested here, not the broader intuition that gravity
may have a directional constitutive response to matter boundaries.

## Why the flat-sheet picture was replaced

The familiar rubber-sheet picture is only a visualization.  A coherent version
of “space itself is a sphere” is a three-dimensional spatial slice with positive
constant curvature.  A geodesic sphere of radius $r$ in that space has area

$$
A(r)=4\pi L^2\sin^2(r/L),
$$

where $L$ is the curvature radius.  Conserved radial flux then gives

$$
g(r)=g_{\rm bar}(r)
\left[{r/L\over\sin(r/L)}\right]^2.
$$

For $r\ll L$,

$$
{g\over g_{\rm bar}}
=1+{r^2\over3L^2}+O(r^4/L^4).
$$

This automatically recovers ordinary gravity locally and strengthens it farther
out.  It also has an unavoidable antipodal pole at $r=\pi L$.  Requiring the
same model to remain valid through the 3,000-kpc raw-lensing integral forces
$L>1,005$ kpc.  The fitted value reaches the frozen lower bound,
$L=10^{3.04}=1,096$ kpc, yet is then nearly flat across galaxy radii.

## Galaxy and cluster-matter results

All parameters were calibrated only to the fixed-RAR response on 2,066 inner
SPARC points.  They were then scored against 968 untouched outer points from
131 galaxies and against 44 BCG plus 72 cluster acceleration summaries.

| Model | Outer SPARC RMSE | BCG RMSE | Cluster RMSE | Outcome |
|---|---:|---:|---:|---|
| Fixed RAR reference | 10.348 km/s | 0.299 dex | -- | galaxy benchmark |
| Global closed sphere, cluster-safe | 72.387 km/s | 0.398 dex | 0.872 dex | fails; $L$ at lower bound |
| Global sphere, galaxy-only diagnostic | 88.741 km/s | 0.374 dex | invalid | antipodal pole before cluster scale |
| GR-strength local-curvature control | 72.399 km/s | 0.398 dex | 0.880 dex | effectively baryons only |
| Unscreened amplified local curvature | 72.399 km/s | 0.398 dex | 0.880 dex | Solar limit removes useful amplitude |
| Screened amplified local curvature | 177.972 km/s | 0.248 dex | invalid | outer overgrowth and antipodal pole |

The screened variant used

$$
{g\over g_{\rm bar}}={E(x_{\rm total})\over E(x_{\rm GR})},
\qquad E(x)=\left({x\over\sin x}\right)^2,
$$

$$
x_{\rm total}^2={g_{\rm bar}r\over c^2}
\left[1+{\lambda\over1+(g_{\rm bar}/a_s)^n}\right].
$$

It demonstrates that Solar screening alone is not enough.  The large curvature
needed by inner galaxies accelerates too quickly with radius, makes the outer
galaxy prediction worse, and becomes mathematically invalid on cluster scales.

## Exact hard-cavity result

For a uniform potential flow $U$ around an impermeable sphere of radius $a$,

$$
{u_r\over U}=\left[1-(a/r)^3\right]\cos\theta,
$$

$$
{u_\theta\over U}=-\left[1+{1\over2}(a/r)^3\right]\sin\theta.
$$

At the surface, the radial component is zero: the flow goes around the body.
That is already in tension with the premise that the same gravity flow also
enters the Earth and pulls it normally.  An absorbing or sourced boundary would
be needed in addition to the impermeable-flow condition.

The linear dipole terms cancel over solid angle.  The isotropic RMS factor is

$$
{u_{\rm RMS}\over U}=\sqrt{1+{1\over2}(a/r)^6},
$$

while the most favorable direction is bounded by

$$
{u_{\rm max}\over U}=1+{1\over2}(a/r)^3.
$$

The test deliberately treated a whole galaxy disk scale as one perfectly hard
cavity, an enormous overestimate of the effect.  Across 960 outer SPARC points:

- median favorable-axis acceleration factor: 1.002715;
- 95th percentile favorable-axis factor: 1.071672;
- median required observed-to-baryonic factor: 3.847626;
- fraction for which the upper bound reaches the required factor: 0%; and
- favorable-axis velocity RMSE: 71.990 km/s.

If individual stars are the actual cavities, their projected covering fraction
has median $9.06\times10^{-13}$ and maximum $2.15\times10^{-11}$ under the
already generous assumption that every solar mass has one solar radius.  Such
cavities cannot accumulate an order-unity effect through a galaxy.

An inviscid, incompressible, irrotational flow around a sphere also has zero net
drag.  Therefore a direct “flow pressure pushes the Earth” interpretation needs
dissipation, absorption, a wake, or another field; the exact conservative
cavity solution supplies none of them.

## Raw cluster lensing

No model passed the galaxy and cluster-matter gates.  Following the frozen
post-failure rule, the least-bad exact candidate--the cluster-safe global
sphere--was transferred to raw lensing with $L$ held fixed.  Only the ordinary
six geometry nuisances were fit per cluster.

| Unseen-cluster model | Equal-system raw-image RMS |
|---|---:|
| Global closed sphere | 25.153 arcsec |
| Baryons in GR | 25.199 arcsec |
| Fixed simple MOND | 25.636 arcsec |
| Compact cluster halo | 9.989 arcsec |

The sphere is indistinguishable from baryons for the raw observable and has
2.52 times the compact-halo error.  MACS1115 and MACS1931 score 29.850 and
19.348 arcsec.  RXJ2129 gives 17.917 arcsec on seven held-out images.  Geometry
parameters frequently reach their bounds, so the model is not merely missing a
small normalization.

Changing the field integration cutoff among 600, 1,000, 2,000, and 3,000 kpc
changes the validation RMS by only 0.75%.  The poor result is therefore robust
to the tested path length.

## What was learned

The concept separates into three cases:

- **Global spherical topology:** gives a universal radius scale.  A scale small
  enough to affect galaxies encounters its antipode before cluster scales; a
  cluster-safe scale is too large to affect galaxies.
- **Local spherical curvature:** naturally depends on potential depth
  $g_{\rm bar}r/c^2$, not on the low-acceleration pattern needed by rotation
  curves.  Amplifying it creates excessive radial growth unless an additional
  response law is invented.
- **Matter as hard cavities:** creates a dipolar redistribution, not a new
  monopole attraction.  Averaging many cavities produces a constitutive or
  permittivity description; the scalar version is closely related to known
  refracted-gravity ideas already tested elsewhere in this project.

The surviving intuition is directional rather than spherical: matter
boundaries might change a tensor response.  A next equation would need both a
source/absorption rule and a conservative energy law, for example

$$
\nabla_i\!\left(K^{ij}\nabla_j\Phi\right)=4\pi G\rho_b,
$$

where $K^{ij}$ is calculated from the baryonic tidal or boundary geometry and
has universal constants.  A scalar $K$ returns to refracted gravity and has
already failed the project's morphology tests.  A tensor $K^{ij}$ would make
new directional predictions, but it should be tested only with explicit gas,
member-galaxy, and resolved velocity maps; otherwise it can absorb missing
baryonic structure.

## Reproduction

- `configs/spherical_spacetime_cavity_protocol.json`
- `src/voidscreen/spherical_spacetime.py`
- `scripts/run_spherical_spacetime_cavity.py`
- `scripts/run_spherical_spacetime_raw_lensing.py`
- `results/spherical_spacetime_cavity/galaxy_report.json`
- `results/spherical_spacetime_cavity/hard_cavity_points.csv`
- `results/spherical_spacetime_cavity/raw_lensing_report.json`
- `results/spherical_spacetime_cavity/raw_lensing_predictions.csv`
