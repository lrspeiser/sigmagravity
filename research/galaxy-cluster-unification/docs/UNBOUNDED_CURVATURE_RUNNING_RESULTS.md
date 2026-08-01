# Unbounded curvature-running gravity: galaxy, cluster, and Cassini results

Status: completed exploratory cycle, 2026-07-29.

## Bottom line

The strongest formula in this cycle is a simple curvature-controlled power law.
It is a meaningful universal phenomenological candidate, but it is not yet a
unified relativistic theory and it does not pass the predeclared galaxy-accuracy
gate.

The search tested 10 broad families, 6 curvature refinements, and 15 fixed-shape
sensitivity settings. The stable galaxy/cluster compromise occurs at power
sharpness `p=2` to `p=3`; it is not a numerical boundary artifact. A separate
additive interpolation creates a smooth tradeoff: more logarithmic weight helps
clusters and raw lensing while predictably hurting galaxy rotation curves. No
tested value reaches both the prior-Sigma bridge score and fixed-RAR galaxy score.

## Current candidate

Define the local tidal/curvature proxy

$$
T=\frac{g_{\rm bar}}{r},
$$

and let

$$
g_{\rm pred}=g_{\rm bar}\left[1+
\left(\frac{T_*}{T}\right)^p\right]^\epsilon.
$$

The best balanced fixed-shape setting is

$$
T_*=4.915\times10^{-29}\ {\rm s}^{-2},\qquad
p=2,\qquad \epsilon=0.12125.
$$

In plain language, ordinary baryonic Newtonian gravity is retained wherever
the gravitational field changes rapidly across space. In shallow, slowly
changing fields, the effective coupling increases. There is no imposed maximum:
as `T` tends to zero, the multiplier grows without a finite ceiling. The
candidate is one equation with three universal constants; no galaxy receives
its own gravity parameters.

The five bridge folds put `log10(T_*)` between -28.375 and -28.240 and `epsilon`
between 0.11695 and 0.12439. That is a reasonably stable numerical basin rather
than a single fragile optimum.

## What was tested

- Bridge: 44 BCG points plus 72 radial cluster-lensing points from 20 clusters,
  grouped into five held-out system folds. Universal gravity parameters were fit
  here only.
- Galaxies: 131 SPARC galaxies. Ordinary distance, inclination, and stellar
  mass-to-light nuisance values were fit on 2,066 inner points. The locked law
  was scored on 968 untouched outer points.
- Raw lensing: RXJ2129 image positions, with the gravity/lensing amplitude locked.
  The test fits lens geometry but no lensing-only gravity multiplier.
- Solar System: zero gravitational slip was imposed (`gamma=1`) and the running
  coupling was checked from 1.6 solar radii through Saturn against a conservative
  `2.3e-5` fractional-change proxy.

## Main scores

| Universal setting | BCG+cluster bridge (dex) | BCG (dex) | cluster (dex) | SPARC outer (km/s) | RXJ2129 held-out (arcsec) | raw reduced chi-square |
|---|---:|---:|---:|---:|---:|---:|
| curvature power, `p=2` | 0.1377 | 0.0992 | 0.1675 | **14.403** | 1.611 | 33.70 |
| curvature power, `p=3` | 0.1376 | 0.0990 | 0.1676 | 14.407 | 1.663 | 35.91 |
| additive power, `alpha=3` | 0.1305 | 0.0943 | 0.1586 | 16.007 | 1.221 | 19.36 |
| additive power, `alpha=10` | **0.1267** | 0.0938 | **0.1526** | 16.765 | **1.150** | **17.17** |

Relevant in-project references are:

| Reference | Score |
|---|---:|
| prior Sigma bridge | 0.11735 dex |
| simple MOND bridge | 0.41840 dex |
| fixed RAR SPARC outer | 10.348 km/s |
| fitted NFW SPARC outer | 17.804 km/s |
| compact-halo RXJ2129 held-out | 2.536 arcsec |

The `p=2` candidate is 18% worse than prior Sigma on the bridge and 39% worse
than fixed RAR on the galaxy test, but 19% better than the fitted NFW reference
on that same galaxy metric. Its raw RXJ2129 error is 37% smaller than the compact
halo reference. These are encouraging controlled comparisons, not evidence that
the candidate beats dark matter generally: the NFW and compact-halo references
are deliberately limited baselines, and the candidate's raw reduced chi-square
is still unacceptable.

## Cassini interpretation

For `p=2`, the predicted fractional coupling change is approximately
`1.86e-31` at Earth and `6.69e-26` at Saturn, far below the conservative
`2.3e-5` proxy gate. The numerical implementation preserves these values even
though adding them directly to 1.0 would round back to exactly 1.0 in standard
floating-point arithmetic.

This does **not** constitute a full Cassini proof. Cassini measured the PPN
curvature parameter `gamma`; this phenomenology assumes zero gravitational slip
and therefore sets `gamma=1`. A covariant action, its PPN limit, preferred-frame
parameters, and time-delay calculation remain to be derived. The correct claim
is “screened well enough to be Cassini-compatible under the zero-slip
assumption,” not “Cassini validates the theory.”

## What the variations taught us

1. Scalar curvature running is consistently better than the tested explicit
   path and tensor-direction variants. The latter produced 20--28 km/s SPARC
   outer errors and did not remove the cluster tradeoff.
2. Power sharpness is weakly identified above about `p=2`; values 2--12 give
   nearly identical bridge scores. `p=2` is preferred because it has the best
   transferred galaxy score and is the simpler smooth choice.
3. The additive family exposes a real Pareto frontier, not a missed optimizer.
   Moving `alpha` from 3 to 300 improves the bridge from 0.1305 to 0.1244 dex but
   worsens SPARC from 16.0 to 18.0 km/s.
4. The present scalar variable `T=g/r` lacks enough information to distinguish
   a galaxy outskirts point from a cluster point when they share the same local
   curvature. Extra curve-fitting exponents do not fix that information deficit.

## Decision and next falsification

Retain `p=2` as the representative **balanced phenomenology** and `alpha=10` as
the **lensing-favored control**. Do not promote either to a theory survivor.

The next useful work was not another interpolation sweep. The declared
multi-cluster raw-image transfer and its first mass-conserving spatial-vector
extension have now both been completed and failed; see
`docs/UNBOUNDED_MULTICLUSTER_RAW_RESULTS.md` and
`docs/UNBOUNDED_SPATIAL_VECTOR_RESULTS.md`. Four raw coordinate likelihoods
gave 18.2--18.6 arcsec equal-system held-out RMS for the locked spherical laws,
versus 9.05 arcsec for an already inadequate compact halo. Redistributing a
universal fraction of the same baryonic monopole into the observed member-light
directions gave 18.2--18.7 arcsec and made every predictive score slightly
worse. Even a forbidden post-failure held-out oracle improved the best
all-root-converged setting by only 1.6%. A spent-data robustness correction that
normalizes every member template at the same 200 kpc aperture gives a best score
of 18.210 arcsec versus its 18.165 arcsec spherical parent, so the unequal
outer-anchor aperture in the first spatial diagnostic does not change the
conclusion. Therefore the updated work sequence is:

1. obtain complete spatially resolved baryon-only cluster maps and explicitly
   distinguish the BCG, gas, ICL, and member-galaxy source terms; member light
   alone is now known to be insufficient;
2. test dynamics and held-out lensing in the same systems with that common
   baryonic model;
3. derive a covariant field equation and complete PPN parameters only if the
   spatial weak-field rule survives;
4. only then test cosmological background growth, since isolated point-mass
   extrapolation is not a model of an expanding homogeneous universe.

## Reproducible artifacts

- `configs/unbounded_running_full_test_protocol.json`
- `configs/unbounded_running_refinement_protocol.json`
- `configs/unbounded_running_sensitivity_protocol.json`
- `configs/unbounded_running_sensitivity_raw_protocol.json`
- `results/unbounded_running_full_test/report.json`
- `results/unbounded_running_refinement/report.json`
- `results/unbounded_running_sensitivity/report.json`
- `results/unbounded_running_sensitivity_raw/report.json`
- `scripts/run_unbounded_running_full_test.py`
- `scripts/run_unbounded_running_sensitivity.py`
- `scripts/run_unbounded_running_sensitivity_raw.py`
