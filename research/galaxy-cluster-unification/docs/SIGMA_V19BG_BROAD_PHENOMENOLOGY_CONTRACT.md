# Sigma V19BG broad phenomenology contract

## Decision

The project will not optimize the long-wave idea around the Solar System or a
single successful galaxy or cluster.  Local gravity remains a mandatory final
exclusion gate, but the active development priority is transfer across diverse
galaxies and clusters.  The same eventual covariant action must then make
forward predictions for other phenomena commonly attributed to dark matter.

This checkpoint passes as a **coverage contract**, not as a gravity result.  No
action, wavelength, amplitude, source operator or holdout result is selected.

## What is already closed

The simplest fixed-range point-source implementation is

\[
{g(r)\over g_{\rm bar}(r)}=
1+A\left[1-\left(1+{r\over L}\right)e^{-r/L}\right].
\]

This is the published STVG/MOG radial shape, not a new Sigma formula.  The
existing EMOG-Q0 screen challenged that mechanism on 131 SPARC galaxies with
3,034 points, 20 CLASH systems with 84 model-derived radial points, and 34 BCG
systems.  The constant-field control failed its five-percent structural gate,
and the action was retired before an observational fit.  One fixed Yukawa
transition also resembles the extra acceleration needed for a flat rotation
curve over only 0.055 dex in radius.

Therefore a new calculation that changes only `A`, `L`, or a Fourier exponent
would duplicate a closed control.  The surviving hypothesis is narrower and
harder: a nonlinear, baryon-forced, source-state-sensitive long-wave response
whose direction and activation follow from baryonic density, current,
pressure, stress, overlap and causal state.  Matter and photons must still use
one physical metric.

## Core breadth required before promotion

The current development evidence already exposes several distinct regimes:

| Regime | Current evidence | Scientific role |
|---|---:|---|
| Full resolved disk-galaxy curves | 131 SPARC galaxies, 3,034 points | Development and comparator evidence |
| Galaxy morphology/residual structure | 32 regime rows, 22 continuous correlations | Detect dwarf/giant, gas, surface-brightness and bulge failures |
| Pressure-supported giant galaxies | 34 BCG systems | Test transfer beyond rotating disks |
| Cluster radial field | 20 CLASH systems, 84 points | Model-derived discovery target only |
| Raw strong-lensing geometry | 5 spent systems, 9 geometry variants each | Development rejection of radial/fixed-orientation laws |
| Resolved cluster baryons | 4 RELICS stellar-plus-gas maps | Forward inputs and inverse discovery; not blind validation |
| Merger source state | Bullet and Abell 2146 | Collisionless current/stress uncertainty without lensing targets |

The eventual blind sample is strengthened from “several clusters” to at least
six: at least two relaxed systems and two disturbed or merging systems, with
both lower- and higher-mass lenses represented.  Every system must have stars,
gas, BCG, intracluster light, members, at least three secure image families,
one spectroscopic family, eight images and defensible positional
uncertainties.  The equation and constants must be frozen before those data are
opened.

The galaxy result must be reported separately for low- and high-mass systems,
gas-rich and gas-poor systems, low- and high-surface-brightness systems,
bulgeless disks, and bulge-dominated or pressure-supported systems.  A good
average cannot conceal one failed class.

## Consequences beyond rotation curves and strong lensing

The long-wave idea would affect much more than the two headline tests.  These
are not optional stories to add later; each is a forward-prediction obligation
of the same action.

| Phenomenon | What the theory must calculate | Distinctive pressure on the idea |
|---|---|---|
| Galaxy-galaxy and cluster weak lensing | Tangential/cross shear, reduced shear and magnification | Tests the same Weyl potential far beyond strong-lens cores and can reveal a common physical-scale bend. |
| Colliding-cluster offsets | Time-dependent convergence, shear, critical curves and image roots relative to gas and galaxies | A causal field may lag or redirect, but the offset and orientation must follow from measured source state rather than a fitted halo map. |
| Dwarf spheroidals and satellites | Dispersion profiles, escape speed, tidal radii and host-distance dependence | Overlapping host/satellite long modes must produce any environmental effect without a halo per satellite. |
| Stellar streams and substructure lensing | Stream precession, width and gaps; compact lens perturbations | A smooth tens-of-kpc mode may change global precession but cannot automatically imitate compact dark subhalos. This is a high-risk discriminator. |
| Dynamical friction and merger times | Orbital energy and angular-momentum loss | With no particle dark-matter wake, the Sigma action must predict whether field radiation/backreaction supplies the observed drag. |
| Cosmic structure growth | Background expansion, growth, slip, RSD, clustering and cosmic shear | A fixed physical or comoving length gives a forced scale/redshift dependence; it cannot be retuned at each epoch. |
| Primary CMB and CMB lensing | Acoustic peaks, ISW effects, damping and lensing power | The field must maintain pre-recombination gravitational potentials without inserting cold dark matter. This may be the strongest eventual test. |

The first two rows become calculable once the quasistatic one-metric equations
and resolved baryonic source state exist.  Satellite and stream tests require a
three-dimensional, time-dependent solver.  Dynamical friction requires the
field energy and radiation terms from the action.  Structure growth and the
CMB cannot be honestly scored until the same covariant theory has a background
solution and cosmological perturbation equations.

## What the current evidence says

The dimensionless long-wave scale precheck remains viable:

\[
5.4243\ {\rm kpc}\le L_\Sigma\le6.5639\ {\rm kpc}.
\]

But this interval was imposed by a chosen 10-kpc transition requirement; it is
not a measurement.  Earlier cross-domain diagnostics also show why broad
testing matters: a fixed phenomenological candidate scored 21.5% worse than
RAR on the SPARC outer points, failed the raw-cluster halo gate by a factor
1.91, and did not have a transferable universal orientation.  The current
Bullet/Abell source-only analysis demonstrates measurable and different
current orientation, but it has not yet established a gravity response.

The next material evidence is therefore the V19W/V19X gas-state product.  It
will determine whether one covariant source invariant can distinguish density,
thermal pressure, shocks and collisionless current across two mergers before a
source operator is selected.

## Reproduction

```powershell
python scripts/check_sigma_v19bg_broad_phenomenology_contract.py
python -m pytest tests/test_sigma_v19bg_broad_phenomenology_contract.py -q
```

The machine-readable result is
`results/sigma_v19bg_broad_phenomenology_contract/report.json`.
