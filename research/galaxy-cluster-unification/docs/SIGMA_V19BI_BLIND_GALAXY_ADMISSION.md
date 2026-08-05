# Sigma V19BI blind-galaxy admission protocol

## Decision

The eventual galaxy claim will not be based only on SPARC, and it will not
reuse LITTLE THINGS as though it were still untouched.  V19BI freezes a new
survey-level admission protocol before reading any new velocity target.  It
requires at least 48 fresh galaxies, at least six representatives in every
predeclared morphology stratum, and several kinds of kinematic evidence rather
than one aggregate rotation-curve number.

No galaxy is selected here.  No new rotation speed, velocity field, stellar
dispersion or model residual is opened.  The checkpoint says what evidence
must exist and how it will be scored after the action and universal constants
are frozen.

## Independent candidate pools

WALLABY is the primary outer-disk pool.  Its first public kinematic release
contains 109 homogeneous H I models with rotation curves, H I surface-density
profiles, geometry and uncertainty products in the Hydra, Norma and NGC 4636
fields.  This is larger and more environmentally varied than the project's
spent 13-galaxy LITTLE THINGS sample.

WALLABY also prevents an easy overclaim.  Many sources are resolved by only a
few beams, the released modeling fixes the gas velocity dispersion to 10
km/s, and the authors warn about inclination, beam-smearing and inner
surface-density limitations.  Our primary comparison will therefore be
object-balanced, and at least 12 systems must be forward modeled in the
released cube or velocity-field observation space.  A good fit to a
deprojected catalog curve that fails this control is a measurement-model
artifact, not evidence for new gravity.

PHANGS supplies a different regime: 67 CO rotation curves at about 150-pc
resolution in massive star-forming disks.  They resolve inner baryonic
features, bulges, bars and spiral structure much better than the WALLABY pilot
data, but generally do not extend far enough to replace outer H I tests.

DiskMass supplies the geometric cross-check most relevant to the user's
flat-disk-versus-bulge concern.  Thirty published nearly face-on galaxies have
rotation curves, surface-brightness profiles and stellar line-of-sight
dispersion profiles.  The same metric must predict both radial attraction and
vertical restoring force under one scale-height and stellar-population
nuisance ensemble.  A formula that fixes rotation only by making the vertical
force too large fails.

## Frozen breadth

The holdout must contain at least 48 unique galaxies, including at least 32
from the independent WALLABY pool and at least eight eligible non-WALLABY
systems.  Every one of the following strata needs at least six members:

- low- and high-baryonic-mass galaxies;
- gas-rich and gas-poor galaxies;
- low- and high-surface-brightness galaxies;
- bulgeless disks and bulge-dominated or pressure-supported galaxies.

The strata may overlap, but the sample must also include group/cluster and
low-density field environments.  At least 12 systems need raw cube or velocity
field scores, at least eight need high-resolution inner curves, and at least
eight need simultaneous radial and vertical dynamics.

Selection uses only identity, imaging, baryonic structure, environment,
instrument quality and measurement priors.  It may not use rotation-curve
residuals or whether Sigma, MOND or a halo model happens to work.

## Fairness and nuisance parameters

Distance, inclination, position angle, systemic velocity, stellar
mass-to-light ratio, scale height, gas conversion, beam/PSF and asymmetric
drift are measurement nuisances.  They may vary only under the same external
priors for Sigma, fixed MOND/RAR, Newtonian baryons and halo comparators.  They
cannot alter a Sigma coupling, range, exponent or metric coefficient per
galaxy.

The primary aggregate will weight galaxies equally so a long, high-resolution
curve cannot dominate dozens of shorter curves.  A point-weighted score will
also be reported.  The frozen success gate remains no worse than 1.05 times
fixed MOND/RAR overall and 1.25 times it in every predefined stratum.  Halo
comparators may retain their conventional per-galaxy freedom, but every fitted
parameter is counted.

## Reproduction

```powershell
python scripts/check_sigma_v19bi_blind_galaxy_admission.py
python -m pytest tests/test_sigma_v19bi_blind_galaxy_admission.py -q
```

The machine-readable result is
`results/sigma_v19bi_blind_galaxy_admission/report.json`.

## Public sources

- WALLABY PDR1 kinematic release: <https://arxiv.org/abs/2211.07333>
- WALLABY data portal: <https://wallaby-survey.org/data/>
- PHANGS public data: <https://www.phangs.org/home/data>
- PHANGS CO kinematics: <https://arxiv.org/abs/2005.11709>
- DiskMass radial/vertical analysis: <https://academic.oup.com/mnras/article/451/4/3551/1104023>
