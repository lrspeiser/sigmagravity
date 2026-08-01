# Solar-screened law across galaxy morphologies

## Outcome

The cluster-selected, Solar-screened isothermal law does **not** transfer to
SPARC galaxies with the same universal setting. With `lambda = 10.5` locked,
its outer-rotation-curve RMSE is `18.602 km/s`, compared with `10.348 km/s` for
fixed RAR and `10.385 km/s` for simple MOND. It fails both frozen accuracy
gates and is worse than RAR in 90 of 131 galaxies.

This is a useful falsification. The equation can pass the limited Mercury
diagnostic and perform well on the two-cluster aggregate, but it is not the
universal galaxy-and-cluster law sought by this project.

The Solar qualification matters: “passes Mercury” currently means a
first-order supplementary-perihelion calculation and a Cassini force-fraction
proxy. It does not mean that the equation has passed a raw, simultaneous
multi-planet ephemeris fit.

## Frozen test

The tested law was

\[
g(r)=g_{\rm bar}(r)+
\lambda\frac{G M_{\rm bar}}{r_*r}
\frac{a_0}{a_0+g_{\rm bar}(r)},
\]

with `lambda = 10.5`, `a0 = 1.2e-10 m/s^2`, and `r* = 200 kpc` fixed before
the galaxy scores. No gravity parameter was fit per galaxy or to the SPARC
sample. As in the existing independent SPARC comparison, each model could
calibrate only four ordinary measurement/baryonic nuisances on the inner 70%
of each curve: disk and bulge mass-to-light ratios, distance, and inclination.
The outer 30%—968 points in 131 galaxies—was then predicted.

The morphology definitions were also fixed before scoring. The primary shape
split contains 104 disk-dominated galaxies (`B/T <= 0.05`), 14 mixed
disk–bulge galaxies, and 13 bulge-dominated galaxies (`B/T >= 0.30`).

## Overall comparison

| Inner-calibrated model | Outer RMSE (km/s) | Equal-galaxy RMSE (km/s) | Mean residual (km/s) | chi-square/point |
|---|---:|---:|---:|---:|
| Fixed RAR | **10.348** | **10.716** | +1.743 | **4.780** |
| Simple MOND | 10.385 | **10.708** | +1.474 | 4.836 |
| NFW inner-to-outer control | 17.804 | 14.498 | +1.646 | 15.973 |
| **Locked screened tail** | **18.602** | **18.441** | **-12.287** | **19.712** |

The negative residual means that the screened law generally predicts outer
stars to move too slowly. Its RMSE is 79.8% higher than fixed RAR and 79.1%
higher than simple MOND. It is also 4.5% worse than the limited NFW radial
extrapolation control in pooled RMSE, although that NFW number must not be
generalized to all dark-matter models.

All 131 optimizations were finite and reported success. However, 7.63% touched
at least one nuisance boundary, exceeding the frozen 5% ceiling. Relative to
the independent RAR fits, the screened-law fits increased mean disk
mass-to-light ratio by `0.298`, mean distance scale by `0.270`, and mean fitted
inclination by `5.15 degrees`. Those shifts tend to give the equation more
baryonic source mass and/or lower the deprojected target speed. The failure
therefore remains even while standard nuisances move in favorable directions.

## Disk, bulge, and Hubble type

| Galaxy family | Galaxies | Screened tail | Fixed RAR | Simple MOND | NFW control | Tail/RAR |
|---|---:|---:|---:|---:|---:|---:|
| Disk-dominated | 104 | **20.324** | 9.942 | 9.931 | 13.081 | **2.044** |
| Mixed disk–bulge | 14 | **10.591** | 9.401 | 9.344 | 13.723 | **1.127** |
| Bulge-dominated | 13 | **17.286** | 12.306 | 12.537 | 30.614 | **1.405** |
| Early S0–Sb | 26 | **15.297** | 10.726 | 10.929 | 25.037 | **1.426** |
| Sbc–Scd spiral | 44 | **22.043** | 11.802 | 11.726 | 16.928 | **1.868** |
| Late Sd–BCD | 61 | **17.534** | 8.132 | 8.128 | 7.346 | **2.156** |

The added term does not fail because a large bulge was omitted. The worst
aggregate group is the nearly pure, flat disk group. Mixed disk–bulge galaxies
are the closest to the trusted galaxy controls. A direct “bulge causes error”
interpretation would nevertheless be misleading because structure, mass, gas
fraction, and Hubble type are correlated in SPARC.

The per-galaxy fractional error has a raw Spearman correlation of `-0.409`
with stellar bulge fraction, but baryonic mass has a stronger correlation of
`-0.641`; gas fraction correlates at `+0.569`. Every dwarf in this frozen sample
is disk-dominated. After controlling for ranked log baryonic mass, the partial
rank correlation between bulge fraction and fractional error falls to `-0.049`
with `p = 0.578`. There is no detected independent bulge trend in this sample;
the dominant failure pattern is low-mass, gas-rich, rising/flat disks rather
than bulge geometry itself.

## Mass and outer-curve shape

| Family | Screened tail | Fixed RAR | Simple MOND | NFW control | Tail/RAR |
|---|---:|---:|---:|---:|---:|
| Dwarf, `Mbar < 1e9 Msun` | **16.520** | 6.315 | 6.319 | 5.955 | **2.616** |
| Intermediate mass | **19.217** | 9.828 | 9.868 | 16.082 | **1.955** |
| Giant, `Mbar >= 1e11 Msun` | **17.588** | 12.808 | 12.847 | 24.421 | **1.373** |
| Declining outer curve | **14.120** | 12.923 | 12.953 | 19.319 | **1.093** |
| Approximately flat | **18.678** | 9.453 | 9.503 | 20.814 | **1.976** |
| Rising outer curve | **21.164** | 9.898 | 9.917 | 6.905 | **2.138** |

The shape split is descriptive because it uses the observed outer velocities
to assign “rising,” “flat,” and “declining.” It is not an independent
selection variable. It does show exactly where prediction fails: the law is
close only for already-declining curves and substantially underpredicts rising
and flat curves.

## Why the mass scaling fails

Where the acceleration screen is open, the added term approaches

\[
v_{\rm tail}^2=\lambda\frac{G M_{\rm bar}}{r_*}.
\]

Thus `v_tail^4` is proportional to `Mbar^2`. By comparison, the observed
baryonic Tully–Fisher/deep-MOND scaling is

\[
v^4\simeq G M_{\rm bar}a_0,
\]

which is proportional to `Mbar`. To make the current tail mimic that scaling,
its supposedly universal constant would have to become

\[
\lambda(M_{\rm bar})=r_*\sqrt{\frac{a_0}{G M_{\rm bar}}}
\propto M_{\rm bar}^{-1/2}.
\]

For example, matching the deep-MOND speed would require approximately
`lambda = 587` at `1e8 Msun`, `58.7` at `1e10 Msun`, `18.6` at `1e11 Msun`,
and `5.87` at `1e12 Msun`. The locked value `10.5` naturally matches the scale
near `3.12e11 Msun`; it cannot also lift dwarf and ordinary spiral outskirts.
Making lambda galaxy-mass dependent would improve fits but would surrender the
single-universal-setting goal and largely rediscover the required
baryonic-Tully-Fisher scaling by hand.

## What geometry is and is not included

This test does account for disk versus bulge geometry in the Newtonian source
term. SPARC provides separate radial gas, stellar-disk, and bulge velocity
templates, and those are combined at every measured radius. That is why a
concentrated bulge affects the inner and outer fitted curve differently from a
thin disk.

The new tail itself is not geometry-aware. It uses total baryonic source mass
and points radially toward the galaxy center. It does not separately predict:

- forces above versus within a thin disk plane;
- disk thickness;
- an oblate versus spherical bulge response;
- bars, spiral arms, warps, or environmental neighboring fields.

Consequently, this run can reveal a residual dependence on morphology, but it
cannot validate the proposed three- or four-dimensional mechanism behind that
dependence. Vertical stellar kinematics, polar rings, streams, and lensing of
the same galaxies would be needed for that stronger test.

![Galaxy morphology assessment](../results/solar_screened_galaxy_morphology/galaxy_type_assessment.png)

## Decision

Reject the unchanged Solar-screened isothermal tail as the universal
galaxy-and-cluster solution. Do not rescue it by assigning one lambda to
dwarfs and another to giant or bulged galaxies; that would violate the stated
universality advantage.

The useful retained pieces are the high-acceleration Solar screen and the idea
of a cluster-side environment/coherence response. A next elegant equation
would need a galaxy baseline with the correct square-root mass normalization,
while an independently derived environment or geometry factor activates extra
cluster lensing without being tuned per object. That new factor must then be
frozen and retested on these same morphology bins and on genuinely new cluster
lenses.

## Reproduction

```powershell
python -m pytest tests/test_sparc_independent_refit.py -q
python scripts/run_solar_screened_galaxy_morphology.py
python scripts/build_formula_scorecard.py
python -m pytest tests/test_solar_screened_galaxy_morphology_results.py tests/test_formula_scorecard.py -q
```

Machine-readable outputs are in
`results/solar_screened_galaxy_morphology/`, including all point predictions,
per-galaxy scores, frozen morphology assignments, and the complete type table.
