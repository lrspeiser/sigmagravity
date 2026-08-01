# P0632 published MOND/RAR replication

## Outcome

The observation-matched simulator reproduces the published Li et al. (2018)
SPARC RAR/MOND benchmark. It recovers the sample, point count, aggregate
scatter, and per-galaxy reduced chi-square values from the authors' source
table.

| Published-replication check | Published | Simulator | Difference |
|---|---:|---:|---:|
| fitted galaxies | 175 | 175 | 0 |
| quality-selected galaxies | 153 | 153 | 0 |
| precision-selected points | 2,694 | 2,694 | 0 |
| nuisance-refit log-acceleration scatter | 0.057 dex | 0.057161 dex | 0.000161 dex |
| fixed-input log-acceleration scatter | approximately 0.13 dex | 0.132766 dex | 0.002766 dex |
| published versus replayed per-galaxy reduced chi-square | — | correlation 0.999976 | median absolute difference 0.0051 |

All predeclared replication gates pass.

![P0632 published MOND replication](../results/p0632_published_mond_replication/p0632_published_mond_replication.png)

## Equation reproduced

The primary relation is Equation 3 of Li et al.:

\[
g_{\rm pred}
=\frac{g_{\rm bar}}
{1-\exp[-\sqrt{g_{\rm bar}/g_\dagger}]},
\qquad
g_\dagger=1.20\times10^{-10}\ {\rm m\,s^{-2}}.
\]

The baryonic acceleration is calculated from the measured gas, disk, and bulge
mass-model contributions:

\[
g_{\rm bar}(R)=
\frac{
\operatorname{sgn}(V_{\rm gas})V_{\rm gas}^2
+\Upsilon_{\rm disk}V_{\rm disk}^2
+\Upsilon_{\rm bulge}V_{\rm bulge}^2}{R}.
\]

This is an empirical RAR and a MOND algebraic circular-orbit interpolation
law. It is not a numerical AQUAL or QUMOND solution for an arbitrary
three-dimensional density field.

## Why two published scatter values exist

The approximately 0.13-dex result holds the catalog inputs fixed:

- disk mass-to-light ratio 0.5;
- bulge mass-to-light ratio 0.7;
- catalog distance;
- catalog inclination;
- one universal acceleration scale.

The 0.057-dex result uses each galaxy's rotation curve to optimize three or
four nuisance quantities: disk mass-to-light ratio, optional bulge
mass-to-light ratio, distance, and inclination. These are ordinary
observational/stellar-population quantities rather than new MOND gravity
constants, but the result is not a blind prediction for a new galaxy.

The simulator implements both modes and labels them separately.

## Strict simulator comparison

Using 153 galaxies and the same 2,694 precision-selected points, with no
per-galaxy refit:

| Law | Acceleration scatter | Point-weighted velocity RMSE | Equal-galaxy velocity RMSE |
|---|---:|---:|---:|
| Newtonian baryons | 0.2809 dex | 63.36 km/s | 54.43 km/s |
| Li 2018 RAR/MOND | **0.1328 dex** | 22.87 km/s | **19.86 km/s** |
| simple-μ MOND | **0.1327 dex** | **22.75 km/s** | 20.00 km/s |
| standard-μ MOND | 0.1365 dex | 28.27 km/s | 22.51 km/s |

This reproduces the published fixed-input result and shows that the two
low-acceleration interpolation shapes are nearly indistinguishable at this
aggregate precision.

## Whole-galaxy holdout

The chronologically frozen P0630 split supplies 23 whole galaxies that are not
used to select any equation or constant in P0632.

| Law | Holdout velocity RMSE | Holdout acceleration RMSE |
|---|---:|---:|
| Newtonian baryons | 52.180 km/s | 0.5309 dex |
| Li 2018 RAR/MOND | 23.326 km/s | 0.2208 dex |
| simple-μ MOND | 23.800 km/s | 0.2223 dex |
| standard-μ MOND | **22.715 km/s** | **0.2136 dex** |

The standard interpolation happens to be best on this particular 23-galaxy
split, but no interpolation function or acceleration scale was selected from
that holdout. This is a useful follow-up hypothesis, not a post-holdout
promotion.

## What the simulator images mean

For DDO154, IC2574, NGC2403, and NGC2841, the same observation-matched light
seed is rendered twice: once using the measured rotation curve and once using
the fixed-input RAR/MOND prediction. The top panels are idealized
axisymmetric line-of-sight velocity maps derived from radial SPARC data, not
independent observed two-dimensional velocity maps.

The contrast between fixed and fitted inputs is substantial for some systems:

| Galaxy | Fixed-input RMSE | Published nuisance-refit RMSE |
|---|---:|---:|
| DDO154 | 4.43 km/s | 1.34 km/s |
| IC2574 | 13.49 km/s | 2.49 km/s |
| NGC2403 | 6.41 km/s | 4.65 km/s |
| NGC2841 | 55.51 km/s | 6.06 km/s |

NGC2841 is the clearest warning against comparing unlike metrics: the
published fit shifts its stellar mass-to-light ratios, distance, and
inclination within their priors, whereas the strict run holds them fixed.

## Claim boundary

P0632 validates:

- the published algebraic RAR/MOND equation;
- signed gas plus stellar mass-model construction;
- distance and inclination transformations;
- publication sample cuts;
- virtual-telescope rendering from the predicted rotation curve.

It does not yet validate:

- an AQUAL or QUMOND field solver for non-axisymmetric density;
- the MOND external-field effect;
- orbit stability or galaxy formation under MOND;
- any relativistic MOND theory or its cluster-lensing prediction.

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0632_li2018_benchmark.ps1
$env:PYTHONPATH='src'
python scripts/run_p0632_published_mond_replication.py
python -m pytest tests/test_mond_benchmark.py tests/test_p0632_published_mond_replication.py -q
```

The machine-readable report is
`results/p0632_published_mond_replication/report.json`.
