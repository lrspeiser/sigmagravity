# P0631 observation-matched replica results

## Correction to the simulator claim

P0630 built a useful gravity-law forward laboratory, but its parametric particle
scenes were not demonstrated to recreate real galaxies. P0631 adds the missing
observation layer. It uses the official SPARC radial 3.6-micron photometry,
nonparametric disk/bulge decompositions, distance, inclination, and rotation
curves to construct deterministic projected light maps, line-of-sight velocity
maps, and luminosity-tracer particle realizations.

This is an **observation-matched axisymmetric replica**, not a galaxy-formation
simulation. The available SPARC products do not constrain the exact two-
dimensional bar, arms, warp, gas clumps, or lopsidedness. Those features cannot
be claimed until resolved image and velocity-map data are added.

![Representative observation-matched replicas](../results/p0631_observation_matched_replicas/p0631_replica_overview.png)

## Results

The frozen P0631 catalog contains the same 131 quality-controlled galaxies as
P0630: 81 train, 27 development, and 23 whole-galaxy holdouts. No fitted
generator coefficient is selected from the holdouts.

| Check | All 131 | 23 holdouts | Meaning |
|---|---:|---:|---|
| angular photometry reconstruction, median RMSE | 0.0000308 dex | 0.0000307 dex | the magnitude profile and generated luminosity profile agree |
| continuous rotation reconstruction, median RMSE | 0.000 km/s | 0.000 km/s | expected: observed rotation is a replica-mode input |
| 257-pixel virtual-camera rotation loss, median RMSE | 0.221 km/s | 0.212 km/s | finite grid interpolation error |
| finite-camera light loss, median RMSE | 0.0407 dex | 0.0382 dex | central-profile loss at fixed grid resolution |
| total rendered-light error, median absolute | 0.367% | 0.318% | numerical luminosity conservation |
| deterministic particle and map replay | pass | pass | the same seed gives bitwise-identical output |

All predeclared replica gates pass. Four 65,536-particle checks—DDO154,
NGC2403, NGC2841, and NGC7814—conserve the nonparametric profile luminosity to
floating-point precision.

The exact continuous scores are not a scientific victory. They verify that the
coordinate transforms, projection, interpolation, and virtual telescope can
faithfully carry supplied observations into a simulated scene and back. They
do not show that any gravity law generated those observations.

## Leakage-safe two-mode design

The code now makes the distinction structural:

1. `render_observed_replica` is explicitly labeled replica mode and supplies
   the observed rotation curve. It validates the generator and renderer.
2. `render_replica` has no observed-speed fallback. It requires an explicit
   circular-speed array from the caller. Blind gravity tests must pass theory
   output here and score it against the hidden observed curve afterward.

The same applies to lensing: a cluster scene may use baryonic light/gas and
ordinary lens geometry, but its hidden image positions or total mass target
cannot enter the gravity prediction.

## How this connects to the held-out physics result

P0630 already performed the blind theory layer on the 23 galaxy holdouts, four
cluster radial holdouts, and two raw strong-lens cluster holdouts. The current
seven-constant transport law did **not** win that test:

| Held-out observable | Fixed RAR | Current transport | Stronger comparator |
|---|---:|---:|---:|
| galaxy velocity RMSE | 23.326 km/s | 29.298 km/s | fixed RAR wins |
| cluster radial acceleration RMSE | 0.4908 dex | 0.1981 dex | transport wins this derived target |
| raw lens image RMS | 25.726 arcsec | 18.008 arcsec | per-cluster compact halo: 9.977 arcsec |

Thus the simulator is now capable of reproducing the measured galaxy layer,
but the present universal gravity law still overpredicts some high-mass galaxy
speeds and cannot reproduce raw cluster lens topology as well as the object-
specific halo comparator.

## What the next fidelity layer needs

To move from radial replicas to genuinely morphology-matched systems, acquire
resolved 3.6-micron/F160W images, H I surface-density maps, and two-dimensional
velocity fields or cubes for a common sample. The generator can then be scored
on pixel morphology, Fourier modes, radial light, gas layout, and velocity-map
residuals before a gravity law is allowed to see the hidden dynamics. Cluster
replicas need member-galaxy light, X-ray gas maps, and raw multiple-image
positions in the same observation-forward framework.

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0631_sparc_replica_data.ps1
$env:PYTHONPATH='src'
python scripts/run_p0631_observation_matched_replicas.py
python -m pytest tests/test_galaxy_replica.py tests/test_p0631_observation_matched_replicas.py -q
```

Machine-readable results are in
`results/p0631_observation_matched_replicas/report.json`; individual images and
compressed scene arrays are under `representatives/`.
