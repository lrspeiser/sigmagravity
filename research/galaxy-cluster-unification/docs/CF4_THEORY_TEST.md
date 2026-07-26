# CF4 test of the smoothly screened void hypothesis

Status: completed confirmatory engineering/statistical test, 2026-07-26.

## Verdict

**The current SPARC + Cosmicflows-4 test does not support the specific claim
that stronger surrounding void density produces a larger anomalous galactic
acceleration.**

![CF4 test summary](../results/cf4_theory_test/cf4_test_summary.png)

The smooth low-acceleration law is decisively better than Newtonian baryons
alone, so it remains a useful phenomenological rotation-curve formula. The
void-specific prediction fails the preregistered robustness conditions:

- The primary grouped score gives positive β in all five galaxy folds, but it
  *worsens* strict held-out-galaxy χ² by 0.991
  per point (95% galaxy-bootstrap interval
  -0.716 to
  2.615).
- Ungrouped 64^3 also worsens held-out prediction by
  1.574 χ²/point.
- The 128^3 reconstruction improves the point estimate by
  0.848 χ²/point, but
  its 95% interval crosses zero and its mean β is
  -0.028, opposite the predicted sign.
- The free exponent is stable across folds at 0.388
  (range 0.373–0.400), below the flat-curve value
  p = 0.5.
- Strict galaxy CV favors fixed empirical RAR over every tested free-p void
  model (21.340
  versus 27.610
  χ²/point without environment).

Positive β in a training fit is therefore not sufficient evidence: in the
primary reconstruction it fails to predict new galaxies, and its sign is not
stable across CF4 releases.

## Data and locked design

- 175 SPARC galaxies received independent CF4 scores; the preregistered cuts
  retain 131 galaxies and 3,034 radial measurements.
- Primary environment: negative grouped 64^3 CF4 density contrast. Ungrouped
  64^3 and the official 128^3 release are frozen sensitivities.
- Radial test: optimize on each galaxy's inner 70%, predict its outer 30%.
- Galaxy test: five folds balanced across the primary environment score. Global
  parameters train on four folds using all their radii. No velocity from the
  held-out galaxies enters optimization; held-out nuisance parameters remain at
  their prior centers.
- Each fitted model receives 5,000 Adam steps in float64 on the RTX 5090.
- Paired uncertainty intervals use 100,000 galaxy-level bootstrap resamples.

The CF4 grids and axis convention come from the [official Cosmicflows
release](https://projets.ip2i.in2p3.fr/cosmicflows/); the reconstruction method
is described by [Courtois et al. (2023)](https://doi.org/10.1051/0004-6361/202245331),
and the 128^3 sensitivity is the [official Zenodo
release](https://doi.org/10.5281/zenodo.20653238).

## Full radial-holdout results

| Model | Train χ²/pt | Outer χ²/pt | Outer RMSE km/s | p | β |
|---|---|---|---|---|---|
| Newtonian | 7.985 | 75.638 | 35.546 | — | — |
| RAR | 2.512 | 4.748 | 10.324 | — | — |
| NFW | 1.662 | 15.992 | 17.811 | — | — |
| Void free p | 2.208 | 5.849 | 10.718 | 0.402 | 0.000 |
| Void p=0.5 | 2.493 | 4.591 | 10.204 | 0.500 | 0.000 |
| Void env grouped 64 | 2.163 | 5.483 | 10.526 | 0.369 | 0.122 |
| Void p=0.5 env grouped 64 | 2.477 | 4.846 | 10.691 | 0.500 | 0.117 |
| Void env ungrouped 64 | 2.111 | 5.895 | 10.781 | 0.393 | 0.191 |
| Void env ungrouped 128 | 2.190 | 5.671 | 10.681 | 0.404 | -0.058 |

The fixed-p model is the best radial extrapolator in this run, slightly ahead of
RAR, but adding grouped environment to that model worsens its outer score. The
free-p environmental improvements seen in one reconstruction do not reproduce
across the other grids.

## Strict held-out-galaxy results

| Model | Held-out χ²/pt | Held-out RMSE km/s |
|---|---|---|
| RAR | 21.340 | 23.085 |
| Void + ungrouped 128 | 26.762 | 25.283 |
| Void, no environment | 27.610 | 25.622 |
| Void + grouped 64 | 28.601 | 25.822 |
| Void + ungrouped 64 | 29.185 | 26.437 |
| Newtonian | 206.780 | 60.721 |

These absolute χ² values are high because the strict test does not calibrate any
nuisance parameter with held-out velocities. The paired environment/no-
environment comparison remains fair because both models receive exactly the
same information.

## Decision-rule audit

| Preregistered requirement | Outcome |
|---|---|
| Better than Newtonian baryons | Pass |
| Free p robustly approaches 0.5 | Fail; fold range 0.373–0.400 |
| Competitive with RAR on new galaxies | Fail |
| Positive β from independent environment | Mixed by reconstruction |
| Environment improves held-out galaxies | Fail for grouped and ungrouped 64^3 |
| Positive β survives catalog sensitivity | Fail; 128^3 mean β is -0.028 |

Overall: **the universal low-acceleration phenomenology remains viable, but the
specific CF4 void-enhancement interpretation is not supported by this test.**

## Limitations

- This is MAP optimization, not a full posterior analysis.
- CF4's published 2-D error products cannot be propagated as voxel-wise 3-D
  uncertainties with the available metadata.
- SPARC rotation curves primarily trace H I/H-alpha gas, not individual outer
  stars.
- The strict galaxy test fixes held-out nuisance parameters at their priors. A
  second hierarchical test could calibrate nuisance parameters from only the
  inner radii, but it must be declared as a distinct design.
- Testing a new potential-screened or differently smoothed model would be a new
  hypothesis, not a rescue of this preregistered equation.
