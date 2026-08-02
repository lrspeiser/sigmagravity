# Qualified 80-step fine-grid AQUAL results (P0729)

Date: 2026-08-02

## Outcome

P0729 passed every frozen engineering and stability gate. The 80-step
Picard/Newton hybrid--which had been preregistered and qualified in P0727
before the P0728 NGC1569 failure was known--converged all four fine-grid AQUAL
fields under one universal numerical setting. All 55 observed circular-speed
points are now scored, and AQUAL can be compared on the same four galaxies as
the other manifests.

This is a robust discrete numerical result on a spent diagnostic sample. It is
not a new theory fit, holdout, or claim that AQUAL is physically correct.

## Numerical closure and independent agreement

The frozen configuration SHA-256 is
`d2f0fcf9b8c3928533994095f88574b92a276170a86a4b55fc385ac65fe8d29b`.
Every field used 80 Picard warm-up steps at damping `0.20`, followed by
Newton--GMRES/Armijo within one 160-iteration budget.

| System | Total iterations | Equation residual | Relative update | Speed normalized RMSE versus independent reference |
|---|---:|---:|---:|---:|
| DDO53 | 83 | `4.08e-11` | `2.83e-9` | `5.00e-9` |
| DDO101 | 87 | `3.35e-12` | `1.40e-10` | `3.24e-8` |
| DDO50 | 82 | `7.21e-14` | `5.58e-12` | `6.26e-10` |
| NGC1569 | 82 | `2.02e-13` | `1.94e-11` | `4.57e-9` |

Potential and acceleration agreement also passed by many orders of magnitude
relative to the frozen 1% limit. No prior solution initialized a P0729 field,
and no per-object gravity or solver parameter was introduced.

The robustness costs time. Each field took approximately 455--458 wall
seconds in this four-process run, compared with roughly 229--235 seconds for
the 40-step P0728 candidate. The 80-step method is therefore a commissioning
candidate for reliability, not yet proof of an optimal production policy.

## Complete fine-grid comparison

Lower equal-galaxy RMSE is better:

| Rank | Manifest | Equal-galaxy RMSE | Reduced chi-square | Universal gravity parameters | Per-galaxy gravity parameters |
|---:|---|---:|---:|---:|---:|
| 1 | QUMOND simple-nu | `16.502 km/s` | `21.525` | 2 | 0 |
| 2 | Refracted Gravity fixture | `18.982 km/s` | `27.468` | 4 | 0 |
| 3 | AQUAL simple-mu | `21.636 km/s` | `23.073` | 2 | 0 |
| 4 | Newtonian baryons | `24.420 km/s` | `45.079` | 1 | 0 |

AQUAL is about 31% worse than QUMOND and 14% worse than the Refracted Gravity
fixture by equal-galaxy RMSE, while improving on Newtonian baryons by about
11%. Every reduced chi-square remains far above one, so none of these four
fixed manifests adequately explains the data under the stated uncertainties.

The AQUAL per-galaxy RMSE values are `2.716 km/s` for DDO53, `28.349 km/s` for
DDO101, `10.999 km/s` for DDO50, and `30.667 km/s` for NGC1569. That spread is
scientifically important: a moderate aggregate score does not mean uniform
performance across galaxy structure.

## Reconstructed resolution result

With AQUAL restored, the fine-grid scenario has all 16 model-system pairs.
The inherited P0724 gates give:

| Metric | Result | Frozen limit | Outcome |
|---|---:|---:|---|
| Median normalized prediction change | `5.84%` | `10%` | pass |
| 90th-percentile change | `23.58%` | `25%` | pass |
| Maximum aggregate-fit change | `2.40%` | `20%` | pass |

The per-galaxy AQUAL changes are `9.06%` (DDO53), `10.71%` (DDO101), `11.53%`
(DDO50), and `24.09%` (NGC1569). NGC1569 lies close to the scenario's 25%
90th-percentile limit. Passing this finite comparison therefore supports the
current grid protocol but does not establish asymptotic convergence. A still
finer NGC1569 run and a second boundary treatment remain warranted.

## Reproduce

```powershell
python scripts/run_p0729_qualified_80step_fine_grid_aqual.py
```

All hashes, predictions, scores, and plots are under
`results/p0729_qualified_80step_fine_grid_aqual/`.
