# Complete fine-grid AQUAL reconstruction results (P0728)

Date: 2026-08-02

## Outcome

P0728 failed its frozen four-system engineering gate. The P0727-selected
40-step Picard/Newton hybrid converged DDO53, DDO101, and DDO50, but not
NGC1569. The incomplete AQUAL row is therefore excluded from the fine-grid
model rank and from a resolution-stability verdict.

NGC1569 was a narrow numerical miss rather than a divergent field. Its
equation residual reached `2.40e-10`, well below the `1e-8` residual target,
but its last relative update was `1.76e-8`, above the independent `1e-8`
update target. Both conditions were required before results were opened, so
the field remains unscored.

## Frozen field results

The P0728 configuration SHA-256 is
`ec257ad79f95520df66e12e1a6ab49f18c9f563fc92663710bd9f2269620dfb1`.
Every galaxy used the same 40-step warm-up, damping, Newton--GMRES controls,
physical manifest, tolerances, and 160-iteration budget.

| System | Result | Total iterations | Equation residual | Relative update | Speed agreement with independent reference |
|---|---|---:|---:|---:|---:|
| DDO53 | passed | 49 | `2.20e-11` | `1.05e-9` | `4.94e-9` |
| DDO101 | passed | 52 | `4.60e-12` | `1.38e-10` | `3.24e-8` |
| DDO50 | passed | 45 | `1.35e-11` | `1.15e-9` | `6.26e-10` |
| NGC1569 | **failed** | 46 | `2.40e-10` | `1.76e-8` | not scored |

The three converged fields reproduce independent Picard results in potential,
acceleration, and predicted speed by much better than the frozen 1% gate. No
per-galaxy gravity or solver setting was used.

## What can and cannot be said about fit

The three converged galaxies produced 45 valid observed points. Their partial
equal-galaxy RMSE is `17.626 km/s`, but it is not comparable to four-galaxy
rows and is not included in the rank. The eligible complete rows remain:

1. QUMOND: `16.502 km/s`;
2. Refracted Gravity fixture: `18.982 km/s`; and
3. Newtonian baryons: `24.420 km/s`.

These numbers do not establish the missing AQUAL position. In particular, the
old two-galaxy AQUAL number and the new three-galaxy number must not be used as
if they covered the same systems as the other models.

The available fine-versus-baseline AQUAL prediction changes are `9.06%` for
DDO53, `10.71%` for DDO101, and `11.53%` for DDO50. With NGC1569 missing, only
15 of the required 16 model-system pairs exist. The inherited stability gate
therefore remains incomplete even though the partial median and 90th
percentile fall below their thresholds.

## Next numerical test

P0727 had already preregistered and qualified an 80-step universal hybrid on
DDO53 and DDO101 before P0728 was opened. Applying that existing alternative
to all four systems is the clean next test: it does not invent a setting from
NGC1569's observed speed, and it keeps the same equation and tolerances. If it
still fails any system, the production default remains unset and a broader
solver study is required.

## Reproduce

```powershell
python scripts/run_p0728_complete_fine_grid_aqual.py
```

The command exits nonzero by design because the engineering gate failed. The
partial scores and explicit missing cells remain under
`results/p0728_complete_fine_grid_aqual/`.
