# Full member-tidal metric test

## Outcome

The remaining honest member-tensor variant has now been tested and retired. It
keeps the circular/radial member stress that the earlier contrast-only tensor
deliberately subtracted. Under the frozen selection protocol, the universal
coupling again selects `t=0` and gives **18.433 arcsec** equal-system held-out
RMS across MACS1115 and MACS1931. The compact-halo comparator gives **9.989
arcsec** on the same images.

The tested weak-field equation is

$$
\partial_i\!\left[\left(\delta_{ij}+tQ^{\rm full}_{ij}\right)
\partial_j\Phi_\Sigma\right]=S_\Sigma,
$$

where $Q^{\rm full}_{ij}$ is the normalized tidal tensor reconstructed from
observed member positions and light weights. Unlike the first member-tensor
test, its circular mean is not removed. The locked matter law is fixed RAR and
the locked scalar photon closure is `s=5`; there is still only one universal
new number, `t`, and no per-cluster gravity amplitude.

## Frozen grid result

The grid was `t = -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9`, selected only on
MACS0329 and MACS0429 before validation on MACS1115 and MACS1931.

| `t` | Selection cost (arcsec) | All selection roots complete? |
|---:|---:|---|
| -0.9 | — | no |
| -0.6 | — | no |
| -0.3 | 3.416 | no |
| **0.0** | **5.791** | **yes** |
| +0.3 | 6.122 | yes |
| +0.6 | 5.962 | yes |
| +0.9 | 4.530 | no |

Negative coupling can lower a local fitting cost, but it does so by destroying
one or more required exact image roots. It is therefore not a valid prediction.
Among settings with complete roots, the frozen rule selects zero coupling.

## Validation and physical audits

| Check | Result |
|---|---:|
| Selected full tensor | 18.433 arcsec |
| Zero tensor | 18.433 arcsec |
| Compact halo | 9.989 arcsec |
| Improvement over zero tensor | 0.0% |
| Error ratio to compact halo | 1.845 |
| Randomized-member control p-value | 1.0 (degenerate at `t=0`) |
| Maximum curl diagnostic | $3.3\times10^{-17}$ |
| Maximum solver-edge tensor eigenvalue | 0.1029 |

The curl audit passes. The edge diagnostic narrowly exceeds the frozen 0.10
limit, while every empirical advancement gate fails. This is the same
structural result as the contrast-only tensor: member light by itself does not
provide the large-scale radial and multi-component field required by the raw
cluster images.

This does not test a complete gas-inclusive tensor. A decisive stronger test
still requires baryonic surface-density maps for gas, BCG, intracluster light,
and member galaxies in one astrometric frame. Adding another member-only
coupling or choosing `t` per cluster would not answer that question.

## Reproduction

```powershell
python -m pytest tests/test_tidal_metric.py -q
python scripts/run_member_full_tidal_metric.py
python scripts/build_formula_scorecard.py
```

Artifacts:

- `configs/member_full_tidal_metric_protocol.json`
- `scripts/run_member_full_tidal_metric.py`
- `results/member_full_tidal_metric/report.json`
- `results/member_full_tidal_metric/selection_grid.csv`
- `results/member_full_tidal_metric/validation_predictions.csv`
- `results/formula_scorecard/formula_scorecard.csv`
