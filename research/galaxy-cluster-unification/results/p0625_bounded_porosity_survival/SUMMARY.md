# P0625 bounded porosity survival

This stage replaces the failed unbounded pair extrapolation with bounded laws and tests parameter-free OR combinations.

| Candidate | Galaxy CV gain | Fold wins | Derived cluster gain | Solar | Raw roots | Diagnostic gate |
|---|---:|---:|---:|:---:|---:|:---:|
| `OR_max_potential_surface` | +10.18% | 4/4 | +0.44% | pass | 17/18 | pass |
| `inverse_hillfloor_m2__mean_surface_R80` | +6.21% | 3/4 | +0.44% | pass | 17/18 | pass |
| `inverse_hill0_m1__potential_depth` | +14.13% | 4/4 | -69.43% | pass | 17/18 | fail |
| `OR_rms_potential_surface` | +12.07% | 4/4 | -2.68% | pass | 14/18 | fail |
| `OR_rms_potential_pair30` | +11.75% | 4/4 | -11.51% | pass | 16/18 | fail |
| `OR_max_potential_pair30` | +10.75% | 4/4 | +1.03% | fail | 13/18 | fail |
| `inverse_hill0_m0.5__pair_count_L30p0kpc` | +8.04% | 4/4 | -5.79% | fail | 14/18 | fail |
| `inverse_hill0_m0.5__pair_surface_L30p0kpc` | +8.04% | 4/4 | +1.03% | fail | 13/18 | fail |
| `inverse_hill0_m0.5__pair_surface_L100p0kpc` | +7.99% | 4/4 | +nan% | fail | 16/18 | fail |
| `inverse_hillfloor_m1__pair_surface_L100p0kpc` | +7.97% | 4/4 | +nan% | fail | 13/18 | fail |
| `inverse_hillfloor_m1__pair_surface_L30p0kpc` | +7.82% | 3/4 | +1.32% | pass | 14/18 | fail |
| `constant` | +0.00% | 0/4 | +0.00% | pass | 17/18 | fail |

The gate is a project-spent diagnostic, not independent validation or a theory claim.
