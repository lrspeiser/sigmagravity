# Sigma V19CY A2319 response-aware spectral result

## Decision

The frozen A2319 response-aware development gate **fails**. All seven primary
source fits and all seven NXB-only prefits completed, six regions passed both
velocity-robustness branches, and every primary velocity interval met the
200-km/s precision requirement. The required no-bound condition nevertheless
failed in every source fit because multiple recommended NXB normalizations
were pinned to the public model's delivered hard bounds.

This is not a Sigma-gravity result. It is a development-only identifiability
failure for the public regional NXB treatment. No signed gas-current map is
constructed, and A3667 validation and A754 holdout remain sealed.

The terminal report is
`results/sigma_v19cy_direct_icm_velocity_evidence/development_response_aware_spectral.json`.
Its finalized SHA-256 is
`f9a2feeb4b478334c0f9ff431952597f1af194676222191f385390f228b535e2`.

## Frozen gate result

| Gate | Result |
|---|---:|
| Ten response/NXB grouping products | pass |
| Seven NXB-only prefits converged | pass |
| Seven primary source fits converged | pass |
| At least five primary half-widths no larger than 200 km/s | pass, 7/7 |
| At least five regions pass both robustness models | pass, 6/7 |
| No free parameter at a hard bound | **fail** |
| Terminal development gate | **fail** |

One of 21 source fits, region D's two-temperature branch, returned XSPEC
profile status `FTFFFFFFF` after detecting non-monotonicity in statistic space.
That branch is retained as non-converged. The primary and narrow-band D fits
remain mutually consistent, and the frozen six-of-seven robustness threshold
still passes.

## Primary velocities

All velocities use the frozen BCG-redshift and heliocentric convention.

| Region | Velocity (km/s) | Profile half-width (km/s) | Published no-SSM benchmark (km/s) |
|---|---:|---:|---:|
| A | -78.48 | 55.30 | -60.0 |
| B | -89.07 | 40.03 | -76.3 |
| D | +0.08 | 34.84 | +13.8 |
| B' | -224.71 | 40.80 | -140.0 |
| C' | -86.02 | 51.07 | +20.1 |
| D' | +55.81 | 113.49 | -46.0 |
| E' | +16.33 | 39.23 | +58.6 |

The P1 regions A, B and D reproduce the published sign and approximate
amplitude. P2 is less successful: B' is substantially more negative, C' and
D' disagree in sign, and E' is lower but retains the published sign.

Across all seven regions, the diagnostic comparison gives:

- inverse-combined-variance weighted RMS difference: 61.34 km/s;
- unweighted RMS difference: 66.85 km/s;
- mean difference: -25.18 km/s;
- Pearson velocity correlation: 0.695;
- Spearman rank correlation: 0.643;
- pairwise rank agreement: 76.2%;
- sign agreement: 71.4%; and
- agreement within the paper's directional one-sigma error: 42.9%.

These are reproduction diagnostics, not validation scores, because the paper
and its reported A2319 velocities were known in advance.

## Robustness across source models

| Region | Primary | Narrow Fe-K | Two-temperature | Robust? |
|---|---:|---:|---:|---:|
| A | -78.48 | -71.69 | -82.09 | yes |
| B | -89.07 | -91.07 | -89.43 | yes |
| D | +0.08 | -9.27 | +0.46 | no; two-temperature profile flag |
| B' | -224.71 | -223.93 | -222.59 | yes |
| C' | -86.02 | -82.17 | -94.62 | yes |
| D' | +55.81 | +79.24 | +41.63 | yes |
| E' | +16.33 | +25.59 | +13.57 | yes |

The strong within-region stability means the P2 discrepancy is not repaired by
choosing a narrow band or adding a second temperature. It is more consistent
with pointing calibration/spatial-spectral mixing and NXB-model limitations.

## The decisive background result

| Region | NXB chi-square | DOF | Reduced chi-square | Numeric values at bounds |
|---|---:|---:|---:|---:|
| A | 1010.99 | 900 | 1.12 | 22 |
| B | 6236.79 | 439 | 14.21 | 39 |
| D | 5274.91 | 464 | 11.37 | 39 |
| B' | 2940.16 | 274 | 10.73 | 18 |
| C' | 3387.06 | 256 | 13.23 | 18 |
| D' | 4238.89 | 242 | 17.52 | 20 |
| E' | 2913.04 | 270 | 10.79 | 19 |

Region A shows that the mixed-statistic and transfer machinery can produce an
acceptable NXB constraint fit. The other detector subsets do not support the
same public empirical line template within its frozen bounds. The public NXB
database and `rsl_nxb_model_v1.mo` were already declared as not byte-identical
to the collaboration-internal NXB v2 used by the paper. The regional pattern
now shows that this limitation is material rather than merely documentary.

The current closure cannot be rescued by widening bounds, dropping line
components, changing the fit band, regrouping, or selecting a favorable source
model. Those would be post-outcome model changes.

## What this means for the theory program

The data contain an internally robust signed velocity pattern, especially in
P1, but the frozen background model does not identify it cleanly enough to
authorize a physical source term. Consequently:

1. P2 causal/current-memory action placement remains unsupported by our own
   prospective evidence.
2. The time-even component-overlap and anisotropic-stress clues remain the
   leading theory directions, but no action is selected from this result.
3. A3667 and A754 cannot be opened under V19CY.
4. The failure does not count as a raw-lensing or gravity-formula failure.

## Next admissible evidence

The most direct route is an independently specified, region-aware Resolve NXB
model or public reproduction recipe that is demonstrably applicable to pixel
subsets. It must be frozen before reading A3667/A754 velocities. A new protocol
may use that external model to reproduce A2319 and then restart validation,
but it may not tune line ratios or bounds from these A2319 residuals.

If no independent regional NXB treatment becomes available, the honest route
is to retain the published A2319 result as known development context and seek
an independent time-odd observable with a validated background likelihood,
rather than treating the current source velocities as Sigma evidence.

## Reproduction

```powershell
python scripts/fit_sigma_v19cy_a2319_spectra.py
python -m pytest tests/test_fit_sigma_v19cy_a2319_spectra.py `
  tests/test_sigma_v19cy_a2319_response_aware_spectral_protocol.py -q
```

The cleaned installed spectral artifacts contain 56 XSPEC decks/sessions, 28
logs, 10 grouped NXB spectra and 30 local PFILES parameter files, totaling
11,758,779 bytes. All 124 files are indexed at
`results/sigma_v19cy_direct_icm_velocity_evidence/development_response_aware_spectral_artifacts.json`;
the index SHA-256 is
`fa5280597b9304b3ab45f070547f1b6fc3854bdc890dadad836f5ba2922742dc`.
No validation, holdout, lensing, halo or gravity target was accessed.
