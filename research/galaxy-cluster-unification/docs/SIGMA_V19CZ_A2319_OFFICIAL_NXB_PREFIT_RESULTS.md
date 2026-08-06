# Sigma V19CZ A2319 official-state NXB prefit result

## Decision

The independently specified XRISM Resolve-v1 regional-background retry
**fails**.  All seven NXB-only sessions completed, no source spectrum or source
energy distribution was loaded, and the delivered photon-index and line-width
shape freedoms were retained during the second fit exactly as required by the
official recipe.  Six regions nevertheless have reduced chi-square between
10.97 and 17.96; only region A is near acceptable at 1.138.  Free parameters
also reach their allowed hard bounds in every region.

No A2319 source refit is authorized.  A3667 validation and A754 holdout remain
sealed.  The public Resolve-v1 small-subarray NXB route is retired unless an
independently released region-aware model or reproduction recipe becomes
available.

## Why this was a valid retry

V19CY froze every shape parameter after its first NXB fit and then thawed the
twelve recommended individual normalizations.  The current official XRISM
documentation instead says to freeze the common scale parameter, thaw those
twelve normalizations, and leave the delivered-free photon index and Au-line
widths free.  V19CZ froze that externally specified distinction before the
retry and changed nothing else:

- the same ten COR-weighted, optimally grouped NXB RATE spectra;
- the same 1--17 keV band and standard chi-square statistic;
- the same public `rsl_nxb_model_v1.mo` and diagonal response;
- the same delivered parameter bounds;
- no source spectrum, velocity, lensing field, halo map, gravity parameter,
  validation target, or holdout target.

This addresses the only concrete recipe mismatch discovered after V19CY.  It
does not tune the model using the A2319 residuals.

## Result

| Region | Chi-square | DOF | Reduced chi-square | Free parameters at bounds |
|---|---:|---:|---:|---:|
| A | 1010.890257 | 888 | 1.138390 | 17 |
| B | 6236.708942 | 427 | 14.605876 | 33 |
| D | 5274.911995 | 452 | 11.670159 | 35 |
| B' | 2940.057584 | 268 | 10.970364 | 17 |
| C' | 3367.709761 | 250 | 13.470839 | 18 |
| D' | 4238.888429 | 236 | 17.961392 | 18 |
| E' | 2913.013275 | 264 | 11.034141 | 18 |

The official additional freedoms produce only negligible statistic changes
relative to V19CY.  They do not explain the detector-region mismatch.  The
failure is therefore not caused by our earlier decision to freeze those shape
terms.

## Scientific consequence

The A2319 velocities remain useful known-development context, but our public
background likelihood does not identify them cleanly enough to construct a
new Sigma source.  In particular:

1. no causal/current-memory action is selected;
2. no source-fit result may be promoted from V19CY;
3. changing bounds, dropping lines, narrowing the band, or selecting favorable
   pixels would be an outcome-driven rescue and is forbidden; and
4. the next evidence must be a new public region-aware Resolve NXB treatment
   or a different direct time-odd observable with a validated likelihood.

This is an observational-route failure, not a galaxy, lensing, or gravity-law
failure.

## Reproducibility

```powershell
python scripts/run_sigma_v19cz_a2319_official_nxb_prefit.py
python -m pytest tests/test_sigma_v19cz_a2319_official_nxb_prefit.py -q
```

The machine-readable result is
`results/sigma_v19cz_a2319_official_nxb_prefit/report.json`, SHA-256
`6ae50188cf5f1a947b39187ded407c0413f1214c14c630f08b2fefd8faca2868`.
Its 21 XSPEC decks, sessions, and logs total 745,514 bytes and are hash-indexed
inside the report.

Official sources:

- [XRISM Resolve NXB spectral-model instructions](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/nxb/nxb_spectral_models.html)
- [Resolve NXB database and extraction recipe](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/nxb/resolve_nxb_db.html)
- [Resolve data-analysis guide](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/Resolve_Data_Analysis.html)
