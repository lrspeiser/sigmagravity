# Frontiers revision verification report

## Scope

This report records the regression, analysis, build, and visual checks performed for the Frontiers resubmission package. It is not part of the manuscript. The manuscript preserves the canonical response and locked fixed-point galaxy predictor; the catalog photometric scale-length formula is evaluated only as a secondary, no-refit hypothesis.

## Publication-facing regression

Command:

```powershell
python -m pytest -q research/reviewer_derivation_audit/tests `
  research/sparc_statistical_validation/tests `
  Publications/Frontiers/scripts/test_sparc_scale_length_sensitivity.py
```

Result: **24 passed, 0 failed, 0 errors**.

This suite covers the analytic response and QUMOND derivative, dimensions and asymptotic limits, identifiability, grouped cluster handling, coherence-feature leakage, axisymmetric field diagnostics, SPARC sample construction, paired statistics, and the new scale-length calculation and controls.

## Original-paper regression

The two documented repository regression runners were also executed with the baseline model settings.

### Full experimental runner

Command:

```powershell
python scripts/run_regression_experimental.py
```

Result reported by the runner: **19/19 tests passed**. It loaded 171 SPARC galaxies, 42 Fox clusters, and 28,368 Gaia/Milky Way stars. The core numerical baselines remained at their archived values, including SPARC RMS 17.42 km/s on all usable points, Fox median ratio 0.987, Fox holdout exponent (0.27\pm0.01), and Milky Way RMS 29.8 km/s.

The runner's pass count is a software-regression result, not evidence that all 19 items are independent observational validations. In particular:

- the BRAVA item was skipped because its optional module was unavailable;
- the CMB and structure-formation items are explicitly informational;
- the Bullet Cluster item is labeled a challenge even though the runner treats execution as a pass;
- the counterrotation item uses the old unmatched catalog comparison, which the revised paper supersedes with a balanced matched-control result consistent with zero; and
- the Fox cluster item reproduces an in-sample calibration, not external validation.

These limitations are why the manuscript makes narrower claims than the legacy regression runner.

### Comparative runner

Command:

```powershell
python scripts/run_regression_extended.py
```

The runner completed all nine active comparative checks and reproduced the manuscript-facing disk-only SPARC values: 16.366 km/s for the locked predictor and 16.056 km/s for the tested MOND prescription, on 164 galaxies. It also reproduced the Fox calibration ratio 0.987 and loaded all 28,368 Gaia/Milky Way stars.

The output line `Guardrails: FAIL` is not a failure of the locked baseline. That runner applies strict improvement guardrails for an optional extended-phase model even when the extension is disabled; an unchanged prediction has zero improvement and fails a strict `<0` improvement condition. No extended-phase result is used in the manuscript.

## Photometric scale-length experiment

Command:

```powershell
python "Publications/Frontiers/scripts/run_sparc_scale_length_sensitivity.py"
```

Primary 164-galaxy results:

| Model | Mean galaxy RMS (km/s) |
|---|---:|
| Locked Equation (6) | 16.3657 |
| Catalog-(R_d) window | 16.5126 |
| Acceleration only | 16.8823 |
| Tested MOND prescription | 16.0563 |

The catalog-(R_d)-minus-locked paired mean is +0.1468 km/s, with galaxy-bootstrap 95% interval [−0.0881, +0.3854] km/s. The catalog window improves on acceleration only by 0.3697 km/s, with interval [0.1108, 0.6197] km/s, but is worse than the tested MOND prescription by 0.4562 km/s, with interval [0.0753, 0.8433] km/s.

The actual scale-length assignment does not beat the 2,000-permutation control (one-sided (p=0.7111)). A common median (R_d=2.3) kpc gives mean RMS 16.4125 km/s, also better than the galaxy-specific catalog assignment. On all 171 usable galaxies and 3,373 valid points, the locked, catalog-window, and MOND mean RMS values are 17.4155, 17.5776, and 17.1540 km/s.

Verdict: radial suppression contributes useful behavior, but the measured photometric scale length is not identified as the controlling variable by this window. The test supports retaining the core fixed-point formula and presenting source geometry as a future hypothesis.

## Statistical sensitivities

- Observational-error weighting gives mean RMS 15.8473 km/s for the locked predictor and 15.6544 km/s for MOND. Their paired mean is +0.1930 km/s, with 95% interval [−0.1773, +0.5637] km/s.
- The mean locked-minus-MOND contrast remains positive at 20%, 30%, and 40% bulge thresholds and with all valid points: +0.1106, +0.3094, +0.4414, and +0.2615 km/s, respectively.
- These checks retain the paper's “comparable aggregate performance” wording and do not establish superiority.

## Build and visual verification

The following compiled successfully:

- 9-page US Letter REVTeX reviewer-continuity manuscript in the original two-column format with continuous line numbers;
- 16-page Frontiers-template manuscript with line numbers; and
- 9-page Frontiers Supplementary Material.

Every page of all three PDFs was rendered to PNG and visually inspected. Equations, tables, figures, captions, references, line numbers, page breaks, the scale-length section, and the staged future-investigations section are legible. The reviewer-response page and section references were checked against the final line-numbered REVTeX proof.

The original and revised manuscript sources use the same `revtex4-2` class options (`aps,prd,reprint,superscriptaddress,showpacs,floatfix,longbibliography`) and both compile at US Letter size. The only submission-format additions to the revised source are the `lineno` package, continuous-numbering command, and line-number spacing setting.

## Final feedback audit

The final audit of the proposed revision identified four clarifications that could be added without changing the fitted formula or selecting a more favorable result. They are now present in the manuscript and in the point-by-point responses:

- the Fox result is consistently described as a secondary illustrative calibration and is not used as a positive result in the Abstract;
- the asymptotic motivation and non-uniqueness of `h(g_N)`, `g^\dagger`, `A_0`, `F`, and `\sigma` are stated explicitly;
- “locked” is defined as retrospective non-reoptimization of the submitted equations and constants, not prospective registration or independent validation; and
- the QUMOND action is limited explicitly to independently specified `B` and is not presented as an action for the endogenous rotational-support prescription.

The audit did not trigger a new cluster amplitude, a replacement path-length formula, or a claim that an external statistical consultation had occurred. The path-length relation was withdrawn from the canonical response because no operational three-dimensional functional was established; it was not relabeled as solved.

The partially resolved reviewer issues are now presented as a conditional research program rather than deferred claims. The manuscript specifies advancement gates for independently measured coherence, no-refit cluster transfer, source-field construction, exact field solutions, relativistic completion, cosmology, and independent statistical review. The two reviewer responses separately identify completed revisions, remaining limitations, and these future tests.
