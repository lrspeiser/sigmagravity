# Grid, box, and vertical-prior sensitivity results (P0724)

Date: 2026-08-02

## Outcome

P0724 submitted 96 formula--galaxy field jobs through the same asynchronous,
formula-neutral API used in P0723: four frozen manifests, four deliberately
selected sentinel galaxies, and six numerical or reconstruction scenarios.
Ninety-four jobs converged and retained valid artifact hashes. Two high-
resolution AQUAL jobs reached the frozen 200-iteration limit and were retained
as `failed_nonconvergence` results rather than being dropped or scored.

The overall P0724 result is therefore **numerical failure**, not a stability
pass. The expanded-box and alternate vertical-prior cases passed their frozen
stability gates. The coarse-grid case failed the aggregate-fit stability gate,
and the fine-grid case is incomplete because its AQUAL comparison has only two
of four galaxies. No partial row is used in a cross-model rank.

## Frozen design

The protocol was committed before the results were opened in
`configs/p0724_grid_box_vertical_sensitivity.json`. Its SHA-256 is
`99ccea87be1e7b21147a8d13752189fb81d9fe8c4642061e8cffeec03f352837`.

- Systems: DDO53, DDO101, DDO50, and NGC1569.
- Models: Newtonian Poisson, AQUAL simple mu, QUMOND simple nu, and the
  published Refracted Gravity fixture.
- Parameter policy: published fixed; zero per-object gravity parameters.
- Baseline volume: `33 x 33 x 9`.
- Scenarios: `25 x 25 x 9`, `49 x 49 x 17`, a 1.5-times wider
  `49 x 49 x 9` box at the baseline transverse cell spacing, and two
  independent vertical-prior seeds.
- Numerical gates: all jobs converge, every galaxy is scored, maximum equation
  residual at most `1e-7`, artifact hashes validate, and no per-object gravity
  parameters appear.
- Stability gates: scenario median normalized prediction change at most 10%,
  90th percentile at most 25%, and maximum model aggregate-fit RMSE change at
  most 20%.

The LITTLE THINGS kinematics archive SHA-256 is
`967110269d59357ee3a94d1d6e46c2402aef38da3f674180d42044ceaf094173`.

## Scenario results

| Scenario | Complete model--galaxy pairs | Median prediction change | 90th percentile | Largest aggregate-fit change | Result |
|---|---:|---:|---:|---:|---|
| Coarse `25 x 25 x 9` | 16/16 | 3.46% | 18.53% | 77.53% | sensitive |
| Fine `49 x 49 x 17` | 14/16 | 5.20%* | 23.78%* | 1.86% over 3 complete models* | incomplete |
| Expanded box `49 x 49 x 9` | 16/16 | 0.32% | 1.57% | 0.56% | stable |
| Vertical draw B | 16/16 | 1.42% | 5.24% | 2.84% | stable |
| Vertical draw C | 16/16 | 1.99% | 6.57% | 2.65% | stable |

An asterisk marks diagnostics over available complete pairs, not an acceptance
score. The deterministic reporter requires all 16 paired predictions and all
four complete model fits before any fine-grid stability gate can pass.

The wider-box result is strong evidence that the current transverse boundary
proximity is not driving the baseline predictions at the tested scale. The two
vertical draws show modest sensitivity to allowed 3D thickness and flaring
priors on these four systems. They do not establish the galaxies' true depth.

The coarse-grid failure is concentrated. AQUAL's equal-galaxy RMSE changes
from `22.169 km/s` at baseline to `39.355 km/s`, a 77.5% increase, and its
NGC1569 prediction changes by 66.5% on the normalized metric. The coarse grid
must not be used for model ranking even though the scenario-wide median and
90th percentile prediction gates pass.

## Retained nonconvergence diagnostics

Both failures occur only for AQUAL on the fine `49 x 49 x 17` grid:

| Galaxy | Iterations | Maximum relative update | Maximum equation residual | Wall time | CPU time |
|---|---:|---:|---:|---:|---:|
| DDO53 | 200 | `4.467e-3` | `3.925e-2` | 1,185.8 s | 1,144.8 s |
| DDO101 | 200 | `2.983e-3` | `1.549e-2` | 1,585.5 s | 1,296.8 s |

Fine-grid AQUAL converged for DDO50 and NGC1569. Newtonian, QUMOND, and
Refracted Gravity converged for all four fine-grid galaxies. AQUAL also
converged for all galaxies at baseline, on the expanded box, and for both
alternate vertical draws. This pattern points to an instability or slow mode
in the current nonlinear AQUAL fixed-point iteration at smaller cell spacing;
it is not evidence that the AQUAL equation itself has failed observationally.

## Fair-reporting correction

The first generated diagnostic contained a partial AQUAL fine-grid aggregate
computed over its two successful galaxies. Although P0724 already failed its
engineering gates, ranking that number beside four-galaxy scores would have
been misleading. The reporter now:

1. requires identical complete galaxy coverage for stability gates;
2. excludes incomplete rows from cross-model rank orders;
3. records rank comparability explicitly; and
4. leaves incomplete plot bars blank while retaining the partial diagnostics
   in the CSV and JSON artifacts.

This changes reporting only. It does not change a manifest, source map,
parameter, solver setting, frozen threshold, or field result.

## What P0724 does and does not establish

P0724 establishes that the generic API can expose resolution, boundary, and
3D-prior sensitivity instead of hiding it. It also demonstrates that
nonconverged scientific jobs remain auditable and cannot receive observational
scores. The result does not validate any gravity theory, provide a blind
holdout, test resolved velocity fields, or test photon lensing. These four
galaxies were deliberately chosen after P0723 and are a spent diagnostic set.

## Required next stage

P0725 should freeze an AQUAL solver-robustness matrix before opening its new
results. It should vary numerical controls only--under-relaxation, iteration
budget, warm starts or continuation, and a better nonlinear/preconditioned
method--while keeping the equation, units, galaxy packages, boundary, and
published parameters unchanged. It must include known-answer nonlinear
problems plus DDO53 and DDO101, report time and residual histories, and require
the converged solutions to agree across successful methods. If merely raising
the iteration limit does not reduce both update and equation residual, the
fixed-point implementation should be replaced rather than declared adequate.

Only after a complete fine-grid comparison should the project freeze a
production grid and proceed to beam-aware resolved velocity fields and the
separately typed photon-lensing adapter.

## Reproduce

With the local development service running:

```powershell
python scripts/run_p0724_grid_box_vertical_sensitivity.py `
  --base-url http://127.0.0.1:4189
```

The command intentionally exits nonzero because the frozen numerical gate
failed. The deterministic JSON, CSV tables, and plots are retained under
`results/p0724_grid_box_vertical_sensitivity/`.
