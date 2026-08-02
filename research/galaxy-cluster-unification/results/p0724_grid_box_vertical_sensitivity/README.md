# P0724 numerical-sensitivity artifacts

These are the deterministic outputs of
`scripts/run_p0724_grid_box_vertical_sensitivity.py` under the preregistered
`configs/p0724_grid_box_vertical_sensitivity.json` protocol.

- `report.json`: authoritative gates, hashes, batch/job identities, scenarios,
  model summaries, coverage-aware ranks, and claim boundary.
- `scenario_model_summary.csv`: one row per model and numerical scenario.
- `per_galaxy_scores.csv`: retained state, diagnostics, and observational score
  for every successful or failed model--galaxy solve.
- `point_predictions.csv`: all valid circular-speed point predictions.
- `paired_prediction_sensitivity.csv`: prediction changes paired to baseline at
  identical observed radii.
- `scenario_sensitivity.csv`: frozen stability metrics and coverage gates.
- `aggregate_fit_sensitivity.png`: equal-galaxy fit changes; incomplete bars are
  intentionally blank.
- `prediction_sensitivity_heatmap.png`: model--galaxy prediction sensitivity.

P0724 is a spent-sample numerical diagnostic, not a blind scientific
validation. The overall result is `numerical_failure`: 94/96 solves converged,
with two retained fine-grid AQUAL nonconvergences. See
`docs/P0724_GRID_BOX_VERTICAL_SENSITIVITY_RESULTS.md`.
