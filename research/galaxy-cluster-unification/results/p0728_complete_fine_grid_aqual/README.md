# P0728 complete fine-grid AQUAL artifacts

- `report.json`: frozen solver, hashes, gates, field results, partial scores,
  and explicit incomplete-coverage status.
- `run_summary.csv`: four universal hybrid field runs.
- `reference_comparison.csv`: potential, acceleration, and speed agreement;
  NGC1569 is deliberately unscored.
- `per_galaxy_scores.csv` and `point_predictions.csv`: only the three
  converged systems.
- `complete_fine_grid_model_summary.csv`: complete eligible rows only; AQUAL
  is excluded rather than ranked on a partial sample.
- `aqual_baseline_prediction_changes.csv`: three available paired resolution
  changes.
- `aqual_reference_agreement.png` and `fine_grid_model_comparison.png`:
  visualizations that preserve the missing AQUAL result.

P0728 failed because NGC1569 missed the relative-update tolerance. See
`docs/P0728_COMPLETE_FINE_GRID_AQUAL_RESULTS.md`.
