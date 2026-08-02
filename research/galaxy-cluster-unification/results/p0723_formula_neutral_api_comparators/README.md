# P0723 formula-neutral API comparator artifacts

These artifacts are the deterministic output of
`scripts/run_p0723_formula_neutral_api_comparators.py` using the frozen
`configs/p0723_formula_neutral_api_comparators.json` protocol.

- `report.json`: authoritative aggregate result, hashes, gates, batch IDs, and
  claim boundary.
- `model_summary.csv`: one row per confirmed model manifest.
- `per_galaxy_scores.csv`: convergence, parameter accounting, and
  observation-space scores for every model--galaxy pair.
- `point_predictions.csv`: all retained circular-speed observations and
  predictions.
- `model_score_comparison.png`: equal-galaxy RMSE comparison.
- `rotation_curve_atlas.png`: all 13 published curves and four API predictions.

P0723 is a spent-sample engineering-conformance run, not a new blind
validation. See `docs/P0723_FORMULA_NEUTRAL_RESOLVED_COMPARATOR_RESULTS.md`.
