# P0726 independent nonlinear cross-check artifacts

- `report.json`: frozen gates, hashes, run results, reference comparisons, and
  claim boundary.
- `run_summary.csv`: six real-system Newton--Krylov runs.
- `known_answer_summary.csv`: three manufactured nonlinear validation runs.
- `reference_comparison.csv`: potential, acceleration, and circular-speed
  agreement with converged P0725 Picard references.
- `residual_history.csv`: full Newton--Krylov iteration diagnostics.
- `reference_prediction_agreement.png`: observation-space agreement where a
  root converged; blank cells are deliberately unscored failures.

P0726 independently verifies DDO53 but fails its two-system universal gate
because DDO101 did not converge. See
`docs/P0726_INDEPENDENT_NONLINEAR_CROSSCHECK_RESULTS.md`.
