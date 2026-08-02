# P0727 hybrid nonlinear cross-check artifacts

- `report.json`: frozen gates, content hashes, six real-field results,
  known-answer results, and selected universal solver.
- `run_summary.csv`: convergence and resource diagnostics for each hybrid and
  galaxy.
- `known_answer_summary.csv`: manufactured nonlinear validation results.
- `reference_comparison.csv`: full-field and circular-speed agreement with the
  locked P0725 Picard references.
- `residual_history.csv`: every Picard warm-up and Newton--GMRES residual step.
- `hybrid_reference_agreement.png`: observation-space agreement heatmap; the
  blank cell is the deliberately unscored 20-step DDO101 nonconvergence.

P0727 passed. The smallest universal qualifying method is 40 Picard steps at
damping `0.20`, followed by Newton--GMRES/Armijo. See
`docs/P0727_HYBRID_NONLINEAR_CROSSCHECK_RESULTS.md`.
