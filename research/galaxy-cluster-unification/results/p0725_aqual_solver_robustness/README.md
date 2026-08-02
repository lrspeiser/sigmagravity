# P0725 AQUAL solver-robustness artifacts

These files are deterministic summaries from the frozen
`configs/p0725_aqual_solver_robustness.json` matrix.

- `report.json`: authoritative hashes, gates, manifests, run metrics, and claim
  boundary.
- `run_summary.csv`: all 12 real-system solver runs.
- `known_answer_summary.csv`: six manufactured nonlinear known-answer runs.
- `residual_history.csv`: full iteration histories for every real-system run.
- `prediction_agreement.csv`: preregistered cross-variant comparability table.
- `solver_residual_matrix.png`: final normalized equation residual by variant
  and galaxy.

`linearized_d020` converged both failed P0724 inputs, but P0725 has status
`fail` because no second successful variant was available for independent
solution agreement. See `docs/P0725_AQUAL_SOLVER_ROBUSTNESS_RESULTS.md`.
