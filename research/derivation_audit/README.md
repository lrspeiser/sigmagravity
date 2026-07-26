# Bounded Sigma-Gravity derivation audit

This directory is an isolated research package.  It does not alter the
submitted manuscript or the production regression scripts.  Its purpose is
to decide whether the coherence and path-length parts of Sigma-Gravity survive
tests in which the relevant quantities are measured independently.

The canonical response is written in terms of the identifiable combination
`B = A C`:

```text
g = g_N [1 + B h(g_N)]
h(g_N) = sqrt(g_dagger/g_N) g_dagger/(g_dagger + g_N).
```

The package includes an action-based QUMOND embedding, independent phase-space
coherence estimators, grouped data loaders with provenance, cluster
calibration/validation diagnostics, an axisymmetric QUMOND approximation
check, and counterrotation data-readiness checks.

Run from the repository root:

```powershell
python research/derivation_audit/run_sprint.py
python -m pytest research/derivation_audit/tests -q
```

Downloaded public data and all generated tables are confined to this
directory.  `results/sprint_summary.json` is the machine-readable summary.
The interpretation and go/no-go decision are in
`docs/DERIVATION_SPRINT_REPORT_2026-07-18.md`.
