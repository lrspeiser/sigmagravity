# Hybrid nonlinear field cross-check results (P0727)

Date: 2026-08-02

## Outcome

P0727 passed every frozen gate. A single universal numerical method--40
Picard warm-up steps at damping `0.20`, followed by Newton--GMRES with an
Armijo line search--converged both difficult fine-grid AQUAL fields. It began
from the unit-coefficient linearized field and did not use the P0725 solutions
as initial conditions.

The independently completed fields reproduce the long P0725 Picard solutions
far more closely than the preregistered 1% threshold:

| System | Total iterations | Equation residual | Speed normalized RMSE | Potential normalized RMSE | Acceleration normalized RMSE |
|---|---:|---:|---:|---:|---:|
| DDO53 | 49 | `2.20e-11` | `4.94e-9` | `1.06e-9` | `4.12e-9` |
| DDO101 | 52 | `4.60e-12` | `3.24e-8` | `9.18e-9` | `5.40e-8` |

In percentage terms, the circular-speed RMS differences are approximately
`0.00000049%` for DDO53 and `0.00000324%` for DDO101. These are numerical
agreement figures, not observational fit scores.

## Frozen protocol

The preregistered configuration SHA-256 is
`b012dd64164ce07953d6f8f00dc2218c17e1fd0500656e8d79275d5fad1cf499`.
All three methods used the same equation, physical parameters, density arrays,
grid, boundary, initialization, tolerances, damping, and Newton--GMRES
controls. Only the declared number of Picard basin-approach steps differed.
Warm-up steps counted against the common 160-iteration total.

| Universal hybrid | DDO53 | DDO101 | Decision |
|---|---:|---:|---|
| 20 Picard + Newton | passed at 36 | failed at 160 | rejected |
| 40 Picard + Newton | passed at 49 | passed at 52 | **selected** |
| 80 Picard + Newton | passed at 83 | passed at 87 | passed, but not minimal |

The 20-step DDO101 residual was `2.54e-3`, so a short warm-up did not merely
miss the tolerance by rounding error. The 40- and 80-step variants both
reached the same locked reference root. Selecting the smaller successful
warm-up follows the frozen policy and does not inspect observed speeds.

All three hybrid variants also passed the nonlinear manufactured known-answer
problem with relative field errors below `4.8e-15`. Requested and executed
iteration limits matched, and no per-object gravity or numerical parameter was
introduced.

## What this establishes

P0725's DDO53 and DDO101 fine-grid fields now have cross-method support. A
damped fixed-point route and a hybrid residual-root route arrive at the same
potential, acceleration, and predicted circular speed. This closes the narrow
solver-basin question that left DDO101 provisional after P0726.

It does **not** show that AQUAL is the correct gravity law, turn these spent
galaxies into a holdout, or prove continuum convergence. The four-galaxy
fine-grid sensitivity comparison must now be rerun with this locked universal
solver so AQUAL is represented by a complete row rather than two successful
systems.

## Reproduce

```powershell
python scripts/run_p0727_hybrid_nonlinear_crosscheck.py
```

The complete hashes, residual histories, reference comparisons, and known-
answer results are under `results/p0727_hybrid_nonlinear_crosscheck/`.
