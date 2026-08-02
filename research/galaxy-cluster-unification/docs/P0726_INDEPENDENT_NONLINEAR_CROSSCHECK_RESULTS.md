# Independent nonlinear field cross-check results (P0726)

Date: 2026-08-02

## Outcome

P0726 directly solved the discrete nonlinear AQUAL residual with three frozen
Newton--Krylov variants, starting from the same unit-coefficient linearized
field rather than from the P0725 answer. Newton--GMRES with an Armijo line
search converged DDO53 in 47 iterations and about 15 wall seconds. Its full 3D
field and observation-space result agree with the converged P0725 Picard
solution to far better than the preregistered 1% threshold:

- potential normalized RMSE: `8.22e-10`;
- acceleration normalized RMSE: `3.96e-9`; and
- circular-speed normalized RMSE: `4.98e-9`.

This independently verifies the DDO53 numerical root. DDO101 did not converge
under any of the three direct-from-linearized variants, so P0726 has status
**fail**, no universal method qualifies, and no production solver is selected.

## Frozen protocol and results

The preregistered configuration SHA-256 is
`2aef4e250afde8ccaebc27e91d9d6c5b3a50889d255cdeeb92f6b2fc9b733ef0`.
All runs retained the P0725 equation, parameters, density arrays, grid,
boundary, and observation targets. The P0725 Picard fields were locked
references and were never initial fields for Newton--Krylov.

| Method | DDO53 | DDO53 residual | DDO101 | DDO101 residual |
|---|---|---:|---|---:|
| LGMRES, Armijo, 20 inner | failed at 80 | `3.454e-3` | failed at 80 | `1.205e-2` |
| LGMRES, no line search, 20 inner | failed at 80 | `2.647e-1` | failed at 80 | `3.944e-1` |
| GMRES, Armijo, 30 inner | **passed at 47** | `1.041e-11` | failed at 80 | `5.149e-2` |

All three methods passed the nonlinear manufactured known-answer problem.
GMRES solved it in seven iterations with `1.80e-13` relative field error. The
real DDO101 failure is therefore input/regime-specific numerical behavior, not
a broken root-solver implementation on the known-answer fixture.

The no-line-search result is materially worse and rules out unguarded Newton
steps for this regime. LGMRES reduces the residual but stalls on both systems.
GMRES finds the DDO53 basin rapidly but does not reach the DDO101 basin from the
linearized field.

## Interpretation

P0725's DDO53 solution is no longer supported by only one algorithm. Picard
and direct Newton--GMRES reach the same potential, acceleration, and predicted
speeds from different numerical routes. That is strong numerical evidence for
one discrete root, not evidence for AQUAL as physical law.

DDO101 remains the narrow blocker. The next test should not alter its physics,
tolerance, or data. A generic bounded Picard warm-up can move the field close
to the nonlinear basin, after which Newton--GMRES can independently finish the
root. The P0725 converged DDO101 field must remain comparison-only and must not
be used as an initializer.

## Reproduce

```powershell
python scripts/run_p0726_independent_nonlinear_crosscheck.py
```

The command exits nonzero because no root method converged both galaxies. All
terminal diagnostics and the DDO53 agreement artifacts remain under
`results/p0726_independent_nonlinear_crosscheck/`.
