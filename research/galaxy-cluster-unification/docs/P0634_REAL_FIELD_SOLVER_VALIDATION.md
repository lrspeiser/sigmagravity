# P0634: real Newtonian, QUMOND, and AQUAL field solvers

## Outcome

P0634 replaces the earlier algebraic radial shortcuts with three equations
solved on the same three-dimensional Cartesian density grid:

1. Newtonian gravity solves `div(grad Phi_N) = 4 pi G rho`.
2. QUMOND first solves for `Phi_N`, constructs the full vector source
   `div[nu(|grad Phi_N|/a0) grad Phi_N]`, and solves a second Poisson equation.
3. AQUAL iterates the nonlinear equation
   `div[mu(|grad Phi|/a0) grad Phi] = 4 pi G rho` to a measured residual.

All twelve gates inherited from the frozen P0633 preregistration pass.

## Numerical controls

The Poisson solver is tested against a smooth manufactured three-dimensional
solution on four grids. Its measured convergence order is recorded in
`results/p0634_field_solver_validation/report.json`; second-order behavior is
required, so a plausible-looking answer at one resolution is insufficient.

A Plummer sphere then supplies an independent analytic gravitational field.
The Newtonian solver must reproduce its force and both MOND solvers must recover
the spherical simple-interpolation result. A separate high-acceleration run
requires AQUAL and QUMOND to return to Newtonian gravity.

The QUMOND implementation constructs fluxes at cell faces. This matters at the
exact center of a symmetric system: `nu` diverges as the Newtonian field tends
to zero, but the complete boosted flux has a finite zero-field limit. Treating
`nu` as an isolated cell multiplier creates a numerical singularity that is not
present in the field equation.

## What this establishes

These tests establish that the code solves the stated nonrelativistic partial
differential equations with controlled grid error. The same machinery can now
consume a lumpy, asymmetric surface-density map rather than pretending every
galaxy is spherical.

They do **not** show that MOND or a SigmaGravity transport term fits a real
galaxy, and they do not derive how photons couple to the field. The P0633
galaxy kinematics and cluster image constraints remain sealed.

## Reproduce

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0634_field_solver_validation.py
python -m pytest tests/test_field_solvers.py tests/test_p0634_field_solver_validation.py -q
```
