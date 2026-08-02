# P0658 exact-root basin audit

## Frozen numerical question

P0657 failed source-family cross-validation because images `1b` and `6b` did
not converge from the ordinary single Newton start at their observed
coordinates. P0658 holds the P0657 fold-1 geometry, profiled source positions,
field, amplitude, and `1e-6 arcsec` root-closure threshold fixed. The recomputed
optimizer cost is exactly `5.692377432355883`.

For each of the four validation images, it tries four algorithms from 25
preregistered starts: the observed coordinate plus eight angles at offsets
`0.05`, `0.2`, and `1.0 arcsec`. Hybrid and Levenberg-Marquardt root solvers
use the lens Jacobian; bounded trust-region and dogleg least-squares solvers
search within five arcseconds. A root counts only if its source-plane closure
is at most `1e-6 arcsec` and it lies within five arcseconds of the observation.

This produces 100 attempts per image and 400 total attempts without changing
any scientific parameter.

## Result

Both originally converged controls are recovered by every attempt:

- `2a`: `100/100` accepted, one distinct local root;
- `7a`: `100/100` accepted, one distinct local root.

Neither failed image has an accepted local root:

- `6b`: `0/100` accepted. The best closure is `0.972149 arcsec`, essentially
  unchanged from the original `0.972176 arcsec`. Multiple algorithms settle
  at the same roughly `1.916 arcsec` displacement, supporting a genuine local
  minimum without a root.
- `1b`: `0/100` accepted. Unbounded root methods do locate an exact mathematical
  solution with closure `1.1e-16 arcsec`, but it is `5.28594 arcsec` from the
  observed image and outside the frozen local-correspondence radius. Bounded
  methods cannot close the source-plane gap inside the neighborhood.

The status is therefore `local_topology_failure_supported`.

## Scientific consequence

The ordinary single-start solver is not the reason P0657 fails. A distant
branch for `1b` would already be a very poor positional prediction, and `6b`
has no exact root across the full local audit. Even replacing the root solver
would leave the all-root CV gate failed and the pooled score non-finite.

P0657 remains an instructive near miss: it is mathematically conservative and
improves the spent holdout, but it cannot reproduce source-family topology.
Neither P0657 nor P0658 opens the sealed P0633/P0640 targets.

## Reproduction

```powershell
python scripts/run_p0658_exact_root_basin_audit.py
python -m pytest tests/test_p0658_exact_root_basin_audit.py -q
```
