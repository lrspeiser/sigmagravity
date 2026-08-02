# P0680 compact-halo exact-Hessian results

## Frozen result: fail on direct-agreement tolerances

The exact NIE Hessian has normalized curl exactly zero, and both exact and
direct `0.01 arcsec` derivatives identify six negative-Jacobian points among
the 92 strong-lens points. Two frozen gates nevertheless fail:

- exact-versus-direct convergence relative RMS: `1.68e-5`, above `1e-5`;
- exact-versus-direct Jacobian-determinant relative RMS: `2.06e-5`, above
  `1e-5`.

The failure does not indicate a physical curl. It shows that the `0.01 arcsec`
central difference has not reached the stringent exact-Hessian agreement
threshold, even though its own curl is only `1.04e-7`. P0678 remains formally
unqualified.

One final numerical audit is justified: compare direct steps
`0.01, 0.005, 0.002, 0.001 arcsec` against the already fixed exact Hessian and
require monotonic convergence plus sub-`1e-6` agreement. This is a derivative
audit, not another scientific candidate or moving P0680's thresholds.

## Reproduction

```powershell
python scripts/run_p0680_compact_halo_exact_hessian.py
python -m pytest tests/test_p0680_compact_halo_exact_hessian.py -q
```
