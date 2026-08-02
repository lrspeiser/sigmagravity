# P0681 compact-halo Hessian step-convergence results

## Frozen result: fail; derivative refinement closed

Direct curl converges quadratically as expected:

- steps `0.01 / 0.005 / 0.002 / 0.001 arcsec`;
- normalized curl `1.04e-7 / 2.61e-8 / 4.17e-9 / 1.04e-9`.

The exact-Hessian differences do not converge:

- convergence relative RMS remains `1.68e-5` at every step;
- Jacobian-determinant relative RMS remains `2.06-2.07e-5`;
- observed convergence orders are effectively zero.

Four frozen gates fail: monotonic convergence and smallest-step agreement for
both convergence and determinant. Under the preregistered terminal rule, no
further derivative threshold or step refinement is allowed. P0678 is not
formally qualified as a convergence/criticality target.

## Supported boundary after the failure

The following P0678 quantities use deflections directly and remain usable as
spent development diagnostics:

- compact-halo/scalar RMS ratio `3.317`;
- target/scalar RMS ratio `4.760`;
- halo/scalar vector alignment cosine `0.9946`;
- radial-monopole RMS fraction `0.9882`;
- angular-residual RMS fraction `0.1535`; and
- the nearly constant `3.27-3.43` halo/scalar magnitude ratio across the
  strong-lens annulus.

The following remain provisional and must not drive a formula without an
independent derivative implementation: spatial convergence correlations,
positive-kappa radii, and coarse critical sign-cell counts.

The next theory stage should therefore use deflection-level radial strength
and alignment across multiple already spent clusters, not continue tuning the
RX J2129 NIE derivative.

## Reproduction

```powershell
python scripts/run_p0681_compact_halo_hessian_richardson.py
python -m pytest tests/test_p0681_compact_halo_hessian_richardson.py -q
```
