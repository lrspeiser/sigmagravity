# P0679 compact-halo derivative convergence results

## Frozen result: fail by one step-count gate

The expected convergence is present:

- nested-grid normalized curl at 33/65/129/257 cells:
  `0.01254 / 0.00379 / 0.000997 / 0.000253`;
- 257-cell improvement over 33 cells: `49.7x`;
- direct central-difference curl at steps
  `0.5/0.2/0.1/0.05/0.02/0.01 arcsec`:
  `2.60e-4 / 4.17e-5 / 1.04e-5 / 2.61e-6 / 4.17e-7 / 1.04e-7`;
- smallest-two-step convergence stability: `4.83e-7`; and
- smallest-two-step Jacobian-determinant stability: `7.20e-7`.

Every gate except `direct_threshold_count` passes. The frozen rule required
four of six direct steps below `1e-5`; only three pass because the `0.1 arcsec`
step gives `1.04e-5`. The threshold is not rounded or relaxed, so P0678 is not
formally promoted by P0679.

The numerical evidence nevertheless identifies the mechanism: curl scales
down rapidly with grid spacing and direct step, while convergence and the
Jacobian stabilize. Per the frozen selection rule, the next audit must replace
sampled-image differentiation with the NIE Hessian and compare it against the
converged direct derivative. No lens images need to be revisited.

## Reproduction

```powershell
python scripts/run_p0679_compact_halo_derivative_convergence.py
python -m pytest tests/test_p0679_compact_halo_derivative_convergence.py -q
```
