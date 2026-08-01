# P0560-P0561 physical tensor response

## Outcome

The physical ACCEPT baryon tensor does **not** have a stable universal coupling
sign or strength. An initial two-sided scan appeared to improve the four-cluster
score by **2.74% at `t=-1`**, but the optimum sat on the scan boundary. A frozen
extended-range run with twice as many optimizer starts overturned it: **zero is
best**, while the best root-complete nonzero choice, `t=-1`, is **2.83% worse**.

This reversal is not evidence that extra optimizer starts make physics worse.
It reveals competing lens-geometry/source basins. The allowed training
criterion can select a lower-cost basin whose unseen image prediction is much
worse. A response smaller than this basin variability cannot support a gravity
claim.

## Equation and mathematical range

Both diagnostics keep P0559's map and equation fixed:

$$
\partial_i\left[(\delta^{ij}+tQ_{\rm ACCEPT}^{ij})
\partial_j\phi_\Sigma\right]=S_\Sigma .
$$

P0560 scanned `t = -1 ... +1`. The apparent optimum at its negative boundary
motivated P0561, which froze `t = -6, -4, -2, -1, 0, 1, 2, 4, 6` before the
extended scores were calculated and used eight independent starts per exact
fit.

The measured maps have only `max |Q| = 0.124-0.150`. Consequently the smallest
eigenvalue of `I+tQ` remains **0.101 or larger** even at `|t|=6`, above the
frozen 0.05 gate. Large-coupling failures are therefore not caused by an
ill-posed differential equation.

## Common-coupling result

| Run | Optimizer starts | Best common coupling | Four-cluster RMS | Change vs zero | All roots? |
|---|---:|---:|---:|---:|---|
| P0560 narrow response | 4 | -1 | 17.453 | +2.74% | yes |
| P0561 extended robustness | 8 | **0** | **17.970** | baseline | yes |
| P0561 best nonzero | 8 | -1 | 18.479 | **-2.83%** | yes |

In P0561, only `t=0` and `t=-1` retain every held-out image root in every
cluster. Every larger magnitude/sign choice loses at least one required image
somewhere even though all optimizations report success and the field operator
remains elliptic.

## Per-cluster response

The best spent-grid points are incompatible:

| Cluster | Best grid `t` | Held-out change | Important caution |
|---|---:|---:|---|
| MACS0329 | -1 | +8.83% | larger magnitudes lose roots |
| MACS0429 | -2 | +3.79% | `t=-1` is 31.16% worse in the robust basin |
| MACS1115 | +4 | +40.24% | other clusters cannot retain roots there |
| MACS1931 | -1 | +11.09% | most larger magnitudes lose roots |

The two-sided near-zero diagnostic splits **two positive versus two negative**.
These per-cluster optima are observations about sensitivity, not permissible
parameters.

## Cross-system transfer

For each cluster, P0561 selected a coupling using the other three and then
looked up the excluded system's exact score. Three folds select zero and make
no change. The one nonzero fold selects `t=-1` and makes excluded MACS0429
**31.16% worse**. Thus the apparent individual improvements do not define a
transferable observable-to-coupling rule.

## Optimizer-basin audit

At the same `t=-1` and physical map, increasing starts found a lower MACS0429
training cost (`125.04 -> 123.92`) but its held-out RMS jumped from `14.29` to
`19.24` arcsec. At `t=+1`, the four-start run retained all roots; the
eight-start run's selected lower-cost basins lost held-out roots in MACS0329
and MACS1931. Even the zero baseline varies by roughly 0.1-0.5% across these
seed sets.

This establishes a practical truth for subsequent raw-lens work: a single
best-of-multistart solution is insufficient when formula gains are at the
percent level. Future comparisons must report either basin ensembles or a
fully specified deterministic global inference, plus root topology.

## What survives

1. **Tensor direction can be high leverage.** Individual clusters move by
   4-40%, much more than the P0557/P0559 locked weak-coupling signal.
2. **That leverage is not universal.** Signs, optima, and root topology conflict
   between clusters.
3. **Mathematical stability is not observational stability.** Positive
   ellipticity through `|t|=6` does not prevent caustic/root failures.
4. **One spectacular cluster improvement is insufficient.** MACS1115's 40%
   response cannot be used without a prospectively measured environmental
   variable that predicts it and transfers elsewhere.
5. **The current physical tensor geometry should not be promoted.** A next
   experiment should diagnose source/geometry basins or use a direct
   image-position response statistic before inventing another amplitude law.

Galaxy rotation and Solar values are unchanged only because this term is
defined as an external non-circular cluster tensor and is zero by scope in
those controls. That preservation is not a covariant explanation.

## Reproduce

```powershell
python scripts/run_p0560_accept_tensor_coupling_response.py
python scripts/run_p0560_accept_tensor_coupling_response.py --config configs/p0561_accept_tensor_extended_response_protocol.json
python -m pytest tests/test_p0560_p0561_accept_tensor_response_results.py -q
```

Machine-readable artifacts are in
`results/p0560_accept_tensor_coupling_response/` and
`results/p0561_accept_tensor_extended_response/`.
