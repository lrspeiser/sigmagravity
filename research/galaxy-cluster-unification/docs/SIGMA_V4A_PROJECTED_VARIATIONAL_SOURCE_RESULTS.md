# Sigma v4A projected variational Weyl-source results

## Decision

The exact Sigma v4A projected source is **retired**. Its derivation and
numerics pass, its action-fixed sign is favorable, and every convergence/shear
channel improves. The improvement is nevertheless far too small and does not
transfer under the frozen gates.

The shared fit reaches normalized Fourier RMSE `0.903971`, compared with the
unmodified AQUAL joint baseline `0.907582` and the preregistered requirement
`0.500`. It removes only `0.794%` of the weighted residual power. This is a
weak positive projection, not the missing cluster field.

No untouched observation was opened, and this result does not increment the
raw action-level holdout-failure count.

## Frozen equation

The projected low-field interaction was

\[
V(\widehat E,M)=
\frac{\|[\widehat E,M]\|_F^2}
{2(1+\|\widehat E\|_F^2)(1+\|M\|_F^2)},
\qquad
M=(1-L_\Sigma^2\nabla^2)^{-1}\widehat E,
\]

with `Ehat=E/tau_sigma` and `E_ij=D_ij psi`. The complete self-adjoint memory
pullback gives

\[
R_W=D_{ij}\left[
\frac1{\tau_\Sigma}
\left(P_E^{ij}+(1-L_\Sigma^2\nabla^2)^{-1}P_M^{ij}\right)
\right],
\qquad
\delta\kappa=-\frac{\eta_\Sigma}{2}R_W.
\]

The minus sign was fixed by the committed projected functional before the
scores were calculated. All three constants were shared by both clusters and
all three lensing channels.

## Numerical result

The joint fit selects

\[
L_\Sigma=50.0735\ {\rm kpc},\qquad
\tau_\Sigma=0.0570702,\qquad
\eta_\Sigma=2.24366\ {\rm kpc^2}.
\]

All values are interior to their logarithmic bounds and the unconstrained
amplitude is positive.

| Map score | AQUAL baseline | v4A | Relative RMSE change | Gate |
|---|---:|---:|---:|---:|
| AS295 | 0.899451 | 0.893976 | -0.609% | at least -20%: fail |
| PLCKG287 | 0.915641 | 0.913857 | -0.195% | at least -20%: fail |
| Joint | 0.907582 | 0.903971 | -0.398% | v4A at most 0.500: fail |

Every individual channel moves in the favorable direction:

| Cluster | Channel | AQUAL | v4A |
|---|---|---:|---:|
| AS295 | convergence | 0.899437 | 0.894871 |
| AS295 | shear 1 | 0.953890 | 0.947869 |
| AS295 | shear 2 | 0.841513 | 0.835661 |
| PLCKG287 | convergence | 0.918069 | 0.915978 |
| PLCKG287 | shear 1 | 0.944478 | 0.944399 |
| PLCKG287 | shear 2 | 0.883349 | 0.880056 |

The favorable sign is therefore not a numerical accident confined to one map
component. Its effective weighted alignment with the missing field is only
about `0.089`, so optimizing the amplitude cannot provide a large correction.

## Cross-system and boundary tests

| Training cluster | Unchanged test cluster | Transfer NRMSE | Gate |
|---|---|---:|---:|
| AS295 | PLCKG287 | 0.913907 | at most 0.800: fail |
| PLCKG287 | AS295 | 0.894538 | at most 0.800: fail |

The independently preferred amplitudes are `2.676` and `0.970 kpc^2`, a
factor `2.76` apart. Those are diagnostics, not accepted per-cluster
parameters.

Changing the zero-padding factor from two to three changes the joint optimum
score by only `1.42e-8` fraction. The failure is not a periodic boundary
artifact. The full periodic source has mean/RMS below `5e-18` in both maps,
and the cropped fields contain almost exactly half positive and half negative
pixels.

All manufactured checks pass:

| Check | Result | Gate |
|---|---:|---:|
| Full composed functional derivative | `6.27e-12` relative | at most `1e-6` |
| Local potential derivative | `1.36e-11` relative | at most `1e-6` |
| Maximum periodic source mean/RMS | `4.94e-18` | at most `1e-10` |
| Padding-score change | `1.42e-8` | at most 5% |

![Sigma v4A projected variational source](../results/sigma_v4a_projected_variational_source_audit/projected_variational_source_audit.png)

## Physical lesson

Varying the scalar commutator potential answers an ambiguity left by the v3
synthetic audit: its signed field does point weakly in the correct direction.
It still does not create the broad convergence and coherent shear structure
of the halo comparator. The double divergence acts mainly on rapid changes in
the local-versus-memory eigenframe mismatch. The resulting correction is
concentrated around baryonic edges and member structures, while much of the
required field is broad and spatially displaced.

This closes the exact v3E rescue route. The scalar action did not merely fail
because the earlier test integrated away its sign; its correctly varied source
also lacks the required phase and power.

The next mechanism cannot be another bounded algebraic function of one total
local STF tide and one linearly smoothed copy. A justified successor must
change at least one physical ingredient:

1. retain local baryonic stress/component-overlap information before it is
   compressed into the total Weyl tide;
2. make a dynamical gravitational polarization field carry and redistribute a
   broad source rather than differentiating a local mismatch potential; or
3. include the trace/convergence channel in a covariant way while still
   deriving matter and photon response from one metric.

A new candidate must first show on the same spent maps that its source has
broad-band phase coherence and cross-cluster transfer. No holdout should be
opened until that source-level gate passes.

## Reproduction

```powershell
python scripts/check_sigma_v4a_projected_variational_source.py
python -m pytest tests/test_sigma_v4a_variational_source.py -q
python -m ruff check src/voidscreen/sigma_variational_source.py scripts/check_sigma_v4a_projected_variational_source.py tests/test_sigma_v4a_variational_source.py
```

Machine-readable results are under
`results/sigma_v4a_projected_variational_source_audit/`.

