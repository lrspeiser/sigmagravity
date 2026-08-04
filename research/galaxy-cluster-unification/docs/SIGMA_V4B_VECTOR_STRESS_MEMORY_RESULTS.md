# Sigma v4B vector-stress memory results

## Decision

The exact Sigma v4B projected source is **retired**. It passes the analytic
variation, conservation, signed-support, broad-power, action-sign, parameter,
and padding gates. It fails the shared accuracy, per-cluster improvement,
all-channel, and cross-cluster transfer gates.

The shared fit reaches normalized Fourier RMSE `0.882874`, compared with the
unmodified AQUAL joint baseline `0.907582` and the preregistered requirement
`0.500`. This is a `2.72%` reduction in RMSE and removes `5.37%` of the
weighted residual power. The source is materially broader than v4A, but its
spatial phase and shear response are still not the missing cluster field.

No untouched observation was opened, and this result does not increment the
raw action-level holdout-failure count.

## Frozen equation

The normalized AQUAL deflection vector and its trace-free field stress were

\[
u_i={\partial_i\psi\over\ell_\Sigma},
\qquad
S_{ij}=u_i u_j-\frac12\delta_{ij}u^2,
\]

with one-scale memory

\[
M=(1-L_\Sigma^2\nabla^2)^{-1}S.
\]

The projected interaction was

\[
V(S,M)=
{\|[S,M]\|_F^2
\over 2(1+\|S\|_F^2)(1+\|M\|_F^2)}.
\]

For `P=P_S+(1-L_sigma^2 Laplacian)^-1 P_M`, the complete chain rule gives

\[
q_i={2\over\ell_\Sigma}P_{ij}u_j,
\qquad
R=-\partial_iq_i,
\qquad
\delta\kappa={\eta_\Sigma\over2}\partial_iq_i.
\]

The sign follows from the preregistered positive interaction in the projected
action. All three constants were shared by both clusters and all three
lensing channels. No target field entered the source calculation.

## Numerical result

The joint fit selects

\[
L_\Sigma=61.0946\ {\rm kpc},\qquad
\ell_\Sigma=15.0235\ {\rm kpc},\qquad
\eta_\Sigma=960.823\ {\rm kpc^2}.
\]

All values are interior to their frozen bounds and the unconstrained amplitude
has the action-fixed positive sign.

| Map score | AQUAL baseline | v4B | Relative RMSE change | Gate |
|---|---:|---:|---:|---:|
| AS295 | 0.899451 | 0.858423 | -4.56% | at least -20%: fail |
| PLCKG287 | 0.915641 | 0.906667 | -0.98% | at least -20%: fail |
| Joint | 0.907582 | 0.882874 | -2.72% | v4B at most 0.500: fail |

Five of the six individual map channels improve:

| Cluster | Channel | AQUAL | v4B | Direction |
|---|---|---:|---:|---|
| AS295 | convergence | 0.899437 | 0.861950 | improves |
| AS295 | shear 1 | 0.953890 | 0.902037 | improves |
| AS295 | shear 2 | 0.841513 | 0.808727 | improves |
| PLCKG287 | convergence | 0.918069 | 0.907737 | improves |
| PLCKG287 | shear 1 | 0.944478 | 0.948112 | worsens |
| PLCKG287 | shear 2 | 0.883349 | 0.862108 | improves |

The two clusters prefer different memory lengths: `119.388 kpc` for AS295
and `33.8952 kpc` for PLCKG287 when each is fit independently. Their preferred
amplitudes are closer (`1301.19` and `1190.63 kpc^2`), so the principal
transfer problem is geometry/scale rather than merely a missing common
normalization.

## Broad power, transfer, and numerical checks

Unlike v4A, the lower-derivative source passes the broad-power gate. The
fractions of correction power at wavelengths at least 50 kpc are `0.8086` for
AS295 and `0.8146` for PLCKG287. Broadness alone is therefore not sufficient:
the field must also put convergence and shear in the correct phase.

| Training cluster | Unchanged test cluster | Transfer NRMSE | Gate |
|---|---|---:|---:|
| AS295 | PLCKG287 | 0.928731 | at most 0.800: fail |
| PLCKG287 | AS295 | 0.868558 | at most 0.800: fail |

| Check | Result | Gate |
|---|---:|---:|
| Local stress directional derivative | `1.19e-11` relative | at most `1e-6` |
| Full composed functional derivative | `2.44e-10` relative | at most `1e-6` |
| Maximum periodic source mean/RMS | `3.15e-18` | at most `1e-10` |
| Minimum signed-pixel fraction | `0.446` | at least `0.100` |
| Minimum broad correction-power fraction | `0.809` | at least `0.500` |
| Padding-score change | `3.28e-6` fraction | at most `0.05` |

![Sigma v4B vector-stress memory](../results/sigma_v4b_vector_stress_memory_audit/vector_stress_memory_audit.png)

## Physical lesson

Moving from the tidal Hessian to the gravitational vector stress solved one
specific problem: the conservative action-derived correction is no longer
mostly an edge detector. Quadratic stress also retains cross terms between
overlapping baryonic fields without identifying galaxies as separate objects.

It did not solve the decisive problem. A single Helmholtz-smoothed memory of
the total vector stress produces broad positive and negative spokes tied to
member structures, but does not reproduce the smooth displaced convergence
or coherent shear of both clusters with one scale. Since the formula is a
divergence of a bounded polarization current, its periodic integral is zero;
it redistributes the AQUAL field but cannot supply a broad monopole-like
response without compensating negative regions.

The next candidate should not add an exponent or choose a cluster-dependent
memory length. A physically distinct route would require a genuinely sourced,
baryon-unique polarization/trace state whose total response need not be a
zero-integral local redistribution, while still forbidding a free halo-like
initial condition. Its source-level phase and transfer must first pass on
these same spent maps before any covariant completion or untouched holdout.

## Reproduction

```powershell
python scripts/check_sigma_v4b_vector_stress_memory.py
python -m pytest tests/test_sigma_v4b_vector_stress.py -q
python -m ruff check src/voidscreen/sigma_vector_stress.py scripts/check_sigma_v4b_vector_stress_memory.py tests/test_sigma_v4b_vector_stress.py
```

Machine-readable results are under
`results/sigma_v4b_vector_stress_memory_audit/`.

