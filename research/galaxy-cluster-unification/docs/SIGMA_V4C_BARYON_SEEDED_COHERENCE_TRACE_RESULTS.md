# Sigma v4C baryon-seeded coherence-trace results

## Decision

The exact Sigma v4C projected closure is **retired**. Its constrained source is
unique, nonnegative, broad, high-field suppressed, rotationally covariant,
and numerically stable. It materially improves one cluster but fails the
shared accuracy, all-channel, cross-transfer, and interior-parameter gates.

The shared fit reaches normalized Fourier RMSE `0.814737`, compared with the
unmodified AQUAL joint baseline `0.907582` and the preregistered requirement
`0.500`. This reduces joint RMSE by `10.23%` and removes `19.41%` of the
weighted residual power. It is the strongest of the v4 projected source
screens, but it is not universal.

No untouched observation was opened, and this result does not increment the
raw action-level holdout-failure count.

## Frozen closure

For the total AQUAL deflection vector `a`, the one-scale memory was

\[
H_L=(1-L_\Sigma^2\nabla^2)^{-1},
\qquad
m_i=H_L[a_i],
\qquad
e=H_L[a_i a_i].
\]

The directional disorder and high-field activation were

\[
D={\max(e-m_i m_i,0)\over e},
\qquad
A={\ell_\Sigma^2\over\ell_\Sigma^2+e}.
\]

Observed baryons then seeded the positive trace state

\[
J=\max(\kappa_b,0)AD,
\qquad
\Sigma=H_L[J],
\qquad
\delta\kappa=\eta_\Sigma\Sigma.
\]

The shear was the unique periodic E-mode shear of `delta_kappa`. The same
`L_sigma`, `ell_sigma`, and `eta_sigma` were used for both clusters and all
three channels. No center, member catalog, direction, ellipticity, or
per-cluster gravity quantity entered the formula.

## Shared result

The optimum is

\[
L_\Sigma=52.3039\ {\rm kpc},
\qquad
\ell_\Sigma=299.999\ {\rm kpc},
\qquad
\eta_\Sigma=9.98895.
\]

`ell_sigma` lies on the frozen upper bound, failing the interior gate. The
baryon-weighted activation is `0.9931` in AS295 and `0.9878` in PLCKG287: the
fit is trying to remove the high-field gate and use almost pure directional
disorder. Increasing the bound would therefore add no identified transition;
it would only approach `A=1`.

| Map score | AQUAL baseline | v4C | Relative RMSE change | Gate |
|---|---:|---:|---:|---:|
| AS295 | 0.899451 | 0.894351 | -0.57% | at least -20%: fail |
| PLCKG287 | 0.915641 | 0.726450 | -20.66% | pass |
| Joint | 0.907582 | 0.814737 | -10.23% | v4C at most 0.500: fail |

Five of six individual channels improve:

| Cluster | Channel | AQUAL | v4C | Direction |
|---|---|---:|---:|---|
| AS295 | convergence | 0.899437 | 0.870620 | improves |
| AS295 | shear 1 | 0.953890 | 1.047457 | worsens |
| AS295 | shear 2 | 0.841513 | 0.737868 | improves |
| PLCKG287 | convergence | 0.918069 | 0.734174 | improves |
| PLCKG287 | shear 1 | 0.944478 | 0.790630 | improves |
| PLCKG287 | shear 2 | 0.883349 | 0.647365 | improves |

## Transfer and source diagnostics

| Training cluster | Unchanged test cluster | Transfer NRMSE | Gate |
|---|---|---:|---:|
| AS295 | PLCKG287 | 0.826098 | at most 0.800: fail |
| PLCKG287 | AS295 | 1.017377 | at most 0.800: fail |

Independently, AS295 prefers `L_sigma=133.10 kpc`, `eta_sigma=8.81`, while
PLCKG287 prefers `L_sigma=36.12 kpc`, `eta_sigma=16.59`. This is not merely an
amplitude discrepancy. The required spatial scale and morphology differ.

| Check | Result | Gate |
|---|---:|---:|
| Helmholtz equation residual | `2.38e-14` | at most `1e-10` |
| Uniform-vector disorder | `0` | at most `1e-10` |
| Rotation-covariance error | `2.09e-16` | at most `1e-10` |
| Seed after `1000x` field scaling | `1.00e-6` fraction | at most `1e-4` |
| Maximum trace-integral mismatch | `6.82e-16` | at most `1e-10` |
| Minimum full trace/RMS | `1.30e-5` | at least `-1e-10` |
| Minimum broad-power fraction | `0.999992` | at least `0.5` |
| Padding-score change | `5.52e-4` | at most `0.05` |

![Sigma v4C baryon-seeded coherence trace](../results/sigma_v4c_baryon_seeded_coherence_trace_audit/coherence_trace_audit.png)

## Physical lesson

The positive trace state solves the monopole problem. It creates an extremely
broad, smooth convergence component and works well for PLCKG287, whose missing
field is comparatively central in this map. The same isotropic smoothing
collapses AS295's distributed baryonic geometry into one central blob. AS295's
target contains displaced and oriented structures, so its first shear channel
is driven in the wrong direction.

This separates two requirements that earlier radial tests mixed together:

1. **trace capacity:** the theory must generate enough broad positive
   convergence; and
2. **tensor transport:** it must retain the locations and orientations of
   separated baryonic structures while generating that trace.

v4C supplies the first and loses the second. Its failure is not repaired by a
larger amplitude or by weakening the high-field gate. A successor would need
an anisotropic, baryon-directed propagation operator or a genuinely dynamical
trace-free carrier coupled to the sourced trace. Another scalar activation or
second isotropic smoothing length is not justified.

## Reproduction

```powershell
python scripts/check_sigma_v4c_baryon_seeded_coherence_trace.py
python -m pytest tests/test_sigma_v4c_coherence_trace.py -q
python -m ruff check src/voidscreen/sigma_coherence_trace.py scripts/check_sigma_v4c_baryon_seeded_coherence_trace.py tests/test_sigma_v4c_coherence_trace.py
```

Machine-readable results are under
`results/sigma_v4c_baryon_seeded_coherence_trace_audit/`.

