# NBM0 nonlocal basin metric: first completed gate

## Decision

The MOND/BTFR-like mass-scaling formula is retired as a physical-theory branch.
NBM0, a nonlocal two-potential basin metric, is now the active theory candidate.
It passes its algebraic weak-field metric gate but cannot yet be fit or claimed as
a galaxy-cluster unification because the current data contain zero strict-ready
same systems.

This is not a negative fit result. It is a successful equation-level result and
a failed empirical-identifiability gate.

## What replaces the acceleration law

NBM0 does not use a MOND `mu` function, a universal `a0`, a fitted baryonic
Tully-Fisher exponent, or the retired mass-velocity formula. It starts from a
dimensionless basin field sourced relative to the cosmological background:

```
(Box - L_X^-2) X = -kappa_X (T-T_background)/M_Pl^2.
```

The retarded solution is nonlocal: an extended overdense or underdense basin and
its boundary conditions determine `X`. Matter and light follow the same physical
metric,

```
g_tilde_mn = exp(2 alpha X) [g_mn + 2 beta X U_m U_n].
```

At weak field this gives

```
Psi = U_N + c^2 (alpha-beta) X,
Phi = U_N - c^2 alpha X,
(Phi+Psi)/2 = U_N - c^2 beta X/2.
```

Therefore the additional lensing-to-dynamics response is fixed by the metric:

```
q_X = -beta/[2(alpha-beta)].
```

It is not permissible to multiply the cluster lensing result by a fitted factor
after fitting galaxies.

## Algebraic result

| Metric limit | Dynamics coefficient | Lensing coefficient | Lensing/dynamics ratio |
|---|---:|---:|---:|
| Pure conformal: `beta=0` | `alpha` | 0 | 0 |
| Pure disformal: `alpha=0` | `-beta` | `-beta/2` | 1/2 |
| No slip: `beta=2 alpha` | `-alpha` | `-alpha` | 1 |
| Example with lensing twice dynamics | `-alpha/3` | `-2 alpha/3` | 2 |

Pure conformal coupling is therefore structurally unsuitable: it can alter
massive-particle dynamics but supplies no additional lensing. A vector/disformal
component is mandatory. The no-slip limit predicts that the same additional
potential changes dynamics and the effective lensing acceleration equally. Other
ratios require different *theory couplings*, not a lensing normalization.

When `X` is dynamically screened to zero, both potentials reduce exactly to the
GR/Newtonian potential. A completed action still has to produce that screening
and pass kinetic, gradient, wave-speed, conservation, and Solar-System gates.

## Why it is not fit yet

The response supports currently available in the repository are not a valid
joint sample:

| Domain | Systems / points | Central 90% `log10 chi` support | Evidence |
|---|---:|---:|---|
| SPARC dynamics | 131 / 3,034 | -8.126 to -5.877 | Resolved rotation speeds |
| BCG dynamics | 34 / 34 | -5.749 to -5.195 | 11 summaries and 23 calibrated proxies |
| CLASH lensing | 20 / 84 | -5.618 to -4.893 | GR/NFW-deprojected summaries |

SPARC and CLASH have a 0.260-dex gap between their central 90% compactness
ranges. BCG dynamics moves into the cluster range, but those systems do not have
same-object raw lensing likelihoods. Dividing the CLASH enhancement by the SPARC
or BCG enhancement would therefore confuse object population, compactness, and
probe type with gravitational slip.

The same-system audit evaluated 15 candidates. Two pass the preliminary 3+3
radial-overlap structure, but zero have complete baryonic forward inputs and
zero have theory-neutral joint covariance. The frozen target is ten.

## Current status and next outcome

- Algebraic metric gate: **pass**.
- Explicit cluster-lensing path: **pass at equation level**.
- Pure conformal version: **reject**.
- Empirical two-potential identifiability: **fail with current data**.
- Parameter fit: **not authorized**.
- Unified galaxy-cluster claim: **withheld**.

The already-running RX J2129 measurement pipeline is the next legitimate input.
If it produces accepted dynamics covariance, baryonic profiles, and
theory-neutral image-position/lensing covariance, it can serve as a single-system
NBM0 engineering demonstration. It cannot establish a population law; nine more
strict systems would still be required.

The next theory derivation must complete the reciprocal equations for `X`, the
timelike vector, the metric, and matter while retaining no more than four global
parameters: `kappa_X`, `L_X`, `alpha`, and `beta`. No galaxy or cluster fit should
precede that action or the same-system data gate.

## Prior-art boundary

Conformal/disformal matter metrics, scalar-vector-tensor gravity, and nonlocal
gravity are established theory classes. The potentially distinctive proposal is
the particular density-contrast basin source, its use as the gravity-valley
variable, and the frozen same-system dynamics/lensing test. NBM0 should not be
described as inventing disformal gravity.
