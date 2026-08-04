# Sigma v7C cubic-Hessian construction gate

## Decision

Sigma v7C passes its numerical construction gate but does **not** advance to an
observational-map test.  A subsequent physical-metric projection shows that the
frozen scalar-only equation has no derived leading Weyl response.  It is
retained as a scalar dynamics and nonlinear-solver control.  No observational
array or raw holdout was opened in reaching either decision.

This is the first renewed candidate that simultaneously has:

- a positive local carrier inherited from the ghost-free massive-spin-2 class;
- a nonlinear three-dimensional response that is not reducible to enclosed
  mean density;
- positive temporal and spatial perturbation coefficients on every tested
  branch cell;
- a verified nonlinear solver; and
- material non-additivity for separated baryonic sources.

It is not a unified Sigma theory.  The construction does not overcome the
factor-`1.5` maximum bimetric lensing amplitude found in v7B, and it has not been
combined with a covariant low-acceleration trace sector in one complete action.
Passing the construction checks authorized only the physical-metric derivation,
which then failed.  It never authorized treating `pi` as a lensing potential.

## Field equation

In dimensionless units scaled by the one Vainshtein length, the frozen static
equation is

$$
\boxed{
3\nabla^2\pi
+\kappa\left[(\nabla^2\pi)^2
-\partial_i\partial_j\pi\,\partial_i\partial_j\pi\right]
=J_b,
\qquad \kappa=1.
}
$$

This is the cubic helicity-zero Hessian structure in the decoupling limit of
Vainshtein-screened massive gravity.  It is established massive-gravity/Galileon
physics, not a new equation invented by this project.  The relevant nonlinear
screening mechanism is reviewed by
[Babichev & Deffayet](https://arxiv.org/abs/1304.7240), and the ghost-free
massive/bimetric action boundary is the
[Hassan--Rosen construction](https://arxiv.org/abs/1109.3515).

The term that matters for the present hypothesis is

$$
(\operatorname{tr}H)^2-\operatorname{tr}(H^2),
\qquad H_{ij}=\partial_i\partial_j\pi.
$$

It depends on the complete Hessian eigenstructure, not only on `|grad pi|`,
density, or total enclosed mass.  Therefore

$$
\pi[J_1+J_2]\ne\pi[J_1]+\pi[J_2]
$$

when the nonlinear term is active.  This is the mathematical feature requested
by the earlier cluster experiments: separated components interact through the
field equation *before* their lensing effects are summed.

## Perturbation-health gate

Linearizing the equation around a static background gives the spatial principal
matrix

$$
Z_{ij}=3\delta_{ij}
+2\kappa\left[(\nabla^2\pi)\delta_{ij}-H_{ij}\right].
$$

Static ellipticity requires every eigenvalue of `Z_ij` to be positive.  The time-
derivative coefficient of the corresponding covariant cubic branch is

$$
Z_t=3+2\kappa\nabla^2\pi,
$$

which must also be positive to avoid a negative kinetic direction for small
perturbations.

Across the analytic and separated-source construction suite, the minima were

$$
\boxed{\min Z_t=3.00305,\qquad
\min\operatorname{eig}(Z_{ij})=2.07945.}
$$

These are support-specific construction results, not a proof that every source
or cosmological branch is stable.  Any later map solve must repeat both checks
cell by cell and reject a parameter setting at the first loss of positivity.

## Analytic spherical recovery

For the exact quadratic spherical potential

$$
\pi=A(x^2+y^2+z^2),
$$

the Hessian is `H_ij=2A delta_ij`, so

$$
\nabla^2\pi=6A,
$$

$$
(\nabla^2\pi)^2-\operatorname{tr}(H^2)=24A^2,
$$

and the exact source is

$$
J_b=18A+24\kappa A^2.
$$

With `A=0.02`, the nonlinear solver recovered the potential with

- relative potential error `6.09e-11`;
- normalized equation residual `8.12e-11`; and
- 83 damped iterations.

This checks the complete finite-difference Hessian, nonlinear invariant,
Dirichlet Poisson substep, and iteration—not merely one helper function.

## Separated-component result

The no-label morphology test used two identical smooth positive sources at
`x=-1` and `x=+1`, with no center, orientation, or response scale fitted from a
target.  Three solutions were compared:

1. the first source alone;
2. the second source alone; and
3. both sources in one nonlinear equation.

The relative difference

$$
{\|\pi[J_1+J_2]-\pi[J_1]-\pi[J_2]\|_2
\over\|\pi[J_1+J_2]\|_2}
$$

was

$$
\boxed{0.0722319.}
$$

The preregistered materiality threshold was `0.05`.  Rotating the entire source
pair from the x axis to the y axis and rotating the solution back changed the
field by only `1.26e-16`, demonstrating that the effect comes from the source
geometry rather than a grid-axis preference.

## Numerical gate

| Check | Result | Gate |
|---|---:|---:|
| Maximum normalized equation residual | `7.9974e-7` | at most `1e-6` |
| Analytic relative potential error | `6.0914e-11` | at most `1e-7` |
| Minimum temporal kinetic coefficient | `3.00305` | above zero |
| Minimum spatial ellipticity eigenvalue | `2.07945` | above zero |
| Component non-additivity | `7.223%` | at least `5%` |
| Rotation covariance error | `1.263e-16` | at most `1e-10` |
| Change after doubling grid resolution | `1.165%` | at most `2%` |

All nonlinear solves converged.  The fine grid contained exactly twice the
spatial resolution over the same physical domain and was downsampled onto the
coarse grid for the comparison.

## What this does and does not mean physically

The result supplies a concrete version of the “gravity vectors do not simply
sum in a distributed system” idea.  The individual baryonic sources still add
linearly in `J_b`; what fails to add is the *field response*, because the
curvatures of the response interact through Hessian invariants.

In a nearly spherical or coherent system, the Hessian eigenvectors line up and
the response approaches the familiar spherical Vainshtein behavior.  Around
separated components, cross terms change the local principal directions and
screening strength.  That is a physically derived route by which a cluster can
behave differently from a smooth galaxy without asking the equation whether it
is looking at a “cluster.”

Several gaps remain:

1. `pi` is only the helicity-zero decoupling-limit field, not the complete two-
   metric solution.
2. The conversion from `pi` to both physical metric potentials must be derived;
   no free lensing multiplier is permitted.
3. The bimetric carrier alone cannot exceed the v7B factor-`1.5` light-
   deflection ceiling.
4. A low-acceleration trace sector is still needed for sustained galaxy curves.
5. Combining that trace sector with the ghost-free bimetric action may change
   the constraint algebra and must be audited before it can be called a theory.
6. Global branch uniqueness, cosmology, `c_T`, PPN, and well-posed dynamical
   evolution remain open.

## Subsequent physical-metric gate

The decoupling-limit field redefinition begins with

$$
\delta h_{\mu\nu}^{(0)}=-\eta_{\mu\nu}\pi.
$$

For

$$
ds^2=-(1+2\Psi)dt^2+(1-2\Phi)d\mathbf{x}^2,
$$

this gives

$$
\delta\Psi=-\frac{\pi}{2},\qquad
\delta\Phi=+\frac{\pi}{2},\qquad
\delta W=\frac{\delta\Psi+\delta\Phi}{2}=0.
$$

The physical-metric audit also confirms that a static disformal term can have
a direction-dependent null contraction.  But v7C did not freeze that mapping,
the complete action-linked scalar equation, or the coupled tensor equation.
Consequently its `7.223%` scalar nonadditivity is not a prediction for
convergence or shear.  See
[`SIGMA_V7C_PHYSICAL_METRIC_PROJECTION_GATE.md`](SIGMA_V7C_PHYSICAL_METRIC_PROJECTION_GATE.md).

The next step is the required three-formulation falsification synthesis for the
positive-spin-2 carrier sequence.  No spent or held-out map may be opened for
v7C.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v7c_cubic_hessian.py
python -m pytest -q tests/test_sigma_v7_cubic_hessian.py
```

Machine-readable evidence is stored in
`results/sigma_v7c_cubic_hessian_gate/report.json`.
