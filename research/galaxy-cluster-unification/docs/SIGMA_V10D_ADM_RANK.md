# Sigma v10D aether-rest ADM Legendre-rank gate

> **Superseded gate result (2026-08-04):** the exact nonzero-carrier TT
> characteristic is `c_TT^2=1+c_P^2(p_parallel-p_perp)^2`, so anisotropic
> carrier backgrounds are outside the one-metric null cone. Exact v10D is
> retired and the aether-tidal carrier family is reset. See
> [`SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md`](SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md).

## Decision

V10D passes its aether-rest nonlinear ADM Legendre-rank subgate. The carrier's
metric-velocity dependence is an invertible perfect-square shift, so it does
not create an irregular constraint surface at any finite carrier amplitude.
The completed aether block and selected AeST scalar clock block remain
positive, and the generic degree count is twelve: the published six AeST modes
plus six hyperbolic carrier modes.

This is not the complete arbitrary-foliation constraint or full metric
characteristic proof. V10D advances only to that gate. No observational product
or holdout was opened.

## Published AeST baseline

The nonlinear Hamiltonian formulation of the AeST base uses six spatial metric
variables, three spatial aether variables, the scalar, and two auxiliary
variables. It has four first-class and four second-class constraints, leaving
six physical degrees of freedom. At the selected high-field row the aether
electric and scalar clock velocity blocks are invertible and positive.

V10D's completion does not depend on the two scalar auxiliary variables, so it
does not remove their primary momenta or their two secondary algebraic
constraints. General covariance and one-metric minimal matter coupling retain
the four diffeomorphism generators.

The baseline is Bataki, Skordis, and Zlosnik,
[Aether scalar tensor theory: Hamiltonian Formalism](https://arxiv.org/abs/2307.15126).

## Exact carrier velocity square

At one event choose an ADM foliation whose unit normal equals the local aether.
For a symmetric spatial carrier,

$$
\boxed{
W_{ij}=\dot P_{ij}-K_i{}^kP_{kj}-K_j{}^kP_{ik}
}
$$

is its projected covariant time derivative. The carrier kinetic density is

$$
\mathcal L_{P,\rm kin}={1\over2}W:W.
$$

In six-component orthonormal symmetric-tensor notation, write

$$
W=\dot P-L(P)\dot h.
$$

The velocity transformation is

$$
\begin{pmatrix}\dot h\\W\end{pmatrix}
=T(P)
\begin{pmatrix}\dot h\\\dot P\end{pmatrix},
\qquad
T(P)=
\begin{pmatrix}I&0\\-L(P)&I\end{pmatrix}.
$$

For every finite `P`,

$$
\boxed{\det T(P)=1}.
$$

The metric--carrier Hessian is therefore the congruence

$$
H_{hP}=T^{\mathsf T}
\begin{pmatrix}H_{\rm DeWitt}&0\\0&I_6\end{pmatrix}T.
$$

Sylvester's law of inertia fixes it to one negative DeWitt conformal direction,
zero null directions, and eleven positive directions for every carrier
amplitude and orientation. The negative DeWitt direction is the usual
Hamiltonian-constraint structure; it is not a new propagating ghost.

This also explains why examining only the path `dot(P)=0` can be misleading.
Its quadratic coefficient can vanish for some backgrounds, but the full
Hessian is not singular because the off-diagonal metric--carrier momentum is
nonzero.

## Remaining velocity blocks

In the same local frame, v10D's aether velocity Hessian is

$$
2K_B[e^X-X]\succeq2K_BI,
$$

and the selected high-field scalar clock coefficient is `nu=2K_2=4>0`.
The complete sixteen-velocity Hessian in

$$
(\dot h_{ij},\dot P_{ij},\dot A_i,\dot\phi)
$$

therefore has constant inertia

$$
\boxed{(n_-,n_0,n_+)=(1,0,15)}.
$$

One thousand random mixed-sign carrier tensors retain this inertia. The
analytic triangular congruence, not the scan, establishes the all-amplitude
result.

## Generic constraint count

Using six independent aether-spatial carrier components adds six configuration
variables and six regular momenta, with no new primary constraint. The phase
space dimension becomes

$$
2(12+6)=36.
$$

Retaining four first-class and four second-class constraints gives

$$
{36-2(4)-4\over2}=12
$$

physical degrees of freedom: six from AeST and six from the carrier.

## What remains unresolved

An aether-rest time direction is a valid local Legendre test but need not form
a global foliation when the aether has vorticity. The next gate must therefore
derive the arbitrary-foliation constraint rank and the complete
metric--aether--scalar--carrier characteristic determinant on anisotropic
carrier, inhomogeneous, and FLRW backgrounds. In particular it must verify that
the physical TT metric cone remains exactly luminal when `P` is nonzero.

PPN/Solar limits, cosmology, and numerical PDE convergence remain later gates.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10d_adm_rank.py
python -m pytest -q tests/test_sigma_v10d_adm_rank.py
```

Machine-readable evidence is in `results/sigma_v10d_adm_rank/report.json`.
