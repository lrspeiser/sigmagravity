# Sigma v8B covariant-variation and FLRW clock gate

## Decision

The v8B completion passes its scalar/vector variation subgate and exposes a new
mandatory cosmological stability bound. It advances only to the exact metric
stress tensor, Noether identity, nonlinear Hamiltonian count, and full
time-dependent characteristic determinant. It is not authorized for
observational fitting.

## Definitions

Write the completion as

$$
\mathcal L_C=C B^2H_\perp,
\qquad
C=(\alpha-1)L_H^2,
$$

$$
B=Q-Q_0,
\qquad
Q=A^\mu\nabla_\mu\phi,
$$

$$
q^{\mu\nu}=g^{\mu\nu}+A^\mu A^\nu,
\qquad
H_\perp=q^{\mu\nu}\nabla_\mu\nabla_\nu\phi.
$$

The scalar and vector are gravitational fields. All ordinary matter remains
minimally coupled to the single metric `g_mn`.

## Exact scalar variation

Holding `g_mn` and `A^m` fixed while varying `phi` gives

$$
\boxed{
{\mathcal E_\phi^{(C)}\over C}
=\nabla_\nu\nabla_\mu(B^2q^{\mu\nu})
-2\nabla_\mu(BH_\perp A^\mu)
}.
$$

Each term appears separately to contain third derivatives of `phi`. Their
principal parts are

$$
-2B A^\rho q^{\mu\nu}
\nabla_\rho\nabla_\mu\nabla_\nu\phi
$$

and

$$
+2B q^{\mu\nu}A^\rho
\nabla_\mu\nabla_\nu\nabla_\rho\phi.
$$

They cancel because the principal third derivative of a scalar is symmetric.
Commutators leave curvature times lower derivatives, not a third scalar time
derivative. The machine audit contracts arbitrary fully symmetric third-order
tensors and obtains cancellation below `1e-12`.

This proves a necessary second-order scalar-principal identity. It is not yet a
Hamiltonian degree-of-freedom proof.

## Exact vector variation

Taking `A^sigma` as the independent vector variable gives

$$
\boxed{
{\mathcal E_{A^\sigma}^{(C)}\over C}
=2BH_\perp\nabla_\sigma\phi
+2B^2A^\mu\nabla_\mu\nabla_\sigma\phi
}.
$$

The completion contains no derivative of `A^mu` in its unintegrated form, so
this contribution is algebraic in the vector. It does not add a new vector
velocity or change the rank of the vector kinetic block by itself. It can still
change coupled constraints through the metric and scalar equations, which is
why the Hamiltonian gate remains open.

At `Q=Q0`, both new Euler derivatives vanish on a static background. This
confirms that the v8B causal partner leaves the selected static geometry equation
unchanged.

## Homogeneous time-dependent background

Use

$$
ds^2=-N^2dt^2+a^2(t)d\mathbf x^2,
\qquad
A^\mu=(1/N,0,0,0),
\qquad
Q={\dot\phi\over N}.
$$

Then

$$
H_\perp=-3H Q,
\qquad
H={\dot a\over aN},
$$

and the reduced completion Lagrangian per coordinate volume is

$$
L_C=-3C a^2\dot a\,Q(Q-Q_0)^2.
$$

There is no lapse velocity. At `Q=Q0`, the new mixed velocity Hessian between
`dot(a)` and `dot(phi)` vanishes, but the scalar clock kinetic coefficient is
shifted. Combining it with the AeST `K_2` term gives

$$
\boxed{
K_{\rm clock}=2K_2-3(\alpha-1)L_H^2H Q_0
}.
$$

For the selected values `K_2=2` and `alpha=16/9`, positivity requires

$$
\boxed{L_H^2H Q_0<{12\over7}}.
$$

Since the selected AeST relation gives `Q0=mu_sigma/2`, this is equivalently

$$
L_H^2H\mu_\Sigma<{24\over7}.
$$

The bound is not a fitted constant. It is a constraint that every eventual
choice of the existing `L_H` and `mu_sigma` must satisfy over the claimed
cosmological validity range. The audit verifies positive kinetic energy below
the boundary and negative kinetic energy just above it.

## Tensor cone on FLRW

On the aligned homogeneous background, `H_perp` is proportional to the trace of
the extrinsic curvature. A term of the form `sqrt(h) f(t,N) K` is linear in that
trace and is removable from the tensor kinetic sector by integration by parts.
Thus this completion does not change the transverse-traceless tensor speed on
FLRW; the Einstein-Hilbert result `c_T=1` is retained at this subgate.

This is not a proof for every tilted or inhomogeneous aether background.

## Remaining kill gates

Before observations, v8B must still provide:

1. the exact metric stress tensor from `L_C`;
2. the off-shell diffeomorphism/Noether identity and matter conservation;
3. a nonlinear ADM or Hamiltonian constraint count, including off-`Q0` mixing;
4. the full metric-vector-scalar characteristic determinant on evolving,
   tilted, rotating, and inhomogeneous backgrounds;
5. a declared cosmological validity range satisfying the clock bound;
6. the weak-field, Solar, PPN, and unique baryon-forced solutions.

Spatially covariant theories can hide a mode at linear order and recover it on
inhomogeneous or nonlinear backgrounds, so a homogeneous kinetic pass is not
sufficient. This general warning is discussed by
[Gao, Kang, and Yao](https://arxiv.org/abs/1902.07702). No claim is made that
v8B belongs to or escapes every class analyzed there.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v8b_covariant_variation.py
python -m pytest -q tests/test_sigma_v8b_covariant_variation.py
```

Machine-readable evidence is stored in
`results/sigma_v8b_covariant_variation_gate/report.json`.
