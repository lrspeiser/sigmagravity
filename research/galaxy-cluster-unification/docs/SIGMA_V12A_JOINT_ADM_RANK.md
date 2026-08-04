# Sigma v12A joint AeST--DHOST ADM kinetic rank

## Decision

V12A passes the unreduced joint kinetic-rank subgate. In the auxiliary-variable
ADM action, the DHOST primary null direction is not lifted by the AeST sector.
The aether Maxwell term adds three positive velocity directions, while the
remaining AeST terms only shift canonical momenta.

This is not yet a constraint-count or theory-health pass. The existence of a
kinematic primary constraint does not prove that preserving it in time produces
the required regular secondary constraint after it is coupled to the AeST
auxiliary-field constraints (or treated in an equivalent unreduced unit-vector
representation).

## Extended ADM variables

Introduce the first-derivative scalar covector

$$
B_\mu=\nabla_\mu\phi
$$

as an auxiliary coordinate, with a constraint enforcing its definition. On an
arbitrary ADM foliation, the second-scalar-derivative velocity is

$$
V_*=\mathcal L_n(n^\mu B_\mu),
$$

and the six metric velocities are the components of `K_ij`. The quadratic
DHOST kinetic density has the standard form

$$
\mathcal L_{\rm D,kin}
=\mathcal A V_*^2+2\mathcal B^{ij}V_*K_{ij}
+\mathcal K^{ij,kl}K_{ij}K_{kl}.
$$

In a six-component basis for `K_ij`, write this Hessian as

$$
H_D=
\begin{pmatrix}
\mathcal A&B^T\\
B&K
\end{pmatrix}.
$$

The Class-Ia identities selected in v12A are precisely the Schur condition

$$
\boxed{\mathcal A=B^TK^{-1}B.}
$$

Hence

$$
\boxed{n_D=(1,-K^{-1}B),\qquad H_Dn_D=0.}
$$

This is the primary degeneracy that removes the would-be Ostrogradsky scalar.

## Why AeST does not lift the null vector

Use the aether covector as the independent vector variable.

### Maxwell term

The AeST field strength is

$$
F_{\mu\nu}=\nabla_\mu A_\nu-\nabla_\nu A_\mu
=\partial_\mu A_\nu-\partial_\nu A_\mu.
$$

The Levi-Civita connection cancels identically. Consequently the Maxwell term
contains the three physical aether velocities but neither `K_ij` nor `V_*`.
At `K_B=1`, its Hessian is a positive `I_3` block in a local orthonormal ADM
basis. With a general spatial metric it is a positive congruence `G_E`, which
does not change its three-positive-direction inertia.

### Scalar first-derivative terms

After `B_mu` is introduced, the AeST invariants

$$
Q=A^\mu B_\mu,
\qquad
Y=(g^{\mu\nu}+A^\mu A^\nu)B_\mu B_\nu
$$

are configuration variables. The free function `F(Y,Q)` therefore contains no
`V_*`, `K_ij`, or aether velocity.

### Aether--scalar mixing

The remaining derivative mixing is

$$
J^\mu B_\mu,
\qquad
J^\mu=A^\nu\nabla_\nu A^\mu.
$$

`J` is linear in the aether velocity and the metric connection. It is therefore
affine in the ADM velocity set and contains no `V_*`. It shifts momenta but its
second velocity derivative is zero on every background. The unit constraint
`A^2=-1` is algebraic and adds no velocity.

The exact combined Hessian is thus

$$
\boxed{
H_{12A}=\operatorname{diag}(H_D,K_BG_E),
\qquad G_E>0.
}
$$

The joint null vector `(n_D,0,0,0)` therefore survives before the algebraic
unit-vector and DHOST constraints are reduced. Background aether tilt and
scalar-gradient orientation can change configuration coefficients and affine
momentum shifts, but they cannot change this unreduced second-velocity block.
Whether the same constraint survives the coupled reduction is the next gate,
not a result of this one.

## Executable audit

Two thousand random nonsingular metric blocks were generated with the one-
negative-direction DeWitt inertia. For each, a random DHOST mixing vector fixed
the scalar entry by the exact Schur identity.

| quantity | result |
|---|---:|
| DHOST inertia | `(1 negative, 1 zero, 5 positive)` in every trial |
| Joint inertia | `(1 negative, 1 zero, 8 positive)` in every trial |
| Added aether eigenvalues | three positive values under random spatial-metric congruences |
| Maximum DHOST null residual | recorded in machine report; gate `<10^-11` |
| Maximum joint null residual | recorded in machine report; gate `<10^-11` |
| Finite-difference residual after arbitrary linear momentum shift | gate `<10^-6` |

The numerical matrices illustrate the exact block proof; they are not a
substitute for it.

## What this does and does not establish

Established:

- the v12A Class-Ia primary degeneracy is not lifted in the unreduced kinetic Hessian by AeST;
- no lapse or shift velocity is added;
- the aether Maxwell modes add a positive independent block; and
- the AeST mixing changes momenta but not velocity rank.

Not established:

- the explicit primary constraint in canonical variables;
- its Poisson brackets with the AeST auxiliary-field primary and secondary
  constraints;
- the regular DHOST secondary constraint;
- the final first-/second-class constraint count or six-mode AeST count;
- positivity of the reduced physical Hamiltonian;
- arbitrary-background characteristic cones; or
- the complete metric stress and weak-field equations.

In particular, this subgate does not claim health on an arbitrarily tilted
background after solving the unit-vector constraints. That statement requires
the explicit coupled Dirac reduction below.

The next kill gate must derive the primary constraint, preserve it in time, and
show that it generates one regular secondary instead of fixing an otherwise
required multiplier or causing field-dependent rank. Only then can the complete
degree count be claimed.

The ADM kinetic form and degeneracy logic are established DHOST results; see
[Ben Achour, Langlois, and Noui](https://arxiv.org/abs/1602.08398). The baseline
AeST nonlinear count of four first-class and four second-class constraints is
published in [Bataki, Skordis, and Zlosnik](https://arxiv.org/abs/2307.15126).
This report establishes only the compatibility of their highest-velocity
blocks for the selected combined action.

No observational product or holdout was opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_joint_adm_rank.py
python -m pytest -q tests/test_sigma_v12a_joint_adm_rank.py
```

Machine-readable evidence is in
`results/sigma_v12a_joint_adm_rank/report.json`.
