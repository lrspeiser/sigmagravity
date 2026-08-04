# Sigma v8B inherited-constraint and source-uniqueness gate

## Decision

V8B is held before observational data. The published AeST constraint count is
now reproduced correctly, and the homogeneous source analysis supplies a
mandatory baryon-only boundary condition:

$$
\boxed{I_0=0.}
$$

This is not a sixth parameter. It is a fixed exclusion of the freely specified
dust-like AeST state. With this boundary, the only homogeneous zero-current v8B
branch having positive clock kinetic slope is $Q=Q_0$. That is a real source
uniqueness result, but only on aligned FLRW. The combined nonlinear Hamiltonian
count and uniqueness for arbitrary three-dimensional baryonic data remain open.

## Primary-source correction to the earlier spectrum wording

The AeST base is from
[Skordis and Zlosnik](https://arxiv.org/abs/2007.00082). Its selected
finite-frequency tensor, vector, and scalar modes have squared speeds `1`, `1`,
and `0.75`. The same paper also finds an `omega=0` sector: a constant mode has
zero Hamiltonian, while a linearly growing mode has positive Hamiltonian only
above a momentum of order `mu` and negative Hamiltonian below it. The authors
describe this as likely Jeans-like and not a low-momentum quantum-vacuum
instability.

Accordingly, the project no longer says that the complete base spectrum or
Hamiltonian is simply positive. The narrower supported statement is that the
selected finite-frequency propagating modes are positive and causal. The
infrared zero-frequency sector is an unresolved health gate.

## What the published nonlinear Hamiltonian establishes

[Bataki, Skordis, and Zlosnik](https://arxiv.org/abs/2307.15126) give the full
non-perturbative Hamiltonian formulation of the published AeST base. Using the
spatial metric, spatial aether, scalar, and two auxiliary fields gives 12
configuration variables and hence a 24-dimensional phase space. They find four
first-class and four second-class constraints, so

$$
N_{\rm dof}
=\frac{24-2(4)-4}{2}
=6.
$$

That count is prior-art evidence for the base action, not for v8B. The v8B term

$$
\mathcal L_C
=(\alpha-1)L_H^2(Q-Q_0)^2
q^{\mu\nu}\nabla_\mu\nabla_\nu\phi
$$

changes the metric-scalar canonical momenta away from $Q=Q_0$. The Hamiltonian
paper explicitly notes that Horndeski or more general higher-derivative
extensions require a new canonical analysis. Therefore, v8B may inherit neither
the six-degree count nor boundedness by assertion.

The combined gate requires:

1. exactly six nonlinear physical degrees of freedom;
2. no change of constraint rank on the claimed phase-space domain;
3. a Hamiltonian bounded in every claimed background regime;
4. resolution or an explicit phenomenological bound for the infrared mode.

## Why the published cosmological state conflicts with the project goal

For the published quadratic clock minimum,

$$
\mathcal K(Q)=K_2(Q-Q_0)^2,
$$

shift symmetry integrates the homogeneous scalar equation once:

$$
a^3\mathcal K_Q=I_0.
$$

Thus

$$
Q-Q_0={I_0\over2K_2a^3},
$$

and

$$
8\pi G\rho_\phi
=Q\mathcal K_Q-\mathcal K
={Q_0I_0\over a^3}
+{I_0^2\over4K_2a^6}.
$$

The leading term behaves exactly as pressureless matter. Its normalization is
the integration constant $I_0$, not something predicted by the baryonic source;
the base paper explicitly says that this density is not classically predicted.
Using it to supply the missing cluster or cosmological gravity would violate
this project's intended advantage over invisible matter: it would replace a
dark-matter density with a freely specified, dust-like gravitational-field
state.

The project therefore freezes $I_0=0$. If v8B later needs nonzero or
object-dependent $I_0$ to fit galaxies, clusters, the CMB, or structure, it
fails the baryon-only gate.

## Exact v8B homogeneous current

On aligned flat FLRW, the selected clock and completion contributions per
physical volume are

$$
\mathcal L_{\rm clock}=2K_2(Q-Q_0)^2,
$$

$$
\mathcal L_C=-3CHQ(Q-Q_0)^2,
\qquad
C=(\alpha-1)L_H^2.
$$

The conserved shift charge is therefore

$$
\boxed{
{I_0\over a^3}
=(Q-Q_0)
\left[4K_2-3CH(3Q-Q_0)\right].
}
$$

For $I_0=0$, the algebraic branches are

$$
Q=Q_0
$$

and, when $CH>0$,

$$
Q={1\over3}
\left(Q_0+{4K_2\over3CH}\right).
$$

The current slope $d(I_0/a^3)/dQ$ is the homogeneous clock Hessian. At the
minimum it is

$$
4K_2-6CHQ_0
=2(2K_2-3CHQ_0),
$$

which is positive precisely below the previously derived FLRW clock bound. At
the other zero-current root it is

$$
-9CH(Q-Q_0)<0.
$$

The completion root is therefore kinetically unstable on the selected positive
branch. At the dimensionless audit point, the two roots are approximately
`0.5` and `4.7381`; their slopes are `+7.4167` and `-7.4167`.

This is why $I_0=0$ selects one stable homogeneous v8B clock state without
adding a parameter.

## What remains unproved

Homogeneous uniqueness does not imply uniqueness in a galaxy or cluster. Before
data, v8B must still demonstrate that fixed baryonic stress-energy and universal
asymptotic conditions select one regular solution for the metric, aether, and
scalar fields. In particular, the audit must exclude:

- multiple stable inhomogeneous scalar branches;
- vector/aether hair not fixed by baryons;
- boundary data that can be varied to imitate a halo;
- a completion-induced change of Hamiltonian constraint rank;
- use of the published infrared mode as the missing gravitating component.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v8b_source_constraint.py
python -m pytest -q tests/test_sigma_v8b_source_constraint.py
```

Machine-readable evidence is stored in
`results/sigma_v8b_source_constraint_gate/report.json`.
