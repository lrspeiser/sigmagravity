# Sigma v6A differentiability result and v6B selection

## Outcome

The exact Sigma v6A orientation closure is retired before a full metric variation
or any observational fit. Its frozen square-root Hessian invariant has no unique
first variation on the GR/Minkowski or isotropic FLRW backgrounds about which the
mode spectrum must be defined.

A minimally related but mathematically distinct v6B closure advances to a full
closed-time-path variation. It replaces the local Hessian magnitude with an
analytic, bounded, twice-retarded tensor coherence. The v6B orientation term
begins at fourth perturbative order, so it does not add a pole to the quadratic
zero-background propagator. This is only a selection result, not a proof that v6B
is healthy on a nonzero astrophysical background.

## Why v6A fails

For a one-parameter metric perturbation

$$
g_{\mu\nu}=\bar g_{\mu\nu}+\varepsilon h_{\mu\nu},
$$

the v6A invariants begin as

$$
X=A\varepsilon^2+O(\varepsilon^3),\qquad
Z=B\varepsilon^2+O(\varepsilon^3),
$$

with $A,B\ge0$. Therefore its frozen invariant is

$$
\chi_{6A}
=A\varepsilon^2+\lambda_\Sigma\sqrt B\,|\varepsilon|+cdots.
$$

Because $f(\chi)=\chi e^{-\sqrt\chi}=\chi+O(\chi^{3/2})$ near zero,

$$
{dS_{6A}\over d\varepsilon}\bigg|_{0^+}
=+\lambda_\Sigma\sqrt B,
\qquad
{dS_{6A}\over d\varepsilon}\bigg|_{0^-}
=-\lambda_\Sigma\sqrt B.
$$

The jump is $2\lambda_\Sigma\sqrt B$. It does not shrink with finite-difference
step. Thus the problem is not numerical stiffness: no unique Euler--Lagrange
equation exists at the required background unless $\lambda_\Sigma=0$. Setting it
to zero removes the proposed orientation physics and leaves the published scalar
nonlocal MOND control.

## What simple powers of the Hessian do

Since $Z^p\sim|\varepsilon|^{2p}$:

| Power | First variation | Quadratic variation | Consequence |
|---:|---|---|---|
| $p=1/2$ | undefined | undefined | v6A cusp |
| $1/2<p<1$ | exists | divergent | no regular spectrum |
| $p=1$ | regular | nonzero | changes the quadratic operator; the direct Hessian term carries additional momentum dependence and needs a new pole/degeneracy proof |
| $p>1$ | regular | zero at the GR background | orientation is nonlinear only |

Simply changing $\sqrt Z$ to $Z$ is therefore not automatically safe. In the
static Fourier limit, a direct local $T_{ij}T^{ij}$ term scales like $k^4h^2$ and
changes the propagator. We instead remove those two derivatives with a second
retarded inverse before constructing the analytic invariant.

## Selected v6B envelope

Retain

$$
U_R=\Box_R^{-1}
\left(R_{\mu\nu}u^\mu u^\nu-\frac12R\right)
$$

and define

$$
T_{\mu\nu}=\operatorname{STF}_h(\nabla_\mu\nabla_\nu U_R),
\qquad
Q_{\mu\nu}=\operatorname{STF}_h(\Box_R^{-1}T_{\mu\nu}),
$$

$$
Z_Q=Q_{\mu\nu}Q^{\mu\nu},
\qquad
C_Q={Z_Q\over Z_Q+\phi_\Sigma^2},
$$

$$
\boxed{\chi_{6B}=X(1+\lambda_\Sigma C_Q)},
\qquad
f(\chi_{6B})=\chi_{6B}e^{-\sqrt{\chi_{6B}}}.
$$

The three provisional universal constants are
$\{a_\Sigma,\lambda_\Sigma,\phi_\Sigma\}$. There are no object labels,
per-object force parameters, lensing multipliers, or free homogeneous memory
states.

The new potential scale $\phi_\Sigma$ is not fitted here. It is required for
analyticity: $C_Q=Z_Q/\phi_\Sigma^2+O(Z_Q^2)$ at $Q=0$. Since
$X=O(\varepsilon^2)$ and $Z_Q=O(\varepsilon^2)$,

$$
\chi_{6B}-X=O(\varepsilon^4).
$$

The executable fit to the perturbation scaling recovers fourth order. Thus the
quadratic scalar MOND control is unchanged, while orientation can become order one
on a finite background when $Z_Q\sim\phi_\Sigma^2$.

## Static momentum-order check

For a static Fourier mode, the two inverse Laplacians cancel the two Hessian
derivatives. The transfer from a scalar metric amplitude into $Q_{ij}$ is
proportional to

$$
P_{ij}(\hat k)=\hat k_i\hat k_j-\frac13\delta_{ij},
$$

whose norm is

$$
P_{ij}P^{ij}={2\over3}
$$

independent of $|k|$. The numerical audit held that norm constant across 24
decades of wavenumber. This avoids the direct $k^4$ growth of the local Hessian
closure. It does not yet prove that the retarded time-dependent kernel has no
secular or background instability.

## Required next gate

Before data, v6B must now:

1. be written as a complete closed-time-path influence action rather than a
   schematic retarded substitution;
2. yield a diffeomorphism Ward identity and conserved physical metric equation;
3. be localized with fixed retarded initial data and no freely specifiable
   halo-like homogeneous modes;
4. pass scalar, vector, and tensor stability on nonzero FLRW, spherical, and
   anisotropic memory backgrounds;
5. show no secular growth from the second inverse d'Alembertian;
6. derive the MOND/BTFR coefficient and two lensing potentials from the same
   equation.

Any failure retires this exact v6B envelope before observational constants are
chosen.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/select_sigma_v6b_differentiable_orientation.py
python -m pytest -q tests/test_sigma_v6_metric_memory.py
```

Machine-readable evidence is in
`results/sigma_v6b_differentiable_orientation_selection/report.json`.
