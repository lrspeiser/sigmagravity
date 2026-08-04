# Sigma v11A bounded anisotropic scalar-memory selection

> **Superseded gate result (2026-08-04):** on a tilted-aether slice the
> bounded `S` alignment depends nonlinearly on the AeST scalar velocity. A
> finite memory gradient drives that scalar Legendre Hessian through zero and
> negative. Exact v11A is retired before data. See
> [`SIGMA_V11A_TILTED_RANK_FALSIFICATION.md`](SIGMA_V11A_TILTED_RANK_FALSIFICATION.md).

## Decision

Sigma v11A passes a narrow theory-only mechanism-selection gate. It is the
first candidate after the required v10 mechanism reset. It replaces the
rank-two aether-tidal carrier with one scalar field whose causal propagation is
directionally shaped by the existing AeST scalar gradient.

The fixed-background results are:

- the memory spatial stiffness lies between `9/44` and `3/11` for every field
  magnitude and direction;
- the aether-memory static Schur margin is at least `1/44`;
- every scanned mixed squared speed is real, positive, and no greater than one;
- the limiting mixed roots move from `(9/44,1)` at zero alignment to
  `(0.156573,0.979790)` when fully aligned and saturated;
- the aether-rest TT metric principal symbol remains Einstein-Hilbert because
  a scalar derivative has no metric connection and `J_i=0` in the TT sector;
- the model uses five physical constants and no object-specific state.

This is not yet a viable theory or an observational result. The coefficient's
dependence on the AeST scalar gradient can create new nonlinear kinetic mixing
when the memory field itself has a background gradient. Complete covariant
variation, global Legendre rank, tilted characteristics, weak metric
potentials, PPN/Solar limits, and numerical convergence remain kill gates. No
astronomical product or holdout was opened.

## Why this is a genuine mechanism reset

The retired v10 family used a six-component spatial tensor `P_ij`. Its
covariant derivative necessarily contained the metric connection. On an
anisotropic background that connection gave the physical TT metric an extra
spatial stiffness and produced

$$
c_{TT}^2=1+c_P^2(p_\parallel-p_\perp)^2.
$$

V11A uses one scalar `chi`. For a scalar,

$$
\nabla_\mu\chi=\partial_\mu\chi,
$$

so there is no analogous `Gamma P` term. Directional propagation is encoded
in an algebraic spatial kinetic tensor, not in a covariant derivative of a
rank-two background.

This also differs from the retired v4 scalar-memory lane. V4 propagated one
isotropic Helmholtz memory and lost distributed shear phase. V11A's operator
is anisotropic, with its principal axis determined locally by the already
baryon-forced AeST field. It may still fail to reproduce cluster topology, but
it does not make the information-destroying isotropy assumption already
falsified in v4.

## Fields and bounded alignment

Retain the one-metric AeST base with unit timelike aether `A^mu` and scalar
`phi`. Define

$$
q_{\mu\nu}=g_{\mu\nu}+A_\mu A_\nu,
\qquad
S_\mu=q_\mu{}^\nu\nabla_\nu\phi,
$$

$$
J_\mu=A^\nu\nabla_\nu A_\mu,
\qquad
Y=S_\mu S^\mu.
$$

The bounded alignment is

$$
z(Y)={Y\over a_\Sigma^2+Y},
\qquad 0\le z<1.
$$

It is regular at `S=0`, needs no unit-vector division, and saturates rather
than growing without bound.

## Candidate addition

Introduce one scalar memory field `chi` and define its spatial kinetic tensor

$$
\boxed{
\mathcal C^{\mu\nu}
=s\left[
q^{\mu\nu}
-(1-u){S^\mu S^\nu\over a_\Sigma^2+Y}
\right].
}
$$

The selected addition is

$$
\boxed{
\begin{aligned}
\Delta\mathcal L_{11A}={}&
{1\over2}(A^\mu\nabla_\mu\chi)^2
-{1\over2}\mathcal C^{\mu\nu}
\nabla_\mu\chi\nabla_\nu\chi\\
&-{\chi^2\over2L_\chi^2}
+\beta D_\mu\chi J^\mu.
\end{aligned}
}
$$

The last term has the first-order boundary form

$$
\beta D_\mu\chi J^\mu
=-\beta\chi D_\mu J^\mu+\hbox{boundary/projector terms}.
$$

In the static weak limit, `J_i` is the gradient of the physical lapse
potential. Its divergence is therefore density/curvature sensitive rather
than only a function of local acceleration magnitude.

For fixed base fields, the static memory equation is

$$
D_\mu(\mathcal C^{\mu\nu}D_\nu\chi)
-L_\chi^{-2}\chi
=-\beta D_\mu J^\mu.
$$

Positive `mathcal C` plus a finite mass and `chi->0` at spatial infinity gives
one static solution. Dynamically, the universal prescription is retarded with
no incoming memory radiation. A freely selected homogeneous profile per
galaxy or cluster is forbidden.

## Coefficients without a sixth constant

Use the frozen sourced AeST speed

$$
u={3\over4}.
$$

Let

$$
s=c_\chi^2,
\qquad
q={\beta^2\over K_B}.
$$

At zero alignment, retain the declared factor-three static response capacity
and saturate, but do not exceed, the one-metric cone:

$$
{q\over s}={2\over3},
\qquad
q=(1-u)(1-s).
$$

These give

$$
\boxed{s={3\over11}},
\qquad
\boxed{q={2\over11}},
\qquad
\boxed{\beta=\sqrt{{2K_B\over11}}}.
$$

The anisotropy fraction is the already-derived complement of the base speed,

$$
1-u={1\over4},
$$

not a fitted sixth number. The five physical constants are provisionally

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,L_\chi\}.
$$

The first four belong to the frozen base; `L_chi` is the finite memory range.
It is not selected from observational data at this stage.

## Exact fixed-background positivity

For a wave making angle `theta` with `S`, write

$$
x={|S|\over a_\Sigma}.
$$

The directional memory stiffness is

$$
s_{\rm eff}(x,\theta)
={3\over11}\left[
1-{1\over4}{x^2\over1+x^2}\cos^2\theta
\right].
$$

Therefore

$$
\boxed{{9\over44}\le s_{\rm eff}\le{3\over11}}.
$$

The worst static principal matrix is

$$
K_{\rm static}=
\begin{pmatrix}
1&-\sqrt{2/11}\\
-\sqrt{2/11}&9/44
\end{pmatrix},
$$

whose Schur margin is

$$
{9\over44}-{2\over11}={1\over44}>0.
$$

Thus the selected cross coupling cannot overturn static ellipticity anywhere
in the bounded alignment domain.

## Fixed-background characteristic roots

The mixed aether-memory channel satisfies

$$
(u-y)(s_{\rm eff}-y)-qy=0,
\qquad y={\omega^2\over k^2}.
$$

Positivity follows from

$$
us_{\rm eff}>q,
$$

and the upper root stays within the metric cone when

$$
q\le(1-u)(1-s_{\rm eff}).
$$

The weakest positivity point is `s_eff=9/44`, where the static margin is
`1/44`. The weakest upper-cone point is the opposite endpoint `s_eff=3/11`,
where the inequality is saturated. Lowering `s_eff` moves the upper root
strictly inside the metric cone.

The endpoint roots are:

| State | `s_eff` | Mixed squared speeds |
|---|---:|---:|
| zero alignment or transverse wave | `3/11` | `9/44`, `1` |
| saturated alignment, parallel wave | `9/44` | `0.156573`, `0.979790` |

The audit evaluates 80,601 magnitude/angle combinations through
`|S|/a_Sigma=10^8`; all roots are real, positive, and at most one.

## Why this might address the empirical target

The galaxy part remains the AeST/RAR-like base. The new field is intended only
to encode the missing finite environment and geometry:

1. `L_chi` lets equal local accelerations respond differently when their
   surrounding mass distributions have different spatial scales.
2. The propagation axis follows `S`, so multiple baryonic components bend,
   converge, and compete for the memory flow rather than being replaced by an
   isotropic blur.
3. Although `chi` is a scalar, its Hessian `D_iD_j chi` has a trace and a
   traceless part, so the metric equations can in principle receive both
   convergence and oriented shear information.
4. The source and zero/retarded boundary rule are universal; there is no
   per-object halo profile.

These are capabilities, not demonstrated predictions. The complete metric
variation must show exactly how `chi` changes both `Psi` and `Phi`. If the Weyl
potential does not receive the required Hessian, v11A fails before data.

## Remaining kill gates

V11A may advance only through the following sequence:

1. derive the complete covariant Euler equations and Hilbert stress;
2. compute the full nonlinear velocity Hessian and constraint count with
   tilted aether, nonzero `S`, and nonzero `nabla chi`;
3. prove all coupled characteristics remain causal on those backgrounds;
4. derive the weak one-metric `Psi`, `Phi`, and Weyl equations;
5. prove high-acceleration Solar screening and PPN limits;
6. build a convergent PDE solver and only then freeze `L_chi` on development
   data before untouched galaxy and raw-cluster holdouts.

Failure at any stage retires the exact action. The empirical bridge scores and
the v10 fixed-background scores are not inherited.

## Prior-art boundary

The AeST base is established work by
[Skordis and Zlosnik](https://arxiv.org/abs/2007.00082). Scalar theories with
noncanonical/effective kinetic metrics are broad prior art, including
[Garriga and Mukhanov](https://arxiv.org/abs/hep-th/9904176), and the need for
background-robust luminal gravitational waves is emphasized by
[Ezquiaga and Zumalacarregui](https://arxiv.org/abs/1710.05901).

The particular bounded AeST-gradient dyad, coefficient derivation, and use as
a baryon-forced anisotropic cluster-memory channel are project hypotheses. No
novelty claim is made before a broader operator-level literature search and
survival of the nonlinear gates.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v11a_anisotropic_scalar_memory.py
python -m pytest -q tests/test_sigma_v11a_anisotropic_scalar_memory.py
```

Machine-readable evidence is in
`results/sigma_v11a_anisotropic_scalar_memory_selection/report.json`.
