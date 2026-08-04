# Sigma v5C degeneracy-first action selection

## Outcome

**Historical selection record; the fixed row is now retired.** The selected
derivation lane was a fixed, four-constant member of the **luminal Class-Ia
quadratic DHOST** family. No observational data were accessed and no constant
was fitted.

The subsequent exterior-law audit proves that its screened vacuum is exactly
GR and its linear massive-scalar exterior falls no slower than inverse-square.
It cannot sustain flat galaxy curves. See
[`SIGMA_V5C_EXTERIOR_LAW_RESULTS.md`](SIGMA_V5C_EXTERIOR_LAW_RESULTS.md).

The exact v5B failure changes the order of work. We no longer invent a source
and ask later whether its action is healthy. v5C first selects a published
degenerate class in which the unwanted higher-derivative mode is removed by an
algebraic identity and the tensor cone is exactly luminal. Only then is one
bounded orientation-carrying coefficient shape inserted.

The action class is established prior art. We do **not** claim DHOST,
beyond-Horndeski gravity, or its degeneracy relations as Sigma inventions. The
only possible novelty is the particular fixed coefficient shape and its use as
a baryon-locked galaxy/cluster lensing mechanism. That novelty remains to be
checked term by term.

## Published action class

Use one physical metric, one dimensionless scalar \(\varphi\), and

\[
X=g^{\mu\nu}\nabla_\mu\varphi\nabla_\nu\varphi,
\qquad
\varphi_{\mu\nu}=\nabla_\mu\nabla_\nu\varphi.
\]

The quadratic DHOST action is

\[
S=\int d^4x\sqrt{-g}\left[
P(\varphi,X)+Q(\varphi,X)\Box\varphi+F(\varphi,X)R
+\sum_{I=1}^{5}A_I(\varphi,X)L_I
\right]+S_b[g,\psi_b],
\]

with

\[
L_1=\varphi_{\mu\nu}\varphi^{\mu\nu},
\qquad
L_2=(\Box\varphi)^2,
\]

\[
L_3=(\Box\varphi)\varphi^\mu\varphi_{\mu\nu}\varphi^\nu,
\]

\[
L_4=\varphi^\mu\varphi_{\mu\rho}
\varphi^{\rho\nu}\varphi_\nu,
\qquad
L_5=(\varphi^\mu\varphi_{\mu\nu}\varphi^\nu)^2.
\]

For the luminal Class-Ia branch, the tensor-speed condition and degeneracy
relations are

\[
\boxed{A_1=A_2=0,}
\]

\[
\boxed{
A_4={48F_X^2-8(F-XF_X)A_3-X^2A_3^2\over8F},
}
\]

\[
\boxed{
A_5={(4F_X+XA_3)A_3\over2F}.
}
\]

These are the relations derived for the `c_T=c` DHOST class by
[Langlois et al.](https://arxiv.org/abs/1711.07403). The broader quadratic
classification and its conformal/disformal equivalence structure are given by
[Ben Achour, Langlois and Noui](https://arxiv.org/abs/1602.08398). The reason
degeneracy, rather than merely higher-order equations, is decisive is developed
in the effective lapse-velocity analysis of
[Langlois et al.](https://arxiv.org/abs/1703.03797).

## Provisional fixed v5C row

Let

\[
F_0={c^4\over16\pi G},
\qquad
q_\Sigma={a_\Sigma\over c^2},
\qquad
\widehat X={X\over q_\Sigma^2}.
\]

The selected functions are

\[
\boxed{
F(\varphi)=F_0(1+2\beta_\Sigma\varphi),
\qquad F_X=0,
}
\]

\[
\boxed{
P(\varphi,X)=-F_0X-{F_0\over L_\Sigma^2}\varphi^2,
\qquad Q=0,
}
\]

and

\[
\mathcal A(\widehat X)
={\widehat X^2\over(1+\widehat X^2)^{3/2}},
\]

\[
\boxed{
A_3(\varphi,X)
={\lambda_\Sigma F_0\over q_\Sigma^4}
\mathcal A(\widehat X).
}
\]

The functions \(A_4\) and \(A_5\) are not independently chosen: they are the
exact degenerate values above. Since `F_X=0`,

\[
A_4=-A_3-{X^2A_3^2\over8F},
\qquad
A_5={XA_3^2\over2F}.
\]

The provisional universal constants are

\[
\boxed{
\{a_\Sigma,L_\Sigma,\beta_\Sigma,\lambda_\Sigma\}.
}
\]

No scalar amplitude, center, orientation, or boundary profile may be fitted per
object.

## Why this particular coefficient shape

The function \(\mathcal A\) is not chosen from a data fit. It satisfies four
theory requirements:

1. it is real, smooth and even on both timelike and spacelike `X` branches;
2. it begins as \(\widehat X^2\), so Minkowski/weak backgrounds are regular;
3. it falls as \(1/|\widehat X|\) at large magnitude; and
4. because the degeneracy relations contain \(X^2A_3^2\), that final power
   keeps `A4` and `A5` bounded rather than allowing them to grow as powers of
   the kinetic invariant.

For the deliberately broad scan `-10<=lambda<=10` and
`1e-12<=|X_hat|<=1e6`, all dependent coefficient shapes remain finite and
their largest normalized magnitude is `13.1549`. The maximum normalized
degeneracy residual is `2.10e-16`. The activation is exactly even, is
`9.999999999985e-13` at `|X_hat|=1e-6`, and
`9.999999999985e-7` at `|X_hat|=1e6`.

Those results prove only algebraic class membership and regular coefficient
shapes. They do not prove scalar hyperbolicity on a solution.

## First-principles interpretation

The scalar is sourced through the curvature derivative of
\(F(\varphi)R\). In the weak matter regime, curvature is tied to the baryonic
stress trace, so no halo density is supplied as an independent source.

The canonical `P` term gives the scalar a positive local kinetic term and a
universal range. The `A3`, `A4`, and `A5` operators depend on the scalar
Hessian. A scalar Hessian carries more information than a scalar amplitude: it
has principal directions and changes when multiple baryonic components
overlap. These operators can therefore generate anisotropic metric stress and
metric slip while preserving one physical matter/light metric.

This is the action-level version of the clue from the cluster tests: a broad
trace response is insufficient unless the propagation law retains the
directions of the baryonic field. Here that directional information comes from
the dynamically baryon-sourced scalar Hessian, not from an inserted cluster
orientation.

## Why the simpler alternatives were not selected

### Individual nonmetricity traces

Exact v5B is rejected by its full-rank static kinetic Hessian. The recent
teleparallel Hamiltonian literature also warns that nonlinear nonmetricity
degree-of-freedom counts are subtle; see
[D'Ambrosio, Heisenberg and Zentarra](https://arxiv.org/abs/2308.02250).
Our rejection does not depend on adopting a disputed global `f(Q)` mode count:
the explicit v5B necessary degeneracy identity fails.

### Pure derivative-screened `P(X)` scalar

For a static spacelike scalar background `X<0`, a `P(X)` scalar has

\[
c_\parallel^2={P_X+2XP_{XX}\over P_X}.
\]

Derivative screening requires `P_X` to increase as the static gradient
`-X` increases, which means `P_XX<0`. Therefore

\[
2XP_{XX}>0,
\qquad
c_\parallel^2>1.
\]

The executable representative scans from `1.00000002` to `2.99999998`.
Pure k-mouflage is therefore incompatible with this project's strict
no-superluminal-characteristic gate.

### Conformal scalar alone

The previously derived linear `F(varphi)R` response shifts the two weak metric
potentials oppositely and cancels from their Weyl average. It can create a
fifth force for matter but is not by itself a cluster-lensing mechanism. v5C
retains `F(varphi)R` as the baryonic source while using the degenerate Hessian
operators to test whether the one-metric Weyl response can become nonzero.

### Aether or generalized Proca

These carry direction naturally, but earlier project actions failed the
combined PPN/required-response gate. Published vector-lensing work also shows
that coherent vector fluctuations can behave as independent halo-like states,
contrary to the required unique baryon-forced profile. They remain controls,
not the selected lane.

### Retarded nonlocal curvature

It remains a conceptual backup, but a causal retarded kernel is not obtained
from an ordinary time-symmetric variational principle without an explicit
initial-state construction. It is deferred until a local degenerate lane has
been exhausted.

## What remains unknown

This selection does not establish that v5C works. Before any data access it
must still:

1. derive the complete metric and scalar equations for the fixed functions;
2. solve the FLRW branch and all scalar/tensor quadratic kinetic matrices;
3. prove `F>0`, hyperbolicity, and causal scalar characteristics throughout the
   relevant signed-`X` branch;
4. derive the static spherical and nonspherical weak equations;
5. show Solar screening and calculate PPN parameters;
6. prove that the retarded baryon-forced branch is unique and has no regular
   source-free lump; and
7. compare the fixed functions against published DHOST, beyond-Horndeski,
   disformal and Vainshtein models.

Failure of any mathematical or Solar gate retires the row before a galaxy or
cluster score.

## Reproduction

```powershell
python scripts/check_sigma_v5c_degeneracy_first_selection.py
python -m pytest tests/test_sigma_degenerate_action.py -q
python -m ruff check src/voidscreen/sigma_degenerate_action.py scripts/check_sigma_v5c_degeneracy_first_selection.py tests/test_sigma_degenerate_action.py
```
