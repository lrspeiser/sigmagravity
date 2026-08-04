# Sigma v17H susceptibility-screened pressure action

## Selection result

There is a parameter-economical way to remove v17G's Solar obstruction without
changing the pressure source or inventing a Solar/cluster label. The coupling
can follow the susceptibility of a dynamical local gravitational frame. The
fixed response is nearly one in a diffuse low-acceleration region and decreases
as \(a_\Sigma/g\) in a high-acceleration region.

This candidate passes only a **theory-selection** gate. It has not passed the
full vector--metric constraint calculation, PPN preferred-frame limits, or any
cluster data test. It authorizes exact variation, not a holdout.

## The covariant fields

Let \(U^\mu\) be a dynamical unit timelike aether and define its proper
acceleration,

\[
\mathcal A_\mu=U^\nu\nabla_\nu U_\mu,
\qquad
Z={c^4\mathcal A_\mu\mathcal A^\mu\over a_\Sigma^2}.
\]

For a static frame, \(c^2|\mathcal A|\) reduces to the local gravitational
acceleration. This is a local covariant field, not an analyst's classification
of an object.

Add the fixed Born--Infeld-shaped aether density

\[
F_A(Z)=\sqrt{1+Z}-1
\]

and define its normalized susceptibility

\[
\boxed{
\chi(Z)=\left(2{dF_A\over dZ}\right)^{1/2}
=(1+Z)^{-1/4}.
}
\]

The proposed action is

\[
\begin{split}
S={}&\int d^4x\sqrt{-g}\left\{
{M_{\rm Pl}^2R\over2}
-{M_{\rm Pl}^2\over2}
[(\nabla X)^2+L_\Sigma^{-2}X^2]
-{M_{\rm Pl}^2c_U\over4}F^{(U)}_{\mu\nu}F_{(U)}^{\mu\nu}
\\
&\hspace{2.5cm}
-M_{\rm Pl}^2{a_\Sigma^2\over c^4}F_A(Z)
+\lambda(U_\mu U^\mu+1)\right\}
+S_m[\widetilde g,\psi_m],
\end{split}
\]

with one physical metric for matter and light,

\[
\boxed{
\widetilde g_{\mu\nu}
=e^{2\alpha\chi(Z)X}
\left[g_{\mu\nu}+2\alpha\chi(Z)XU_\mu U_\nu\right].
}
\]

This is a derivative-dependent disformal metric and must pass an exact
degeneracy/constraint calculation. Writing a covariant action does not by
itself prove that calculation.

## Why it screens without a separate Solar rule

At \(X=0\), variation of the same matter metric gives

\[
J_X=\alpha\chi T+\alpha\chi E.
\]

Cold rest mass cancels when the conformal and disformal coefficients are equal.
For isotropic pressure,

\[
\boxed{J_X=3\alpha\chi(Z)p.}
\]

The weak metric potentials are

\[
\Psi=U_N,
\qquad
\Phi=U_N-c^2\alpha\chi X,
\qquad
W=U_N-{c^2\alpha\chi X\over2}.
\]

The source contains one factor of \(\chi\), and the metric response contains a
second. Where the source and response occupy the same acceleration regime,

\[
\boxed{
\mathcal R(Z)=\chi^2={1\over\sqrt{1+Z}}
={1\over\sqrt{1+(g/a_\Sigma)^2}}.
}
\]

Consequently:

- at \(g=0.1a_\Sigma\), 99.50% of the pressure response remains;
- at \(g=a_\Sigma\), 70.71% remains;
- at \(g=10a_\Sigma\), 9.95% remains; and
- at \(g=10^5a_\Sigma\), the response is \(10^{-5}\).

The last identity gives the project's required high-acceleration limit without
adding another scale or fitting the exponent.

## Solar calculation

The source must be screened throughout the Sun, not merely at its surface. The
executable control uses a uniform-density hydrostatic pressure shape
\(p(r)\propto1-r^2/R_\odot^2\), normalized to the deliberately conservative
v17G pressure compactness. It computes

\[
\langle\chi\rangle_p
={\int p(r)\chi[g(r)/a_\Sigma]dV\over\int p(r)dV}
=8.82\times10^{-7}.
\]

The metric factor is conservatively evaluated at 10 AU, where it is larger
than nearer the Sun. Even using the largest coupling required over the declared
cluster selection envelope gives

\[
|\gamma-1|_{\rm proxy}=3.00\times10^{-10},
\]

more than four orders below Cassini. The exact report value depends on the
largest envelope coupling and is intentionally more conservative than using
the unscreened-cluster coupling alone.

This is not a PPN proof. A realistic solar pressure profile, the complete
radio-path integral, scalar/aether stress, \(\beta\), \(\alpha_1\),
\(\alpha_2\), and Mercury remain required.

## Reduced health result

Eliminating no fields, the square-root aether density has acceleration-block
eigenvalues proportional to

\[
\lambda_\perp={1\over\sqrt{1+Z}},
\qquad
\lambda_\parallel={1\over(1+Z)^{3/2}}.
\]

Both are positive for every finite \(Z\). The frozen Maxwell-aether coefficient
\(c_U=1\) adds a positive floor, so the reduced block never loses rank in the
scan through \(Z=10^{30}\). This avoids the immediate superluminal
\(P(X)\)-screen shortcut that v5C already rejected.

It does not prove the full result. The derivative-dependent matter metric can
mix metric, aether, scalar, and matter velocities. Its full tilted and
time-dependent principal symbol must be calculated before this action can be
called healthy.

## How halo size would emerge

The conditional projected equation is

\[
\boxed{
(1-L_\Sigma^2\nabla_\perp^2)s_\Sigma
\propto\chi(Z)\mathcal J_T,
\qquad
\Delta W\propto\alpha\chi(Z)s_\Sigma.
}
\]

Three measured baryonic structures then determine the apparent halo:

1. \(\mathcal J_T(\mathbf x)\) supplies the pressure or random-stress extent;
2. \(Z(\mathbf x)\) suppresses the response where the local dynamical frame is
   already strongly accelerated; and
3. a single common \(L_\Sigma\), only if v17F requires it, broadens the source.

There is no fitted halo radius. Different \(R_{50}\) and \(R_{80}\) values must
come from the baryonic stress and acceleration maps. Before an empirical test,
\(Z\) must be computed from a target-blind three-dimensional baryonic/aether
solution; a dark-halo or lensing target may not define it.

## Prior-art boundary

The broad ingredients are published: Bekenstein's TeVeS physical metric,
generalized Einstein-aether MOND actions, the vector-tensor rewriting of TeVeS,
and derivative-dependent conformal/disformal transformations. Relevant primary
sources are [Bekenstein](https://arxiv.org/abs/astro-ph/0403694),
[Zlosnik--Ferreira--Starkman](https://arxiv.org/abs/astro-ph/0607411), their
[vector-tensor TeVeS analysis](https://arxiv.org/abs/gr-qc/0606039), and
[Zumalacárregui--García-Bellido](https://arxiv.org/abs/1308.4685).

The square-root susceptibility is therefore not claimed as a new field-theory
class. A potentially distinctive result would be the full combination:
pressure-sourced halo geometry, the same susceptibility in source and metric,
one transferable scale, raw lens topology, and Solar/PPN consistency with no
per-object gravity parameter.
