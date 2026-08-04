# Sigma v17I localized covariant variation

> **Subsequent gate:** the
> [v17J flat-kinetic audit](SIGMA_V17J_FLAT_KINETIC_FALSIFICATION.md) finds that
> the classically correct localized equations have an unhealthy flat-vacuum
> quadratic sector. V17I's variation bookkeeping remains valid, but the frozen
> v17H/v17I action is retired before data.

## Result

The v17H susceptibility metric has a classically equivalent first-order
localized action. Its reciprocal pressure source, scalar and aether equations,
Einstein-frame matter stress, constraint stress, and off-shell diffeomorphism
identity are explicit. Four independent directional variations agree with
five-point finite differences within the frozen tolerance, and the
perfect-fluid source reduces exactly to \(3p\).

The first numerical run used a relative error divided only by the projected
directional derivative. A nearly orthogonal acceleration perturbation had an
absolute discrepancy of \(1.22\times10^{-11}\), but division by its
\(1.16\times10^{-8}\) projected derivative reported a misleading \(0.001\)
error. Version 1.0.1 normalizes by the full chain-rule derivative norm. The
frozen \(2\times10^{-6}\) tolerance, equations, and physical choices are
unchanged.

This closes the bookkeeping problem, not the theory-health problem. The
localized multiplier system must still pass a full Dirac constraint and
principal-symbol calculation. No observational or holdout data were opened.

## Localized action

V17H makes the susceptibility a function of aether proper acceleration, which
contains a derivative of \(U_\mu\). Introduce an independent covector
\(A_\mu\) and multiplier \(B^\mu\):

\[
\begin{split}
S_{\rm loc}=\int d^4x\sqrt{-g}\{&{M_{\rm Pl}^2R\over2}
-{M_{\rm Pl}^2\over2}[(\nabla X)^2+L_\Sigma^{-2}X^2]
-{M_{\rm Pl}^2c_U\over4}F^{(U)2}\\
&-M_{\rm Pl}^2{a_\Sigma^2\over c^4}F_A(Z)
+\lambda(U^2+1)
+B^\mu[A_\mu-U^\nu\nabla_\nu U_\mu]\}
+S_m[\widetilde g,\psi].
\end{split}
\]

Here

\[
Z={c^4A_\mu A^\mu\over a_\Sigma^2},\quad
F_A=\sqrt{1+Z}-1,\quad
\chi=(1+Z)^{-1/4},\quad q=\alpha\chi X,
\]

and

\[
\widetilde g_{\mu\nu}=e^{2q}(g_{\mu\nu}+2qU_\mu U_\nu).
\]

The \(B^\mu\) equation gives

\[
A_\mu=U^\nu\nabla_\nu U_\mu,
\]

so eliminating the auxiliaries returns v17H. Neither auxiliary is an
observational parameter or an independently selectable halo state.

## Reciprocal matter variation

Define

\[
H^{\mu\nu}={\sqrt{-\widetilde g}\over\sqrt{-g}}
\widetilde T^{\mu\nu},
\]

\[
D_{\mu\nu}={\partial\widetilde g_{\mu\nu}\over\partial q}
=2e^{2q}[g_{\mu\nu}+(2q+1)U_\mu U_\nu],
\]

and

\[
\boxed{\mathcal J={1\over2}H^{\mu\nu}D_{\mu\nu}.}
\]

At \(q=0\), \(\mathcal J=T+E\). A comoving perfect fluid has
\(T=-\varepsilon+3p\) and \(E=\varepsilon\), hence

\[
\boxed{\mathcal J=3p.}
\]

Cold dust cancels exactly. This pressure source comes from varying the same
physical metric that sets matter and photon trajectories.

The remaining matter variations are

\[
Q_X=\alpha\chi\mathcal J,
\]

\[
Q_A^\mu=2\alpha X\chi_Z\mathcal J
{c^4A^\mu\over a_\Sigma^2},
\]

\[
Q_U^\mu=2q e^{2q}H^{\mu\nu}U_\nu,
\]

and

\[
T_{(m)}^{\mu\nu}=e^{2q}H^{\mu\nu}
-2\alpha X\chi_Z\mathcal J
{c^4A^\mu A^\nu\over a_\Sigma^2}.
\]

The executable audit treats \(H^{\mu\nu}\) as an arbitrary symmetric response
tensor and independently perturbs \(X\), \(A_\mu\), \(U_\mu\), and
\(g_{\mu\nu}\). This tests the metric chain rule without assuming an equation
of state.

## Euler--Lagrange system

The scalar equation is

\[
M_{\rm Pl}^2(\Box-L_\Sigma^{-2})X+\alpha\chi\mathcal J=0.
\]

The independent acceleration equation is algebraic,

\[
B^\mu-2M_{\rm Pl}^2F_{A,Z}A^\mu+Q_A^\mu=0.
\]

Together with the acceleration constraint, the aether equation is

\[
M_{\rm Pl}^2c_U\nabla_\nu F_{(U)}^{\nu\mu}
+\nabla_\nu(B^\mu U^\nu)-B^\rho\nabla^\mu U_\rho
+2\lambda U^\mu+Q_U^\mu=0.
\]

The metric equation is

\[
M_{\rm Pl}^2G^{\mu\nu}=T_X^{\mu\nu}+T_F^{\mu\nu}
+T_A^{\mu\nu}+T_\lambda^{\mu\nu}+T_B^{\mu\nu}
+T_{(m)}^{\mu\nu}.
\]

In addition to the standard scalar and Maxwell stresses,

\[
T_A^{\mu\nu}=2M_{\rm Pl}^2F_{A,Z}A^\mu A^\nu
-M_{\rm Pl}^2{a_\Sigma^2\over c^4}F_Ag^{\mu\nu},
\]

\[
T_\lambda^{\mu\nu}=-2\lambda U^\mu U^\nu
\quad\hbox{on }U^2=-1,
\]

and

\[
T_B^{\mu\nu}=g^{\mu\nu}\mathcal L_B
+2B^\rho U^{(\mu}\nabla^{\nu)}U_\rho
-\nabla_\rho(B^\rho U^\mu U^\nu).
\]

Every localized gravitational Euler equation is at most second differential
order. That fact alone does not determine how many constrained modes propagate.

## Conservation identity

Let \(E_g,E_X,E_U,E_A,E_B,E_\lambda,E_\psi\) denote the Euler derivatives.
Diffeomorphism invariance gives

\[
\begin{split}
0={}&-\nabla_\mu E_g{}^\mu{}_\nu
+E_X\nabla_\nu X+E_\lambda\nabla_\nu\lambda
+E_U^\mu\nabla_\nu U_\mu-\nabla_\mu(E_U^\mu U_\nu)\\
&+E_A^\mu\nabla_\nu A_\mu-\nabla_\mu(E_A^\mu A_\nu)
+E_{B\mu}\nabla_\nu B^\mu+\nabla_\rho(E_{B\nu}B^\rho)
+E_\psi\mathcal L_{\partial_\nu}\psi.
\end{split}
\]

Thus the metric Euler tensor is conserved when every other equation holds.
Matter alone is covariantly conserved with \(\widetilde g\); in Einstein
variables it exchanges stress with \(X\) and \(U\). This is reciprocal
bookkeeping, not a one-way modification of gravity.

## Remaining decisive gate

The next calculation must construct the full tilted, time-dependent kinetic
matrix for \((g_{ij},U_\mu,A_\mu,B^\mu,X)\), identify all primary and secondary
constraints, and prove the reduced Hamiltonian is bounded. It must then derive
every characteristic cone and the preferred-frame PPN parameters.

Only after those gates and a target-blind construction of \(Z(\mathbf x)\) may
this action touch a cluster lensing target.
