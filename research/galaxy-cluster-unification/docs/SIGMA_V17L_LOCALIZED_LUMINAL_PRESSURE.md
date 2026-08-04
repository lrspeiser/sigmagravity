# Sigma v17L localized luminal-pressure equations

## Result

The v17K physical metric has an exact first-order localization with the selected
luminal Einstein--aether carrier. The pressure source, reciprocal matter stress,
auxiliary equations, carrier equation, stress tensors, and off-shell
diffeomorphism identity are explicit. The hash-locked matter-chain audit again
matches finite differences, and every localized Euler equation is at most
second differential order.

This is bookkeeping, not yet the active-background health test. The next gate
must determine whether ordinary pressure changes the positive vacuum kinetic
matrix enough to create a zero or negative eigenvalue.

## Localized action

Introduce an independent covector (A_\mu) and multiplier (B^\mu):

\[
\begin{split}
S_{\rm loc}=\int d^4x\sqrt{-g}\{&{M_{\rm Pl}^2R\over2}
-{M_{\rm Pl}^2\over2}[(\nabla X)^2+L_\Sigma^{-2}X^2]\\
&-{M_{\rm Pl}^2\over2}K^{\mu\nu}{}_{\alpha\beta}
\nabla_\mu U^\alpha\nabla_\nu U^\beta
+\lambda(U^2+1)\\
&+B^\mu[A_\mu-U^\nu\nabla_\nu U_\mu]\}
+S_m[\widetilde g,\psi].
\end{split}
\]

Here

\[
K^{\mu\nu}{}_{\alpha\beta}
=c_1g^{\mu\nu}g_{\alpha\beta}
+c_2\delta^\mu_\alpha\delta^\nu_\beta
+c_3\delta^\mu_\beta\delta^\nu_\alpha
-c_4U^\mu U^\nu g_{\alpha\beta},
\]

with the frozen v17K coefficients

\[
c_1=\varepsilon,
\quad c_3=-\varepsilon,
\quad c_4=0,
\quad c_2={\varepsilon\over1-2\varepsilon},
\quad \varepsilon=10^{-7}.
\]

The physical metric is unchanged:

\[
\widetilde g_{\mu\nu}=e^{2q}
(g_{\mu\nu}+2qU_\mu U_\nu),
\quad
q=\alpha\chi(Z)X,
\quad
\chi=(1+Z)^{-1/4},
\]

\[
Z={c^4A_\mu A^\mu\over a_\Sigma^2}.
\]

The (B^\mu) equation restores

\[
A_\mu=U^\nu\nabla_\nu U_\mu.
\]

Unlike v17I, there is no Born--Infeld acceleration density. Eliminating the
auxiliaries therefore returns v17K, not the retired v17H action.

## Exact reciprocal matter terms

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
\mathcal J={1\over2}H^{\mu\nu}D_{\mu\nu}.
\]

At (q=0), a comoving perfect fluid gives

\[
\boxed{\mathcal J=T+E=3p,}
\]

while cold dust gives zero. The remaining variations are

\[
Q_X=\alpha\chi\mathcal J,
\]

\[
Q_A^\mu=2\alpha X\chi_Z\mathcal J
{c^4A^\mu\over a_\Sigma^2},
\]

\[
Q_U^\mu=2qe^{2q}H^{\mu\nu}U_\nu,
\]

and

\[
T_{(m)}^{\mu\nu}=e^{2q}H^{\mu\nu}
-2\alpha X\chi_Z\mathcal J
{c^4A^\mu A^\nu\over a_\Sigma^2}.
\]

The executable audit reuses the already verified v17I matter-variation kernel
by exact file hash because v17K intentionally retains the same physical metric.
It does not reuse the retired carrier dynamics.

## Euler equations

The scalar equation is

\[
M_{\rm Pl}^2(\Box-L_\Sigma^{-2})X+\alpha\chi\mathcal J=0.
\]

The independent acceleration equation is now simply

\[
\boxed{B^\mu+Q_A^\mu=0.}
\]

This is the central difference from v17I. There is no
(-2M_{\rm Pl}^2F_{A,Z}A^\mu) term. Consequently (B^\mu=0)
continuously in vacuum, at (X=0), or for cold dust.

Define the standard aether current

\[
J_{(U)}{}^\mu{}_\alpha
=K^{\mu\nu}{}_{\alpha\beta}\nabla_\nu U^\beta.
\]

The aether equation is

\[
M_{\rm Pl}^2\nabla_\nu J_{(U)}{}^\nu{}_\mu
+\nabla_\nu(B_\mu U^\nu)-B^\rho\nabla_\mu U_\rho
+2\lambda U_\mu+Q_{U\mu}=0.
\]

The metric equation is

\[
M_{\rm Pl}^2G^{\mu\nu}
=T_X^{\mu\nu}+T_U^{\mu\nu}+T_\lambda^{\mu\nu}
+T_B^{\mu\nu}+T_{(m)}^{\mu\nu}.
\]

The standard aether stress can be written compactly as

\[
\begin{split}
T^{(U)}_{\mu\nu}=M_{\rm Pl}^2\{&\nabla_\rho[
J_{(\mu}{}^\rho U_{\nu)}-J^\rho{}_{(\mu}U_{\nu)}
-J_{(\mu\nu)}U^\rho]\\
&+c_1[(\nabla_\rho U_\mu)(\nabla^\rho U_\nu)
-(\nabla_\mu U_\rho)(\nabla_\nu U^\rho)]\\
&+[U_\alpha\nabla_\rho J^{\rho\alpha}]U_\mu U_\nu
-\tfrac12g_{\mu\nu}\mathcal L_U\},
\end{split}
\]

where the selected (c_4) is zero. This current, equation, and stress are
published Einstein--aether results, not new Sigma formulas. See
[Eling--Jacobson--Mattingly](https://arxiv.org/abs/gr-qc/0410001).

The multiplier stress remains

\[
T_B^{\mu\nu}=g^{\mu\nu}\mathcal L_B
+2B^\rho U^{(\mu}\nabla^{\nu)}U_\rho
-\nabla_\rho(B^\rho U^\mu U^\nu).
\]

## Conservation and differential order

The manifestly covariant localized action gives the same off-shell identity as
v17I:

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

Thus the metric Euler tensor is conserved when all other equations hold, and
matter is conserved with the one physical metric. Every localized gravitational
Euler equation is at most second order. Neither statement determines the sign
or rank of the reduced kinetic matrix.

## Next decisive calculation

In vacuum, (Q_A=B=0), so v17K's exactly luminal carrier result is recovered.
In pressure-supported matter with (X\ne0), however,

\[
B^\mu=-2\alpha X\chi_Z\mathcal J
{c^4A^\mu\over a_\Sigma^2}.
\]

Substituting (A_\mu=U^\nu\nabla_\nu U_\mu) changes the carrier's local kinetic
matrix. Because (\varepsilon=10^{-7}) is small, even a weak matter-induced
correction could be important. The next gate must compute that correction with
an explicit ordinary-matter action, reduce all constraints, and locate any
zero-eigenvalue surface before using cluster data.
