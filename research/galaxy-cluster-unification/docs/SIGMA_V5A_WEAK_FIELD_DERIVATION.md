# Sigma v5A complete static weak-field derivation

## Outcome

The covariant Sigma v5A action now has compact metric, connection, and scalar
Euler equations and a complete leading static variation for the two metric
potentials and polarization field. The anisotropic transport chain rule,
transition-source derivative, combined local polarization variation, and
Weyl-trace identity pass independent centered finite differences.

No observational data were accessed. This closes the formal weak-variation
item, but not the full nonlinear constraint, cosmological-background,
`c_T`, or PPN gates. No empirical fit is authorized.

## Exact covariant variational form

Write the complete action as

\[
S=-{c^4\over16\pi G}\int d^4x\sqrt{-g}\,f_5+S_b[g,\psi_b],
\]

where

\[
f_5=\mathbb Q+2q_\Sigma^2\mathcal H(Y)
+{\eta_\Sigma\over L_\Sigma^2}
\left[
L_\Sigma^2\mathcal G_\sigma^{ab}\sigma_a\sigma_b
+\sigma^2-2\sigma J(Z)
\right],
\]

\[
\sigma_a=\nabla_a\sigma,
\quad
Y={\widetilde Q_a\widetilde Q^a\over4q_\Sigma^2},
\quad
Z=Y^2,
\quad
J_Z={1-Z\over(1+Z)^3}.
\]

Define the complete nonmetricity conjugate

\[
\Pi^{a mn}={\partial f_5\over\partial Q_{a mn}}.
\]

It separates into the Sigma-v2 conjugate and polarization contribution,

\[
\Pi^{a mn}=P^{a mn}+{1\over2}\mathcal H_Y Z_V^{a mn}
+{\eta_\Sigma\over L_\Sigma^2}
\left[
L_\Sigma^2
{\partial\mathcal G_\sigma^{bc}\over\partial Q_{a mn}}
\sigma_b\sigma_c
-2\sigma J_Z{\partial Z\over\partial Q_{a mn}}
\right],
\]

where

\[
Z_V^{a mn}=\widetilde Q^m g^{an}+\widetilde Q^n g^{am},
\qquad
{\partial Z\over\partial Q_{a mn}}
={Y\over2q_\Sigma^2}Z_V^{a mn}.
\]

For the disformal derivative, let

\[
\mathcal W_d=Q_d-4\widetilde Q_d,
\quad
S=\mathcal W_d\mathcal W^d,
\quad
A={\alpha_\Sigma\over1+\alpha_\Sigma},
\quad
\Delta=S^2+(4q_\Sigma)^4,
\quad
\beta={A\over\sqrt\Delta}.
\]

Holding the metric fixed in the nonmetricity derivative,

\[
C_d{}^{a mn}
={\partial\mathcal W_d\over\partial Q_{a mn}}
=\delta_d^a g^{mn}-4g^{a(m}\delta_d^{n)},
\]

\[
{\partial S\over\partial Q_{a mn}}=2\mathcal W^dC_d{}^{a mn}.
\]

Therefore

\[
{\partial\mathcal G_\sigma^{bc}\over\partial Q_{a mn}}
=-\beta\left(
g^{bd}C_d{}^{a mn}\mathcal W^c
+\mathcal W^b g^{cd}C_d{}^{a mn}
\right)
+{A S\over\Delta^{3/2}}
{\partial S\over\partial Q_{a mn}}
\mathcal W^b\mathcal W^c.
\]

These expressions include every derivative through the transition source and
orientation-dependent kinetic tensor.

The exact scalar equation is

\[
\boxed{
\sigma-L_\Sigma^2\nabla_a
\left(\mathcal G_\sigma^{ab}\nabla_b\sigma\right)=J(Z).
}
\]

The metric equation is compactly

\[
\boxed{
{1\over2}f_5 g^{mn}
+{\partial f_5\over\partial g_{mn}}
-{1\over\sqrt{-g}}\nabla_a
\left(\sqrt{-g}\,\Pi^{a mn}\right)
={8\pi G\over c^4}T_b^{mn}.
}
\]

The algebraic metric derivative is taken at fixed \(Q_{a mn}\) and fixed
\(\sigma_a\); it includes index raising in \(Y\), \(\mathcal W^a\), and
\(\mathcal G_\sigma^{ab}\). This partial-derivative form is exact and avoids
hiding terms inside an “effective density.”

Variation of the flat, torsion-free connection gives

\[
\boxed{
\nabla_m\nabla_n
\left(\sqrt{-g}\,\Pi_a{}^{mn}\right)=0.
}
\]

The connection equation and metric equation supply the diffeomorphism Noether
identity. Since baryons couple minimally to the same metric and carry no
independent connection charge,

\[
\nabla_mT_b^{mn}=0
\]

on shell.

## Complete static weak action

Use

\[
ds^2=-(1+2\Psi/c^2)c^2dt^2
+(1-2\Phi/c^2)d\mathbf x^2,
\]

and define

\[
p_i=\partial_i\Psi,
\qquad
f_i=\partial_i\Phi,
\qquad
W={\Psi+\Phi\over2},
\qquad
u_i={\partial_iW\over a_\Sigma},
\qquad
y={f_if_i\over a_\Sigma^2}.
\]

The fixed source and its derivative are

\[
J(y)={y^2\over(1+y^2)^2},
\qquad
J_y={2y(1-y^2)\over(1+y^2)^3}.
\]

The spatial polarization transport tensor becomes

\[
\boxed{
K^{ij}(u)=\delta^{ij}
-{\alpha_\Sigma\over1+\alpha_\Sigma}
{u^iu^j\over\sqrt{1+(u_ku^k)^2}}.
}
\]

Let

\[
\Lambda_\Sigma={\eta_\Sigma c^4\over2L_\Sigma^2}.
\]

The complete leading static action is

\[
S_{\rm wf}=\int dt\,d^3x\left\{
-{1\over8\pi G}\mathcal L_{\rm wf}-\rho_b\Psi
\right\},
\]

\[
\mathcal L_{\rm wf}
=2p_if_i-f_if_i+a_\Sigma^2\mathcal H(y)
+\Lambda_\Sigma\left[
L_\Sigma^2K^{ij}\sigma_i\sigma_j
+\sigma^2-2\sigma J(y)
\right].
\]

This is one variational system. \(\sigma\) is held independent during metric
variation and obeys its own Euler equation; no memory pullback is omitted.

## Analytic anisotropic chain rule

Define

\[
s=\sqrt{1+(u_ku^k)^2},
\qquad
d=u_i\sigma_i,
\qquad
c_\alpha={\alpha_\Sigma\over1+\alpha_\Sigma}.
\]

Then

\[
C_i
={\partial K^{jk}\over\partial u_i}\sigma_j\sigma_k
=-2c_\alpha\left[
{\sigma_i d\over s}
-{u_i(u_ku^k)d^2\over s^3}
\right],
\]

and define

\[
B_i={L_\Sigma^2\over2a_\Sigma}C_i.
\]

The finite-difference implementation verifies this derivative rather than
treating the anisotropic tensor as fixed while varying the metric.

## Three coupled weak equations

Independent variation gives

\[
\boxed{
\nabla\cdot\left[
\nabla\Phi+{\Lambda_\Sigma\over2}\mathbf B
\right]=4\pi G\rho_b,
}
\]

\[
\boxed{
\nabla\cdot\left\{
2\nabla\Psi+2(\mathcal H_y-1)\nabla\Phi
+\Lambda_\Sigma\left[
\mathbf B-{4\sigma J_y\over a_\Sigma^2}\nabla\Phi
\right]
\right\}=0,
}
\]

and

\[
\boxed{
\sigma-L_\Sigma^2\nabla_i
\left(K^{ij}\nabla_j\sigma\right)=J(y).
}
\]

Massive tracers respond to

\[
\mathbf a_m=-\nabla\Psi,
\]

while photons respond to the same metric through

\[
\boxed{W={\Psi+\Phi\over2}.}
\]

There is no photon-only multiplier. The polarization changes the two metric
potentials through its action, and lensing follows from their derived average.

## Executable variation checks

| Check | Relative or absolute error | Gate |
|---|---:|---:|
| Anisotropic transport chain | `1.09e-8` | at most `1e-7` |
| Transition-source derivative | `1.53e-9` | at most `1e-7` |
| Complete local polarization variation | `4.24e-10` | at most `1e-7` |
| Weyl-trace weak identity | `4.05e-13` absolute | at most `1e-12` |

All gates pass. Machine-readable results are in
`results/sigma_v5a_weak_field_derivation/report.json`.

## Remaining theory gates

This derivation closes the static scalar variation, not the whole goal. The
next calculations are:

1. expand the tensor/vector/scalar perturbations of the complete action around
   Minkowski and FLRW backgrounds and count physical modes;
2. prove the base nonmetricity sector plus polarization has no ghost or strong
   coupling;
3. derive the transverse-tensor quadratic action on FLRW and verify
   `c_T=c` exactly;
4. define the globally real cosmological continuation of the Sigma-v2
   primitive;
5. solve the PPN/high-field expansion with the universal cosmological
   polarization boundary state; and
6. complete a prior-art/field-redefinition audit.

Only after these pass may a numerical weak-field solver be preregistered.

## Reproduction

```powershell
python scripts/check_sigma_v5a_weak_field_derivation.py
python -m pytest tests/test_sigma_causal_polarization.py -q
python -m ruff check src/voidscreen/sigma_causal_polarization.py scripts/check_sigma_v5a_weak_field_derivation.py tests/test_sigma_causal_polarization.py
```
