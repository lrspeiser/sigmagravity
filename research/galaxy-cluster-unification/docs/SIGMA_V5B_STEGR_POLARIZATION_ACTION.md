# Sigma v5B STEGR causal-polarization action

## Status

Sigma v5B is the theory successor selected by the v5A cosmological failure. It
uses the same already-screened causal polarization sector but removes the
nonanalytic Sigma-v2 base. No observational data have been used and no
constant has been fitted.

The action, background, tensor, scalar, static-uniqueness, high-field, and weak
equations are specified below. A full nonlinear Hamiltonian count, PPN
solution, and prior-art audit remain mandatory before any map fit.

## Covariant action

With a flat, torsion-free connection, define the STEGR scalar \(\mathbb Q\),

\[
\widetilde Q_a=Q^m{}_{am},
\qquad
\mathcal W_a=Q_a-4\widetilde Q_a,
\qquad
q_\Sigma={a_\Sigma\over c^2},
\]

\[
Y={\widetilde Q_a\widetilde Q^a\over4q_\Sigma^2},
\qquad
Z=Y^2,
\qquad
J(Z)={Z\over(1+Z)^2}.
\]

The bounded inverse metric for the polarization is

\[
\mathcal G_\sigma^{ab}=g^{ab}
-{\alpha_\Sigma\over1+\alpha_\Sigma}
{\mathcal W^a\mathcal W^b
\over\sqrt{(\mathcal W_c\mathcal W^c)^2+(4q_\Sigma)^4}}.
\]

The v5B action is

\[
\boxed{
S_{\Sigma5B}=-{c^4\over16\pi G}\int d^4x\sqrt{-g}
\left\{
\mathbb Q
+{\eta_\Sigma\over L_\Sigma^2}
\left[
L_\Sigma^2\mathcal G_\sigma^{ab}
\nabla_a\sigma\nabla_b\sigma
+\sigma^2-2\sigma J(Z)
\right]
\right\}
+S_b[g,\psi_b].
}
\]

It has four provisional universal constants,

\[
\{a_\Sigma,L_\Sigma,\alpha_\Sigma,\eta_\Sigma\},
\qquad
0\le\alpha_\Sigma\le10,
\qquad
\eta_\Sigma>0.
\]

There is one physical metric, no material dark component, no object label, and
no per-object boundary or gravity quantity.

## Exact field equations

Let \(f_{5B}\) denote the expression inside braces and

\[
\Pi^{a mn}_{5B}={\partial f_{5B}\over\partial Q_{a mn}}.
\]

The scalar, metric, and flat-connection equations are

\[
\boxed{
\sigma-L_\Sigma^2\nabla_a
\left(\mathcal G_\sigma^{ab}\nabla_b\sigma\right)=J(Z),
}
\]

\[
\boxed{
{1\over2}f_{5B}g^{mn}
+{\partial f_{5B}\over\partial g_{mn}}
-{1\over\sqrt{-g}}\nabla_a
\left(\sqrt{-g}\,\Pi_{5B}^{a mn}\right)
={8\pi G\over c^4}T_b^{mn},
}
\]

\[
\boxed{
\nabla_m\nabla_n
\left(\sqrt{-g}\,\Pi_{5B\,a}{}^{mn}\right)=0.
}
\]

The polarization part of \(\Pi_{5B}\) is exactly the chain expression derived
for v5A; only the retired \(\mathcal H_Y\) term is absent. Diffeomorphism
invariance and the connection equation give

\[
\nabla_mT_b^{mn}=0
\]

for minimally coupled on-shell matter.

## Background and tensor limit

On homogeneous FLRW in coincident gauge,

\[
\widetilde Q_a=0,
\qquad
Z=J=0.
\]

The universal retarded state \(\sigma=0\) is an exact background solution.
The action then reduces exactly to STEGR, hence to the GR Friedmann equations
up to the standard boundary identity.

For a transverse-traceless perturbation, both nonmetricity traces have zero
linear TT perturbation. With background \(\sigma=0\), the polarization sector
has no quadratic tensor mixing. The tensor quadratic action is therefore the
STEGR/GR one and

\[
\boxed{c_T=c}
\]

on this background. The independent polarization perturbation is massive,
has positive time and spatial kinetic eigenvalues in the frozen anisotropy
range, and its local cone lies inside the metric cone.

## Static weak equations

Use the notation from the v5A weak derivation:

\[
W={\Psi+\Phi\over2},
\quad
u_i={\partial_iW\over a_\Sigma},
\quad
y={|\nabla\Phi|^2\over a_\Sigma^2},
\quad
J={y^2\over(1+y^2)^2},
\]

\[
\Lambda_\Sigma={\eta_\Sigma c^4\over2L_\Sigma^2},
\qquad
B_i={L_\Sigma^2\over2a_\Sigma}
{\partial K^{jk}\over\partial u_i}\sigma_j\sigma_k.
\]

Removing the Sigma-v2 primitive gives

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
2\nabla\Psi-2\nabla\Phi
+\Lambda_\Sigma\left[
\mathbf B-{4\sigma J_y\over a_\Sigma^2}\nabla\Phi
\right]
\right\}=0,
}
\]

\[
\boxed{
\sigma-L_\Sigma^2\nabla_i
\left(K^{ij}\nabla_j\sigma\right)=J(y).
}
\]

When the polarization source vanishes, \(\sigma=0\) and the regular branch is
\(\Psi=\Phi\) with Newtonian/GR gravity. Galaxy and cluster effects must both
come from the same polarization action. Matter follows \(-\nabla\Psi\), while
light follows \(W=(\Psi+\Phi)/2\).

## Why v5B is more falsifiable

v5B cannot inherit a MOND fit and then use a second mechanism only for
clusters. If its four frozen constants eventually reproduce galaxy dynamics,
cluster lensing, and Solar limits, they do so through one polarization field.
If it fails galaxies, it is rejected even if it helps clusters. If it requires
different `L`, `alpha`, or `eta` by system, it is rejected.

## Remaining pre-fit gates

1. Perform the full Hamiltonian/constraint count beyond the FLRW quadratic
   screen.
2. Derive PPN parameters and solve the Solar transition-shell response with
   the universal retarded boundary state.
3. Audit cosmological scalar production and demonstrate that \(\sigma=0\) is
   stable under realistic perturbations.
4. Complete prior-art and invertible-field-redefinition comparisons.
5. Only then preregister a numerical solver and spent-data source/phase test.

No observational fitting is currently permitted.
