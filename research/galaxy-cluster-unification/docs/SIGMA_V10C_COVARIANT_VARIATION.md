# Sigma v10C covariant first-order variation subgate

## Decision

V10C passes its covariant variation/order subgate and advances to the nonlinear
ADM constraint gate. The projected carrier momentum is correct on a tilted
aether background, the spatial constraint leaves exactly six carrier
components, the interaction has a first-derivative action form, all Euler
equations are at most second order, and the all-field diffeomorphism identity
establishes on-shell stress conservation.

This is not yet a degree-of-freedom, arbitrary-background hyperbolicity, or
Solar-System pass. No astronomical product or holdout was opened.

## Exact covariant definitions

Let

$$
q_{\mu\nu}=g_{\mu\nu}+A_\mu A_\nu,
\qquad A^\mu A_\mu=-1,
$$

and impose

$$
P_{\mu\nu}=P_{\nu\mu},\qquad A^\mu P_{\mu\nu}=0.
$$

The projected derivatives are

$$
\dot P_{\mu\nu}
=q_\mu{}^\alpha q_\nu{}^\beta
A^\rho\nabla_\rho P_{\alpha\beta},
$$

$$
D_\lambda P_{\mu\nu}
=q_\lambda{}^\rho q_\mu{}^\alpha q_\nu{}^\beta
\nabla_\rho P_{\alpha\beta}.
$$

The exact carrier derivative momentum is

$$
\boxed{
\Pi^{\rho|\mu\nu}
=A^\rho\dot P^{\mu\nu}-c_P^2D^\rho P^{\mu\nu}
}.
$$

A central finite-difference audit of the kinetic Lagrangian on a Lorentz-tilted
unit-aether background agrees with this expression to a relative error below
`1e-9`. All three projected indices are numerically orthogonal to the aether.

## Carrier equation

With

$$
V(P)={1\over L_P^2}
\left[{P:P\over2}+{(P:P)^2\over4}\right]
$$

and `H_mn=D_(m J_n)`, variation with respect to the symmetric carrier gives

$$
\boxed{
-\nabla_\rho\Pi^{\rho|\mu\nu}
-{1\over L_P^2}(1+P:P)P^{\mu\nu}
+\beta H^{\mu\nu}
+A^{(\mu}\zeta^{\nu)}=0
}.
$$

Variation of `zeta` gives `A^mu P_mn=0`. As a linear map on a symmetric
four-tensor this constraint has rank four, reducing ten components to the six
spatial components used in the selection calculation.

## First-order interaction and aether equation

On the spatial constraint,

$$
\int\sqrt{-g}\,\beta P^{\mu\nu}\nabla_\mu J_\nu
=-\int\sqrt{-g}\,\beta C^\nu J_\nu,
\qquad C^\nu=\nabla_\mu P^{\mu\nu},
$$

up to a boundary term. Holding the metric and `P` fixed, this contributes

$$
\boxed{
\mathcal E_A^\sigma\big|_{PJ}
=\beta\nabla_\rho(A^\rho C^\sigma)
-\beta C^\nu\nabla^\sigma A_\nu
}
$$

modulo spatial-constraint terms absorbed by `zeta`. The remaining dependence
of the carrier kinetic terms on `A` is algebraic through projectors and
`A^rho`; the magnetic term is first derivative. Therefore the complete aether
equation is at most second order.

The v10C addition has no explicit `phi`, so it does not add a scalar Euler
term. Metric connection variations integrate by parts once and likewise give
at most second derivatives. The stress tensor is unambiguously

$$
T^{(10C)}_{\mu\nu}
=-{2\over\sqrt{-g}}
{\delta(\sqrt{-g}\,\Delta\mathcal L_{10C})\over\delta g^{\mu\nu}}.
$$

Expanding every component of this Hilbert tensor remains useful for the PPN
and numerical implementations, but its variational definition and derivative
order are now fixed.

## Conservation identity

For Euler derivatives of `g`, `A`, `P`, `phi`, `lambda`, and the vector
multiplier `zeta`, diffeomorphism invariance gives

$$
\begin{aligned}
0={}&-2\nabla_\mu E_g{}^\mu{}_\nu
+E_{A\mu}\nabla_\nu A^\mu
+\nabla_\mu(E_{A\nu}A^\mu)\\
&+E_P^{\alpha\beta}\nabla_\nu P_{\alpha\beta}
-2\nabla_\alpha(E_P^{\alpha\beta}P_{\nu\beta})
+E_\phi\nabla_\nu\phi+E_\lambda\nabla_\nu\lambda\\
&+E_{\zeta\mu}\nabla_\nu\zeta^\mu
+\nabla_\mu(E_{\zeta\nu}\zeta^\mu).
\end{aligned}
$$

On the nonmetric field equations, the gravitational metric source is
covariantly conserved. Minimal coupling to the one physical metric separately
gives matter conservation. This closes the action/conservation requirement
without introducing separate dynamics and lensing metrics.

## Remaining kill gates

The next gate must compute the full nonlinear ADM velocity Hessian and primary
and secondary constraints with tilted `A` and nonzero `P`. Six spatial
components are not the same as six healthy physical degrees of freedom; the
constraint algebra must determine that.

After that, v10C still needs arbitrary-background characteristics, an expanded
weak metric and PPN solution, Solar/compact-source limits, FLRW stability, and
a convergent PDE solver. No astronomical fitting is authorized before those
gates.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10c_covariant_variation.py
python -m pytest -q tests/test_sigma_v10c_covariant_variation.py
```

Machine-readable evidence is in
`results/sigma_v10c_covariant_variation/report.json`.
