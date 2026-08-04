# Sigma v12A canonical primary and conditional Dirac chain

## Decision

V12A passes the canonical-primary subgate. In the published reduced AeST ADM
variables, the metric momentum is exactly the GR momentum. The AeST sector
therefore does not shift the Class-Ia DHOST primary constraint at all. The
primary commutes with the AeST auxiliary primaries, so preserving it in time
necessarily produces one secondary constraint.

This still is not a complete Dirac pass. The explicit secondary density and
its effective Poisson bracket after eliminating the AeST auxiliary pairs have
not been calculated for the fixed v12A coefficient function. That bracket is
now the precise next kill gate.

## Canonical primary

Use `B_mu=nabla_mu phi` for the auxiliary scalar covector, reserving `A_mu` for
the AeST aether. Let `V_*` be the normal velocity of `B_*`, and collect the six
components of `K_ij` into `K_A`. The quadratic DHOST kinetic density is

$$
{\mathcal L_{\rm D,kin}\over N\sqrt h}
=\mathcal A V_*^2+2\mathcal B_A V_*K_A
+\mathcal K_{AB}K_AK_B+2\mathcal C_AK_A+2\mathcal C_0V_*.
$$

Its momenta are

$$
p_*=2\sqrt h(\mathcal A V_*+\mathcal B_AK_A+\mathcal C_0),
$$

$$
\pi_A=\sqrt h(\mathcal K_{AB}K_B+\mathcal B_AV_*+\mathcal C_A).
$$

Define

$$
n^A=(\mathcal K^{-1})^{AB}\mathcal B_B.
$$

Class-Ia degeneracy is

$$
\mathcal A=\mathcal B_A(\mathcal K^{-1})^{AB}\mathcal B_B.
$$

Eliminating both velocities gives the explicit primary

$$
\boxed{
\Psi_\Sigma
=p_*-2n^A\pi_A+2\sqrt h(n^A\mathcal C_A-\mathcal C_0)
\approx0.
}
$$

The executable audit evaluates this identity on 2,000 arbitrary velocity,
affine-coefficient, volume, mixing, and DeWitt-signature metric blocks.

## Why the AeST base does not alter it

The nonlinear AeST Hamiltonian paper first solves the unit-timelike condition
as `chi=sqrt(1+A_i A^i)`. Its exact metric momentum is

$$
\Pi^{ij}_{\rm AeST}={\sqrt h\over16\pi\widetilde G}
(K^{ij}-Kh^{ij}),
$$

with no aether- or scalar-dependent affine shift. The aether and scalar
momenta do mix with one another, but neither contains `V_*`. Consequently:

- the total `mathcal K` and `mathcal B` are the standard Class-Ia metric/scalar
  blocks already selected in v12A;
- the AeST contribution to `mathcal C_A` is zero in these variables;
- `Psi_Sigma` is independent of the AeST auxiliary fields `mu,nu` and their
  momenta.

The last point gives the strong brackets

$$
\{\Psi_\Sigma,\Pi_{(\mu)}\}=0,
\qquad
\{\Psi_\Sigma,\Pi_{(\nu)}\}=0.
$$

The standard auxiliary-gradient constraints tying `B_i` to `D_i phi` are the
same second-class pairs used in the published DHOST derivation and may be
eliminated first.

## Secondary existence

The total primary Hamiltonian contains multipliers for
`Pi_(mu),Pi_(nu),Psi_Sigma`. Since the brackets above vanish and
`{Psi_Sigma,Psi_Sigma}=0`, preserving `Psi_Sigma` cannot determine any of those
multipliers. It produces

$$
\boxed{
\Omega_\Sigma=\{\Psi_\Sigma,H_0\}\approx0.
}
$$

As in the published DHOST Hamiltonian,

$$
\Omega_\Sigma=p_\phi+\Omega_{\rm rest}.
$$

The first-derivative AeST terms modify `Omega_rest`, but they do not change the
unit coefficient of `p_phi`. Thus the secondary removes the redundant scalar
momentum rather than introducing a new initial datum.

## Exact remaining regularity condition

Let the AeST auxiliary primaries be `p_A=(Pi_mu,Pi_nu)` and their secondaries
be `S_A=(S_mu,S_nu)`. Define

$$
C_{AB}=\{p_A,S_B\},\quad
D_A=\{p_A,\Omega_\Sigma\},\quad
E_B=\{\Psi_\Sigma,S_B\},\quad
\Delta=\{\Psi_\Sigma,\Omega_\Sigma\}.
$$

The primary-to-secondary bracket block is

$$
M=
\begin{pmatrix}
C&D\\
E&\Delta
\end{pmatrix}.
$$

Assuming the already-required AeST block `C` is invertible, its determinant is

$$
\det M=\det C\,\Delta_{\rm eff},
$$

where

$$
\boxed{
\Delta_{\rm eff}=\Delta-EC^{-1}D.
}
$$

For any secondary-secondary bracket block, the complete antisymmetric Dirac
matrix obeys

$$
\boxed{
\det\mathcal D=(\det C)^2\Delta_{\rm eff}^2.
}
$$

Two thousand random regular blocks verify this identity numerically. A test
setting `Delta_eff=0` makes the full matrix singular exactly as required.

If the actual differential operator `Delta_eff` is invertible, the reduced
count changes from the AeST result

$$
{2(12)-2(4)-4\over2}=6
$$

to

$$
{2(13)-2(4)-6\over2}=6.
$$

The new auxiliary coordinate `B_*` is then removed by the new second-class
pair and no Ostrogradsky mode is added. This count is conditional, not yet a
v12A result.

## What remains

The next calculation must derive `Omega_Sigma` for the fixed

$$
A_3(X)={\lambda_DF_0\over q_\Sigma^4}\mathcal A(X/q_\Sigma^2,x_0)
$$

and evaluate `Delta_eff` as a spatial differential operator. It must remain
regular on all admitted timelike scalar-gradient and aether-tilt backgrounds,
including the flat AeST clock where the DHOST activation vanishes. A zero,
sign-changing principal coefficient, lost boundary condition, or
field-dependent rank surface retires the exact v12A row before data.

The primary formula and pure-DHOST secondary structure follow the
[Hamiltonian analysis of higher-derivative scalar-tensor theories](https://arxiv.org/abs/1512.06820).
The reduced AeST momenta and its four-first-/four-second-class result follow
the published [AeST Hamiltonian formulation](https://arxiv.org/abs/2307.15126).
Their combination and the `Delta_eff` gate above are the project-specific
audit. No novelty claim is made.

No observational product or holdout was opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_primary_dirac.py
python -m pytest -q tests/test_sigma_v12a_primary_dirac.py
```

Machine-readable evidence is in
`results/sigma_v12a_primary_dirac/report.json`.
