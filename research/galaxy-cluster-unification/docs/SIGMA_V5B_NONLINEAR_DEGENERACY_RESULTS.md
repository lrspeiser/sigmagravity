# Sigma v5B nonlinear degeneracy result

## Decision

The exact Sigma v5B action is **retired before any observational fit**. Its
`sigma=0` FLRW branch and free polarization mode are healthy at quadratic
order, but the nonlinear action fails a necessary kinetic-degeneracy condition
on the static polarized backgrounds it was designed to describe.

Both proposed new ingredients independently cause the failure:

1. the band-pass source built from the second nonmetricity trace; and
2. the orientation-dependent scalar kinetic metric built from the Weyl trace.

Each gives the lapse/flat-connection combination a velocity on a generic
polarized background. The ordinary STEGR lapse-null direction is lost, the
reduced kinetic Hessian becomes full rank, and its extra eigenvalue is not
positive throughout the transition. No galaxy, cluster, Solar, or other
observational array was opened.

## Why the linear FLRW screen passed

On the v5B FLRW background,

\[
\widetilde Q_a=0,
\qquad
\sigma=0.
\]

For a metric perturbation of amplitude \(\lambda\),

\[
\widetilde Q_a=O(\lambda),
\qquad
Y={\widetilde Q_a\widetilde Q^a\over4q_\Sigma^2}=O(\lambda^2),
\]

\[
Z=Y^2=O(\lambda^4),
\qquad
J={Z\over(1+Z)^2}=O(\lambda^4).
\]

Consequently the retarded polarization response begins at
\(\sigma=O(\lambda^4)\), and its tree-level metric feedback begins at eighth
order. The executable logarithmic probes return orders `3.999992` and
`7.999985`. Linear cosmology and the TT sector are therefore exactly GR.

The independent free polarization mode is also locally healthy. With

\[
r_H={9\over4}\left({H\over q_\Sigma}\right)^2,
\qquad
f_H={\alpha_\Sigma\over1+\alpha_\Sigma}
{r_H\over\sqrt{1+r_H^2}},
\]

its quadratic time kinetic magnitude is \(1+f_H\), its spatial coefficient is
one, and

\[
c_\sigma^2={1\over1+f_H}\le1,
\qquad
m_{\rm eff}^2L_\Sigma^2={1\over1+f_H}>0.
\]

Across `0<=alpha<=10` and `1e-12<=H/q_sigma<=1e12`, the minimum value is
`0.523810`, with no ghost or gradient instability at this order.

This explains why the previous background screen did not see the nonlinear
problem. The offending coefficients vanish when `sigma=0` and enter at higher
perturbative order.

## Necessary nonlinear degeneracy screen

Use a local isotropic ADM reduction with zero shift and define

\[
x=\dot{\ln N},
\qquad
h=\dot{\ln a},
\qquad
s=\dot\sigma.
\]

At a locally static background in coincident gauge,

\[
\widetilde Q_0=2x,
\qquad
\mathcal W_0=6(h-x).
\]

Let the fixed spatial background quantities be

\[
v_i=\widetilde Q_i,
\qquad
w_i=\mathcal W_i,
\qquad
p_i=\partial_i\sigma,
\qquad
Y_0={v_iv_i\over4q_\Sigma^2}.
\]

After positive local rescalings set `q_sigma=L_sigma=eta_sigma=1`. Omitting
terms independent of \((x,h,s)\), the exact local reduced Lagrangian is

\[
\mathcal L_{\rm red}
=-6h^2-\mathcal G_\sigma^{ab}\sigma_a\sigma_b
+2\sigma J(Y),
\]

where

\[
Y=Y_0-x^2.
\]

Define

\[
A={\alpha\over1+\alpha},
\quad
D=\sqrt{(w_iw_i)^2+4^4},
\quad
b=w_ip_i,
\]

\[
c={Ab\over D},
\qquad
d={Ab^2(w_iw_i)\over D^3},
\qquad
e=-4\sigma J_Y(Y_0).
\]

The exact Hessian at \((x,h,s)=(0,0,0)\) is

\[
\boxed{
H=
\begin{pmatrix}
72d+e & -72d & 12c\\
-72d & -12+72d & -12c\\
12c & -12c & 2
\end{pmatrix}.
}
\]

Its determinant is

\[
\boxed{
\det H=-24\left(6c^2e-72c^2-6de+72d+e\right).
}
\]

STEGR plus a canonical scalar has `c=d=e=0`, rank two, and a null lapse row.
The negative scale-factor direction is then removed by the Hamiltonian
constraint. A viable extension must retain an identically degenerate Hessian,
although its null vector need not remain the bare lapse direction.

## The source fails by itself

Set `alpha=0`, so `c=d=0`. Then

\[
\det H=-24e=96\sigma J_Y(Y_0).
\]

The source derivative is

\[
J_Y={2Y(1-Y^2)\over(1+Y^2)^3}.
\]

For the physical positive retarded solution \(\sigma>0\):

- on the low-field side `0<Y<1`, the new lapse kinetic coefficient is
  negative;
- at `Y=1`, it crosses zero and is strongly coupled; and
- on the high-field side `Y>1`, it becomes positive.

It is therefore neither degenerate nor sign-definite across the band that is
supposed to generate the modification. The representative `Y=0.5` source-only
matrix has

\[
\operatorname{eig}(H)=(-12,-0.3072,2),
\qquad
\det H=7.3728,
\]

and rank three.

## The orientation transport also fails by itself

Set the background source amplitude to zero but retain a spatial polarization
gradient. Then `e=0` and

\[
\det H=1728(c^2-d)
=1728{Ab^2\over D^3}(AD-w_iw_i).
\]

This is zero only on special lower-dimensional surfaces such as `alpha=0`,
`w.p=0`, or `AD=w^2`; it is not an action identity. A frozen random scan of
5,000 spatial backgrounds found 5,000 full-rank Hessians at the declared
`1e-10` determinant threshold.

The representative transport-only matrix has determinant `-1.08507` and rank
three. The combined representative has eigenvalues

\[
(-11.5842,-0.178473,2.49808),
\]

determinant `5.16467`, and two negative directions.

The analytic Hessian agrees with an independent centered finite difference to
`1.17e-8` maximum absolute error.

## Connection interpretation

The result is not an artifact that disappears by restoring the flat
connection. In a Stückelberg representation,

\[
\Gamma^0{}_{00}={\ddot\xi^0\over\dot\xi^0},
\]

and the coincident-gauge lapse velocity is replaced by the combination

\[
x\longrightarrow x-\Gamma^0{}_{00}.
\]

Thus the same nonlinear terms depend on a connection Stückelberg acceleration.
Restoring the gauge variable supplies the coordinate null combination but
leaves an additional lapse-connection combination with the nonzero Hessian
shown above. This is the familiar place where a nondegenerate nonlinear
nonmetricity interaction introduces an extra mode.

The project does not need to classify every possible nonlinear completion of
this mode. Its preregistered requirement was stronger: positive kinetic terms,
a proved constraint count, and a well-posed initial-value problem before data.
The exact v5B action fails the necessary degeneracy condition and therefore
cannot pass that gate.

## Why parameter tuning cannot repair v5B

- Setting `eta=0` restores GR but eliminates the proposed effect.
- Setting `alpha=0` removes orientation transport but leaves the source-only
  failure.
- Selecting `Y=1` suppresses the source Hessian at one point only and places
  the system on a zero-kinetic surface.
- Choosing a special alignment `w.p=0` cannot hold for arbitrary galaxies and
  clusters.
- Adjusting `a_sigma`, `L_sigma`, or the positive amplitude moves or rescales
  the problem; it does not make the determinant an identity.

Repair therefore requires a materially different action, not a new exponent
or fitted constant.

## Constraint on the next action

The useful physical target survives: a baryon-unique broad trace plus
orientation-preserving transport. The direct implementation with nonlinear
individual nonmetricity traces does not.

Any successor must be **degeneracy-first**:

1. start from GR/STEGR;
2. use only an established degenerate scalar/vector/tensor kinetic class, or
   prove its ADM/Stückelberg degeneracy symbolically before selecting a source;
3. avoid nonlinear dependence on `tilde(Q)^2` and `W^a W^b` inside an otherwise
   canonical scalar action;
4. retain one physical metric and a unique baryon-forced retarded state; and
5. pass the same static-background Hessian screen before Solar or data work.

The immediate next task is an action-envelope comparison between a
degeneracy-preserving auxiliary carrier and known scalar-tensor/vector-tensor
classes. No v5C empirical formula is authorized yet.

## Reproduction

```powershell
python scripts/check_sigma_v5b_nonlinear_degeneracy.py
python -m pytest tests/test_sigma_causal_polarization.py -q
python -m ruff check src/voidscreen/sigma_causal_polarization.py scripts/check_sigma_v5b_nonlinear_degeneracy.py tests/test_sigma_causal_polarization.py
```
