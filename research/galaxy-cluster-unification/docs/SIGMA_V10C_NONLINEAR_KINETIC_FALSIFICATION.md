# Sigma v10C nonlinear kinetic falsification

## Decision

Exact Sigma v10C is retired before observational data. Its first-order
aether--carrier interaction makes the kinetic energy of the physical AeST
vector modes depend linearly on the carrier background. At a finite allowed
carrier amplitude, the kinetic matrix becomes singular; immediately beyond
that surface it has negative eigenvalues.

This is a structural failure. It cannot be repaired by changing the five
constants, imposing an object-specific amplitude cutoff, or invoking the
quartic potential. No astronomical product or holdout was opened.

## The local nonlinear subblock

Work at one event in a locally inertial aether-rest frame,

$$
g_{\mu\nu}=\eta_{\mu\nu},\qquad
A^\mu=(1,0,0,0),\qquad
P_{0\mu}=0.
$$

Let `a_i` be the spatial aether perturbation and `p_ij=P_ij` the instantaneous
carrier background. Differentiating the exact spatiality constraint gives

$$
{d\over dt}(A^\mu P_{\mu i})=0
\quad\Longrightarrow\quad
\dot P^{0i}=p_{ij}\dot a_j.
$$

For a homogeneous high-frequency kinetic probe,

$$
J_i=\dot a_i,
\qquad
C^i=\nabla_\mu P^{\mu i}=p_{ij}\dot a_j.
$$

The first-order interaction derived at the previous gate therefore contains

$$
-\beta C^iJ_i=-\beta p_{ij}\dot a_i\dot a_j.
$$

Combining it with the positive Maxwell-aether and carrier kinetic terms gives

$$
\boxed{
\mathcal L_{\rm kin}
=\dot{\boldsymbol a}^{\mathsf T}
\left(K_B I-\beta p\right)
\dot{\boldsymbol a}
+{1\over2}\dot P:\dot P
}.
$$

The code differentiates this reduced Lagrangian numerically in all nine
velocities and reproduces its analytic Hessian below `1e-9`.

## Finite strong-coupling and ghost surfaces

For the isotropic allowed carrier background

$$
p_{ij}=p\,\delta_{ij},
$$

the three vector kinetic eigenvalues are proportional to

$$
K_B-\beta p.
$$

The selected relation `beta^2/K_B=2/11` puts the zero at

$$
\boxed{
p_\star={K_B\over\beta}=\sqrt{{11K_B\over2}}
}.
$$

At the selection row `K_B=1`,

$$
p_\star=2.34521.
$$

The velocity-Hessian minimum eigenvalue is positive at `0.99 p_star`, zero at
`p_star`, and negative at `1.01 p_star`. All three physical aether-vector
directions are negative above an isotropic threshold.

The carrier potential is

$$
V={1\over L_P^2}
\left[{P:P\over2}+{(P:P)^2\over4}\right].
$$

It is convex and grows at large amplitude, but it is finite at `p_star`. A
potential suppresses high-amplitude states energetically; it does not remove
them from the Cauchy data or change the sign of their high-frequency kinetic
term. The spatiality constraint restricts orientation, not amplitude.

## Why metric constraints do not rescue the action

The failed directions are the same transverse aether-vector modes that are
physical and positive in the published flat AeST spectrum. The calculation is
a local homogeneous decoupling subblock with fixed metric and scalar clock;
lapse and shift do not supply kinetic terms that could turn a negative
physical transverse-vector coefficient positive. At the zero surface the
Legendre map loses rank, and beyond it the physical vector sector is a ghost.

A complete ADM count could reveal additional failures, but it is not needed
to overturn this necessary positivity gate.

## Consequence for the mechanism

Setting `beta=0` removes the problem and also removes the tidal response.
Declaring `|P|<p_star` would be a new unexplained state cutoff and would not be
preserved automatically by arbitrary sources or free carrier waves.

A possible successor must alter the action itself, for example by embedding
the aether--carrier kinetic dependence in a globally positive square or a
bounded nonlinear function whose full Hessian stays positive. Such a theory
is materially different and must rederive its static capacity and cones; it
cannot inherit v10C's numerical selection results by assertion.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10c_nonlinear_kinetic.py
python -m pytest -q tests/test_sigma_v10c_nonlinear_kinetic.py
```

Machine-readable evidence is in
`results/sigma_v10c_nonlinear_kinetic/report.json`.
