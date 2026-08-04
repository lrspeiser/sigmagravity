# Sigma v12A homogeneous aligned Dirac branch

## Decision

V12A passes its homogeneous aether-aligned primary-secondary test. The result
closes a specific strong-coupling concern: at the intended flat AeST clock the
new DHOST activation vanishes, but the constraint pair does not vanish with it.
The ordinary AeST clock susceptibility leaves a nonzero bracket of magnitude
`4 K2`.

This is not the arbitrary-gradient or tilted-background result. The exact
v12A row advances only to that calculation.

## Exact homogeneous reduction

Take the scalar gradient to be normal to the ADM slice and align the aether
with that normal. Write

$$
q=\nabla_n\phi,\qquad
V_*=\phi_{nn},\qquad
K=h^{ij}K_{ij}.
$$

With vanishing spatial scalar gradient, the three v12A invariants reduce to

$$
L_3=-q^2V_*^2-q^3KV_*,
$$

$$
L_4=-q^2V_*^2,
\qquad
L_5=q^4V_*^2.
$$

For the pure trace of the metric velocity, the Einstein term is

$$
\kappa K^2,
\qquad
\kappa=-{2F_0\over3}.
$$

Substituting the luminal Class-Ia relations gives

$$
\mathcal L_{\rm kin}
=\kappa K^2+2bKV_*+aV_*^2,
$$

where

$$
b=-{q^3A_3\over2},
\qquad
a=-{3q^6A_3^2\over8F_0}
={b^2\over\kappa}.
$$

Thus the entire homogeneous higher-derivative kinetic sector is an exact
degenerate square for every finite value of `q` and `A3(q)`.

## Primary and reduced Hamiltonian

The two momenta obey

$$
p_q=2(aV_*+bK),
\qquad
\pi_K=2(bV_*+\kappa K),
$$

and hence

$$
\boxed{\Psi=p_q-{b\over\kappa}\pi_K\approx0.}
$$

The Legendre transform is

$$
p_qV_*+\pi_KK-\mathcal L_{\rm kin}
={\pi_K^2\over4\kappa}.
$$

Both `V_*` and the function `b(q)` disappear. Including the homogeneous AeST
clock and the definition of `q` gives

$$
H={\pi_K^2\over4\kappa}+p_\phi q-L_{\rm AeST}(q)+u\Psi.
$$

Therefore

$$
\Omega=\{\Psi,H\}
=-p_\phi+{dL_{\rm AeST}\over dq}\approx0,
$$

and

$$
\boxed{
\{\Psi,\Omega\}
=-{d^2L_{\rm AeST}\over dq^2}.
}
$$

The v12A DHOST coefficient drops out of this bracket exactly on the aligned
homogeneous branch.

## Flat-clock result

The AeST convention is

$$
K(Q)=-{1\over2}F(0,Q),
$$

with

$$
K(Q)=K_2(Q-Q_0)^2+\cdots.
$$

Consequently

$$
L_{\rm AeST}(q)=-F(0,q)=2K_2(q-Q_0)^2+\cdots
$$

and at the clock background

$$
\boxed{
\{\Psi,\Omega\}_{Q_0}=-4K_2.
}
$$

For the selected row `K2=2`, the bracket is `-8`. Meanwhile the v12A shape has

$$
A_3(Q_0)=0,
\qquad
b(Q_0)=0,
\qquad
\Psi(Q_0)=p_q.
$$

The auxiliary scalar-Hessian coordinate is therefore removed by a regular
second-class pair even where the new interaction turns off. There is no flat-
clock loss of rank or extra strongly coupled mode from this pair.

## Executable audit

The frozen scan covers 4,001 signed clock values from `-4` to `4`, with two
independent velocity draws at each point. It checks:

- direct `L3-L5` reduction against `a=b^2/kappa`;
- the canonical primary identity;
- independence of the reduced Hamiltonian from `V_*`;
- exact disappearance of `A3` and `b` at `Q0`; and
- the nonzero `-4K2` bracket at that point.

These are algebraic synthetic checks. No astronomical data were opened.

## Remaining kill gate

The next calculation must restore a nonzero spatial scalar gradient, finite
aether tilt, anisotropic `K_ij`, and the spatial derivative terms in
`Delta_eff`. That operator must remain invertible throughout the admitted
background domain. A finite field value where its principal rank changes
retires v12A before observations.

The AeST clock expansion comes from the published
[AeST Hamiltonian formulation](https://arxiv.org/abs/2307.15126). The
Class-Ia degeneracy mechanism follows the
[DHOST Hamiltonian analysis](https://arxiv.org/abs/1512.06820). The combined
homogeneous reduction is the project-specific calculation; no novelty claim is
made.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_homogeneous_dirac.py
python -m pytest -q tests/test_sigma_v12a_homogeneous_dirac.py
```

Machine-readable evidence is in
`results/sigma_v12a_homogeneous_dirac/report.json`.
