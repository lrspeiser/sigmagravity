# Sigma v10C covariant coefficient and PPN-applicability precheck

## Decision

The v10C spatial-aether counterterm has an exact covariant coefficient map and
is **not independently retired** by the standard Einstein-aether PPN formulas.
This is not a Solar-System pass. It instead identifies why importing those
formulas as if v10C were pure Einstein-aether would be invalid, and fixes the
calculation that must be done next.

No astronomical product or holdout was opened.

## Exact electric/magnetic identity

Use signature `(-,+,+,+)`, a unit timelike aether

$$A^\mu A_\mu=-1,$$

and define

$$
J_\nu=A^\mu F_{\mu\nu},\qquad
B_{\mu\nu}=q_\mu{}^\alpha q_\nu{}^\beta F_{\alpha\beta},\qquad
q_\mu{}^\nu=\delta_\mu{}^\nu+A_\mu A^\nu.
$$

The exact decomposition is

$$
F_{\mu\nu}F^{\mu\nu}
=B_{\mu\nu}B^{\mu\nu}-2J_\mu J^\mu.
$$

Therefore the selected aether terms

$$
-{K_B\over2}F^2+{K_B(1-u)\over2}B^2
$$

are exactly

$$
\boxed{-{K_Bu\over2}F^2+K_B(1-u)J^2}.
$$

This proves that the counterterm is a covariant change in the aether's
electric/magnetic stiffness, not a coordinate-dependent instruction.

## Map to the pure Einstein-aether coefficients

With the local project's convention

$$
\mathcal L_{\rm ae}=-c_1(\nabla A)^2-c_2(\nabla\cdot A)^2
-c_3\nabla_\mu A_\nu\nabla^\nu A^\mu+c_4J^2,
$$

the exact map is

$$
(c_1,c_2,c_3,c_4)
=\big(K_Bu,0,-K_Bu,K_B(1-u)\big).
$$

Consequently,

$$
c_{13}=0,\qquad c_{14}=K_B,\qquad c_{123}=0,
$$

and the pure-aether vector speed proxy is

$$
{c_1\over c_{14}}=u={3\over4}.
$$

The counterterm thus leaves the luminal tensor condition `c13=0` and the
electric/Newton normalization `c14=K_B` unchanged while reducing the spatial
vector stiffness to the selected value.

## What the standard PPN formula does and does not say

For pure Einstein-aether theory, Foster and Jacobson derived

$$
\alpha_1=-8{c_3^2+c_1c_4\over2c_1-c_1^2+c_3^2}.
$$

Substitution gives

$$
\boxed{\alpha_1^{\rm pure\ ae}=-4K_B}.
$$

Importantly, the original Maxwell row `u=1` gives exactly the same proxy.
The v10C counterterm therefore does not create a new `alpha1` failure relative
to its AeST base.

That number is **not** v10C's PPN prediction. The Foster--Jacobson theory has
only the metric and unit aether. V10C also has the AeST scalar and the
hyperbolic `P` carrier, both of which enter the moving-source constraint and
gravitomagnetic equations. The mapped pure-aether row also has `c123=0`, where
the published pure-aether `alpha2` expression and spin-0 gradient sector are
singular. AeST supplies the missing scalar dynamics; v10C further mixes the
aether acceleration with `P`.

Thus a pure-aether substitution can neither pass nor retire v10C. It does
expose a serious unresolved requirement: neither the base row nor v10C has yet
derived its complete preferred-frame parameters.

## First-order carrier interaction

On the spatial constraint `A^mu P_mn=0`, the selected source is boundary
equivalent to

$$
\int\!\sqrt{-g}\,\beta P^{\mu\nu}\nabla_\mu J_\nu
=-\int\!\sqrt{-g}\,\beta(\nabla_\mu P^{\mu\nu})J_\nu.
$$

The right-hand form contains only first derivatives of `P`, `A`, and the
metric. Defining `C^nu=nabla_mu P^{mu nu}`, its aether Euler contribution,
up to terms proportional to the spatiality constraint that are absorbed by
the multiplier, is

$$
\mathcal E_A^{\sigma}\big|_{P J}
=\beta\nabla_\rho(A^\rho C^\sigma)
-\beta C^\nu\nabla^\sigma A_\nu.
$$

This confirms the interaction is second-order at the Euler-equation level. A
complete variation must still include every projector, multiplier, and metric
connection term.

## Next decisive calculation

The next PPN gate must solve one frozen moving-source expansion of the complete
AeST-plus-`P` action through the orders that determine
`gamma`, `beta`, `alpha1`, and `alpha2`. It must retain the retarded/static
boundary rule and may not tune `K_B`, `3/11`, `2/11`, or the magnetic fraction
after seeing the result.

The theory advances only if the same row then passes the declared Cassini,
Mercury, preferred-frame, propagation, and stability limits. Until that
calculation exists, the Solar/PPN gate remains false.

The pure-aether formula and its domain are from Foster and Jacobson,
[Post-Newtonian parameters and constraints on Einstein-aether theory](https://arxiv.org/abs/gr-qc/0509083).
The AeST action and its scalar/vector content are from Skordis and Zlosnik,
[A new relativistic theory for Modified Newtonian Dynamics](https://arxiv.org/abs/2007.00082)
and [Linear stability on Minkowski space](https://arxiv.org/abs/2109.13287).

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10c_covariant_ppn_precheck.py
python -m pytest -q tests/test_sigma_v10c_covariant_ppn_precheck.py
```

Machine-readable evidence is in
`results/sigma_v10c_covariant_ppn_precheck/report.json`.
