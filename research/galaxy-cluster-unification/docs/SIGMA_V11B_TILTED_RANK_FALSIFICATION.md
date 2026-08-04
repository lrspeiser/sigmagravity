# Sigma v11B tilted-flow kinetic falsification

## Decision

Exact Sigma v11B is retired before observational data. Its flat phonons are
healthy, but the quadratic strain becomes quartic in a material-coordinate
velocity on a slice tilted relative to the aether. The resulting physical
phonon Legendre Hessian crosses zero and becomes negative while the material
flow is still timelike.

This is the second post-v10-reset formulation to fail the nonlinear kinetic
rank gate. V11A failed because a bounded dynamical alignment multiplied an
unbounded gradient; v11B fails because a spatial strain square becomes a
negative quartic in tilted coordinate velocity. One further materially
distinct failure at this same gate will trigger another mechanism reset.

## Exact local configuration

Use Minkowski space with

$$
A^\mu=\gamma(1,v,0,0),\qquad v={1\over2}.
$$

The unstrained material map in these coordinates is

$$
X^1=\gamma(x-vt),\qquad X^2=y,\qquad X^3=z.
$$

Perturb only

$$
X^1\rightarrow X^1+w t.
$$

The internal strain is

$$
E_{11}=2aw+a^2w^2,\qquad a=\gamma v,
$$

and the velocity-dependent Lagrangian is

$$
\mathcal L(w)
={\gamma^2w^2\over2}
-{s\over4}\left({2\over3}+b\right)E_{11}^2.
$$

Therefore

$$
\boxed{
H(w)={d^2\mathcal L\over dw^2}
=\gamma^2-s\left({2\over3}+b\right)
[2a^2+6a^3w+3a^4w^2].
}
$$

For every nonzero tilt and positive rigidity,

$$
H(w)\longrightarrow
-3s\left({2\over3}+b\right)a^4w^2
$$

at large velocity. A finite rank surface is unavoidable.

## Frozen numerical point

With

$$
s={3\over11},\qquad b={17\over24},\qquad v={1\over2},
$$

the positive root is

$$
\boxed{w_*=1.68359944775}.
$$

The material-coordinate flow velocity is

$$
v_{\rm mat}=v-{w\over\gamma}=-0.9580399,
$$

strictly inside the physical light cone. At `0.99`, `1`, and `1.01` times the
critical velocity:

| `w/w_*` | Hessian | material velocity |
|---:|---:|---:|
| 0.99 | `0.0143410` | `-0.943459` |
| 1.00 | `0` | `-0.958040` |
| 1.01 | `-0.0144119` | `-0.972620` |

The Lagrangian at the crossing is finite, `1.10726` in normalized units.

## Why constraints cannot rescue it

This is a Rayleigh direction in which one physical material scalar velocity
changes while the metric and aether velocities are fixed. A negative value of
that quadratic form means the complete Hessian cannot be positive definite,
regardless of omitted off-diagonal entries. The crossing occurs before the
material flow becomes null, so it cannot be excluded by the standard timelike
material-domain condition.

Restricting strain or velocity by hand would add an unexplained state cutoff.
Changing the positive speed or bulk coefficients moves the finite surface but
does not remove the negative large-velocity term. Setting rigidity to zero
removes the geometry mechanism.

## Consequence

The flat selection result remains mathematically correct but insufficient.
V11B's exact quadratic strain-square action is closed. The next candidate must
have an all-velocity convex/degenerate material action, not merely positive
small-strain elastic moduli.

No observational product or holdout was opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v11b_tilted_rank.py
python -m pytest -q tests/test_sigma_v11b_tilted_rank.py
```

Machine-readable evidence is in
`results/sigma_v11b_tilted_rank/report.json`.
