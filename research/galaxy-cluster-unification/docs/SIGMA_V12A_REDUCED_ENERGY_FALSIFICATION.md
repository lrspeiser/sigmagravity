# Sigma v12A constraint-solved modal-energy falsification

> **Subsequent result:** the minimal action-level clock constraint does not
> remove this sector cleanly. It converts it into a freely specifiable signed
> dust-like charge; a finite regularization only renormalizes `K2`. Exact v13A
> is also retired before data. See
> [`SIGMA_V13A_CLOCK_CONSTRAINT_FALSIFICATION.md`](SIGMA_V13A_CLOCK_CONSTRAINT_FALSIFICATION.md).

## Decision

Exact Sigma v12A is retired before observational data under the project's
strict bounded-physical-Hamiltonian gate.

The previously selected constant-background cone row

$$
K_B=1,\qquad K_2=4,\qquad \lambda_D=-1
$$

has finite characteristic roots and common Cauchy times, but at least one
finite physical oscillator has negative canonical energy for every sampled
common time. The same negative branch remains when `lambda_D=0`, so it is
inherited from the AeST zero/Jeans sector rather than created by the new DHOST
operator. Neither the signed `lambda_D` scan nor the AeST parameter screen
provides a positive-energy rescue.

This is the first materially failed formulation after the v12 mechanism
reset. It does not trigger the three-failure reset rule.

## Why this is a constraint-solved energy

After the complete Euler matrices are formed and the spatial gauge is applied,
the real quadratic mode Lagrangian is

$$
L={1\over2}\dot q^T K\dot q+\dot q^TAq+{1\over2}q^TBq,
$$

with

$$
C=A-A^T.
$$

The kinetic matrix is singular because lapse, shift, gauge and DHOST
constraints are present. Those variables are not discarded before variation.
Instead, the singular generalized pencil

$$
(s^2K+sC-B)u=0
$$

is solved homogeneously. Its `24` finite roots satisfy all algebraic Euler
constraints, while the `16` primary/gauge roots remain at generalized infinity.
Evaluating a finite eigenvector therefore includes the lapse, shift, aether and
DHOST constraint response. It is equivalent to eliminating those algebraic
variables mode by mode.

For an oscillatory root $s=i\omega$, the time-averaged canonical energy is

$$
\boxed{
E={1\over4}u^\dagger(\omega^2K-B)u.
}
$$

The eigenvalue equation independently gives the Krein derivative identity

$$
\boxed{
E={\omega\over4}
u^\dagger(2\omega K-iC)u.
}
$$

The recorded maximum normalized difference between these expressions is
`4.27e-9`. The modal Euler-polynomial residuals also pass `1e-7`. The negative
sign is therefore not an unevaluated constraint or a sign convention in one
energy formula.

## Positive flat control

The same code was first applied to the near-flat clock background, selecting
the ten positive-frequency tensor, vector and finite scalar real-phase modes
above frequency fraction `0.05`. All ten have positive energy. The minimum
normalized energy is positive, and the canonical/Krein residual is below the
declared tolerance.

The flat zero-frequency sector is intentionally excluded from this control;
it is the sector being tested after it is deformed by a finite on-shell clock
misalignment.

## On-shell sentinel

The decisive sentinel has constant on-shell aether tilt `A=0.5`. Its scalar
clock is solved from the projected aether equation before the quadratic action
is formed. The audit scans:

- 19 boosts parallel to the aether, from `-0.875` through `0.8` with additional
  resolution near the aether rest frame;
- 13 wave directions from `0` through `180` degrees in `15`-degree steps;
- wave number `k=300`;
- all finite positive-frequency modes down to `|omega|/k=1e-8`; and
- normalized negative-energy tolerance `1e-8`.

Every kinematically valid candidate time retains the `24+16` root structure.
No candidate makes all directions positive. The best maximin time in the
declared normalized score is

| Quantity | Result |
|---|---:|
| Boost | `-0.25` |
| Worst normalized energy | `-0.783010` |
| Worst canonical energy | `-0.0571711` |
| Worst direction | `90 deg` |

The result is far outside numerical tolerance.

Near the aether rest frame (`v=0.447`), the worst direction is `75 deg` and the
mode decomposition is

| Contribution | Value |
|---|---:|
| Kinetic | `+0.00241650` |
| Potential | `-0.0472409` |
| Canonical/Krein total | `-0.0448244` |
| Normalized total | `-0.902673` |
| Growth divided by `k` | `4.50e-13` |
| Modal polynomial residual | `1.64e-15` |

Thus the mode is not an exponentially growing high-frequency kinetic ghost.
It is a real-frequency oscillator with positive instantaneous kinetic
contribution, a more negative potential contribution, and negative Krein
signature. Gyroscopic mixing keeps it oscillatory while the quadratic
Hamiltonian remains unbounded.

## Wave-number persistence

At the resolved boost `v=0.446`, the two axial directions were evaluated from
`k=100` to `k=2000`:

| `k` | Minimum normalized energy | Minimum canonical energy | Negative roots |
|---:|---:|---:|---:|
| `100` | `-0.457876` | `-0.161741` | `4` |
| `300` | `-0.460774` | `-0.209080` | `4` |
| `600` | `-0.305364` | `-0.267496` | `4` |
| `1000` | `-0.164321` | `-0.298755` | `4` |
| `2000` | `-0.0516446` | `-0.317455` | `4` |

All rows preserve the descriptor root structure. The normalized value tends
toward zero because the positive luminal blocks scale as `k^2`, but the raw
negative energy approaches a finite negative value rather than changing sign.

## The new DHOST term is not the source

The scan varied

$$
\lambda_D=-8,-4,-2,-1,0,1,2,4,8.
$$

For every value through `+4`, at least one kinematically valid common time
exists, but its best worst-direction energy remains negative. At `+8`, none of
the scanned times remains hyperbolic. Crucially, `lambda_D=0` has best sampled
normalized energy `-0.829917`. Removing the v12A DHOST operator therefore does
not remove the failing branch.

## Existing AeST constants do not rescue the screen

The aether-rest-frame screen sets `lambda_D=0` and spans 43 healthy-flat rows:

- `K_B=0.5,0.75,1,1.25,1.5,1.7,1.8,1.9,1.95,1.98,1.99`;
- `K2=2,4,8,32`; and
- only pairs having `0<c_s^2<=1`.

Every row retains a negative finite mode in at least one direction. As
`K_B` approaches `2`, the frequency and negative energy can become extremely
small. They remain resolved by the `1e-8` frequency threshold. At the exact
endpoint,

$$
c_s^2={2-K_B\over K_2K_B}=0,
$$

so the putative escape is a nonpropagating strong-coupling limit, not a
strictly positive Hamiltonian.

This finite grid is not a proof over the entire continuous parameter plane.
It is sufficient to reject the already selected v12A row, and it shows that
ordinary tuning of its existing constants does not provide an identified
rescue.

## Relation to published AeST stability work

The published AeST Minkowski analysis finds bounded Hamiltonians for its
massive propagating modes, plus a nonpropagating linearly time-dependent mode
whose Hamiltonian can be unbounded for wave numbers below a scale `mu`. The
authors estimate that scale to be at most approximately inverse megaparsec and
interpret the issue as potentially Jeans-like rather than a short-distance
quantum-vacuum instability:
[Skordis and Zlosnik, *Aether scalar tensor theory: Linear stability on
Minkowski space*](https://arxiv.org/abs/2109.13287).

Our result does not contradict that statement. The flat finite-frequency
control passes, while finite scalar/aether misalignment deforms the inherited
zero sector into a negative-Krein oscillator. The project goal deliberately
uses a stricter rule: the claimed physical-background domain must have a
bounded modal Hamiltonian. Under that rule, v12A fails.

The full nonlinear AeST constraint count of six degrees of freedom is prior
work, not proof that the v12A extension has positive energy:
[Bataki, Skordis and Zlosnik, *Aether scalar-tensor theory: Hamiltonian
formalism*](https://arxiv.org/abs/2307.15126).

## Falsification accounting

- Exact formulation rejected: `v12A`.
- Gate: bounded constraint-solved physical modal energy.
- Observational data opened: no.
- Post-v12-reset materially distinct failures: `1`.
- Three-failure mechanism reset triggered: no.

The previous constant-cone pass remains correct but insufficient. A theory can
be hyperbolic and causal while containing negative-energy modes.

## Requirement for the next formulation

Further numerical tuning of v12A stops here. A materially distinct successor
must alter the action-level origin of the inherited zero/Jeans sector. It must:

1. remove it with a regular constraint or give it a strictly positive
   Hamiltonian, without a freely chosen dust-like integration constant;
2. retain one physical metric for matter and light;
3. preserve a regular degeneracy structure and no extra ghost degree of
   freedom;
4. keep tensor propagation luminal;
5. retain a baryon-sourced nonlinear geometric response capable of producing
   cluster shear topology; and
6. remain within five universal constants.

Two legitimate next lanes are an action-level second-class constraint that
removes the zero mode, or replacement of the AeST clock/aether kinetic base by
a convex positive-Hamiltonian carrier. Either is a new formulation and must
repeat the flat, tilted, common-cone and energy gates before observations.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_reduced_energy.py
python -m pytest -q tests/test_sigma_v12a_reduced_energy.py
```

Machine-readable evidence is in
`results/sigma_v12a_reduced_energy/report.json`.
