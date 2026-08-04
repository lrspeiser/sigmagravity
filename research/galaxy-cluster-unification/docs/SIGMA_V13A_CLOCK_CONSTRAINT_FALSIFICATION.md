# Sigma v13A exact clock-current constraint falsification

## Decision

Exact Sigma v13A is retired before observational data. The proposed covariant
constraint

$$
\Delta S_{13A}=\int d^4x\sqrt{-g}\,
\Lambda\left(U^\mu\nabla_\mu\phi-Q_0\right)
$$

does fix the AeST scalar clock exactly, but it does not simply remove the
inherited zero/Jeans sector. Shift symmetry converts the multiplier into a
freely specifiable conserved, pressureless state. Its homogeneous reduced
Hamiltonian is linear in a signed integration charge and therefore is not
bounded below on the unrestricted phase space.

Restricting the charge to be nonnegative removes the negative branch but does
not satisfy the project goal: two systems with identical baryons can retain
different positive multiplier densities. That is operationally invisible dust,
not gravity predicted by baryonic matter.

Giving the auxiliary field finite quadratic curvature is not a rescue. It
integrates out exactly into another positive contribution to the existing
AeST clock coefficient `K2`, returning to the soft clock family already shown
not to remove the v12A negative-energy mode.

This is the second materially distinct formulation rejected after the v12
reset and the second rejection at the bounded-Hamiltonian gate. It does **not**
trigger the three-failure mechanism reset. No galaxy, cluster, Solar-System,
or holdout observation was opened.

## Why the constraint looked promising

Define

$$
Q=U^\mu\nabla_\mu\phi.
$$

The v12A negative mode came from the AeST clock/Jeans sector rather than from
the new DHOST coefficient. The most direct action-level response is therefore
to impose

$$
Q=Q_0
$$

with a scalar multiplier. This uses the existing scalar and aether, adds no
new physical length, preserves one metric for matter and light, and leaves the
spatial invariant

$$
Y=(g^{\mu\nu}+U^\mu U^\nu)
\nabla_\mu\phi\nabla_\nu\phi
$$

available for the MOND-like quasistatic response. It is therefore a genuine
test of the action-level constraint lane identified after v12A, not a numerical
retuning of `K2` or `lambda_D`.

## Exact homogeneous reduction

On an aligned homogeneous background with lapse one,

$$
Q=\dot\phi,
\qquad
L_\Lambda=a^3\Lambda(\dot\phi-Q_0).
$$

Variation with respect to `Lambda` gives

$$
\dot\phi=Q_0.
$$

Because the full candidate remains shift symmetric in `phi`, the scalar
equation is a current conservation law. Calling the non-multiplier scalar
current $J_{\rm base}$,

$$
\boxed{
{d\over dt}\left[a^3(J_{\rm base}+\Lambda)\right]=0.
}
$$

Hence

$$
\boxed{
\Lambda={I\over a^3}-J_{\rm base},
}
$$

where $I$ is an arbitrary integration charge. On the aligned vacuum clock,
$J_{\rm base}=0$, and the canonical momentum and reduced Hamiltonian are

$$
p_\phi=a^3\Lambda=I,
\qquad
\boxed{H_\Lambda=Q_0 I.}
$$

The associated physical density and pressure are

$$
\rho_\Lambda={Q_0 I\over a^3},
\qquad
p_\Lambda=0.
$$

This is exactly the scaling of pressureless dust. The executable audit verifies
the conserved current and $a^{-3}$ scaling to machine precision.

## Two independent failures

### 1. Unrestricted energy is not bounded

For the selected positive clock orientation $Q_0>0$, the action does not
restrict the sign of $I$. Therefore

$$
I\rightarrow-\infty
\quad\Longrightarrow\quad
H_\Lambda\rightarrow-\infty.
$$

This is an analytic statement, not an inference from the finite signed-charge
sentinels. The sentinels merely verify the implementation.

A positivity restriction on initial multiplier density is discussed in the
mimetic-gravity Hamiltonian literature. It is an extra phase-space restriction,
not a derivation of the density from baryons; see
[Ganz et al.](https://arxiv.org/abs/1812.02667) and the earlier
[Hamiltonian formulation](https://arxiv.org/abs/1404.4195). The Sigma term is
not claimed to be identical to every mimetic theory, but the conserved
multiplier-dust mechanism is the same relevant warning.

### 2. Positive charge is still an invisible state

Suppose one permits only $I\ge0$. At fixed scale factor and identical baryonic
sources, the solutions

$$
I=0,qquad I=1,qquad I=2
$$

all obey the same exact clock constraint but carry different gravitational
energy densities. The baryons do not choose among them.

Using $I$ to supply missing galaxy or cluster gravity would therefore
reintroduce precisely what the project forbids: a system-dependent gravitating
profile not predicted by visible matter. Calling it a boundary condition rather
than dark matter does not change its predictive role.

Freezing $I=0$ everywhere would prevent using it as missing gravity, but the
full theory still admits the charge sector and its signed Hamiltonian. The goal
requires a healthy action, not only a preferred initial condition.

## Why a regular auxiliary potential does not repair it

The minimal regularization is

$$
\Delta L_{\rm reg}
=\Lambda\,\delta Q-{\chi\over2}\Lambda^2,
\qquad
\delta Q=Q-Q_0,
\qquad \chi>0.
$$

The multiplier equation is now algebraic rather than constraining:

$$
\Lambda={\delta Q\over\chi}.
$$

Substitution gives

$$
\boxed{
\Delta L_{\rm reg,eff}={\delta Q^2\over2\chi}.
}
$$

In the AeST normalization $L_{\rm clock}=2K_2\delta Q^2$, this is exactly

$$
\boxed{
K_2\longrightarrow K_2+{1\over4\chi}.
}
$$

It adds no constraint and no new mechanism. Direct substitution and an
independent centered finite difference agree below `4.5e-10` in the frozen
audit. The v12A energy screen already found the negative branch at positive
`K2=2,4,8,32`, with the endpoint escape becoming a zero-speed strong-coupling
limit rather than a positive Hamiltonian.

The two branches therefore have a sharp tradeoff:

- `chi=0`: an exact constraint, but a free signed dust charge;
- finite `chi>0`: no exact constraint, only another soft `K2` coefficient.

Changing the shape of a regular algebraic potential for `Lambda` changes the
Legendre-transformed function of $Q$ but not this structural distinction.

## Falsification accounting

- Exact formulation rejected: `v13A`.
- Primary gates: bounded physical Hamiltonian and source uniqueness.
- Post-v12 total materially distinct formulation failures: `2`.
- Same bounded-Hamiltonian gate failures: `2`.
- Same source-uniqueness gate failures: `1`.
- Three-failure mechanism reset triggered: no.
- Observational data opened: no.

The v12A DHOST geometry mechanism is not separately falsified by this result.
What is closed is the minimal linear clock-multiplier repair.

## Requirement for v13B

The next successor must use a convex reduced carrier rather than another
linear clock multiplier. It must:

1. have a Hamiltonian bounded below without a sign restriction on an arbitrary
   object-level charge;
2. make every extra state uniquely baryon-forced under universal boundary
   conditions;
3. retain one physical metric and luminal tensor propagation;
4. preserve a nonlinear spatial response and directional curvature capability;
5. use at most five universal constants; and
6. pass flat, tilted, common-cone, and reduced-energy gates before observations.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v13a_clock_constraint.py
python -m pytest -q tests/test_sigma_v13a_clock_constraint.py
```

Machine-readable evidence is in
`results/sigma_v13a_clock_constraint/report.json`.
