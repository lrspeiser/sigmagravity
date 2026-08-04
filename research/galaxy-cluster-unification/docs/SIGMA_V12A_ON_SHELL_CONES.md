# Sigma v12A on-shell constant-background common-cone gate

> **Subsequent result:** the constraint-solved modal-energy audit confirms that
> the finite modes satisfy the quadratic constraints but finds a negative
> canonical/Krein-energy oscillator for every sampled common time. Exact v12A
> is therefore retired before data. See
> [`SIGMA_V12A_REDUCED_ENERGY_FALSIFICATION.md`](SIGMA_V12A_REDUCED_ENERGY_FALSIFICATION.md).
> The cone pass below remains correct but is not sufficient for viability.

## Decision

The original frozen `K_B=1,K2=2` v12A row is rejected by an invariant
on-shell characteristic test. The same action has one provisional theory-side
rescue row on the frozen parameter screen:

$$
K_B=1,\qquad K_2=4,\qquad \lambda_D=-1.
$$

This changes no equation and adds no constant. It changes the flat scalar
principal speed from $c_s^2=1/2$ to $c_s^2=1/4$. The rescued row passes all
eight declared constant-background gates, including a common sampled time
direction and principal-cone convergence through aether tilt magnitude 8.

This is not a viable theory verdict. Negative coordinate-time mode energies
remain on some moving backgrounds. A reduced physical Hamiltonian on the same
common time direction, followed by nonconstant and curved backgrounds, is the
next kill gate. No observation or holdout was opened.

## Why the earlier off-shell warning was insufficient

For the AeST scalar invariants

$$
X=g^{\mu\nu}\phi_\mu\phi_\nu,\qquad
Q=U^\mu\phi_\mu,\qquad
Y=X+Q^2\ge0,
$$

the constant-gradient scalar equation is automatic because the action is
shift symmetric. A misaligned constant aether is not arbitrary, however. Its
projected equation is

$$
\sqrt{Y}\,\frac{\partial L}{\partial Q}=0,
$$

where the selected simple AeST interpolation gives

$$
\frac{\partial L}{\partial Q}
=-2(2-K_B)Q\left[1+
\frac{\sqrt{Y}/a_\Sigma}{1+\sqrt{Y}/a_\Sigma}\right]
+4K_2(Q-Q_0).
$$

The earlier arbitrary `q=0.25`, tilt `0.5` rows that reached roughly `1.89c`
do not satisfy this equation. They remain a warning about the action away from
its physical branch, but they cannot by themselves retire the branch used by a
solution.

The new audit solves this equation numerically at every nonzero tilt before
forming the characteristic pencil. The maximum projected aether-equation
residual stays below the declared `1e-10` tolerance.

## Manifest covariant implementation

The action density was rewritten for an arbitrary constant timelike scalar
covector rather than assuming scalar-unitary gauge. Three nontrivial unitary
sentinels reproduce the established ADM Hessian with maximum normalized
residual `1.10e-17`. A general Lorentz boost preserves

$$
X,\quad U^2=-1,\quad Q,\quad Y
$$

with maximum residual `4.44e-16`.

The same full Euler system is formed before imposing the spatial gauge. A
valid time direction must preserve `24` finite generalized roots and `16`
constraint roots at infinity for every sampled wave direction.

## Invariant cone criterion

For an oscillatory characteristic root $s=i\omega$ and spatial wave number
$k$, the physical-metric norm of its characteristic covector is

$$
g^{\mu\nu}\xi_\mu\xi_\nu=k^2-\omega^2.
$$

If $|\omega|/k>1$ in the principal limit, the covector is metric timelike. The
sign of this norm is Lorentz invariant, so a change of slicing cannot remove
a genuine superluminal characteristic. By contrast, complex frequency in one
time coordinate can disappear when another metric-timelike Cauchy covector is
used. The audit therefore treats growth and cone norm separately.

Finite wave number can split a multiple luminal root. The cone decision is
made from a linear extrapolation of the maximum frequency excess in `1/k`, not
from a single finite-`k` value.

## Existing-constant screen

The frozen theory-only screen used:

- `K_B = 0.5,1,1.5`;
- `K2 = 2,4,8,16,32`;
- on-shell tilts `0.1,0.5,1`;
- wave/aether angles `0,45,90,135,180` degrees;
- the negative DHOST orientation;
- `k=1000`;
- normalized growth tolerance `0.01`; and
- finite-`k` metric-frequency tolerance `0.001`.

Only one of the 15 existing-constant pairs passed every screen:

| $K_B$ | $K_2$ | Flat $c_s^2$ | Max growth | Max frequency/$c$ | Screen |
|---:|---:|---:|---:|---:|---|
| `1` | `2` | `0.5` | `1.23e-8` | `1.002780` | fail cone |
| `1` | `4` | `0.25` | `2.04e-8` | `1.000614` | **pass** |
| `1` | `8` | `0.125` | `0.01793` | `1.000550` | fail growth |
| `1` | `16` | `0.0625` | `0.01989` | `1.000523` | fail growth |
| `1` | `32` | `0.03125` | `0.01695` | `1.000511` | fail growth |

The `K_B=0.5` rows either have a superluminal flat scalar or miss the finite-`k`
cone tolerance. Every `K_B=1.5` row misses the growth threshold. The complete
15-row table is retained in the machine-readable report.

## Original-row rejection

For the on-shell `K_B=1,K2=2`, tilt `0.5` perpendicular sentinel, the fastest
stable frequency was evaluated at `k=300,1000,3000`. Its extrapolated principal
frequency excess is

$$
\lim_{k\rightarrow\infty}\left(\frac{|\omega|}{k}-1\right)
=0.00257605.
$$

This is a roughly `0.258%` speed excess, not a finite-wave splitting. Because
the corresponding characteristic-covector norm is invariant, a Lorentz boost
cannot rescue the original `K2=2` row.

## Common-time scan of the rescued row

For `K_B=1,K2=4`, both coupling signs were tested at on-shell tilts
`0.1,0.5,1,2,5,8`. For each background a single boost parallel to the aether
had to work simultaneously for five wave directions. Both signs have a common
sampled time at the declared `k=300` thresholds. The negative sign is retained
provisionally because it is already hyperbolic in less extreme slicings over
the moderate-tilt branch.

The negative-sign principal sentinels use `k=300,600,1000`:

| Tilt | Common boost | Max growth | Extrapolated frequency excess |
|---:|---:|---:|---:|
| `0.5` | `0` | `1.19e-8` | `-2.72e-7` |
| `1` | `0` | `2.04e-8` | `2.05e-6` |
| `2` | `-0.925` | `6.80e-4` | `-5.03e-6` |
| `5` | `-0.9` | `1.67e-3` | `-5.98e-7` |
| `8` | `-0.98` | `8.08e-3` | `-3.05e-7` |

All 15 convergence rows retain exactly `24+16` roots. The largest absolute
intercept is `5.03e-6`, inside the final-audit `1e-4` tolerance. The largest
normalized growth is `0.00808`, inside the `0.01` conditioning threshold.

## Unresolved energy gate

Some common-time rows have negative values of the quadratic coordinate-time
energy diagnostic; the most negative best-frame value in the current scan is
approximately `-34.4`. This is reported but not declared a ghost because:

1. the backgrounds are local inertial representatives that may require
   curvature or external stress to satisfy the metric equation;
2. the diagnostic has not completed the lapse, shift, aether, and DHOST
   constraint reduction; and
3. energy associated with a coordinate time can change on a background with
   momentum relative to that time.

The next calculation must construct the Dirac-reduced physical Hamiltonian
with respect to the selected common Cauchy covector. If its physical kinetic
form is negative for any on-shell sentinel, this rescued row is retired before
observations.

## Scope and next gate

This audit establishes only constant local backgrounds. It does not include
nonzero scalar Hessian, aether gradient, extrinsic curvature, spacetime
curvature, PPN limits, Solar solutions, numerics, or observations.

The next kill sequence is:

1. reduce the physical Hamiltonian in the common time and decide its energy
   sign;
2. add nonzero scalar Hessian and aether gradient;
3. add extrinsic curvature and local spacetime curvature;
4. require the same root, hyperbolicity, causal-cone, and energy gates; and
5. only then derive the weak-field galaxy and lensing equations.

No formulation-failure count is incremented here. The `K2=2` parameter row is
rejected, but the same v12A action has an existing-constant survivor. A v12A
formulation failure is recorded only if the rescued row fails a later required
gate.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_on_shell_cones.py
python -m pytest -q tests/test_sigma_v12a_general_covector.py tests/test_sigma_v12a_on_shell_cones.py
```

Machine-readable evidence is in
`results/sigma_v12a_on_shell_cones/report.json`.
