# Sigma v12A finite-tilt scalar-unitary characteristic gate

## Decision

Scalar-unitary metric time is **not** a universally healthy slicing of the
frozen v12A action. The finite physical/constraint root structure survives on
all 72 constant backgrounds, but the grid contains three independently
diagnosed failures:

- exponential roots whose growth remains proportional to wave number;
- real coordinate frequencies that remain outside the matter-metric light
  scale; and
- negative quadratic energy for modes that remain oscillatory in this time
  coordinate.

This blocks Solar, galaxy, and cluster calculations. It does not yet retire
the covariant action: a failure of the scalar-unitary time covector does not
prove that no other metric-timelike covector is simultaneously Cauchy for all
physical modes.

## Frozen grid

The calculation uses the same complete quadratic Euler pencil as the flat
regression. Homogeneous generalized eigenpairs are classified before dividing
`alpha/beta`, preventing roundoff from turning exact Class-Ia constraint roots
at infinity into enormous spurious physical frequencies.

The grid contains:

- both `lambda_D` signs;
- scalar clock ratios `0.5,1,2` relative to the preferred clock;
- aether tilt magnitudes `0.1,0.5,1,2`;
- wave/aether angles `0,45,90` degrees; and
- wave number `k=300`.

That gives `2 x 3 x 4 x 3 = 72` constant backgrounds. The preregistered
failure thresholds were normalized exponential growth above `0.01`, real
frequency above `1.01` times the metric light scale, negative oscillatory
quadratic energy below `-10^-8`, or Euler-polynomial residual above `10^-7`.

## Results

| Gate | Failures among 72 |
|---|---:|
| Finite/infinite root structure | `0` |
| Euler-polynomial residual | `0` |
| Principal exponential growth | `22` |
| Metric-cone coordinate frequency | `31` |
| Negative oscillatory coordinate-time energy | `38` |

Every row retains exactly `24` finite roots and `16` constraint roots at
infinity. The failures therefore do not come from accidentally propagating
the degenerate higher-derivative constraint mode.

Both v12A signs fail:

| Branch | Growth failures | Frequency failures | Energy failures |
|---|---:|---:|---:|
| `lambda_D=-1` | `9` | `15` | `21` |
| `lambda_D=+1` | `13` | `16` | `17` |

The extrema on the frozen grid are:

| Diagnostic | Value | Background `(q, tilt, angle, sign)` |
|---|---:|---|
| Maximum normalized exponential growth | `0.90355` | `(2,2,90 deg,+)` |
| Maximum absolute frequency/light ratio | `1.36698` | `(2,2,0 deg,-)` |
| Minimum oscillatory quadratic energy | `-48.7730` | `(0.5,1,45 deg,-)` |

## Wave-number convergence

Three independent sentinels were rerun at `k=100,300,1000`.

### Preferred clock, perpendicular large tilt

At `(q,tilt,angle,sign)=(1,2,90 deg,+)`, the maximum growth divided by wave
number is

$$
0.45108,\quad0.45157,\quad0.45162.
$$

It approaches a nonzero high-frequency value rather than decaying as a
lower-order instability.

### Off-clock frequency

At `(0.5,0.5,90 deg,+)`, the maximum real-frequency magnitude divided by the
metric light frequency is

$$
1.19334,\quad1.19320,\quad1.19319.
$$

The roughly 19.3% excess is a principal effect, not the small finite-wave
splitting seen near the flat vacuum.

### Parallel negative energy

At `(2,0.5,0 deg,-)`, the minimum oscillatory quadratic energy is

$$
-3.5651,\quad-3.7579,\quad-3.7808.
$$

Its sign and magnitude persist under increasing wave number.

## Why this is not yet a covariant falsification

The current calculation uses the scalar as time:

$$
\phi=q t.
$$

For a multi-cone field theory, a covector can be timelike for the matter metric
but fail to be timelike for a narrower or tilted effective mode cone. Complex
frequency in that slicing proves that this particular initial surface is not
Cauchy for the mode. It does not prove that every metric-timelike initial
surface fails.

Likewise, coordinate-time energy can change sign when a background carries
momentum relative to the chosen time direction. A defensible retirement
requires the invariant next calculation:

1. allow a constant scalar background with both temporal and spatial first
   derivatives;
2. construct the fully covariant principal polynomial for a general covector;
3. scan all metric-timelike candidate time covectors;
4. require one common hyperbolicity cone for the metric, aether, scalar, and
   v12A modes; and
5. evaluate the reduced physical energy with respect to that same time
   direction.

If no common metric-timelike covector exists for any frozen failure sentinel,
or if the physical energy remains negative throughout the common cone, exact
v12A is retired before observations. Until that calculation, the honest
status is **serious characteristic warning, invariant verdict pending**.

No observational data or holdout were opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_tilted_characteristics.py
python -m pytest -q tests/test_sigma_v12a_tilted_characteristics.py
```

Machine-readable evidence is in
`results/sigma_v12a_tilted_characteristics/report.json`.
