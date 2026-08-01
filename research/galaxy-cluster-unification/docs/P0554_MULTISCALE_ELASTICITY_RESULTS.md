# P0554 multi-scale elasticity results

## Outcome

The P0554 field law was perturbed at four progressively smaller scales to test
whether earlier parameter rankings were genuine local effects or consequences
of one arbitrary low/high interval. The experiment evaluated 89 formulas on
131 SPARC galaxies, 20 CLASH systems, five raw strong-lensing clusters, and the
Solar proxies. No gravity coefficient or ordinary lens geometry was fit.

The main result is a split between smooth and topological observables:

- galaxy and derived radial-cluster errors have stable, nearly linear local
  derivatives; but
- raw multiple-image solutions repeatedly cross caustics, so a 1--2% formula
  change can create or destroy an image root.

This is stronger evidence that raw cluster lensing is not just asking for a
different scalar amplitude. It is testing the shape and topology of the
two-dimensional field.

No candidate was selected.

## Frozen experiment

Each parameter was assigned a declared reference move. Central differences
were evaluated at 0.1, 0.25, 0.5, and 1.0 times that move.

| Parameter | Full reference move | Smallest tested move |
|---|---:|---:|
| universal amplitude $q$ | 10% | 1% |
| radial exponent $\alpha$ | 10% | 1% |
| apogee ratio $\zeta$ | 20% | 2% |
| screen exponent $n$ | 20% | 2% |
| screen location $s$ | 20% | 2% |
| potential power $p$ | 10% | 1% |
| potential scale $\chi_t$ | 20% | 2% |
| path-ratio power $\beta$ | 20% | 2% |
| photon multiplier $m_\gamma$ | 0.25 | 0.025 |
| mass-radius exponent $\delta$ | 0.05 | 0.005 |
| extent exponent $\epsilon$ | 0.05 | 0.005 |

For a normalized observable error $y$, the reported central slope is

$$
D_y(u)=\frac{y(+u)-y(-u)}{2u\,y_0}.
$$

Mercury uses absolute supplementary precession divided by its 3.1
mas/century margin. Raw slopes use only clusters with every held-out root for
$-u$, the parent, and $+u$. Root gains and losses are recorded separately.

## Stable local controls

A direction was called stable only if at least three step sizes had finite
comparisons and at least 75% of their nonzero slopes had the same sign.

| Domain | Strongest stable lever | Median normalized slope | Better direction |
|---|---|---:|---|
| SPARC rotation | radial exponent $\alpha$ | 0.0892 | increase |
| CLASH radial lensing | mass-radius exponent $\delta$ | 0.1711 | increase |
| RX J2129 fixed geometry | path-ratio power $\beta$ | 0.7755 | decrease |
| other four raw clusters | mass-radius exponent $\delta$ | 0.1016 | decrease |
| Mercury | screen exponent $n$ | 3.0125 | increase |

The SPARC and CLASH rankings are particularly robust. Across all four scales,
the absolute galaxy slope varies by only 10.6% for $\alpha$, and the CLASH
slope varies by only 9.8% for $\delta$. The latter is not a universal raw-lens
direction: increasing $\delta$ improves RX J2129 at the smallest comparable
step but decreasing it improves the other four clusters.

The nominal apogee ratio is the least consequential parameter. It never
changes raw-root count and has only 0.0084, 0.0040, and 0.0822 normalized
median slopes for galaxies, CLASH, and RX J2129. Increasing field reach is not
the missing mechanism near P0554.

## Strong-lensing bifurcations

The parent solves 17 of 18 held-out images. Across the grid, solutions range
from 13 to 18 roots. Four parameters change topology at the smallest tested
step:

| Small change | Root response |
|---|---|
| $\delta=\pm0.005$ | 16 roots for the decrease, 18 for the increase |
| $\chi_t$ by $\pm2\%$ | the increase produces 18 roots |
| $m_\gamma$ by $\pm0.025$ | the decrease produces 18 roots |
| $p$ by $\pm1\%$ | the decrease produces 18 roots |

This does not mean all four changes improve lensing. It means P0554 places at
least one held-out image close to a caustic boundary. Once a root disappears,
the continuous RX RMS derivative is no longer defined, which is why the very
large one-step slopes of $\delta$, $\chi_t$, and $m_\gamma$ are not ranked as
stable multi-scale controls.

Extent leakage behaves differently. It has a smooth RX direction at all four
scales, but does not recover the missing root until $\epsilon=+0.025$. This
supports interpreting it as a topology threshold rather than the main
continuous amplitude lever.

## Solar boundary

The screen exponent remains the dominant Solar control and is highly
nonlinear. A 2% decrease from $n=1$ remains safe, but a 5% decrease to
$n=0.95$ predicts -4.443 mas/century for Mercury and crosses the frozen 3.1
mas/century margin. The full 20% decrease predicts -74.666 mas/century.

The other Solar crossings occur only at the full declared moves:

- $\alpha=0.675$ gives -3.510 mas/century; and
- $\delta=+0.05$ gives -4.102 mas/century.

This tightens the earlier conclusion: the high-acceleration screen is not a
loosely adjustable fitting term. Around this parent, its exponent must remain
above roughly 0.95 before any proper ephemeris calculation is even attempted.

## Cross-domain directions

Only the screen location has a material, directionally consistent fixed-
geometry response in at least two non-Solar domains: lowering $s$ improves
SPARC, CLASH, and the finite RX J2129 comparison. This is the closest local
candidate for a common scalar direction.

That statement has an important qualification. In the previous exact geometry
refit, lowering $s$ from 1.0 to 0.8 recovered the missing MACS1931 root and
improved the matched multi-cluster aggregate, but changed RX J2129 from 1.256
to 1.260 arcsec. Thus the common fixed-geometry direction does not survive as
an RX accuracy gain after nuisance freedom is restored. It remains a useful
transition/topology control, not a promoted law.

The photon multiplier is an especially clean negative lesson. It changes
neither galaxy dynamics nor Mercury, yet a 0.025 decrease changes the raw-root
topology. A separate light-only strength is therefore not an independent
smooth lensing knob near a caustic.

## Universal lessons retained

1. **Galaxy response is primarily radial.** The $\alpha$ derivative is stable
   over a factor of ten in perturbation size.
2. **Radial cluster amplitude is primarily a mass-scale/potential problem.**
   $\delta$, $\chi_t$, and $p$ dominate CLASH, while apogee barely matters.
3. **Raw lensing is a topology problem.** Several percent-level changes alter
   image multiplicity before they establish a stable RMS trend.
4. **One mass scaling does not transfer across raw clusters.** The preferred
   sign of $\delta$ reverses between RX J2129 and the other four systems.
5. **Solar decoupling must be sharp and tightly controlled.** A 5% decrease in
   the screen exponent already violates the analytic Mercury margin.
6. **Environmental extent is a threshold control.** It can recover a root but
   is not the largest smooth scalar derivative.
7. **Light-only amplification is not a free rescue.** It can change caustic
   topology without improving the dynamical domains.

These results favor a future field equation with a smooth radial/potential
sector plus a separate derived two-dimensional tidal or routing operator. The
latter must be calculated from baryonic geometry rather than chosen per
cluster.

## Limits

The systems and parent formula are spent exploratory data, CLASH accelerations
are derived through conventional GR/NFW profiles, raw geometries are held
fixed, and Solar tests are analytic proxies rather than an ephemeris
likelihood. Parameter reference moves are declared comparison scales, not
physical priors. A topology change is not evidence that the corresponding
parameter is physically real.

## Reproduction

```powershell
python scripts/run_p0554_multiscale_elasticity.py
python -m pytest tests/test_p0554_multiscale_elasticity.py -q
```

Machine-readable results are in `results/p0554_multiscale_elasticity/`.

## Structural continuation

Nine parent-preserving algebraic deformations have now been tested in a second
73-formula grid. Generalized dynamics and lensing addition laws are the largest
structural controls, but create galaxy/cluster and cluster-to-cluster sign
conflicts. Screen shape is the only partially shared direction. See
[`P0554_STRUCTURAL_MICROVARIATIONS_RESULTS.md`](P0554_STRUCTURAL_MICROVARIATIONS_RESULTS.md).
