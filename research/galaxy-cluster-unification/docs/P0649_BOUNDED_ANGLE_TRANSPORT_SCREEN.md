# P0649 bounded angle-transport screen

## The change in plain language

The rejected P0647 formula measured how much two gravity vectors cancel. Near
alignment that cancellation is proportional to the *square* of their angle, so
the raw signal was tiny and a fitted multiplier kept growing.

P0649 instead measures the chord between the two unit directions. Its primary
invariant is

\[
D_\angle=
{2\sqrt{g_\star g_{\rm gas}}\over g_\star+g_{\rm gas}}
\sin{\theta\over2},
\qquad 0\le D_\angle\le1.
\]

The square-root mixing factor is one when the two component magnitudes are
equal and vanishes when either component disappears. Thus the response is
first order in a small angle, exactly zero for aligned vectors or a
one-component Solar field, and cannot grow without bound.

The full activation remains

\[
A={a_0\over a_0+g_b}
D_\angle
\left(1-e^{-\ell/(10\,{\rm kpc})}\right).
\]

There is no fitted amplitude and no setting for an individual galaxy or
cluster. The earlier universal `a0` and 10 kpc coherence length remain.

## Frozen map-screen result

The primary chord invariant passes all 11 preregistered gates without opening a
velocity or lensing outcome:

- synthetic co-centered radial activation: `1.73e-9`;
- synthetic large/small offset ratio: `35.83`;
- 13-galaxy median activation: `0.010839`;
- four-cluster median activation: `0.085496`;
- registered cluster/galaxy ratio: `7.888`;
- cluster/galaxy ratios under low, nominal, and high mass maps:
  `9.42`, `7.89`, and `7.30`;
- observed-map activation range: `2.69e-9` to `0.9929`, within its mathematical
  zero-to-one bound;
- one-component Solar activation: exactly zero; and
- rotation and translation errors below `5e-10`.

The first-order invariant deliberately increases the galaxy response relative
to the quadratic formula. The old nominal medians were `0.001127` for galaxies
and `0.021022` for clusters; the new values are about 9.6 and 4.1 times larger.
Consequently, domain separation falls from `18.66` to `7.89` but remains above
the frozen fourfold requirement. This is the tradeoff expected from replacing
a quadratic small-angle signal with a bounded linear one.

## Controls and limitations

The legacy quadratic cancellation and a bounded cross-product measure were
computed as predeclared diagnostics. They were not eligible to replace the
primary after scores were seen. The chord result authorizes exactly one next
step: use amplitude one on the already-spent RX J2129 lens and refit the same
ordinary geometry. No strength grid is allowed.

A map-screen pass does not show that the direction produces correct lens
topology, that galaxy rotation curves improve, or that photons and matter obey
the same metric. The formula is a projected spatial closure, not yet a
four-dimensional generally covariant action.

## Reproduction

```powershell
python scripts/run_p0649_bounded_angle_transport_screen.py
python -m pytest tests/test_p0649_bounded_angle_transport_screen.py -q
```
