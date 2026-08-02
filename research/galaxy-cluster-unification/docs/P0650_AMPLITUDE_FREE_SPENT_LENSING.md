# P0650 amplitude-free spent-lensing result

## Frozen test

P0650 gave the P0649 bounded angle field no adjustable strength. The candidate
was fixed to

\[
\nabla^2\delta\psi=
\nabla\cdot\left[A_\angle
|\nabla\psi_{\rm carrier}|
\widehat{(\hat{\mathbf g}_{\rm gas}-\hat{\mathbf g}_\star)}\right]
\]

with amplitude exactly one. The same five source-family folds and six ordinary
lens parameters were refit in every fold. There was no strength grid and the
already-spent descriptive heldout split did not select anything.

## Result: reject this closure

The formula recovers all `15/15` CV image roots but fails its two predictive
gates:

- zero-field CV RMS: `2.760255 arcsec`;
- bounded-angle CV RMS: `2.837492 arcsec`, a `2.80%` worsening; and
- matched `m=3` control: `2.599360 arcsec`, making the bounded field `9.16%`
  worse.

The field remains conservative to numerical precision, its activation is
bounded, its strength is exactly one, and the one-component Solar activation is
zero. The failure is not a numerical or topology failure; it is a predictive
failure on spent data.

## What was learned about magnitude versus placement

The bounded invariant did solve the arbitrary-size problem. On RX J2129 its
unit deflection RMS is `0.782537 arcsec`, compared with `0.068284 arcsec` for
the old quadratic field—a natural amplification of about `11.46`. That is close
to the rejected P0647 boundary strength of `12.5`.

Yet the spatial weighting is wrong. Fold RMS values are:

| Fold | Validation RMS (arcsec) |
|---:|---:|
| 0 | 3.9310 |
| 1 | 3.8233 |
| 2 | 1.4914 |
| 3 | 1.1663 |
| 4 | 1.4773 |

The first-order angle helps some source families and strongly hurts others.
Replacing a quadratic mismatch by a linear one changed not only total strength
but where the response concentrates around the component maps. A viable
geometric law therefore needs a transport/placement principle that generalizes
across source locations; deriving the overall magnitude is insufficient.

## Why the full heldout result does not rescue it

The full refit has `0.446691 arcsec` training RMS and `1.621474 arcsec` on the
already-spent seven-image heldout set, a `10.4%` descriptive improvement over
P0599. That split was not used for selection because it has already influenced
earlier development. The source-family CV result is the frozen decision metric,
so the closure remains rejected.

## Next research direction

Do not fit another multiplier to the chord field. A next candidate must change
placement while retaining bounded unit magnitude. The fold pattern suggests
testing a conservative geometric redistribution that follows finite field
lines or tidal principal directions and explicitly balances positive and
negative source over each connected path. It must be frozen against the same
folds and matched multipole before any blind outcome is opened.

No P0633 galaxy velocity or P0640 cluster-lensing outcome was opened.

## Reproduction

```powershell
python scripts/run_p0650_amplitude_free_spent_lensing.py
python -m pytest tests/test_p0650_amplitude_free_spent_lensing.py -q
```
