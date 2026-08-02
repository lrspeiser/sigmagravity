# P0654 zero-padded nonlocal transport

## Question

P0652 produced the best spent-cluster cross-validation score in this branch,
but its transported flux reached the edge of the finite 121-by-121 map. P0653
forced the flux back to zero with a second physical taper; that conserved the
field but worsened the spent holdout. P0654 asks a narrower question: was the
P0652 leak only caused by an undersized numerical workspace?

The original P0652 law is left intact. The measured 121-by-121 stellar and gas
maps are surrounded by 48 zero-valued cells on every side. Forty-eight is the
existing maximum streamline length in cells, not a value selected from a lens
score. The physical activation still tapers from 50 to 58 arcseconds, the
coherence length remains 10 kpc, and the field amplitude remains exactly one.
No stars, gas, physical length, or fitted gravity parameter are added.

## Result

The larger 217-by-217 numerical domain completely fixes the finite-boundary
integral:

- integrated-source fraction: `6.37e-18`;
- maximum edge flux divided by field RMS: `0.0`;
- normalized curl RMS: `2.82e-17`; and
- all `15/15` full-training and `7/7` spent-holdout roots converge.

The spent-holdout RMS is `1.990285 arcsec`, or `9.94%` worse than P0599. That is
barely inside the frozen `10%` safety ceiling and is better than both P0652's
`12.72%` and P0653's `15.31%` worsening.

The candidate nevertheless fails. In source-family cross-validation, image
`6b` in fold 1 has no exact root under the geometry fitted without that fold.
Only `14/15` validation roots converge, so the preregistered pooled CV score is
infinite. The other four fold RMS values are `3.471`, `1.377`, `1.127`, and
`1.893 arcsec`; they cannot replace the required all-root score.

P0654 therefore passes 12 of 15 frozen gates. The failed gates are all-root CV,
improvement over lambda zero, and improvement over the matched multipole. No
sealed P0633 galaxy or P0640 cluster outcome was opened.

## What this teaches us

The conservation defect in P0652 really was numerical: a sufficiently large
zero domain removes it without a second physical taper. But the attractive
P0652 CV result is not stable to that necessary domain correction. The domain
changes the nonlocal deflection placement enough to remove one required image
root. A formula that depends on truncating its computational map is not a
defensible field law.

This closes simple boundary padding and support retuning as rescue strategies.
The next useful investigation must isolate the transport kernel or direction
that causes the fold-1 topology failure, with a new frozen mechanism and no
use of the sealed galaxy/cluster outcomes.

## Reproduction

```powershell
python scripts/run_p0654_zero_padded_nonlocal_transport.py
python -m pytest tests/test_p0654_zero_padded_nonlocal_transport.py -q
```
