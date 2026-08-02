# P0655 conservative streamline deposition

## Mechanism

P0655 replaces P0652's destination-centered path average with a
source-centered conservative transport. Every cell starts with a local vector
flux calculated from the stellar/gas field mismatch. It divides that same
flux uniformly among its origin and twelve forward plus twelve backward
streamline samples. Bilinear deposition weights sum to one. Consequently, the
sum of each vector-flux component is preserved exactly while the flux is moved
through the map.

The operator uses the same Newtonian streamlines, tidal trace length, 10 kpc
coherence length, 50-to-58-arcsecond physical taper, unit amplitude, twelve
integration steps, and 48-cell zero buffer as the preceding frozen tests. It
adds no physical scale, fitted strength, or per-object gravity parameter.

## Mathematical checks pass

The source-deposition implementation behaves as designed:

- vector-flux sum relative error: `0.0`;
- integrated-source fraction: `1.96e-17`;
- maximum edge flux divided by RMS: `0.0`;
- normalized curl RMS: `2.34e-17`;
- transport changes the original flux by `80.65%` RMS; and
- the final flux retains `67.09%` of the original RMS.

Unit tests also show that reversing every streamline direction leaves the
deposited field unchanged.

## Predictive result fails

The candidate passes 13 of 17 frozen gates but fails all-root CV, both CV
improvement gates, and spent-holdout safety.

Fold 3 loses exact topology: one of thirteen fitting images lacks a root and
held-out image `3b` also lacks a root, leaving `14/15` validation roots overall.
The required pooled CV RMS is therefore infinite. Even among root-complete
folds, fold 1 is poor at `5.002 arcsec`.

The full fit recovers all `15/15` training and `7/7` spent-holdout roots, but
its heldout RMS is `3.090932 arcsec`: `70.73%` worse than the P0599 baseline and
far outside the preregistered `10%` ceiling.

No sealed P0633 or P0640 outcome was opened.

## Interpretation

Conservation is necessary for a defensible field operator, but it is not the
feature that produced P0652's attractive spent-cluster score. Source-centered
deposition changes the nonlocal spatial pattern and can concentrate the unit
field to a `4.344 arcsec` maximum, compared with `2.351 arcsec` for P0654's
padded gather field. That placement breaks both source-family topology and the
spent holdout.

P0655 rejects uniform conservative deposition at unit amplitude. Its result
does not authorize changing deposition weights, path length, support, or
amplitude after seeing the score. The next useful work is a topology diagnostic
of the already-tested fields, not another unconstrained interpolation between
failed operators.

## Reproduction

```powershell
python scripts/run_p0655_conservative_streamline_deposit.py
python -m pytest tests/test_p0655_conservative_streamline_deposit.py -q
```
