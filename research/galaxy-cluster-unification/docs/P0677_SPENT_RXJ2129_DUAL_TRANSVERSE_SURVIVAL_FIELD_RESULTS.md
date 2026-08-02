# P0677 spent RX J2129 dual-transverse-survival field results

## Frozen result: fail and branch retired

The fixed two-transverse-dimension law is numerically healthy but misses both
predeclared strength gates:

- normalized residual: `5.624e-6` in 10 iterations;
- scalar/dual median physical deflection: `3.066 / 4.023 arcsec`;
- dual/scalar RMS ratio: `1.30335`, below the frozen `2.0` minimum;
- dual-minus-scalar relative RMS: `0.318425`, below `1.0`; and
- normalized deflection curl: `3.66e-16`.

Squaring the survival fraction increases the response compared with P0676,
but only from `1.15x` to `1.30x`. Repeating this with arbitrary powers would
turn the exponent into an RX J2129 strength fit. Under the frozen selection
rule, the transverse-confinement family is therefore retired and no raw image
score is computed.

The next development step should be inverse and diagnostic: on already spent
data, decompose the successful compact-halo comparator's added field relative
to the absolute baryonic scalar field. That identifies the amplitude, radial
profile, angular modes, and baryon/required-field alignment that a new
first-principles law must explain before proposing another tensor.

## Reproduction

```powershell
python scripts/run_p0677_spent_rxj2129_dual_transverse_survival_field.py
python -m pytest tests/test_p0677_spent_rxj2129_dual_transverse_survival_field.py -q
```
