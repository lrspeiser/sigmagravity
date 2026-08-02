# P0676 spent RX J2129 transverse-confinement field results

## Frozen result: fail

The field is numerically healthy but misses two predeclared strength gates:

- normalized residual: `6.705e-6` in 10 iterations;
- boundary mismatch: `0`;
- minimum constitutive eigenvalue: positive;
- scalar/confinement median physical deflection: `3.066 / 3.518 arcsec`;
- confinement/scalar RMS ratio: `1.15059`, below the frozen `1.2` minimum;
- confinement-minus-scalar relative RMS: `0.161584`, below `0.2`; and
- normalized deflection curl: `4.09e-16`.

The orientation change has the intended sign: suppressing the two leakage
directions increases rather than decreases the lensing field. Its one-pass
strength is still too small to justify another raw-image audit, so no new raw
score was computed.

The exact survival interpretation suggests one bounded follow-up with no new
constant. Because leakage can occur through either of two transverse
dimensions, require the unrouted fraction to survive both channels. That gives
the transverse eigenvalue `(1-sigma)^2` while the route eigenvalue remains
one. This fixed square is dimensional, not fitted. It must clear a newly frozen
field threshold before seeing images.

## Reproduction

```powershell
python scripts/run_p0676_spent_rxj2129_transverse_confinement_field.py
python -m pytest tests/test_p0676_spent_rxj2129_transverse_confinement_field.py -q
```
