# P0675 spent RX J2129 compound raw-topology results

## Frozen result: fail

The compound field fails the same eight substantive gates as P0672:

- scalar/compound training RMS: `7.0869 / 7.0987 arcsec`;
- scalar/compound spent-heldout RMS: `17.8373 / 17.8266 arcsec`;
- compound training change: `-0.166%` (a worsening);
- compound/compact-halo heldout RMS ratio: `7.029`;
- all seven families still have missing multiplicity and exactly one root;
- no family has both parities;
- no family has a critical-curve sign change; and
- three of four ordinary nuisance parameters reach a bound.

## What the failure isolates

P0673 proved that the coefficient can become nonperturbative. P0674 proved
that its nonlinear field converges and differs measurably from scalar AQUAL.
P0675 proves that coefficient magnitude alone is insufficient: the
`I-sigma h h` constitutive orientation does not produce the required field
topology. It reduces the strong-lens RMS slightly and leaves the raw image
problem essentially unchanged.

The next candidate should not add an amplitude or further steepen the same
activation law. It should test the geometric meaning of `h`: instead of
suppressing mobility along the inferred route, a waveguide-like law can leave
the route direction open while suppressing leakage in the two transverse
directions. That change uses the same P0673 coefficient and constants, but it
requires a newly frozen field solve before any additional raw score.

RX J2129 remains entirely spent, and P0633/P0640 remain sealed.

## Reproduction

```powershell
python scripts/run_p0675_spent_rxj2129_compound_raw_topology.py
python -m pytest tests/test_p0675_spent_rxj2129_compound_raw_topology.py -q
```
