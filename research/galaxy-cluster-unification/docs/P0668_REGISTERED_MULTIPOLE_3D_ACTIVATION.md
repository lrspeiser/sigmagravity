# P0668 registered multipole 3D activation

## Frozen question

P0668 asked whether the P0667 scale-free baryonic multipole gate transfers to
all 13 registered galaxy maps and all four registered cluster maps before any
kinematic or lensing outcome is opened. Each projected stellar and gas map was
conservatively resampled to a common 33-cell grid, lifted with a deterministic
sech-squared vertical profile, and evaluated under three mass scenarios.

The tested coefficient was unchanged from P0667:

\[
\sigma_{\rm final}=\left[1-e^{-(D^2+Q^2)}\right]\sigma_{\rm local}.
\]

It has no per-object gravity parameter and introduces no new universal
constant after P0659.

## Frozen result: fail

The candidate fails exactly one preregistered gate:

- nominal galaxy median `sigma`: `3.70453e-6` (passes the `0.001` maximum);
- nominal cluster median `sigma`: `5.57686e-4` (fails the `0.001` minimum);
- nominal cluster/galaxy ratio: `150.54x` (passes the `10x` minimum);
- weakest ratio across all mass scenarios: `129.32x` (passes the `5x`
  minimum); and
- cluster/galaxy multipole-gate ratio: `2.5324x` (passes).

Thus the geometry distinguishes the two registered domains strongly, but the
physical-length local factor suppresses the absolute cluster response too
much. The failure is not repaired by lowering the threshold or refitting a
coherence length after seeing the outcome.

## What remains usable

The dimensionless dipole/quadrupole relationship is still a useful measured
feature: its cluster/galaxy separation survives all declared mass scenarios.
It may be reused only inside a genuinely different activation law that is
frozen before its registered-map score is computed.

## Claim boundary

P0668 evaluates a constitutive coefficient, not a nonlinear gravity solution
or a lensing prediction. The sech-squared lift is deterministic rather than a
native three-dimensional baryonic likelihood. No spent RX J2129 lensing image,
sealed galaxy velocity, or sealed cluster lensing constraint was opened.

## Reproduction

```powershell
python scripts/run_p0668_registered_multipole_3d_activation.py
python -m pytest tests/test_p0668_registered_multipole_3d_activation.py -q
```
