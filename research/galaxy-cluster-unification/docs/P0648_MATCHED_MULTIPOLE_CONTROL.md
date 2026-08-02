# P0648 matched multipole control

## Why this control was required

The gas-minus-star flux produced an attractive cross-validation score before
P0647 rejected its unbounded strength. A lens can improve simply because the
baseline lacks some angular structure. P0648 asks whether conventional angular
potential terms can purchase the same gain with one amplitude.

The controls are curl-free `m=3` and `m=4` scalar-potential multipoles. Their
phase is fixed to the gas-centroid minus stellar-centroid axis measured before
P0648; their 30-arcsecond radial scale, 50--58-arcsecond taper, and support are
fixed; and each unit field has exactly the same `0.0682839 arcsec` whole-grid
deflection RMS as the P0646 transport field. Only a signed amplitude changes.

## Exact result

The best three-start, root-complete control is `m=3` at amplitude `-12.5`:

- lambda-zero CV RMS: `2.760255 arcsec`;
- matched `m=3` CV RMS: `2.599360 arcsec`;
- rejected P0647 boundary-field CV RMS: `2.308115 arcsec`; and
- transport-field improvement over the best exact multipole: `11.20%`.

Under the frozen one-percent rule, these fixed one-amplitude multipoles do not
explain the entire transport-field gain. The baryonic component geometry is
therefore more informative on this spent lens than these two simple angular
bases.

That is a specificity result, not a validation result. The `m=3` control also
runs to the negative endpoint, so its amplitude is not identified. The P0647
transport strength remains rejected because it ran to its own endpoint and was
root-unstable.

## Optimizer and topology lesson

The one-start screen scored `m=3, amplitude=-12.5` at `2.02427 arcsec`; the
required three-start exact replay scored it at `2.59936 arcsec`, with all roots
in both cases. The lower optimizer cost selected by additional starts does not
guarantee a lower exact image-plane RMS after source profiling. This reinforces
the project rule that a cheap screen cannot establish a claimed gain.

Positive `m=3` amplitudes and several `m=4` amplitudes lose image roots in the
screen. The exact selected `m=3` full refit recovers `15/15` training and `7/7`
already-spent heldout roots. Its spent-heldout RMS is `1.04414 arcsec`, but that
split was not used for selection and is not independent validation.

## Numerical field audit

Both controls match the unit RMS to floating-point precision and have curl near
`1e-16`. The `m=3` source integral is compensated to `1.6e-17`. The unselected
`m=4` field has a `2.66e-4` discrete source-integral fraction and fails the
frozen `1e-8` numerical gate. That makes `m=4` unsuitable as a precision control
in this implementation, though it does not affect the selected `m=3` result.

## Claim boundary and next move

P0648 compares only fixed-phase `m=3/4` controls; it does not exhaust flexible
lens mass models, free multipole phases, member-galaxy mass freedom, or
line-of-sight structure. No P0633 galaxy velocity or P0640 cluster-lensing
outcome was opened.

The remaining research path should not revive the unbounded multiplier. A new
candidate must make the response finite through geometry or field dynamics and
derive its magnitude from an invariant. It should also be tested against a
numerically compensated generic control before any blind unsealing.

## Reproduction

```powershell
python scripts/run_p0648_matched_multipole_control.py
python -m pytest tests/test_p0648_matched_multipole_control.py -q
```
