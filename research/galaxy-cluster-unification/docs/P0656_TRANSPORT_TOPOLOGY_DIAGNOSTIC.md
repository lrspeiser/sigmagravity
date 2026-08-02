# P0656 transport topology diagnostic

## Purpose

P0656 does not fit, select, blend, or advance a model. It compares the four
already-tested transport fields at every spent RX J2129 image position using
the same unit field amplitude and the same fixed P0599 radial geometry. It asks
whether the changing lens outcomes are associated mainly with field strength,
direction, or local gradients.

For each field and image, the diagnostic records radial/tangential deflection,
convergence, shear, correction-gradient norm, and the determinant of the full
fixed lens mapping. It also compares the vector fields over a shared
111-by-111 central grid.

## Gather fields remain one family

P0652 finite gather, P0653 compact gather, and P0654 padded gather have vector
cosine correlations from `0.879` to `0.929`. At the observed image positions,
their correction RMS increases only from `0.307` to `0.336 arcsec`, while their
maximum gradient norms remain tightly grouped from `0.189` to `0.195`.
Convergence RMS stays near `0.021` and shear RMS near `0.040`.

Their main systematic change is tangential placement: tangential RMS rises
from `0.261` through `0.274` to `0.289 arcsec` as boundary handling changes.
This explains why similar-looking field audits can yield different fitted
image outcomes.

Image `6b`, the root lost by P0654 cross-validation, is already extremely near
a critical boundary under the common fixed geometry. Its full lens-map
determinant is between `-5.2e-4` and `-6.7e-4` for all three gather variants.
Small tangential changes can therefore remove a root after the geometry is
refit without requiring a globally large field gradient.

## Conservative deposition is qualitatively different

P0655 deposition has:

- image-position correction RMS `0.640 arcsec`, roughly twice the gather fields;
- convergence RMS `0.139`, about 6.6 times P0654;
- shear RMS `0.180`, about 4.5 times P0654;
- maximum local gradient norm `0.929`, about 4.9 times P0654; and
- ten negative fixed mapping determinants instead of eight.

Its largest gradient and shear occur at image `7b`. At `6b`, its correction
gradient norm is `0.273` and its fixed mapping determinant moves to `-0.0143`.
The source-centered deposition is conservative in total, but converging paths
focus flux locally. That focusing supplies a clear mechanism for its topology
and heldout failures.

## Consequence for the next equation

The useful P0652 behavior is associated with nonlocal smoothing and
tangential redistribution, not merely with conserving a total flux sum.
Conversely, a conservative operator must prevent local focusing. This points
to a self-adjoint field-aligned diffusion law: a symmetric graph/finite-volume
operator can smooth along Newtonian streamlines, conserve the global flux
exactly, remain invariant to direction reversal, and impose zero-flux
boundaries without an after-the-fact physical taper.

That is a new mechanism and must receive its own frozen unit-amplitude test. No
interpolation weight may be selected from P0656. No sealed P0633 or P0640
outcome was opened.

## Reproduction

```powershell
python scripts/run_p0656_transport_topology_diagnostic.py
python -m pytest tests/test_p0656_transport_topology_diagnostic.py -q
```
