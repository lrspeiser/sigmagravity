# P0644 spent RX J2129 accumulated-tensor result

## Exact lens operator

P0644 is the first raw-lensing use of the P0643 activation.  It does not move a
light template or rotate a fitted shear.  It applies the baryon-derived tensor
to the P0599-minus-baryons lens potential:

\[
\nabla^2\delta\psi=
\nabla\cdot\left[
A(\boldsymbol\theta)\hat{\mathbf h}\hat{\mathbf h}
\nabla\psi_{\rm carrier}\right],
\]

where

\[
A={a_0\over a_0+g_b}
C_{\rm cancel}\left(1-e^{-\ell/(10\,\mathrm{kpc})}\right).
\]

The Fourier Poisson solve makes `delta psi` a scalar potential.  The measured
normalized curl is `2.79e-17`, and the integrated source is zero to
`7.50e-18` of its absolute integral.  One universal non-negative strength
`lambda` multiplies the resulting deflection.

The control is not artificially spherical.  It refits the same six ordinary
pseudo-elliptical geometry and external-shear parameters used in P0601.

## Result

The preregistered experiment fails 2 of 8 progression gates.

With the old zero-tensor geometry held fixed:

| lambda | training RMS | spent-heldout RMS | roots (train/heldout) |
|---:|---:|---:|---:|
| 0 | 0.4952" | 1.8087" | 15/7 |
| 0.5 | 0.5069" | 1.6687" | 15/7 |
| 1 | 0.5234" | 1.5723" | 15/7 |
| 2 | 0.5677" | 1.4157" | 15/7 |
| 5 | 0.7554" | 1.1399" | 15/7 |
| 10 | 1.2472" | undefined | 15/6 |
| 20 | undefined | undefined | 14/3 |

The legal training-only selection is therefore `lambda=0`.  Its exact refit is
statistically the P0599 baseline: 0.4952 arcsec on training and 1.8131 arcsec
on the spent holdout.  The required 0.5-percent training gain is absent, and
large strengths violate complete-root safety.

## Why the downward spent-holdout curve is not a success

The seven spent holdout images improve monotonically through `lambda=5`, by
about 37 percent, while the training images worsen.  It would be easy but
invalid to choose five from that curve.  Those images have been inspected in
many prior stages and cannot select a universal constant.

The opposite trends identify a fair-comparison problem worth testing: the six
ordinary geometry parameters were optimized for `lambda=0` and then frozen.
Adding a structured field without letting the lens geometry readjust gives the
baseline an optimization advantage.  P0644 intentionally records that failure;
it does not alter the selection after seeing it.

## Additional limitation visible in the map

The spent RX J2129 inputs provide registered F160W and X-ray *proxies*, not the
physical mass maps now available for the four sealed clusters.  The cancellation
map emphasizes sharp edges and places where only one proxy is nonzero.  Some of
the visible ring and pixel-scale structure can therefore reflect mask filling,
background subtraction, or the different point-spread functions.  A fair
follow-up must smooth both components to a common physical resolution and test
the 10/90 mass-fraction assumption.

## Correct follow-up

P0645 should:

1. convolve stellar and gas maps to common physical resolutions;
2. refit the same six conventional lens parameters for every fixed `lambda`;
3. select `lambda` by leaving entire source families out, never individual
   images from the same family;
4. compare against an added generic multipole with the same one-parameter
   budget; and
5. transfer the selected value unchanged to other spent clusters.

Only then can a constant and a field-map recipe be frozen for P0640.

## Reproduction

```powershell
python scripts/run_p0644_spent_rxj2129_accumulated_tensor.py
python -m pytest tests/test_p0644_spent_rxj2129_accumulated_tensor.py -q
```
