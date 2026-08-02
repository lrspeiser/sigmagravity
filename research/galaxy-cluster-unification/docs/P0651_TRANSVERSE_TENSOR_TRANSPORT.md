# P0651 transverse-tensor transport

## Hypothesis

P0650 showed that a first-order angle invariant naturally produced the required
overall field magnitude but distributed it poorly. P0651 kept bounded unit
strength while restoring quadratic angular localization:

\[
D_\perp={2\sqrt{g_\star g_{\rm gas}}\over g_\star+g_{\rm gas}}
\sin^2\theta,
\qquad 0\le D_\perp\le1.
\]

At small angle, the legacy cancellation is approximately
`0.5 w(1-w) theta^2`, while this invariant is approximately
`2 sqrt[w(1-w)] theta^2`. Their ratio is
`4/sqrt[w(1-w)]`, or `13.33` for a 10/90 component mix. The size therefore
comes from component-vector geometry, not a fitted lambda.

## Map stage: pass

All 11 frozen baryon-map gates pass:

- radial activation: `2.49e-15`;
- synthetic large/small ratio: `33.13`;
- 13-galaxy median activation: `0.008475`;
- four-cluster median activation: `0.073242`;
- cluster/galaxy ratio: `8.642`;
- low/nominal/high mass-sensitivity ratios: `9.36`, `8.64`, and `7.40`;
- maximum activation: `0.7529`;
- one-component Solar activation: zero; and
- rotation/translation errors below `3e-11`.

The frozen protocol allowed the spent-lens stage only because every map gate
passed.

## Spent-lens stage: decisive failure

The unit-strength field recovers every root but fails both predictive gates:

- zero-field CV RMS: `2.760255 arcsec`;
- matched `m=3` control: `2.599360 arcsec`;
- transverse-tensor CV RMS: `3.188415 arcsec`;
- worsening versus zero field: `15.51%`; and
- worsening versus the multipole: `22.66%`.

Its unit deflection RMS is `0.804095 arcsec`, again the required order of
magnitude. The placement failure is even more concentrated than P0650: fold
zero reaches `5.596 arcsec`, while folds two and three are near `1.26` and
`1.12 arcsec`.

The full refit retains `15/15` training and `7/7` spent-heldout roots. Its
spent-heldout RMS is `1.95663 arcsec`, within the safety gate but not an
improvement. This descriptive split was not used to select the formula.

## Family-level conclusion

P0650 and P0651 tested two bounded, amplitude-free pointwise mappings of the
same stellar/gas vectors. Both pass strong map-domain screens and naturally
generate a field about twelve times the old unit field. Both fail
source-family lens cross-validation because local activation is placed in the
wrong regions.

That closes the simple local-mismatch family. The next candidate must be
nonlocal: response should be transported and balanced along connected field
paths or tidal principal curves, rather than assigned independently at every
map pixel. Merely changing the pointwise power, saturation, or multiplier would
repeat the rejected degree of freedom.

No P0633 or P0640 validation outcome was opened.

## Reproduction

```powershell
python scripts/run_p0651_transverse_tensor_transport.py
python -m pytest tests/test_p0651_transverse_tensor_transport.py -q
```
