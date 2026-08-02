# P0652 nonlocal streamline transport

## First genuinely nonlocal operator

P0652 transports the bounded transverse component flux along the measured
Newtonian field rather than assigning its effect independently at each pixel.
For every cell, the code traces both forward and backward over

\[
\ell={|\mathbf g_N|\over\|\nabla\mathbf g_N\|_F}
\]

and symmetrically averages the sampled vector flux. Reversing the field
direction leaves the result unchanged. The existing tidal length supplies the
integration distance, so the operator adds no physical length or fitted
amplitude. Twelve integration steps are a numerical resolution setting.

## Predictive result

The sole advancing closure produces the strongest fair CV result in this
research branch:

- lambda-zero CV RMS: `2.760255 arcsec`;
- matched `m=3` multipole: `2.599360 arcsec`;
- local transverse tensor from P0651: `3.188415 arcsec`;
- nonlocal path average: `2.075148 arcsec`;
- improvement versus zero field: `24.82%`; and
- improvement versus matched multipole: `20.17%`.

All `15/15` CV roots converge. The path average changes the local flux by
`75.3%` RMS and retains `69.0%` of its flux RMS, so this is a material nonlocal
placement change rather than a numerical no-op. Fold zero falls from `5.596`
in the local P0651 field to `2.833 arcsec`; fold one falls from `3.180` to
`2.465 arcsec`.

The predeclared diagnostic `transported - local` closure scores
`2.781367 arcsec`, showing that the useful signal is the transported flux, not
the residual redistribution alone. The diagnostic was never eligible to
replace the primary.

## Why P0652 does not advance yet

Two of twelve gates fail:

1. The full-refit, already-spent heldout RMS is `2.040761 arcsec`, `12.72%`
   worse than P0599 and beyond the frozen 10% safety margin.
2. The transported flux has a `0.0495` integrated-source fraction, far above
   the `1e-4` conservation gate.

The second failure has an identifiable numerical mechanism. Local activation
was compactly tapered at 58 arcseconds, but path averaging lets map cells
outside that radius sample inward. On the finite grid, some transported flux
then reaches the computational boundary. The Poisson solver removes the zero
mode, but relying on that implicit uniform compensation is not an acceptable
field law.

## Authorized correction

P0652 itself remains failed. A new frozen test may reapply the **same existing
compact taper** after path transport, before taking the divergence. This adds
no scale or fit and forces the transported flux to vanish smoothly at the
declared support boundary. It must rerun all exact folds and the heldout safety
gate. If conservation is restored but predictive performance disappears, the
nonlocal result is rejected rather than retuned.

No P0633 or P0640 validation outcome was opened.

## Reproduction

```powershell
python scripts/run_p0652_nonlocal_streamline_transport.py
python -m pytest tests/test_p0652_nonlocal_streamline_transport.py -q
```
