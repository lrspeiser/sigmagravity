# P0657 self-adjoint field-line diffusion

## Equation

P0657 tests a conservative smoothing law rather than gather or deposition:

\[
(I+L_\parallel)\,\mathbf F_{\rm out}=\mathbf F_{\rm local}.
\]

`L_parallel` is a symmetric graph Laplacian. Each grid cell connects one cell
forward and backward along the measured Newtonian direction using bilinear
weights. Link strength uses the squared, already-defined tidal trace length
and the inherited 50-to-58-arcsecond taper. The graph is unchanged if every
field direction is reversed.

The identity coefficient and final field amplitude are both fixed to one.
There is no fitted diffusion strength, new length, or per-object gravity
parameter.

## Mathematical behavior

The field behaves as intended:

- sparse graph edges: `31,403`;
- both linear solves converge;
- vector-flux sum relative error: `7.34e-13`;
- component overshoot: `0.0`;
- integrated-source fraction: `1.63e-17`;
- edge flux: `0.0`;
- normalized curl RMS: `2.03e-17`;
- flux change: `67.54%` RMS; and
- retained flux RMS: `54.96%`.

It is therefore conservative, self-adjoint, direction-symmetric, smoothing,
and nontrivial.

## Predictive result

P0657 passes 16 of 19 frozen gates. Its full fit converges all `15/15`
training and `7/7` spent-holdout roots. The spent-holdout RMS is
`1.763160 arcsec`, **2.61% better** than P0599 and materially better than the
P0652, P0653, P0654, and P0655 transport variants.

It nevertheless fails the preregistered source-family CV requirement. Fold 1
converges only `2/4` validation roots: images `1b` and `6b` have source-plane
closures `0.0307` and `0.972 arcsec`. The total is `13/15` roots, so pooled CV
RMS and both CV improvement comparisons are infinite by definition.

The only failed gates are all-root CV, improvement over lambda zero, and
improvement over the matched multipole. P0657 does not advance and no sealed
P0633/P0640 outcome is opened.

## Why a numerical audit is warranted

The exact-image solver currently runs one hybrid Newton solve from the observed
coordinate and accepts only closure below `1e-6 arcsec`. This is a valid frozen
scoring rule, but near a critical curve a single Newton basin is not a complete
test of whether a corresponding root exists. P0654's failed `6b`, for example,
stopped at `0.0018 arcsec`, while P0657's two failures are much farther away.

The next step is therefore not to relax the tolerance or tune diffusion. It is
to hold the P0657 fit fixed and try multiple standard root algorithms and
small, preregistered starting offsets at the same `1e-6` closure threshold. If
no root is recovered, the topology failure is physical within this projected
model. If roots are recovered, every comparator must be rescored with the same
improved solver before any scientific conclusion changes.

## Reproduction

```powershell
python scripts/run_p0657_field_line_diffusion.py
python -m pytest tests/test_p0657_field_line_diffusion.py -q
```
