# Sigma V19K pre-science fixture failure

## Outcome

V19K stopped before reopening either cluster science array because its mandatory
curved-step fixture failed the arc-linking gate.  This is an implementation
failure, not a source-data or physics failure.

The important positive result is that the new likelihood asks the intended
question.  The old V19J detector returned `9,230` candidate pixels in a smooth
linear gradient and `3,120` in a smooth radial profile.  V19K evaluated every
automatic seed and produced:

- `0/1,752` passing linear-gradient seeds;
- no retained arc in the smooth radial profile; and
- `64/64` passing seeds on the injected density-compression step.

Thus the explicit smooth null fixes the contour-forest statistic failure.

## Why the injected step still failed

The injected circular step produced 64 valid, significant likelihood seeds but
zero linked arcs.  The local-maximum sampler placed neighboring seeds about
30 kpc apart.  V19K had set maximum node separation to 20 kpc by directly
copying V19H's maximum *empty ridge gap*.

Those quantities are not equivalent.  Two likelihood seeds may be separated
by more than 20 kpc while the complete nonmaximum ridge between them contains
no empty pixels.  The protocol therefore discarded a continuous detected arc
because of its sparse evidence sampling.

## Decision

V19K remains frozen and its fixture failure is recorded in
`results/sigma_v19k_smooth_null_fronts/fixture_failure.json`.  No V19K science
seed, profile or likelihood was evaluated.

V19L must preserve the physical 20-kpc empty-gap rule but evaluate it along the
complete underlying V19J candidate path.  Seed proximity alone cannot define
ridge continuity.  This correction changes no source model, likelihood,
compression bound, score threshold, curvature bound or published-coordinate
rule.
