# Sigma v3D post-failure protocol

The preregistered Sigma v3D structural test failed: its primary median
distributed-to-compact response ratio was `1.1167`, compared with the frozen
minimum of `10`, and its `M=3` ratio was `0.9599`, compared with the per-mass
minimum of `2`.  All algebra, screening, and resolution gates passed.

The v3D decision is therefore fixed as **retired**.  The diagnostics in
`configs/sigma_v3d_post_failure_diagnostics.json` cannot rescue it and will not
be called validation.  They vary only four structural choices on the same
synthetic equal-mass fixtures:

- screen power `2`, `4`, or `8`;
- memory length `0.5`, `1`, or `2` fixture units;
- applying the screen before or after the static Helmholtz memory; and
- integrating within half-widths `0.5`, `1`, `1.5`, or `2` fixture units.

The diagnostic asks which ingredient controls the failure.  A factor-three
change defines material sensitivity.  An isolated ratio above ten is treated
as fragile if fewer than half of its neighboring grid choices also exceed ten.
No observational data are opened, no empirical parameter is fitted, and no
third raw-holdout failure is counted.
