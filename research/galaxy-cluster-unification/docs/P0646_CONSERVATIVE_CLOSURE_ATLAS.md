# P0646 conservative closure atlas

## Fixed part of the hypothesis

P0646 does not retune the P0643 activation.  Every closure uses

\[
A(\mathbf x)={a_0\over a_0+g_b}
C_{\rm cancel}\left[1-e^{-\ell/(10\,{\rm kpc})}\right]
\]

on the same 25 kpc common-resolution stellar and X-ray maps.  Only the vector
or tensor that places the accumulated response is changed.  Every final
deflection is obtained from a scalar Poisson solve and is numerically curl-free.

The atlas includes three positive-semidefinite rank-one tensors, an isotropic
control, and four signed Helmholtz-projected component fluxes.

## Result

The experiment is a scientifically useful **boundary failure**: 9 of 10 gates
pass, but the best strength is the largest value tested.

The leading closure is

\[
\nabla^2\delta\psi=\nabla\cdot\left[
A(\mathbf x)|\nabla\psi_{\rm carrier}|
(\hat{\mathbf g}_{\rm gas}-\hat{\mathbf g}_{\star})_{\rm unit}
\right].
\]

At `lambda=5` it gives:

- audited zero-field CV RMS: `2.760255 arcsec`;
- selected CV RMS: `2.622874 arcsec`;
- improvement over zero field: `4.977%`;
- best matched isotropic-control RMS: `2.744999 arcsec`;
- improvement over the isotropic control: `4.449%`;
- complete CV roots: `15/15`;
- full-refit training RMS: `0.471891 arcsec`;
- descriptive spent-heldout RMS: `1.850970 arcsec`, only `2.24%` worse than
  the P0599 comparison; and
- full roots: `15/15` training and `7/7` spent heldout.

The field has normalized curl below `3e-17` and zero integrated source to
floating-point precision.

## Why the sign result matters

The gas-minus-star direction improves as lambda rises from 0.5 through five in
the one-start screen.  The exact opposite, star-minus-gas direction, loses an
exact root at lambda two and five.  The clockwise perpendicular flux is weaker
and loses a root in the stage-two replay.  This sign specificity is evidence
that the effect is about *where* the flux is placed, not merely the amount of
extra radial convergence.

It is not proof of new gravity.  A mismatch between the effective stellar and
X-ray point-spread functions, foreground contamination, or ordinary unmodeled
mass could also prefer one registered component direction.

## Baseline correction

During the first stage-two run, a new three-start zero-field optimizer chose a
lower profiled cost whose exact root solver subsequently lost roots.  Treating
that accidental infinity as the physical baseline would have manufactured a
100-percent improvement.  The final report instead uses the identical,
root-complete P0645 zero-field folds (`2.760255 arcsec`), which were already
audited before P0646.  The rerun reproduces the closure scores and the corrected
4.977-percent gain.

## Why this does not advance yet

`lambda=5` is the upper boundary of the frozen grid.  A boundary optimum does
not identify a universal constant.  The formula remains ineligible for P0640
until a targeted expansion finds an interior minimum with all roots.

The next frozen test should vary only gas-minus-star lambda above and around
five.  If the score continues improving to the next boundary, root topology
fails, or the optimum is unstable across smoothing and mass fractions, the
closure is rejected.  If an interior optimum survives, it must then beat a
generic one-parameter multipole and transfer unchanged to other spent clusters.

## Reproduction

```powershell
python scripts/run_p0646_conservative_closure_atlas.py
python -m pytest tests/test_p0646_conservative_closure_atlas.py -q
```
