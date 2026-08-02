# P0664 registered tensor field solve

## Projected field construction

P0664 moved from coefficient screens to actual nonlinear field solutions. For
each registered baryonic map it first computed the thin-sheet Newtonian
potential `Phi_N`, then used the same variational graph to define

\[
-L\!\left[\mu\big((1-\sigma)I+\sigma nn\big)\right]\Phi
=-L[I]\Phi_N.
\]

The outer boundary is fixed to `Phi_N`. This source construction has consistent
projected units and recovers `Phi_N` when `mu=1` and `sigma=0`. Each map was
solved twice: ordinary scalar AQUAL and the tensor candidate. The comparison
therefore isolates the new tensor term without using an observed target.

## Frozen result

All 17 gates pass across 13 galaxies and four clusters:

- constant-`mu` Newtonian recovery error: `1.16e-11`;
- rotation covariance error: `2.23e-12`;
- maximum nonlinear residual: `9.65e-6`;
- minimum constitutive eigenvalue: `8.18e-5`;
- maximum normalized acceleration curl: `2.82e-16`;
- galaxy median tensor field effect: `0.0451%`;
- galaxy maximum tensor field effect: `0.346%`;
- cluster median tensor field effect: `3.067%`;
- cluster minimum tensor field effect: `1.544%`; and
- cluster/galaxy median field-effect ratio: `68.055x`.

All 34 registered nonlinear solves converged. There are no object-specific
gravity settings and no new universal constant after P0659.

## Interpretation

This is the strongest structural result in the tensor branch so far. The same
equation behaves almost exactly like scalar AQUAL on the dwarf galaxy maps but
produces a measurable anisotropic response on every cluster map. The response
tracks baryonic component geometry rather than an object label.

It is not yet evidence that the response has the correct **shape** or
**magnitude** for cluster lensing. A three-percent field change may be too small,
or it may be topologically important near critical curves. The next authorized
test must use already-spent lensing systems to check roots, image parity, and
compact-halo gaps before any untouched P0640 cluster is opened.

The scalar AQUAL solutions amplify galaxy RMS fields by a median factor of
`7.14` and cluster fields by `2.07`; those are field-space diagnostics, not
rotation-speed or lensing fits.

## Claim boundary

The calculation is projected, nonrelativistic, and tied to thin-sheet source
maps. It does not supply a photon metric, a covariant action, or PPN
predictions. No P0633 velocity or P0640 lensing target was opened.

## Reproduction

```powershell
python scripts/run_p0664_registered_tensor_field_solve.py
python -m pytest tests/test_registered_tensor_field.py tests/test_p0664_registered_tensor_field_solve.py -q
```
