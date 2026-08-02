# P0678 spent RX J2129 required-field decomposition results

## Frozen result: integrity fail; diagnostic preserved

Fifteen of 16 integrity gates pass. The sole failure is the compact-halo curl
computed by second-order finite differences on the coarse 33-cell P0674 grid:
`0.01254`, above the frozen `1e-5` tolerance. Because the NIE deflection is
potential-derived, this requires a separately frozen step/resolution audit; the
threshold is not relaxed.

Subject to that numerical caveat, the decomposition gives a very clear target:

- reference-source reduced scalar RMS: `2.449 arcsec`;
- compact-halo RMS: `8.124 arcsec`, or `3.317x` scalar;
- external-shear RMS: `3.788 arcsec`;
- scalar+halo+shear RMS: `11.661 arcsec`, or `4.760x` scalar;
- halo monopole RMS fraction: `0.9882`;
- halo angular-residual RMS fraction: `0.1535`;
- halo/scalar vector alignment cosine: `0.9946`;
- scalar/scalar+halo/full-target critical sign-change cells: `0 / 30 / 36`;
- positive-halo-convergence `R50/R80`: `138 / 210 kpc`; and
- exact monopole-plus-angular reconstruction error: numerical zero.

The halo/scalar magnitude ratio is nearly constant across the strong-lens
annulus, falling only from roughly `3.43` to `3.27`. This is predominantly a
broad radial strength deficit, not a missing angular route.

The strongest spatial correlate of required deflection magnitude is the
baryonic tidal trace length (`Spearman rho=+0.930`). Gas surface density is the
strongest listed correlate of required positive convergence (`rho=+0.953`).
Neither predicts the *relative* missing strength well: the best ratio correlate
is compound activation at only `rho=+0.207`. Thus the present path statistic
tracks where a broad field lives better than how much extra field is needed.

No candidate, image root, or sealed target was scored.

## Next numerical audit

Evaluate the same analytic NIE deflection directly on nested grids and compute
derivatives using fixed small central-difference steps at matched physical
points. Require curl to decrease with step and remain below the frozen
tolerance away from the core before promoting the decomposition as a reliable
target specification.

## Reproduction

```powershell
python scripts/run_p0678_spent_rxj2129_required_field_decomposition.py
python -m pytest tests/test_required_field_decomposition.py tests/test_p0678_spent_rxj2129_required_field_decomposition.py -q
```
