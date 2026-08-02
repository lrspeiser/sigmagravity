# P0670 spent RX J2129 absolute 3D map results

## Frozen result: pass

All 20 physical-map progression gates pass:

- baryonic mass inside the exact 200 kpc anchor: `7.88563e12 Msun`;
- component surface/volume mass errors: `2.22e-16 / 8.88e-16`;
- stellar/gas scale heights: `109.31 / 109.07 kpc`, derived from the
  projected RMS radii;
- multipole power/amplitude gates: `0.016476 / 0.128357`;
- mass-weighted tensor `sigma`: `0.00485906`;
- maximum local `sigma`: `0.0405553`;
- minimum constitutive eigenvalue proxy: `0.0173997`; and
- largest declared strong-lens radius: `5.45` grid cells.

The map contains a complete finite-grid simple-MOND boundary and the P0669
transport direction. No empirical radial lens, fitted gravity amplitude, raw
lens score, root, parity, multiplicity, or topology entered the result.

## Interpretation

This is the first project artifact in this branch that can feed an absolute
three-dimensional scalar/tensor field comparison. Earlier RX J2129 experiments
used morphology as a zero-monopole correction to a fitted radial lens. P0670
instead fixes the source mass from an independent baryonic acceleration anchor
and stores the physical density in SI units.

The limitations are substantial: F160W and Chandra maps are morphology proxies,
the 10/90 component split is shared rather than inferred, the depth model is
approximate, and 14.03 kpc cells provide only coarse strong-lens sampling. A
separately frozen solve must therefore test convergence, scalar/tensor
difference, zero-slip normalization, and resolution sensitivity before raw
image topology can be scored.

## Reproduction

```powershell
python scripts/run_p0670_spent_rxj2129_absolute_3d_map_build.py
python -m pytest tests/test_p0670_spent_rxj2129_absolute_3d_map_build.py -q
```
