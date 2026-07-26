# Cosmicflows-4 density grids

This directory contains the grouped (`CF4gp`) and ungrouped (`CF4`) 64^3 density
contrast reconstructions, their published 2-D error products, and the
Cosmicflows-4 group catalog used to validate the grid convention. They are
downloaded directly from the official Cosmicflows and CDS/VizieR archives by
`scripts/download_cosmicflows4.ps1`.

The official 2026 Zenodo release of the ungrouped 128^3 grid is also included
as a sensitivity and convention check (DOI: 10.5281/zenodo.20653238). Its
documented 1000 h^-1 Mpc box has the same 7.8125 h^-1 Mpc voxel scale as the
original 64^3, 500 h^-1 Mpc release.

Primary citation: Courtois et al., "Gravity in the Local Universe: Density and
velocity fields using CosmicFlows-4," *Astronomy & Astrophysics* 670, L15
(2023), DOI: 10.1051/0004-6361/202245331.

These grids are independent of the SPARC rotation-curve residuals. The grouped
grid is the preregistered primary environment reconstruction; the ungrouped grid
is a sensitivity check. Do not infer or tune a void score using SPARC observed
velocities.

The FITS headers do not contain a WCS. `scripts/build_cf4_environment.py` uses
the official page's `(SGZ, SGY, SGX)` FITS-axis declaration, accounting for the
axis reversal when Astropy reads FITS into NumPy as `(SGX, SGY, SGZ)`. It also
checks the sky-to-supergalactic transform against all 38,053 catalog rows and
records the result.
