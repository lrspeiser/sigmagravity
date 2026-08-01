# P0632 published MOND/RAR simulator replication

**Published benchmark replication: PASS**

- Recovered the published sample exactly: 153 galaxies and 2694 points.
- Replayed Li et al. nuisance-fit scatter: 0.057161 dex versus 0.057 dex published.
- Recalculated versus published per-galaxy reduced chi-square correlation: 0.999976.
- Fixed-input RAR/MOND scatter: 0.132766 dex versus approximately 0.13 dex published.
- Strict 23-galaxy holdout velocity RMSE: 23.326 km/s.

The first two comparisons reproduce the paper. The whole-galaxy holdout is our stricter simulator diagnostic: it fixes the published acceleration scale, stellar mass-to-light ratios, catalog distance, and catalog inclination, with no per-galaxy fit.

This validates the algebraic circular-orbit MOND/RAR plugin. It does not validate a complete AQUAL/QUMOND field solver, external-field effect, relativistic MOND lensing theory, or galaxy formation simulation.
