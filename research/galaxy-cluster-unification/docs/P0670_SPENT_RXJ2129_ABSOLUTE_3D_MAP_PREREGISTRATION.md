# P0670 spent RX J2129 absolute 3D map preregistration

P0670 is the bridge from a coefficient audit to an absolute field equation. It
does not reuse the empirical P0599 radial lens or fit a correction amplitude.
It prepares one physical source for the already-spent RX J2129 cluster:

- HST F160W supplies the stellar morphology;
- Chandra 0.7--2.0 keV emission supplies the hot-gas morphology through the
  same square-root emissivity/depth approximation used for P0641;
- the existing Tian `g_bar` anchor fixes the total baryonic mass inside
  200 kpc through `M_bar=g_bar R^2/G`;
- fixed 10/90 stellar/gas fractions divide that independently normalized mass;
- each projected component is lifted with a data-derived RMS scale height; and
- P0669 supplies `sigma` and the transport direction on a common 33-cell cube.

The known spent image positions are used only to mask arc light out of the HST
stellar template. No lens residual, nonlinear root, parity, multiplicity,
critical curve, or topology is computed in this stage.

Progression requires conservative 2D and 3D mass recovery, a nontrivial
positive cluster coefficient, adequate strong-lens sampling, a finite
nonconstant boundary, and zero new or per-object gravity parameters. A pass
authorizes a separately frozen scalar/tensor field solve; it is not a lensing
result.
