# V19CL registered cluster morphology acquisition

V19CL applies the V19CJ rules exactly as frozen before the catalog values were read. It uses two published X-ray morphology tables and only normalized catalog-name equality against the 162 clean V19CH cluster candidates. Coordinates, lensing multiplicities, halo properties, Sigma residuals, and gravity parameters are not used.

The primary Yuan–Han catalog contains 964 Chandra clusters. Twenty-four clean V19CH candidates match one-to-one; 18 have finite morphology index, uncertainty, concentration, centroid shift, and power ratio values. Their internally frozen medians are `logc=-0.65`, `logw=-1.76`, and `log(P3/P0)=-5.71`.

The finite primary set contains three multimetric-confirmed relaxed systems, ten multimetric-confirmed disturbed systems, and five discordant extremes. The discordant cases have a securely disturbed signed morphology index but at least two of the three independent structural indicators point in the other direction. They are retained as a deliberate complexity stratum rather than relabeled.

This satisfies the preregistered minimum morphology-pool gate. It does not admit a final cluster. The next selection stage must use only the already declared source-side axes: redshift, SZ/source-side mass, gas and BCG concentration, single-core versus multipeak layout, and source imaging/spectroscopy completeness. Raw lensing targets remain sealed until a balanced final sample has been fixed.

The secondary Zhang et al. Planck/LOFAR catalog is retained as an independent concentration/centroid-shift cross-check. Exact catalog-name matching produces no same-candidate overlap with the primary matched set, so V19CL does not manufacture aliases using positions or lensing information.

This result increases the severity of the future cluster test: one frozen field law must reproduce image roots and shear geometry in relaxed, disturbed, and structurally discordant baryonic configurations. Solar-System consistency remains a later hard veto rather than a present tuning objective.
