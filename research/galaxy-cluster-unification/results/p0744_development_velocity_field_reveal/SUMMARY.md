# P0744 real development velocity-field reveal

Protocol/execution status: **PASS**

This status means the frozen comparison ran with finite, leakage-audited scores;
it is not a claim that either gravity formula passed every galaxy.

- fixed_simple_mond: median/worst RMSE 26.16/34.00 km/s; median error ratio 1.99 (close)
- newtonian_thin_sheet: median/worst RMSE 41.07/49.70 km/s; median error ratio 3.14 (miss)

- Real galaxies: 4
- Scored velocity pixels: 76,182
- Maximum fake-twin/source prediction transport RMSE: 5.75 km/s
- Validation arrays opened: 0
- Holdout arrays opened: 0
- Gravity parameters fitted: 0
- Dark-matter parameters: 0
- Report SHA-256: `d3feb859f9d50e7a9ecefd80c16f0ed6cee1d0f0ccb6ded307b09c21e84ee9d0`

This is a raw circular-equilibrium development comparison. It does not fit pressure support, bars, warps, streaming motions, M/L, distance, inclination, MOND parameters, or dark halos.
