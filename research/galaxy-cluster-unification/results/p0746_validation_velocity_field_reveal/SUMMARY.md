# P0746 real velocity-field reveal

Protocol/execution status: **FAIL**

This status means the frozen comparison ran with finite, leakage-audited scores;
it is not a claim that either gravity formula passed every galaxy.

- fixed_simple_mond: median/worst RMSE 47.69/71.15 km/s; median error ratio 3.25 (miss)
- newtonian_thin_sheet: median/worst RMSE 60.16/66.01 km/s; median error ratio 3.48 (miss)

- Real galaxies: 2 (validation)
- Scored velocity pixels: 31,029
- Maximum fake-twin/source prediction transport RMSE: 11.65 km/s
- Validation arrays opened: 4
- Holdout arrays opened: 0
- Gravity parameters fitted: 0
- Dark-matter parameters: 0
- Report SHA-256: `3eda055f18c948f43b55de1ae8393edcf6564d62f646190aeee5f26adccf1952`

This is a raw circular-equilibrium comparison. It does not fit pressure support, bars, warps, streaming motions, M/L, distance, inclination, MOND parameters, or dark halos.
