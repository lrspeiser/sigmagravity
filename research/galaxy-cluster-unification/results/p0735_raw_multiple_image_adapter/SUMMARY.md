# P0735 summary

The raw multiple-image adapter passes all frozen P0735 gates. It projects an
explicitly typed 3D photon field, profiles two source-position nuisance
coordinates per family, globally finds roots, performs one-to-one assignment,
and keeps raw image-position residuals in their own arcsecond score channel.
No gravity parameter is added. Missing predicted multiplicity produces an
`incomplete_topology` state and null aggregate fit statistics.

The AS295/PLCKG287 import audit preserves all 65 secure images and 18 families,
but the P0713/P0714 parsed catalog contains no published positional
uncertainties. The platform refuses to invent them, so those real systems are
coordinate-ready but not yet likelihood-ready.

See `report.json` and `../../docs/P0735_RAW_MULTIPLE_IMAGE_ADAPTER.md`.

