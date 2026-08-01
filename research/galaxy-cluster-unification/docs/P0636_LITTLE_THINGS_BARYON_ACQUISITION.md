# P0636: sealed LITTLE THINGS baryonic-input acquisition

## Outcome

All 13 galaxies frozen in P0633 now have their permitted baryonic inputs in the
repository through Git LFS:

- one primary-beam-corrected natural-weight H I moment-0 map;
- one B-band optical image;
- one V-band optical image;
- one UBV photometric-calibration record.

That is 52 products and 300,811,128 bytes. Every file has a source URL, byte
count, and SHA-256 in
`results/p0636_little_things_baryon_acquisition/provenance.json`.

## Readiness checks

All 39 FITS images are finite two-dimensional arrays. Every radio image reports
the expected Jy/beam m/s unit and has a recoverable CLEAN beam, including files
that store beam metadata only in AIPS HISTORY cards. Every calibration record
contains both the B- and V-band photometric transformations.

No failed object was replaced and no target was scored.

## Sealed boundary

The downloader rejects filenames containing cube, moment-1, moment-2,
velocity, dispersion, rotation, or rotmod markers. The acquisition therefore
contains no H I cube, velocity field, velocity-dispersion field, published
circular velocity, or P0633 outcome. Those products remain sealed until a
candidate equation, universal parameter hash, prediction manifest, and solver
gates are committed.

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0636_little_things_baryons.ps1
$env:PYTHONPATH='src'
python scripts/audit_p0636_little_things_baryons.py
python -m pytest tests/test_p0636_little_things_baryon_acquisition.py -q
```

Source: <https://science.nrao.edu/science/surveys/littlethings/data>.
