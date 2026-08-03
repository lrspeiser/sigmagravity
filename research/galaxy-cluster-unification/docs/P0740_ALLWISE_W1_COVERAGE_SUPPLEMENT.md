# P0740: uniform AllWISE W1 coverage supplement

P0740 acquired wider 3.4-micron stellar-light coverage for the same eight-galaxy development/validation/holdout sample. It uses the official [AllWISE Atlas archive](https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/) and its documented [programmatic cutout interface](https://irsa.ipac.caltech.edu/ibe/cutouts.html).

The tile IDs, cutout centers, cutout sizes, products, and 99% header-only WCS union gate were frozen before any WISE pixel array was opened. Every W1 tile returned by the metadata query was retained; tiles were not selected by how their images looked.

Result: **PASS**.

- 8 systems across the unchanged 4/2/2 split
- 14 frozen W1 coadds
- 28 intensity and uncertainty FITS cutouts
- 108,662,400 bytes with SHA-256 hashes
- 99.19% minimum header-only union footprint across the requested square
- zero development, validation, or holdout pixel arrays opened during acquisition
- zero velocity or dispersion targets opened
- zero gravity parameters

The WCS footprint statistic samples equal-area cell centers. Sampling the exact mathematical boundary of an angular cutout is inappropriate because the archive rounds cutouts to whole detector pixels; a zero-area edge can otherwise be reported as one detector pixel short.

P0740 does not validate a theory. It only establishes that the raw stellar-light coverage exists. During development, WISE must still be background-subtracted, foreground-cleaned, cross-calibrated against valid SINGS overlap, mosaicked, beam-matched, and subjected to uncertainty sensitivity tests. Validation and holdout pixel arrays remain sealed.

Reproduce with:

```powershell
python scripts/acquire_p0740_allwise_w1_supplement.py
```

