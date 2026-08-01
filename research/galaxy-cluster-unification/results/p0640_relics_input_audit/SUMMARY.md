# P0640 RELICS baryon and sealed-lensing input audit

- Status: **READY**
- Untouched clusters: 4
- Hashed artifacts: 34 (876,237,350 bytes)
- HST catalog objects: 12,020
- Strict photometric members: 1,230
- Chandra observations: 19 (632.4 ks)
- Opaque sealed constraint containers: 2
- Derived lens maps downloaded: `false`
- Sealed constraint contents opened: `false`

The open inputs are real HST F160W pixels, their matched source segmentation,
photometric member catalogs, and Chandra level-2 count maps. The raw
multiple-image tables are present only as byte-counted, SHA-256-hashed sealed
artifacts; this audit never parses or extracts them.
