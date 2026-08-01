# P0640: real cluster baryons and opaque raw-lensing constraints

P0640 acquires the permitted inputs for the four untouched RELICS clusters in
P0633 without spending the lensing validation set.

## Open baryonic evidence

For each cluster, the repository now contains the official RELICS F160W mosaic,
its pixel-matched segmentation image, and the full multiband source catalog.
These products preserve the measured positions, shapes, overlap, asymmetry, and
relative F160W light of cluster galaxies and the BCG. They replace circular or
radial member-light proxies.

The X-ray morphology comes from every relevant public Chandra level-2 central
count image found by an archive-coordinate query: 19 observations totaling
roughly 632 ks. Counts divided by exposure constrain morphology only. A
published gas mass and uncertainty will set normalization; X-ray brightness is
not directly equated with projected gas mass.

## Blind raw-lensing evidence

The coordinate-bearing Cibirka et al. source package for Abell S295,
MACS J0025.4-1222, and MACS J0159.8-0849 and the D'Addona et al. multiple-image
catalog for PLCK G287.0+32.9 are downloaded into the P0633 sealed directory.
Acquisition records only their URL, byte count, and SHA-256 hash. The source
archive is not listed or extracted and the PLCK catalog is not parsed before
candidate lock.

Published convergence, deflection, magnification, shear, and critical-curve
maps were deliberately not downloaded. Those are model outputs, not baryonic
inputs or raw image constraints.

## Reproduce

```powershell
$env:PYTHONPATH='src'
python scripts/download_p0640_relics_inputs.py
python scripts/audit_p0640_relics_inputs.py
python -m pytest tests/test_p0640_relics_input_acquisition.py -q
```

The candidate equation, universal parameter vector, field-solver gates, and
prediction manifest must still be committed and hashed before the sealed files
can be opened.
