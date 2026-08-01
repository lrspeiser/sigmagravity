# P0641: registered physical cluster baryon maps

P0641 converts the open P0633 cluster inputs into physical projected mass maps
without reading any lensing target.

## Stellar component

The member rule is the same for every cluster: galaxy-like morphology, a
positive five-sigma F160W detection, a BPZ 95% interval containing the cluster
redshift, and ODDS of at least 0.5. Each accepted member receives a mass from
one shared near-infrared old-population conversion. Its mass is then spread over
the actual positive F160W pixels carrying that member's segmentation ID. This
preserves measured ellipticity, overlap, clumps, and BCG shape. The universal
mass-to-light sensitivity is 0.5, 0.8, and 1.1 in solar units; no value is
chosen per cluster.

## Gas component

All 19 Chandra count maps are reprojected to the same physical frame as the
stellar map. A universal compact-source suppression, robust background cutoff,
and 40 kpc smoothing rule
constructs surface-brightness morphology. Because X-ray emissivity is roughly
density squared, the projected gas template uses the square root of brightness
and an explicit spherical line-of-sight depth. The integral is normalized to an
external gas mass measurement:

- AS295: the published X-ray `M500`, core-excised temperature, and the same
  published `M500-YX` relation inverted to recover gas mass;
- MACS J0025: the directly published projected Chandra gas mass inside 500 kpc;
- MACS J0159: the directly published `Mgas,500` and `R500`;
- PLCK G287: the directly published deprojected `Mgas,500` profile.

The nominal exponent 0.5 and sensitivity exponents 0.4 and 0.6 are stored, as
are gas-mass and stellar-mass low/high maps. These are measurement-systematic
brackets, not adjustable gravity parameters.

## What this enables

The maps expose observables that a geometric transport law can use before any
lensing fit: gas/star centroid separation, axis-angle misalignment, dipole and
quadrupole moments, radial extent, and local component overlap. A valid new law
must derive its response from these measured fields with one universal
parameter vector.

Run:

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0641_registered_cluster_baryon_maps.py
python -m pytest tests/test_p0641_registered_cluster_baryon_maps.py -q
```
