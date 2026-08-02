# P0720 gravity-independent resolved-galaxy generator

Date: 2026-08-02

## Outcome

The simulator now has a reusable scientific core for extracting a compact
baryonic description from a registered two-dimensional galaxy map and
rendering it again. It does not read a rotation curve, fit a law of gravity, or
contain a branch for Sigma Gravity, MOND, Refracted Gravity, or dark matter.

Each gas and stellar component contains:

- total baryonic mass and a physical centroid;
- a 24-bin radial surface-density profile;
- Fourier modes 1 through 4 for lopsided, bar/two-fold, and higher angular
  structure; and
- 64 signed Gaussian residual features for clumps, cavities, and local
  departures from the smooth representation.

The package is deterministic JSON with a content hash. Its declared gravity
parameter object is empty and `velocityTargetsUsed` is false. A researcher can
change mass, radial scale, angular-mode strength, residual-feature strength,
orientation, and component offsets and then render a controlled new system.

## Real-map commissioning result

All 13 available P0639 LITTLE THINGS registered gas and stellar maps were
extracted and regenerated.

| Product | Median normalized L2 error | Worst error | Median correlation | Worst correlation |
|---|---:|---:|---:|---:|
| gas | 0.150 | 0.210 | 0.987 | 0.978 |
| stars | 0.295 | 0.501 | 0.956 | 0.860 |
| total baryons | 0.168 | 0.257 | 0.986 | 0.960 |

Mass is conserved to `3.98e-16` relative error. The stellar component is the
weakest part of the representation; compact and irregular light structure is
harder to compress than the smoother H I maps. The atlas also shows low-level
ring/arc structure introduced by the finite radial Fourier basis. Pixel
correlation must therefore not be interpreted as a perceptually exact image.

These are commissioning results. The numerical thresholds were chosen after
exploratory work with the same maps, so this is not a blind validation set.

## Honest 2D-to-3D behavior

One projected map cannot identify a unique depth distribution. P0720 draws
explicit exponential or squared-sech vertical profiles with varied scale
height and flaring. Seventy-eight different realizations (three for gas and
three for stars in every galaxy) projected back to their source 2D generated
maps with a maximum relative discrepancy of `4.15e-16`.

This demonstrates two things simultaneously:

1. the volume-density products are internally mass consistent; and
2. 2D agreement cannot prove that any one of those 3D realizations is the true
   galaxy.

## What this milestone does not prove

- The input maps already passed through P0639 registration, deprojection, and
  baryonic conversion. This is not yet an inverse fit to raw telescope images.
- Stellar mass inherits the fixed V-band mass-to-light assumption of 0.5.
- No uncertainty, PSF/beam, noise, mask, bulge, warp, or distance posterior is
  inferred yet.
- No rotation speed or lensing observable is predicted here.
- The 13 systems are gas-rich dwarfs, not a morphologically complete sample.
- The implementation is a Python library and reproducible research stage; its
  extractor/generator operations are not yet exposed by the public hosted API.

## Reproduce

```powershell
python scripts/run_p0720_resolved_galaxy_parameter_roundtrip.py
python -m pytest tests/test_resolved_galaxy_generator.py tests/test_p0720_resolved_galaxy_parameter_roundtrip_results.py -q
```

Primary outputs are under
`results/p0720_resolved_galaxy_parameter_roundtrip/`: content-hashed parameter
packages, regenerated NPZ maps, per-component scores, a vertical-prior
ensemble, a known-galaxy atlas, and a controlled generated family.
