# P0635: real 2D DDO154 map commissioning

## Outcome

The P0634 field equations now consume real resolved baryonic images. DDO154 was
chosen as the commissioning object because it was already used in earlier
SPARC work; no untouched P0633 target was spent.

The inputs are the official LITTLE THINGS primary-beam-corrected natural-weight
H I moment-0 map and B/V optical images. The download script verifies every
byte against a frozen SHA-256. It deliberately does not request either H I
velocity map, a spectral cube, or velocity-dispersion map.

## Mass and map checks

The radio beam is read from the AIPS HISTORY cards, the Jy/beam m/s moment map
is converted through 21-cm brightness temperature to neutral-hydrogen column
density, and the declared factor 1.33 adds helium. The raw image contains
`3.0462e8` solar masses of H I. The face-on solver grid retains 99.139% of the
corresponding H I-plus-helium mass.

The V-band image supplies resolved stellar morphology. Foreground peaks are
bounded and the aperture is restricted to the visible optical body. For this
commissioning run only, that shape is normalized to the existing project-spent
SPARC photometric stellar mass. This per-object normalization is explicitly
forbidden in P0633 validation, where a universal stellar-population rule must
be frozen before targets are opened.

DDO154 is 94.44% gas in the resulting baryonic map, so its field is primarily
testing an observed, irregular gas distribution rather than a chosen stellar
profile.

## Field results on the spent rotation curve

| Law | RMSE (km/s) | Mean bias (km/s) |
|---|---:|---:|
| Newtonian 3D map | 25.031 | -24.148 |
| algebraic simple MOND on the map's Newtonian field | 2.916 | +1.327 |
| QUMOND 3D map | 3.936 | -1.391 |
| AQUAL 3D map | 3.600 | -0.803 |

The result is a commissioning diagnostic, not an external score. It does show
that the full PDE implementations recover most of the missing acceleration on
a real low-surface-brightness dwarf without a per-galaxy gravity parameter.

## What geometry changes

Five controlled map variants separate data-processing artifacts from the
equations' geometry response.

| Change from baseline | QUMOND RMSE change (km/s) |
|---|---:|
| razor-thin disk | -0.053 |
| thick gas/stars | +0.132 |
| axisymmetrize both baryonic maps | +0.080 |
| remove stars | +0.919 |

Neither thickness nor visible lumpiness explains the difference between the
algebraic shortcut and the full field. After axisymmetrization, QUMOND is still
1.116 km/s worse in RMSE than algebraic simple MOND. The remaining difference
is therefore mainly the non-spherical disk response of the field equation,
including its vector/curl-field structure, rather than a map-registration
mistake or an adjustable constant.

This is precisely why the real field solver matters: an algebraic radial rule
and a field theory can agree for a sphere yet make measurably different
predictions for the same flattened baryons.

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0635_ddo154_maps.ps1
$env:PYTHONPATH='src'
python scripts/run_p0635_ddo154_map_commissioning.py
python scripts/run_p0635_map_geometry_sensitivity.py
python -m pytest tests/test_galaxy_maps.py tests/test_p0635_ddo154_map_commissioning.py -q
```

The NRAO survey page documents the moment-map products and optical images:
<https://science.nrao.edu/science/surveys/littlethings/data>. The DDO154 H I
directory is <https://things.cv.nrao.edu/littlethings/ddo154/HI/>.
