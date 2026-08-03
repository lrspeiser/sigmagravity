# P0738 morphology-diverse resolved acquisition

Date: 2026-08-02

## Outcome

The repository now contains a frozen, integrity-audited sample of eight nearby
spiral galaxies with the raw observations needed for a genuinely two-dimensional
galaxy twin:

- THINGS natural-weighted H I moment 0 intensity maps;
- THINGS moment 1 line-of-sight velocity maps;
- THINGS moment 2 velocity-dispersion maps;
- SINGS IRAC channel-1 3.6 micron images; and
- SINGS IRAC channel-1 weight maps.

All 40 FITS files were downloaded from the public project archives. Their exact
byte counts, SHA-256 hashes, structural FITS metadata, WCS axes, source URLs,
roles, and frozen sample splits are recorded in
`results/p0738_morphology_diverse_resolved_acquisition/manifest.json`.

The acquisition passed all frozen gates:

| Property | Result |
|---|---:|
| Systems | 8 |
| Development / validation / holdout | 4 / 2 / 2 |
| FITS files | 40 |
| Total bytes | 380,151,360 |
| Exact-size matches | 40/40 |
| SHA-256 hashes recorded | 40/40 |
| Readable primary FITS headers | 40/40 |
| Celestial WCS axes present | 40/40 |
| Holdout image arrays opened | 0 |
| Gravity parameters used | 0 |

The manifest hash is
`b74a4b513dfd7e7fad0fac08a1b6b8a85c9a08437a9a95b8134f1d75803de0b2`.

## Why these galaxies

The sample is the intersection of systems with public THINGS maps, public
SINGS 3.6 micron imaging, and an existing SPARC radial record. It contains
NGC2403, NGC2841, NGC3198, NGC3521, NGC5055, NGC6946, NGC7331, and NGC7793.
It spans SPARC Hubble types 3 through 7, inclinations from 38 to 76 degrees,
and roughly a factor of 35 in 3.6 micron luminosity. This adds ordinary and
bulge-rich spirals to the existing dwarf-only resolved sample.

The split was committed before the images were downloaded:

- development: NGC2403, NGC3198, NGC5055, NGC7793;
- validation: NGC3521, NGC6946; and
- untouched whole-galaxy holdout: NGC2841, NGC7331.

The raw holdout bytes were downloaded and hashed, and their FITS headers were
checked. Their image arrays remain unopened. They may be unlocked only after
the registration, background, deprojection, mass-calibration, nuisance,
morphology, and velocity-scoring rules have been frozen.

## Non-circular test boundary

The H I intensity and infrared light are permitted baryonic inputs. The H I
velocity and dispersion maps are withheld targets. They cannot set the twin's
centroid, orientation, mass, angular modes, scale, clumps, or gravity
parameters.

The next stage must first reconstruct the observed baryonic image, including
its 2D concentration, asymmetry, clumpiness, centroid, Fourier structure,
radial profiles, and integrated masses. Only then may it evaluate the same
gravity formula on the observed baryons and generated twin and reveal the
velocity field. Pressure support and non-circular-motion sensitivity must be
reported as declared observation physics, not absorbed into a gravity fit.

This acquisition does not yet claim that the maps are registered,
background-subtracted, deprojected, PSF/beam matched, or converted into
baryonic mass. It proves that the required raw observations exist locally and
that the future holdout is still clean.

## Sources and reproduction

- THINGS public products: <https://things.www3.mpia.de/Data.html>
- THINGS survey paper: <https://arxiv.org/abs/0810.2125>
- SINGS public products and citation instructions:
  <https://irsa.ipac.caltech.edu/data/SPITZER/SINGS/overview.html>
- SINGS dataset DOI: `10.26131/IRSA424`

Run:

```powershell
python scripts/acquire_p0738_morphology_diverse_maps.py
python -m pytest tests/test_p0738_morphology_diverse_resolved_acquisition.py -q
```

Publications using these data must retain the citations and NRAO
acknowledgement recorded in the frozen configuration.
