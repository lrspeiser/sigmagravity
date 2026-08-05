# Sigma V19BY WALLABY moment-zero source maps

## Frozen boundary

V19BY will acquire exactly one public two-dimensional moment-zero FITS map for
each of the 711 V19BU release rows. That includes both Hydra TR1 and TR2 maps
where releases overlap; no alternative is discarded.

The archive selection is limited to the four declared DR1 `source_data_*`
planes and artifact names ending `_mom0.fits`. The runner rejects cubes,
spectral masks, moment-1 velocity maps, moment-2 dispersion maps, spectra,
kinematic models, residuals and rotation products.

## Why this is source information

A moment-zero map integrates H I brightness over the spectral axis. It shows
where the detected neutral gas lies on the sky but does not retain a resolved
velocity coordinate. This spatial footprint is needed to judge whether an
optical candidate overlaps the H I source rather than merely lying nearby.

Every file must match the CADC content length and MD5, have `NAXIS=2`, have
positive sky dimensions and contain no third FITS coordinate axis. The image
pixels are not used to select a counterpart in this checkpoint.

## Acquisition result

The frozen query returned one eligible map for every release row. V19BY saved
711 maps totaling 10,200,960 bytes: 148 Hydra TR1, 272 Hydra TR2, 147 NGC 4636
TR1 and 144 Norma TR1. Every archive byte length and MD5 was reproduced, every
file passed the two-dimensional FITS gate, and there were no missing,
ambiguous or failed products. The manifest SHA-256 is
`871df6aa9db724ad648a08762d619884f326d643c86ecd97414b79d4a2ae7aa7`.

All 119 Hydra release alternatives remain distinct. This acquisition therefore
adds spatial H I source morphology without silently resolving the previously
measured release-policy uncertainty.

## What follows

After the maps pass, a separate frozen protocol can combine H I contours,
SkyMapper cutouts, foreground-star masks, candidate colors, extendedness and
source-release uncertainty into counterpart probabilities. Ambiguous systems
must retain multiple candidates or fail a target-blind eligibility rule.

## Reproduction after the contract commit

```powershell
python scripts/acquire_sigma_v19by_wallaby_moment0_source_maps.py
python -m pytest tests/test_sigma_v19by_wallaby_moment0_source_maps.py -q
```

Primary archive documentation:
<https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/doc/tap/>.
