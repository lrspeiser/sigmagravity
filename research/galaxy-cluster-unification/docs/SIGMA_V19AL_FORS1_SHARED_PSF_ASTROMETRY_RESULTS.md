# Sigma V19AL FORS1 shared-PSF astrometry results

## Decision

**Failed at the preliminary PSF-calibration gate; no WCS was fit.** The frozen protocol required at least 20 acceptable free-width stellar fits in every filter before deriving a shared width. B supplied 32, but I supplied 2 and R supplied 3.

The result report SHA-256 is `d8d5f5201088c4e7ed9040a2cd78bd3705d3dfcd4580870c5e5c3bf56dc8903e`.

## Rejection anatomy

| Filter | Accepted preliminary fits | Median amplitude S/N | Median normalized RMSE | Principal rejection |
|---|---:|---:|---:|---|
| B | 32/37 | 147.2 | 3.06 | 4 at coordinate bound |
| I | 2/30 | 141.7 | 8.05 | 21 RMSE failures; 6 at bound |
| R | 3/32 | 101.0 | 6.30 | 20 RMSE failures; 7 at bound |

The stars are not generally too faint: median fitted amplitude S/N is above 100 in every filter. Instead, a circular Gaussian plus planar background leaves residual structure far above the annulus-derived pixel noise in I and R. Many fits also run to the allowed coordinate boundary. This is direct evidence that the common image model is inadequate for the redder images, which may contain non-Gaussian PSF wings, blends, subpixel undersampling, structured cluster light or some combination.

## What was not done

The RMSE ceiling was not raised and the 20-star PSF-calibration minimum was not lowered. Because the failure happened before a filter width could be accepted, no shared-PSF centroid catalog, WCS, leave-one-out residual or three-filter center comparison was produced.

## Next strategy

If these FORS1 frames remain the chosen observation, the next model must address the identified structure rather than tune a threshold. A defensible option is an empirical or Moffat PSF with spatial variation and explicit joint fitting of nearby blends, validated on withheld foreground stars. The lower-risk alternative is an independently calibrated, higher-resolution image covering the same field.

No member/candidate payload, science photometry, baryonic mass/current model, lensing/halo payload, gravity equation or holdout was opened or changed. This is an observational-preparation result only.
