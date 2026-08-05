# Sigma V19AS DECam forced-photometry development results

## Decision

V19AS completed its development-only comparison with the validation set still
sealed. The frozen ranking recommends the **four-arcsecond area-scaled rule**
for a separately frozen validation run.

This is a measurement-development result, not a validation, mass, lensing or
gravity result.

## Frozen counts and leakage controls

- Development anchors measured: **10/10**
- Development measurement memberships retained: **670/670**
- Development detector groups used: **122**
- Variant/aperture output rows: **4,020**
- Validation anchors measured: **0/5**
- Validation coordinates masked before background/source detection: **yes**
- Lensing, inferred-halo or gravity outcomes read: **no**

## Ranking

| Variant | Diameter | Valid rows | Complete development `griz` | Repeat scatter | Leave-one-out color error |
|---|---:|---:|---:|---:|---:|
| raw | 4 arcsec | 100.000% | 10/10 | 0.03081 mag | 0.03202 mag |
| area-scaled | 4 arcsec | 100.000% | 10/10 | 0.03081 mag | **0.03090 mag** |
| rotate-180 | 4 arcsec | 100.000% | 10/10 | 0.03088 mag | 0.03193 mag |
| raw | 8 arcsec | 99.851% | 10/10 | 0.06255 mag | 0.05450 mag |
| area-scaled | 8 arcsec | 99.851% | 10/10 | 0.06566 mag | 0.06089 mag |
| rotate-180 | 8 arcsec | 99.851% | 10/10 | 0.06882 mag | 0.06290 mag |

The four-arcsecond rules are nearly tied. Area scaling wins only after the
predeclared completeness, valid-fraction and repeatability criteria tie, by a
0.00112-mag advantage in leave-one-development-object-out color error. The
result therefore recommends a rule; it does not establish that deblending is
scientifically superior. The five sealed galaxies, especially the known
catalog-crowded member 57, are the real test.

## Why the absolute catalog offset is not yet a calibrated color result

The median absolute difference between the image-header-calibrated
four-arcsecond values and old NSC catalog values is about 0.52 mag when all
filters are mixed. That aggregate conceals a stable filter pattern for the
area-scaled rule:

| Filter | Image minus NSC four-arcsecond magnitude | Robust scatter |
|---|---:|---:|
| `g` | -1.0638 mag | 0.0736 mag |
| `r` | -0.0351 mag | 0.0363 mag |
| `i` | +0.4836 mag | 0.0497 mag |
| `z` | +0.6855 mag | 0.0390 mag |
| `Y` | +0.7755 mag | 0.0764 mag |

This is consistent with the documented limitation of Community Pipeline image
zeropoints in this southern field. NOIRLab explains that `MAGZERO` is the
correct counts-to-characterization relation, but at declination below -30
degrees the reference is Gaia G regardless of DECam filter, so scientific
photometry needs additional color calibration. The NSC, in contrast, applies
its own calibration and combination after Source Extractor measurement.

Sources:

- [NOIRLab guidance for DECam SkySub photometry](https://datalab.noirlab.edu/help/index.php?qa=1767&qa_1=decam-photometry-for-skysub-image-products)
- [NSC DR2 processing and calibration overview](https://datalab.noirlab.edu/data/nsc)
- [NSC fixed-aperture diameter definitions](https://datalab.noirlab.edu/help/index.php?qa=1021&qa_1=timeout-for-query-python-client-topcat-and-query-interface)

The next validation must therefore assess the color mapping learned solely
from development objects. It must not call the current header values final AB
photometry or use a validation-dependent zeropoint correction.

## Reproducibility

- Frozen protocol: `configs/sigma_v19as_decam_forced_photometry_development.json`
- Runner: `scripts/run_sigma_v19as_decam_forced_photometry_development.py`
- Machine report: `results/sigma_v19as_decam_forced_photometry_development/report.json`
- Per-exposure measurements: `data/derived/sigma_v19as_decam_forced_photometry_development/measurements.csv`
- Aggregates: `data/derived/sigma_v19as_decam_forced_photometry_development/aggregates.csv`
- Frozen ranking: `data/derived/sigma_v19as_decam_forced_photometry_development/ranking.csv`
- Group audit: `data/derived/sigma_v19as_decam_forced_photometry_development/group_audit.csv`
