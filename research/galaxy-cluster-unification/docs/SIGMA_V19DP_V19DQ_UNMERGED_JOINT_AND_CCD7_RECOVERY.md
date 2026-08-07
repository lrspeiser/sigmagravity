# Sigma V19DP--V19DQ unmerged joint likelihood and CCD7 recovery

Status: terminal registered preflight completed 2026-08-06. The real CCD7
blank-sky recovery passes. It authorizes recovery of all 254 affected CCD7
background products, not the 494-region likelihood itself.

## Question

The merged regional spectra passed while some integrated spectra did not. V19DP
therefore asked whether a single plasma state could fit each observation's PHA,
background, ARF and RMF jointly, without merging responses or adding a free
normalization per observation. It used one shared temperature, abundance and
normalization, fixed Galactic absorption and redshift, `chi2xspecvar`, 0.5--7
keV, at least 25 source counts per grouped bin, three starts, two complete
rebuilds and leave-one-observation-out checks.

Only two already-registered regions were opened:

- Bullet Cluster bin 169, nine observation/CCD spectra;
- Abell 2146 bin 62, ten observation/CCD spectra.

The pass gate was reduced statistic at most 1.5, a finite ordered 68% temperature
interval with fractional half-width at most 0.5, all parameters inside their
bounds, agreement with the parent merged temperature, at most 25% temperature
movement under any leave-one-observation-out fit, at most 30% of counts from one
dataset, and deterministic repeats.

## V19DP result

| Cluster | Reduced statistic | Temperature (keV) | 68% interval (keV) | Result |
|---|---:|---:|---:|---|
| Bullet | 0.763083 | 15.1404 | 12.1041--17.6873 | pass |
| Abell 2146 | 1.949652 | 11.1344 | 9.1393--14.7849 | fail |

Abell 2146 failed only the reduced-statistic gate. Its interval, parameter
bounds, repeat, merged-temperature and count-balance gates all passed. Omitting
ObsID 10464 CCD7 reduced the statistic to 1.32714; omitting ObsID 10888 CCD7
reduced it to 1.63371. Other omissions left the fit near 1.98--2.16. The failure
was therefore localized to the two CCD7 spectra, especially ObsID 10464.

## What the provenance audit found

The earlier response commissioning had shown only that a zero-event background
PHA could be generated consistently when the response archive contained no CCD7
background rows. It had not shown that the physical background was zero.

CIAO's original blank-sky construction did include CCD7 and calculated particle
scale factors of 0.086949646 and 0.015882602 for ObsIDs 10464 and 10888. After
astrometric correction and point-source exclusion, the two background files
still contained 1,354,493 and 1,390,824 CCD7 events. The subsequent common-grid
`reproject_events` products contained zero CCD7 events.

The exact frozen bin-62 mask selects substantial real backgrounds before that
loss:

| ObsID | Source events, 0.5--7 keV | Background events, all energy | Background events, 0.5--7 keV | Particle scale |
|---:|---:|---:|---:|---:|
| 10464 | 345 | 9,498 | 1,219 | 0.086949646 |
| 10888 | 84 | 9,509 | 1,204 | 0.015882602 |

V19DQ was frozen before extracting or fitting those backgrounds. It changed only
the background event source from the lossy post-reprojection file to the
astrometry-corrected, point-source-excluded pre-reprojection file. The exact mask,
source events, response settings, particle scaling, spectral model, free
parameters, starts, statistic, band, grouping and every threshold were unchanged.

## Corrected result

Both corrected background PHAs reproduce their event-channel histograms exactly,
use the frozen particle scales to floating-point tolerance and pass all response
audits. Neither uses the zero-background path.

| Cluster | Reduced statistic | Temperature (keV) | 68% interval (keV) | Abundance | Result |
|---|---:|---:|---:|---:|---|
| Bullet | 0.763083 | 15.1404 | 12.1041--17.6873 | 0.01144 | pass |
| Abell 2146 | 1.031211 | 10.1729 | 8.3131--12.5707 | 0.13701 | pass |

Abell's fit statistic falls from 198.8645 to 105.1835 at the same 102 degrees of
freedom. Its maximum leave-one-observation-out temperature shift is 9.97%, and
all ten omission fits have reduced statistics below 1.12. The Bullet result is
unchanged. No fit parameter was added.

This is strong evidence that the V19DP failure was created by loss of real CCD7
background events, not by observation-to-observation thermodynamic
inconsistency. It is a calibration result, not a Sigma-gravity result.

## Exact next stage

The unified response archive still contains zero-background PHAs for 254 Abell
2146 CCD7 cells. The next registered stage must rebuild all 254 using the real
pre-reprojection backgrounds and audit exact event counts, PHA channels,
particle scaling, response links, hashes and archive completeness. Only after
that archive passes may the 494-region unmerged likelihood be run. I4
thermodynamic-gradient stress and I5 baroclinicity remain sealed until the full
regional likelihood passes.

No lensing, halo, action, gravity, validation or holdout payload was opened, and
no gravity formula or parameter changed.

Machine-readable evidence:

- `results/sigma_v19dp_unmerged_regional_joint_likelihood_preflight/report.json`
- `results/sigma_v19dq_ccd7_background_recovery_preflight/report.json`
