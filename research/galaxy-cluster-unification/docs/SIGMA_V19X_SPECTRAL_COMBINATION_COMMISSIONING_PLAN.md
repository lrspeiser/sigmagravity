# Sigma V19X spectral combination commissioning plan

## Decision frozen before execution

V19X is the last end-to-end commissioning step before fitting all 494 accepted
thermodynamic regions. It may run only after V19W reports that all 5,082
source/background PHA, ARF, and RMF cells pass and that its product index is
complete. Replacement-cluster lensing targets remain sealed.

The commissioning apertures are target-blind:

| cluster | integrated cells | accepted regions | selected commissioning region | selected-region cells | exact source events |
|---|---:|---:|---:|---:|---:|
| Bullet Cluster | 3,812 | 366 | 169 | 9 | 2,675 |
| Abell 2146 | 1,270 | 128 | 62 | 10 | 2,503 |

The selected region is the unique region with the greatest sum of frozen
0.5--7 keV source-event rows in its cluster. No response shape, temperature,
fit statistic, shock location, published temperature, halo, or lensing
coordinate enters the selection.

## Count audit correction

The initial V19X draft incorrectly stated that Sherpa's 0.5--7 keV count from a
grouped response-linked PHA must equal the event-table `ENERGY=500:7000` count.
Those are not identical selections. The already-completed V19T engineering fit
provides a direct example: its exact event selection contained 625 rows, while
Sherpa selected 651 counts because detector channels and RMF `EBOUNDS` straddle
the nominal energy endpoints.

This was corrected before any V19X combination or fit. V19X now requires:

1. every cell report's event-energy source and background counts to equal the
   frozen manifest exactly;
2. every PHA, ARF, and RMF byte count and SHA256 to match the V19W report and
   product index;
3. `combine_spectra` to conserve the complete source-PHA channel count exactly;
4. the Sherpa response-energy fit-band count to be reported without falsely
   equating it to an event-row count.

No energy band, spectrum, response, grouping threshold, plasma model, or fit
gate changed.

## Frozen execution sequence

For each cluster, V19X directly combines every registered source PHA in the
integrated aperture with CIAO `combine_spectra 4.18.2`, using `method=sum`,
`bscale_method=asca`, and `exp_origin=pha`. It then groups the combined source
PHA to 25 counts per group. The same direct combination is repeated for the
preselected maximum-count region.

The fit sequence is:

1. fit both integrated spectra with `xstbabs*xsapec`, fixed HI4PI column and
   redshift, and free cluster-wide abundance;
2. freeze each integrated best-fit abundance;
3. fit the selected region in that cluster with the abundance fixed;
4. authorize all 494 regional fits only if every combination, interval,
   statistic, and parameter-bound gate passes.

The published cluster temperature is not a gate. A failure is retained and
reported; it cannot be repaired by changing cells, grouping, absorption,
energy range, abundance handling, optimizer, or bounds after seeing the fit.

## What this stage establishes

A pass proves that the raw response archive can produce reproducible
cluster-wide abundances and stable regional temperatures under one frozen
pipeline. It does not test Sigma Gravity. These measurements are needed to
construct baryonic gas density, pressure, stress, and merger-state inputs for
the later directional long-wavelength metric test without borrowing a dark
matter map or tuning against lensing.
