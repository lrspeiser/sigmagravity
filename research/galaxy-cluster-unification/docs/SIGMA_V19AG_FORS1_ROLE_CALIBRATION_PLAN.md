# Sigma V19AG FORS1 role-specific global calibration plan

## Decision being tested

V19AF correctly failed its frozen requirement that all 46 commissioning files have one exact detector signature. That failure is retained. V19AG asks a different, preregistered question: can the only complete early four-port B/R/I science triplet be globally calibrated with the available biases and twilight flats, without looking at any cluster member or candidate?

This is source-data preparation for the baryonic-current and long-wavelength Sigma tests. It is not a gravity test.

## Frozen selection

The primary science frames are the contiguous 1998-12-14 four-port exposures:

- R: `OFORS.1998-12-14T05:44:56.445`
- I: `OFORS.1998-12-14T05:54:16.814`
- B: `OFORS.1998-12-14T06:05:25.298`

They are the only temporally complete B/R/I set having one exact science readout signature. V19AG also uses all 14 exact-signature bias frames and all 25 four-port B/R/I twilight flats. The two single-output R frames and the later incomplete B/I science pair are excluded at the header gate, before pixel access.

## ESO-derived detector operation

The raw 2080-by-2048 array contains four outputs. Each output supplies a 16-column prescan, 1008 valid columns, 1024 rows and a 16-column overscan. Following the ESO FORS pipeline manual, the runner computes a row-dependent bias level from the prescan of each port and ignores the overscan. It then concatenates only the four valid regions into a 2016-by-2048 active mosaic.

V19AG does not subtract a two-dimensional master bias. Instead, all 14 biases test the FORS cookbook's claim that the residual structure is small enough for prescan-only correction. This choice also avoids transferring a time-dependent residual pattern from the early bias epoch into the later twilight flats.

For each filter, a twilight frame is divided by one global median, not four port medians. Its master flat can therefore correct relative channel gains. Median combining supplies outlier resistance.

## Frozen gates

The run passes only if all of these hold:

- all selected compressed hashes, roles, filters and detector geometries are exact;
- residual bias block peak-to-peak and port-median spread are each at most 2 ADU;
- no bias-frame port median is more than 5 ADU from zero;
- empirical port read noise is between 0.3 and 10 ADU;
- every twilight-flat median is between 1,000 and 60,000 ADU and fewer than 1% of pixels are at the saturation proxy;
- at least 99% of each master flat lies from 0.5 to 1.5 response;
- at least 95% of each master flat has temporal signal-to-noise of 100 or more;
- the maximum residual channel boundary in every flat is at most 3%;
- at least 99% of each calibrated science image remains finite, fewer than 1% of its pixels meet the saturation proxy, its global background is positive, and its maximum channel boundary is at most 5%.

The channel-boundary tests are explicit because ESO documents discontinuities as a known failure mode of old FORS1 four-port processing.

## Leakage controls

The runner opens only the 42 selected raw detector frames and the frozen V19AE/V19AF provenance. It contains no member or candidate coordinate, no cutout, no source detector, no deblender, no catalog matcher, no photometry model, no mass or current inference, and no lensing, halo or gravity input.

If a gate fails, only the metrics report is retained. Full-frame master flats and calibrated science products are persisted only after an all-gates pass. Their inherited WCS remains explicitly approximate until a separately frozen astrometric solution is run.

## Authoritative method sources

- ESO, *FORS Pipeline User Manual*, issue 5.18, Sections 10.15 and 11-12: <https://ftp.eso.org/pub/dfs/pipelines/instruments/fors/fors-pipeline-manual-5.18.pdf>
- ESO, *FORS data reduction cookbook*, Section 5.2.1: <https://www.eso.org/sci/facilities/paranal/instruments/fors/doc/VLT-MAN-ESO-13100-4030_v1.pdf>
- ESO FORS1 calibration data description: <https://www.eso.org/observing/dfo/quality/FORS1/pipeline/pipe_calib.html>
- ESO FORS1 pipeline known problems: <https://www.eso.org/qc/FORS1/pipeline/pipe_problems.html>
