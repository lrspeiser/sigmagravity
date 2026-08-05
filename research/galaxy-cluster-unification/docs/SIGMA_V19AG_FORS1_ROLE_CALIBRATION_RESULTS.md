# Sigma V19AG FORS1 role-specific global calibration results

## Outcome

**Passed all frozen global-calibration gates.**

The only complete early four-port B/R/I FORS1 science triplet can be calibrated with the archived commissioning biases and twilight flats without opening a member coordinate, candidate coordinate, cutout, catalog match, lensing model, halo model, mass map, current map or gravity result.

This passes a detector-preparation gate. It is not evidence for or against the long-wavelength Sigma field.

## Frozen input and method

The pixel run used exactly:

- three contiguous 1998-12-14 science frames: one each in R, I and B;
- all 14 early-signature four-port biases;
- all 25 four-port twilight flats: 5 B, 10 R and 10 I.

Each detector output was corrected row-by-row with its own 16-column prescan. The overscan was ignored. Only the four 1008-by-1024 valid regions were concatenated, producing a 2016-by-2048 active mosaic. Twilight frames were normalized once over the entire active mosaic, not port-by-port, so the master flat retained and corrected relative channel gains.

The frozen runner SHA-256 was `dfcb7159072294ab0afb28cc5881d63566eeac4370d1ae3ad08c271deddb9fcd`.

## Gate margins

| Quantity | Frozen gate | Observed | Result |
|---|---:|---:|---|
| Bias block peak-to-peak | <= 2.0 ADU | 1.75 ADU | Pass |
| Bias port-median spread | <= 2.0 ADU | 1.75 ADU | Pass |
| Largest absolute bias-frame port median | <= 5.0 ADU | 1.0 ADU | Pass |
| Empirical read noise | 0.3-10 ADU | 2.097 ADU in all four ports | Pass |
| B master-flat valid fraction | >= 99% | 99.9129% | Pass |
| I master-flat valid fraction | >= 99% | 99.9992% | Pass |
| R master-flat valid fraction | >= 99% | 99.9479% | Pass |
| B temporal S/N >= 100 fraction | >= 95% | 99.9998% | Pass |
| I temporal S/N >= 100 fraction | >= 95% | 99.9031% | Pass |
| R temporal S/N >= 100 fraction | >= 95% | 99.9562% | Pass |
| Worst calibrated-flat channel boundary | <= 3% | 0.0173% | Pass |
| Worst science channel boundary | <= 5% | 0.3431% | Pass |
| Science finite fraction | >= 99% | 99.9129%-99.9992% | Pass |
| Science saturation proxy | <= 1% | 0.1064%-0.3240% | Pass |
| Science background | positive | B 717.6, R 3594.2, I 6553.5 ADU | Pass |

The residual master-bias median is 0 ADU with robust sigma 1.112 ADU. These measurements support the preregistered prescan-only choice for this dataset; no two-dimensional bias was subtracted after seeing the values.

## Full-frame products

Six compressed FITS products were emitted only after the all-gates pass:

| Product | SHA-256 |
|---|---|
| `master_flat_B_BESS.fits.gz` | `aa58e9b2770626184b6eaa5c9ddd341b357389282075c4e43f45378fcc5ab174` |
| `master_flat_I_BESS.fits.gz` | `ade8c5c21b4b664e713a9c855753c0068f6e1156d50f4d66755a22344719a12d` |
| `master_flat_R_BESS.fits.gz` | `8d8d09134b9df106d8fe4ab4b393a02588ef7881c12687fab0ce5061394e88f4` |
| calibrated R science | `fa0ec4c8ae32030a2b66071df9dd328f373b51e8832cb05edec0cfa9e378e764` |
| calibrated I science | `4d26f11531d7eb112d0229dcb735e72e3c7d1ebbc305f2d67814a8a7e92df677` |
| calibrated B science | `5b8bb3863b0386a0de3b63617e3fabfb39d60136f44517b34501b9e7331e2343` |

Every product reopens as one 2048-by-2016 floating-point primary image and reproduces its recorded hash. Each header marks the WCS as `APPROX`; no astrometric claim is permitted yet.

## Non-scored full-frame QA

A post-pass, full-frame-only visual check found the same gross field geometry in B, R and I, including a dark upper-right field boundary and saturated-star bleed trails. It did not inspect or select any member cutout. These features must be turned into frozen masks before deblending; they cannot be allowed to decide which member measurements are retained after photometry.

## What this resolves

The earlier NSC/DELVE failure was caused by catalog deblending and flags, not by absence of the original imaging. V19AG now provides a defensible path to measure crowded objects directly in the source frames. It also demonstrates that the later flat metadata do not create a measurable four-port discontinuity after the frozen normalization: the worst calibrated-flat boundary is 0.0173%, far below the 3% gate.

## What remains unresolved

V19AG does not establish:

- an independent astrometric solution for the compressed active mosaic;
- a point-spread function in B, R or I;
- photometric zeropoints, atmospheric extinction or B/R/I color terms;
- deblended cluster-member fluxes or colors;
- cluster membership, stellar mass or baryonic current;
- any raw strong-lensing or gravity prediction.

The next protocol must freeze full-field astrometry, bad/saturated/vignetted masks and a PSF/deblending model before opening any member-centered cutout. Only after that can the independent B/R/I measurements feed the baryonic-current map needed to test directional versions of the long-wavelength Sigma field.
