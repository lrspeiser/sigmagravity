# Sigma V19AF FORS1 header-compatibility results

## Decision

V19AF failed the frozen one-detector-signature gate.  All 46 inputs are valid,
complete 2080-by-2048 two-dimensional primary images, but the science, bias and
flat pool is not one readout configuration.  No pixel value was interpreted,
no decompressed payload was retained, and no member or photometric result was
opened.

## Exact header differences

Four detector signatures occur.  The invariant chip, window, binning and image
geometry fields match, but the following readout fields differ:

| Header field | Early four-output configuration | Later four-output configuration | Single-output science configuration |
|---|---:|---:|---:|
| `ESO DET OUTPUTS` | 4 | 4 | 1 |
| `ESO DET READ CLOCK` | `Readout ABCD (l...` | `Readout ABCD (l...` | `Readout A (low...` |
| `ESO DET OUT1 GAIN` | 0.33 | 0.34 | 0.33 or 0.34 by date |
| `ESO DET OUT1 RON` | 0.0 | 5.75 | 0.0 or 5.75 by date |
| `ESO DET OUT1 CONAD` | 3.0 | 2.9 | 3.0 or 2.9 by date |

The two one-output files are the December 14 and December 26 archive-category
science `R_BESS` exposures.  Every acquired bias and flat is four-output, so
those `R` frames cannot enter the frozen calibration pool.

The early four-output subgroup contains the December 14 `R` 500-second,
`I` 600-second and `B` 600-second science frames plus all 14 acquired biases.
It therefore contains a complete same-configuration `B/R/I` science triplet,
but the acquired twilight flats are from the later gain/read-noise header
configuration.

The later four-output subgroup contains the December 22 `I` and `B` science
frames plus all 25 acquired twilight flats.  It lacks a same-signature `R`
science frame and a bias in the current pool.

## Interpretation

This failure prevents blindly reducing all acquired files together.  It does
not reject the original-image route: the complete early `B/R/I` science
triplet and matching biases are promising if compatible early twilight flats
can be recovered.  Header value `RON=0.0` may be a commissioning placeholder,
but changing or ignoring it after the audit would violate the frozen exact
signature rule; any revised compatibility rule must be a new protocol.

The next action is a metadata-only search for earlier FORS1 Bessel flats and
later compatible biases.  Candidate calibration IDs and a header-matching rule
must be frozen before their FITS payloads are downloaded or opened.  No pixel
calibration is authorized by V19AF.

## Reproducibility

- Frozen protocol: `configs/sigma_v19af_fors1_header_compatibility.json`
- Runner: `scripts/run_sigma_v19af_fors1_header_compatibility.py`
- Exact primary cards: `results/sigma_v19af_fors1_header_compatibility/primary_headers.json`
- Machine-readable report: `results/sigma_v19af_fors1_header_compatibility/report.json`
