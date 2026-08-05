# Sigma V19AX DELVE DR3 coadd acquisition results

## Decision

V19AX **failed closed** at its frozen inverse-variance support gate. All 12
requested coadd planes were acquired and every candidate coordinate lies well
inside them, but the `g` and `z` weight planes do not have positive support over
at least 99% of the field.

No anchor or candidate flux was measured and no counterpart was selected.

## Frozen acquisition result

| Gate | Requirement | Result | Pass? |
|---|---:|---:|---:|
| SIA rows | exactly 27 | 27 | yes |
| Products | exactly 12 | 12 | yes |
| Common shape | one shape | 2,331 × 2,333 pixels | yes |
| Candidate geometry | 568 inside every plane | 568 | yes |
| Finite pixels | at least 99% | 100% in every plane | yes |
| Positive weight, `g` | at least 99% | **98.773%** | **no** |
| Positive weight, `r` | at least 99% | 99.985% | yes |
| Positive weight, `i` | at least 99% | 99.970% | yes |
| Positive weight, `z` | at least 99% | **98.928%** | **no** |
| Source photometry/association | forbidden | none | yes |

The frozen acquisition runner stopped on the first failed weight plane. A
separate deterministic post-failure summarizer inspected only WCS, data type,
finite support and weight support so the entire failure could be reported
without measuring source flux.

## Candidate-level upper bound

The unsupported pixels are not confined to irrelevant cutout corners:

| Band | Candidate centers with positive weight |
|---|---:|
| `g` | 441/568 |
| `r` | 568/568 |
| `i` | 568/568 |
| `z` | 566/568 |
| all `griz` | **439/568 (77.29%)** |

The all-band intersection is an optimistic upper bound because it checks only
the center pixel. A real aperture or deblended source footprint can require
additional supported pixels and therefore cannot exceed this count under the
same coadds. Only 52/57 members have an all-band-supported candidate.

A second post-failure upper bound takes the union with the 200 candidates that
already had three-sigma signed stacks in every `griz` band in V19AV. The two
sets overlap for 164 candidates, so their union is 475/568 (83.63%). This is
still below 90%, although all 57 members have at least one candidate in the
union.

## What this closes

The project has now tested four distinct measurement routes for the same broad
candidate population:

- positive single-exposure aperture measurements (V19AU);
- robust signed multi-exposure stacks (V19AV);
- direct DELVE DR3 catalog identities (V19AW); and
- DELVE DR3 coadd pixel support (V19AX).

None can make at least 90% of all 568 candidates complete in four bands without
changing the candidate list or the definition after seeing the result. Even the
optimistic union of the two strongest pixel routes reaches only 83.63%.

The broad candidate-completeness branch should stop here. The next baryonic map
must be probabilistic and explicitly represent missing bands, null identities
and ambiguous candidates. It may use supported coadd pixels in a separately
frozen partial-measurement protocol, but it must not present that partial map as
complete.

## Consequence for the long-wavelength hypothesis

This remains an input-data failure, not a test of the gravity wave. It tells us
how uncertainties must enter the source term. A defensible long-wave equation
cannot be tested against a single best-guess baryonic map; it must propagate an
ensemble of allowed density/current maps through

\[
(1-L_\Sigma^2\Box)X_{\mu\nu}=S_{\mu\nu}[T,j,\Pi]
\]

and report whether the predicted lensing structure survives the missing-source
uncertainty.

Reproducibility:

- `configs/sigma_v19ax_delve_dr3_coadd_acquisition.json`
- `scripts/acquire_sigma_v19ax_delve_dr3_coadds.py`
- `scripts/summarize_sigma_v19ax_delve_dr3_coadds.py`
- `results/sigma_v19ax_delve_dr3_coadd_acquisition/report.json`
- `data/raw/sigma_v19ax_delve_dr3_coadd_acquisition/`
- `data/derived/sigma_v19ax_delve_dr3_coadd_acquisition/`
