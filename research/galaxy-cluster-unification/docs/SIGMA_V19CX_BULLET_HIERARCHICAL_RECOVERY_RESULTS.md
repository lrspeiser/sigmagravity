# Sigma V19CX Bullet hierarchical-recovery results

## Terminal result

V19CX successfully recovered the full Bullet Cluster integrated spectrum from
all 3,812 registered response cells, but the unchanged V19X2 spectral
commissioning protocol failed its preregistered integrated-fit quality gate.
The terminal status is
`unified_spectral_combination_commissioning_gate_failed`; this is a scientific
gate failure, not an execution exception.

The hierarchy was introduced only after V19CW demonstrated direct-versus-
hierarchical equivalence on the complete 1,270-cell Abell 2146 workload. It
partitions the Bullet response calculation by the ten frozen observation IDs,
uses zero intermediate RMF threshold, and applies the unchanged final
`1e-6` RMF threshold. It does not select by cluster outcome, spectrum, fitted
temperature, lensing, halo information, or gravity performance.

## Source-product gates

| Aperture | Cells | Expected full-PHA counts | Combined full-PHA counts | Result |
|---|---:|---:|---:|---|
| Abell 2146 integrated | 1,270 | 259,688 | 259,688 | exact |
| Abell 2146 bin 62 | 10 | 3,886 | 3,886 | exact |
| Bullet integrated | 3,812 | 707,569 | 707,569 | exact |
| Bullet bin 169 | 9 | 2,787 | 2,787 | exact |

Every registered cell was used exactly once. Every frozen input hash matched,
the source/background/ARF/RMF products existed, and the grouped PHA links were
exact. The hierarchical recovery therefore closed the large-stack CIAO failure
without dropping data or changing the scientific fit.

## Frozen fit results

The unchanged model was `xstbabs * xsapec` over 0.5--7.0 keV with
`chi2xspecvar`. Integrated abundance was free and each commissioning region
used its cluster's integrated best-fit abundance.

| Fit | Temperature (keV) | Reduced statistic | Fractional 68% temperature half-width | Gate |
|---|---:|---:|---:|---|
| Abell 2146 integrated | 8.1255 | 1.2232 | 0.0107 | pass |
| Bullet integrated | 16.0636 | 2.7937 | 0.0106 | **fail** |
| Abell 2146 bin 62 | 10.2086 | 0.7508 | 0.1946 | pass |
| Bullet bin 169 | 15.2068 | 1.0103 | 0.1865 | pass |

The frozen integrated-fit limit was a reduced statistic of at most 1.5. The
Bullet integrated fit exceeded it by 1.2937, or 86.2% relative to the limit.
All parameters were finite and strictly inside their bounds, both confidence
intervals were ordered, and both regional fits passed. The sole failed terminal
gate was `both_integrated_fits_pass`.

## What the failure means

The complete Bullet spectrum cannot be accepted as adequately described by the
frozen one-temperature absorbed-plasma model and quality limit. A merging
cluster plausibly contains multiple temperature and abundance components, but
this result alone does not distinguish real multiphase structure from residual
calibration, background, or response-model systematics. It does establish that
the planned use of one integrated abundance as the authority for all 366 Bullet
regions did not pass its own admission test.

The result does **not** show that the hierarchy corrupted the spectrum: source
counts were conserved exactly, response links passed, the hierarchy had already
passed direct-equivalence tests, and the independently selected Bullet region
fit passed with reduced statistic 1.0103.

## Frozen disposition

The V19X2/V19CX failure rule is now binding:

- do not change grouping, energy band, Galactic column, redshift, abundance
  rule, statistic, optimizer sequence, parameter bounds, or cell membership;
- do not start the 494-region V19X3B production;
- do not construct V19X4B gas posteriors or V19BMB stellar controls;
- do not run V19BQ or V19BS from this source chain;
- do not derive a Sigma action from the blocked I4/I5 source route; and
- do not reinterpret the failure as evidence for or against a gravity theory.

The next direction decision must reassess the observational source evidence.
Admissible new evidence includes direct gas velocity information, an
independently clocked merger lag, or an independently preregistered merger
sample. A new plasma/source reconstruction would be a separately named
protocol and could not retroactively convert V19CX into a pass.

## Authoritative artifacts

- Config: `configs/sigma_v19cx_bullet_hierarchical_recovery.json`
- Runner: `scripts/run_sigma_v19cx_bullet_hierarchical_recovery.py`
- Terminal report:
  `results/sigma_v19cx_bullet_hierarchical_recovery/report.json`
- Terminal report SHA-256:
  `c86ecffa2ea36086b6733f717ebb0b4fb59eed2e24d58b813093f18f25f15717`
- Validated 5,082-cell index SHA-256:
  `2f959371745c884eeeb120b525ecb90abe044c59f501efb5ffcf07e2082624bf`
