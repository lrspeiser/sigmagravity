# Sigma V19DO--V19DO2: observation-resolved soft-background audit

Status: terminal source-only diagnostic, 2026-08-06. The Bullet soft-spectrum
failure is not background dominated. No plasma component, regional temperature
map, I4/I5 source, lensing, halo, action, gravity parameter, validation, or
holdout payload was admitted.

## Result

V19DO2 audited all 5,082 registered source/background cells with their exact
exposure, `BACKSCAL`, and `AREASCAL` ratios. It reconstructs the V19DL
integrated Sherpa-band source counts exactly and verifies identical channel
energy bounds across all 30 observation/CCD controls.

The scaled blank-sky contribution to the 0.5--2 keV source counts is only:

| Cluster | Cells | Soft source counts | Scaled soft background | Background fraction | Net soft S/N |
|---|---:|---:|---:|---:|---:|
| Bullet | 3,812 | 406,157 | 8,274.88 | 2.037% | 624.0 |
| Abell 2146 | 1,270 | 147,745 | 2,795.16 | 1.892% | 376.7 |

The Bullet hard-band background fraction is 7.119%, and Abell 2146 is 7.962%.
Thus the band that fails most strongly in Bullet is actually much less
background contaminated than the 2--7 keV band that passes its spectral fit.
Blank-sky amplitude cannot plausibly explain the V19DN soft-only reduced
`chi2=4.1623` by itself.

## Frozen interpretation

V19DO preregistered three background regimes:

- background dominated at a scaled background/source fraction of at least
  0.50;
- source dominated at a fraction of at most 0.25; and
- mixed between those values.

It also defined strong observation heterogeneity as a soft-background-fraction
span of at least 0.25. The outcomes are:

| Cluster | Aggregate regime | Observation/CCD minimum | Maximum | Span | Strong heterogeneity? |
|---|---|---:|---:|---:|---|
| Bullet | source dominated | 1.560% | 11.096% | 9.536% | no |
| Abell 2146 | source dominated | 0.000% | 2.401% | 2.401% | no |

The largest Bullet fractions occur in tiny detector-edge samples containing
only one to five cells. They do not dominate the integrated likelihood. The
main observations consistently have soft background fractions of roughly
1.6--5.3%.

## V19DO implementation failure and V19DO2 remediation

V19DO stopped before producing a terminal scientific output because it
incorrectly asserted that source and blank-sky `BACKSCAL` must be equal to
within `1e-12`. The source and blank-sky extraction footprints may legitimately
have different `BACKSCAL` values. The correct Sherpa scaling is

\[
s_{\rm bkg}=
{t_s\over t_b}
{B_s\over B_b}
{A_s\over A_b},
\]

where `t`, `B`, and `A` are exposure, `BACKSCAL`, and `AREASCAL`. V19DO had
already implemented this ratio correctly.

V19DO2 removed only the invalid equality assertion. It retained the complete
ratio, all files, hashes, bands, thresholds, grouping, interpretation rules and
authorization boundaries. The failed V19DO report remains part of the audit
record and is scientifically discarded.

## What this rules out

This result rules out the simple explanation that the Bullet soft residual is
created primarily by a large or strongly observation-dependent blank-sky
subtraction. It does not rule out:

- response calibration differences that change source spectral shape rather
  than background amplitude;
- foreground or absorption structure not represented by the fixed model;
- spatially distributed temperature or abundance; or
- the mathematical error of representing a heterogeneous merger by one plasma
  spectrum folded through one averaged response.

The last possibility is now the leading observation-model explanation. It is
also consistent with V19DL: both registered local-region fits pass even though
the whole-cluster Bullet fit fails.

## Next required experiment

Freeze an unmerged-response joint-likelihood preflight on the two already
registered regions. Each observation retains its own PHA, background, ARF and
RMF. The physical region shares temperature and abundance, while only the
instrumental normalization may differ as dictated by exposure/coverage. The
preflight must demonstrate parameter recovery, response-link integrity,
repeatability, and agreement with the merged regional result within declared
uncertainty.

Only a successful preflight may authorize a separately frozen spatially
resolved production model. The integrated Bullet fit is not to be repaired by
adding post-hoc global plasma components.

## Artifacts

- `configs/sigma_v19do_observation_resolved_soft_background_audit.json`
- `configs/sigma_v19do2_backscal_ratio_remediation.json`
- `scripts/run_sigma_v19do_observation_resolved_soft_background_audit.py`
- `scripts/run_sigma_v19do2_backscal_ratio_remediation.py`
- `results/sigma_v19do_observation_resolved_soft_background_audit/report.json`
- `results/sigma_v19do2_backscal_ratio_remediation/report.json`
- `results/sigma_v19do2_backscal_ratio_remediation/cell_soft_background_audit.csv`

The cell-level CSV contains one row for every registered response cell and is
hashed by the terminal report.
