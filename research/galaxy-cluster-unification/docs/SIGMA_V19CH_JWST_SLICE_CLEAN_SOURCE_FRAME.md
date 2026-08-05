# Sigma V19CH JWST SLICE clean source frame

## Decision

The project now has a large externally selected cluster source frame that is
independent of Sigma fit quality.  The official JWST SLICE program fixed 182
massive SZ- and X-ray-selected targets to sample cluster evolution.  A
whole-repository identity audit leaves 162 targets with no tracked Sigma name
hit and no raw-coordinate exposure.

No replacement cluster has been selected or admitted.  V19CH authorizes only
the next source-side step: freeze an independent X-ray/SZ morphology rule,
then use it to build a balanced metadata shortlist before any lensing target is
opened.

## Audit result

| State | Systems |
|---|---:|
| Official external source frame | 182 |
| Repository identity hits quarantined | 12 |
| Current source targets failed closed after PDF exposure | 12 |
| Overlap between those groups | 4 |
| Unique quarantined or spent systems | 20 |
| Zero-hit, unexposed candidates | 162 |
| Replacement clusters selected | 0 |
| Clusters admitted | 0 |

The repository scan covers every tracked path name and decodable tracked file
content up to 20 MB.  Larger and binary payload contents remain outside this
text audit, so final admission still requires a separate hash-only payload
manifest and a FITS/header alias audit.  That limitation is recorded rather
than silently treating large files as clean.

## Second exposure correction

During source discovery, an automated name-only extraction parsed the
published 14-cluster SLICE PDF.  The PDF also contains its raw multiple-image
table.  A later search response visibly returned coordinate rows for one of
those systems.  No coordinate value was copied into the repository or used in
a score, target choice, formula, or parameter decision.

Nevertheless, all 14 paper systems are permanently ineligible as future raw
coordinate holdouts.  Twelve are still present in the current 182-target APT
frame and are removed there.  This is intentionally stricter than claiming
that filtered text was never viewed by a person: the payload entered a process
used by the project, so the blind claim is no longer defensible.

## Next selection rule

Before reading morphology values, the next protocol must specify:

- the independent X-ray/SZ catalog and alias resolver;
- a continuous morphology statistic, with relaxed, intermediate and disturbed
  strata fixed by catalog-wide quantiles or published thresholds;
- minimum mass and redshift coverage;
- source-side requirements for gas, BCG, intracluster light, member galaxies,
  imaging and spectroscopy;
- at least eight metadata-complete candidates spanning three morphology states;
- six final clusters with at least two relaxed and two disturbed systems;
- a physically separate, hash-only target container for multiple-image
  positions, uncertainties and same-catalog halo predictions.

The candidates must be chosen by those rules, not by image multiplicity,
reported lensing RMS, inferred halo shape, or Sigma performance.

## Why this serves the broader program

The cluster test is not just an average-amplitude check.  Relaxed and disturbed
systems probe whether one baryon-sourced metric can predict curvature topology
when gas, galaxies and merger structure have very different layouts.  The
same frozen action must later predict resolved weak lensing, magnification,
joint dynamics and lensing, and merger directions.  The broader dark-matter
phenomenology gates remain those frozen in V19CC: satellites, stellar streams,
substructure, dynamical friction and energy transfer, growth, cosmic shear,
cluster abundance, and the CMB.  Solar-System work remains a mandatory later
veto, not the current optimization target.

## Reproduction

```powershell
python scripts/audit_sigma_v19ch_jwst_slice_clean_source_frame.py
python -m pytest tests/test_sigma_v19ch_jwst_slice_clean_source_frame.py -q
```

The machine-readable result is
`results/sigma_v19ch_jwst_slice_clean_source_frame/report.json`.
