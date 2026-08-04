# Sigma v17C blank-sky scaling correction

## Decision

Sigma v17C spectral protocol `1.0.6` is abandoned and preserved. It produced
12 complete AS295 ObsID/CCD response cells, but no combined cluster spectrum,
temperature, abundance, regional spectrum, thermal-stress map, inverse
coefficient, or lensing score. No v1.0.6 product is reused.

Protocol `1.0.7` starts in the fresh `spectral_v17c_v107` namespace. It changes
only the background PHA `AREASCAL` bookkeeping after `specextract`; event rows,
regions, responses, grouping, fit model, thresholds, and physics parameters are
unchanged.

## The invariant that failed

CIAO `blanksky` with `weight_method=particle` records `BKGSCALn`, the measured
9--12 keV observation/background particle-count ratio. The project requires
the analysis software's effective PHA subtraction scale to reproduce that
directly measured ratio. For a source and its blank-sky background, Sherpa's
effective scale is

\[
s_{\rm eff}=
\frac{t_s}{t_b}
\frac{{\tt BACKSCAL}_s}{{\tt BACKSCAL}_b}
\frac{{\tt AREASCAL}_s}{{\tt AREASCAL}_b}.
\]

The v1.0.6 background PHA had
`AREASCAL_b=1/BKGSCALn` while retaining the 600 ks blank-sky exposure. Thus
`s_eff` also contained the source/background exposure ratio and was only
1.50%--7.44% of `BKGSCALn`. This would have under-subtracted the particle
background by factors of approximately 13--66 if allowed into a fit.

The correction is derived rather than fitted:

\[
\boxed{
{\tt AREASCAL}_b=
\frac{t_s}{t_b}
\frac{{\tt BACKSCAL}_s}{{\tt BACKSCAL}_b}
\frac{{\tt AREASCAL}_s}{{\tt BKGSCALn}}
}
\]

After the edit, the runner rereads the header and requires both the encoded
`AREASCAL` and `s_eff/BKGSCALn` to agree with the derived values within
`1e-6`. It never rescales background events.

## Evidence from every abandoned v1.0.6 cell

| Cell | Source exposure (s) | BKGSCALn | Old effective scale | Old / required | Corrected AREASCAL |
|---|---:|---:|---:|---:|---:|
| 12260 CCD3 | 19,769.951 | 0.042446516 | 0.001398609 | 0.032950 | 0.776269 |
| 16127 CCD2 | 43,319.065 | 0.065914959 | 0.004758957 | 0.072198 | 1.095327 |
| 16282 CCD3 | 9,027.093 | 0.014645180 | 0.000220339 | 0.015045 | 1.027311 |
| 16524 CCD0 | 44,611.534 | 0.071173213 | 0.005291910 | 0.074353 | 1.044671 |
| 16524 CCD1 | 44,614.634 | 0.071030915 | 0.005281697 | 0.074358 | 1.046836 |
| 16524 CCD2 | 44,608.434 | 0.070290588 | 0.005225922 | 0.074347 | 1.057715 |
| 16524 CCD3 | 44,605.334 | 0.077410236 | 0.005754849 | 0.074342 | 0.960367 |
| 16525 CCD0 | 44,484.484 | 0.070102535 | 0.005197459 | 0.074141 | 1.057605 |
| 16525 CCD1 | 44,484.484 | 0.070565492 | 0.005231783 | 0.074141 | 1.050667 |
| 16525 CCD2 | 44,478.284 | 0.072534502 | 0.005377017 | 0.074130 | 1.022003 |
| 16525 CCD3 | 44,481.384 | 0.076863632 | 0.005698335 | 0.074136 | 0.964509 |
| 16526 CCD2 | 43,741.418 | 0.067485698 | 0.004919867 | 0.072902 | 1.080264 |

For a copied real `16524 CCD0` PHA pair, the corrected header gave
`AREASCAL_b=1.0446705102504`. Sherpa independently returned
`get_bkg_scale=0.07117321300000123` for measured
`BKGSCALn=0.071173213`, a relative discrepancy of
`1.71e-14`. The source PHA's response and background pointers remained valid.

## First production-cell audit

The first completed v1.0.7 production cell, AS295 ObsID 16282 CCD3, was audited
with the separate read-only
`scripts/audit_sigma_v17c_spectrum_scaling.py` program. The audit obtains the
required `BKGSCAL3` from the frozen v17A cleaning report rather than accepting a
value reported by the v17C extraction runner. It then parses the completed PHA
headers independently, reconstructs the effective scale, validates the
background/ARF/RMF pointers, hashes every product, and asks Sherpa for the scale
it will actually use.

| Quantity | Value |
|---|---:|
| Frozen `BKGSCAL3` | 0.0146451800000000 |
| Corrected background `AREASCAL` | 1.0273109944004 |
| Independently reconstructed effective scale | 0.0146451799999995 |
| Sherpa `get_bkg_scale` | 0.0146451799999995 |
| Relative discrepancy from frozen `BKGSCAL3` | 3.4e-14 |

All five audit checks passed at the frozen `1e-6` tolerance. The machine-readable
evidence is in `results/sigma_v17c_first_cell_scale_audit/report.json`. This
validates the bookkeeping of one real production cell; the extraction runner
still applies and verifies the invariant separately for every later cell.

## Execution and claim boundary

- This is a pre-fit data-reduction correction, not a change to Sigma gravity.
- It was frozen before any v17C temperature or abundance and before any v17D
  thermal map or v17E lensing target was opened.
- All v1.0.6 files remain in their immutable namespace for forensic audit.
- The same correction and tolerance apply to integrated and every regional
  spectrum in both clusters.
- A thermal-stress or lensing result can still fail; this correction only makes
  that later test interpretable.

CIAO reference behavior is documented in the official
[`blanksky`](https://cxc.harvard.edu/ciao/ahelp/blanksky.html),
[`specextract`](https://cxc.harvard.edu/ciao/ahelp/specextract.html), and
[ACIS blank-sky background](https://cxc.harvard.edu/ciao/threads/acisbackground/)
documentation. The acceptance invariant above is additionally checked against
the directly counted high-energy events and Sherpa's own PHA scaling result.
