# Sigma v17C regional calibrated-response support

## Outcome

The frozen v17C integrated spectra and temperature fits are valid, and both
clusters passed the preregistered integrated gate:

| Cluster | Temperature | 68% interval | Abundance | Reduced statistic | Difference from published validation value |
|---|---:|---:|---:|---:|---:|
| AS295 | 9.6036 keV | 9.3832–9.8301 keV | 0.3812 Solar | 1.2731 | 0.98% |
| PLCKG287 | 14.3962 keV | 13.8463–15.0347 keV | 0.3749 Solar | 1.1543 | 13.09% |

The v1.0.7 extraction report SHA256 is
`c2f5708baaed07d477b42679e476b13637e1295ba8abd3de94c22c313e2819bb`.
The successful integrated-temperature report SHA256 is
`9ce3b05dc64e0ee547693921f62b1f38ad0b9974f5c71e44f07a6338381a22e0`.
The core v17C configuration remains byte-identical to its original freeze.

## The regional edge case

Before any regional spectrum was extracted or any regional temperature was
fit, the planning pass reached AS295 region 2, ObsID 16524, CCD 2. The frozen
selection contains one 0.5–7.0 keV event at detector pixel `(39, 1021)`, near
the dithered chip edge. Its registered sky coordinate is
`(4121.963867, 4127.860840)`, or
`(RA, Dec)=(41.35824252, -53.02777512)` degrees. CIAO maps that sky reference
to CCD 3 rather than the selected CCD 2.

This is not a fitted-temperature or signal-to-noise decision. It is a response
existence problem. Three isolated `specextract` trials were run:

1. The unchanged frozen event-mean `refcoord`.
2. A detector-projected reference explicitly on CCD 2.
3. No explicit reference, with `resp_pos=REGION`.

All three created source and background PHA files, but none created an ARF or
RMF. CIAO terminated each with `max() iterable argument is empty`. The exact
logs and hashes are stored under
`results/sigma_v17c_regional_response_support/diagnostics/`.

## Frozen admission rule

The separate protocol
`configs/sigma_v17c_regional_response_support.json` was frozen after the two
integrated gates passed but before any regional spectrum, regional fit,
thermal-stress map, inverse coefficient, or lensing target access.

A regional ObsID/CCD cell is recorded and excluded before fitting only when:

- its unchanged event-mean response reference maps off the selected CCD; or
- the same reference lacks support on the selected CCD in either the science
  or matched blank-sky geometry.

Every record retains the region, ObsID, CCD, source/background event counts,
reference coordinate, and mapped CCDs. Any other response failure aborts the
run. There is no fitted-count, temperature, uncertainty, morphology, lensing,
gravity-residual, or adjustable threshold in this rule.

Keeping this as a separately hashed protocol avoids rewriting the historical
v17C core or cascading new hashes through completed v17G–v18 records. The
regional report records the support-protocol hash, so the exception remains
part of the reproducible evidence chain.

## Runtime response-support supplement

The first complete AS295 extraction pass later produced 193 valid calibrated
cells and one additional response-only failure: region 12, ObsID 16524, CCD 3.
The first attempt was obscured by an operational `AF_UNIX path too long`
failure. Moving only the disposable CIAO `TMPDIR` to the short, isolated path
`/tmp/sv17c/...` removed that problem without changing the command, data,
region, response reference, or any scientific setting.

With the short path, CIAO created nonempty source and background PHA files but
no ARF or RMF, then terminated with the same
`max() iterable argument is empty` response-domain error. The reference was
slightly beyond the nominal detector boundary (`CHIPX=1035`, clipped to 1024),
but a detector-coordinate cutoff would have been invalid: at least three other
cells with clipped coordinates (`CHIPX=-7`, `CHIPX=1031`, and `CHIPY=1029`)
successfully produced all four calibrated products.

Therefore the separately frozen runtime supplement
`configs/sigma_v17c_regional_runtime_response_support.json` admits only the
exact conjunction:

- source and background PHA files both exist and are nonempty;
- ARF and RMF files are both absent; and
- the log contains source extraction, background extraction, and the exact
  empty-response-domain terminal error.

The two PHA files are hashed and moved into a response-support quarantine, an
immutable per-cell marker is written, and the cell is recorded as skipped.
Any mismatch still aborts. The rule contains no coordinate threshold, fitted
quantity, or gravity observable. Its config SHA256 is
`75fe99de5b2e4df2e64791e63090fcbb70ee947bc6910495a82c555710807987`;
the audit-report SHA256 is
`4311d87c37768f60eff0ae6d7f3ac4738dbe625a0690e8db0c587f8a9fe06b89`.
It was frozen before a regional spectra report, any regional temperature, any
thermal-stress map, or any lensing-target access existed.

## Scientific interpretation

The excluded cell has one source event and no usable instrumental response.
It cannot contribute a calibrated temperature likelihood. Excluding it is not
evidence for or against thermal stress as a Sigma source; that question remains
fully controlled by the regional-temperature, v17E transfer, and conditional
v17F extent gates.
