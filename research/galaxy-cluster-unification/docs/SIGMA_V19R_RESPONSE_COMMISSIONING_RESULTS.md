# Sigma V19R response commissioning result

V19R **passed every frozen commissioning gate**.  Full regional-response
production is technically authorized.

The cell was selected before response construction by one deterministic rule:
the unique maximum source-count row in the frozen 5,082-task V19Q manifest.
It was Bullet region 390, ObsID 5356, CCD 2.  No morphology, response quality,
temperature, lensing result, or gravity residual entered the selection.

| Check | Result |
|---|---:|
| Frozen positive-exposure source events, 0.5–7 keV | 625 |
| Extraction source preflight, 0.5–7 keV | 625 |
| Blank-sky preflight, 0.5–7 keV | 232 |
| Source PHA total events / exact PI-channel mismatches | 656 / 0 |
| Background PHA total events / exact PI-channel mismatches | 470 / 0 |
| Positive finite ARF bins | 1,070 |
| Finite nonzero RMF matrix elements | 538,171 |
| Effective blank-sky-scale relative error | \(1.55\times10^{-15}\) |
| Response reference maps to selected CCD in both geometries | pass |
| PHA background, ARF and RMF links | pass |

The difference between the 0.5–7 keV preflight count and each PHA's total is
expected: the ungrouped PHA retains the full event-channel range.  The stronger
audit compared every PHA channel with the PI histogram of its exact input event
selection and found zero mismatched channels.

The frozen source PHA, background PHA, ARF, RMF and `specextract` log are stored
with the report.  The four calibrated products occupy 6.64 MB for this cell,
below the provisional 25 MB planning allowance.  Extrapolating linearly gives a
more realistic roughly 33 GiB product size for 5,082 similar cells, although
region complexity and response sparsity will vary.

This establishes the extraction machinery only.  No temperature, density,
shock speed, lensing prediction, or gravity parameter was fit in V19R.  The
next thermodynamic decision is whether to schedule the full batch or first
commission the frozen spectral fit on this response cell.
