# Sigma V19W live archive audit results

## Decision

The completed half of the V19W response archive is internally clean, but the
base production run cannot finish as frozen. The point-in-time read-only audit
validated all 2,569 completed checkpoints present at the second snapshot:

- 10,276 PHA/ARF/RMF products;
- 16,991,812,800 product bytes;
- exact manifest source and background preflight counts;
- exact saved file sizes and SHA-256 hashes;
- exact PHA event/channel histograms and links; and
- finite, nonzero ARF and RMF audits.

No completed checkpoint was corrupt, duplicated or quarantined. V19X remains
unauthorized because V19W is incomplete.

## Missing-cell diagnosis

A separate target-blind preflight reproduction examined the 57 manifest tasks
that were already more than one full batch behind the largest completed
production index. It found three implementation classes:

| Class | Stable missing tasks | Cause |
|---|---:|---|
| Exact-count mismatch | 35 | DS9/CIAO region filtering includes one boundary event that the authoritative frozen binmap assigns elsewhere. Source, background, or both can differ by exactly one event. |
| Response reference maps to adjacent CCD | 21 | A low-count region's centroid in the reprojected sky grid maps to a neighboring detector chip when transformed through that observation's aspect solution. |
| Preflight passes but response construction fails | 1 | Both unchanged attempts for `BULLET_bin36_obs4984_ccd3` failed because CIAO's internal AF_UNIX socket path exceeded the operating-system length limit. |

The first two classes occur before `specextract`, so they leave no failed
attempt directory. The rolling progress file retains only the latest 64 task
outcomes and its compact record omitted the exception text; consequently the
accumulating missing set was not visible from the headline completed count.

## Correct implementation direction

The frozen V19P/V19Q binmap assignment is the authoritative disjoint event
partition. Updating the manifest to the polygon counts would allow a small
number of boundary events to be double counted or assigned inconsistently.
The successor must instead filter source and blank-sky events with an exact
binary pixel mask derived from `binmap == bin_id`.

CIAO 4.18 explicitly supports `sky=mask(mask_file)` event filters. With a
pixel mask, `specextract` permits `resp_pos=CENTROID` when `refcoord` is left
unset. This simultaneously removes the polygon-boundary ambiguity and the
invalid manually transformed reference coordinate while retaining the same
weighted ARF/RMF construction. A short cell token must also be used for
PFILES, temporary directories and partial outputs to avoid the AF_UNIX limit.

These are implementation corrections, not changes to the accepted regions,
events, energy band, background scaling, response weighting, model, gravity
physics or scientific thresholds.

## Frozen next gate

Before any full recovery is authorized, the exact-binmap implementation must
pass a small commissioning set containing:

1. a source-only boundary-count discrepancy;
2. a background-only boundary-count discrepancy;
3. a source-and-background discrepancy;
4. a low-count adjacent-CCD centroid failure; and
5. the path-length failure.

The commissioning products remain separate from the base V19W archive. Only
an all-gate pass authorizes a successor to reconstruct every task missing from
the final base report and then independently validate all 5,082 checkpoints.

## Claim boundary

This audit tests response-archive integrity and implementation behavior. It
does not combine or fit a spectrum, estimate gas temperature or density, open
a lensing or halo target, select a long-wave source, or change a gravity
formula.
