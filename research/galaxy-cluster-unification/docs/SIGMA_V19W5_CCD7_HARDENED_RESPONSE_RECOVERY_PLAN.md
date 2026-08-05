# Sigma V19W5 CCD7-hardened response-recovery plan

## Decision

V19W5 supersedes the unexecuted V19W4 launcher. It preserves V19W4's terminal
process gate, protected-base byte audit, exact-binmap recovery, 5,082-cell
unified index and second complete product audit. It adds the passing V19W2C CCD7
commissioning report as a mandatory hashed parent.

V19W5 is frozen while the original V19W process is still active. It cannot run
until that process exits and its frozen full-interval report passes. The final
missing count is deliberately unknown at freeze.

## Why V19W4 is not sufficient

V19W4 was correctly frozen after direct recovery tests on CCD0--CCD3. The later
production interval exposed a different implementation boundary: 256 manifest
cells on CCD7, where the corresponding blank-sky files contain no CCD7 events.
At the V19W2C snapshot, 254 of those cells had exhausted both base attempts.

Running V19W4 without a CCD7 commissioning would spend the one frozen recovery
attempt on an untested detector boundary. V19W2C now supplies direct evidence
for six preregistered cases spanning both affected observations and source
counts from 22 to 532.

## Complete detector coverage

| CCD ID | Manifest cells | Direct commissioning source |
|---:|---:|---|
| 0 | 971 | V19W2B |
| 1 | 758 | V19W2B |
| 2 | 1,234 | V19W2B |
| 3 | 1,863 | V19W2 |
| 7 | 256 | V19W2C |
| **Total** | **5,082** | All manifest detector IDs covered |

This is implementation coverage, not a claim that six commissioning cases
prove every recovery cell. V19W5 retains the one-attempt, fail-closed rule.

## Launch and recovery gates

V19W5 refuses to start unless:

1. every V19W4/V19W3/V19W2/V19W2B parent hash and gate still passes;
2. the V19W2C report has six successful CCD7 cells in ObsIDs 10464 and 10888;
3. all six used an exact zero-all-energy background path and passed every
   response/product audit;
4. no process command contains the base V19W production runner;
5. the frozen base final report exists and covers the requested full interval;
6. the base and recovery scratch roots are disjoint; and
7. at least 10 GB remains on the target filesystem.

The workload is the frozen 5,082-cell manifest minus every base checkpoint that
passes the independent auditor. A valid base checkpoint always wins. Each
missing or invalid cell receives exactly one V19W2 recovery attempt in
`/home/henry/sigma-v19w5-response-recovery/v100` with at most two concurrent
cells.

## Protected-base and final audits

Before recovery, V19W5 recursively hashes every file under the base
`completed`, `failed_attempts`, `quarantine` and `partial` trees and records the
independently valid and invalid checkpoint inventories. After recovery and the
unified-index audits, those byte and checkpoint snapshots must be identical.

The unified index must contain exactly 5,082 unique tasks and 20,328 products.
Every index row is checked against its manifest task, checkpoint report,
product name, byte size and SHA-256 hash while the index is written. The index
is then reopened and all 5,082 checkpoints and products are independently
validated a second time.

## Downstream boundary

The shared V19X2 adapter now accepts `v19w5_recovery` only when a caller
explicitly declares that archive label and the V19W5 terminal status; its
default remains the historical V19W4 contract, so the two authorities cannot be
silently interchanged. The V19X2 mechanical freezer and orchestration scaffold
now explicitly require the V19W5 status, report, index and adapter mode. The
already hash-bound V19X3/V19X4 preflight chain is preserved; after V19X2 passes,
a separately named V19X3 successor must propagate the new authority rather than
mutating that preregistered evidence. This is a schema update, not permission to
alter any spectrum, region or fit setting.

V19W5 itself does not combine or fit a spectrum, infer a gas state, open a
lensing target or change a gravity formula.

## Status-only command

```powershell
wsl.exe -e bash -lc 'cd /mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification && /home/henry/miniforge3/bin/conda run --no-capture-output -n sigma-ciao-4.18 python scripts/run_sigma_v19w5_ccd7_hardened_response_recovery.py --status-only'
```

At freeze, this reports the base process as active, the terminal base report as
absent, the CCD7 parent as passed, and the V19W5 scratch as absent. Therefore no
terminal recovery has started.
