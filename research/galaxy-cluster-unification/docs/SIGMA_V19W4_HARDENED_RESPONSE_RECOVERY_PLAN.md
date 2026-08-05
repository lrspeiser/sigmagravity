# Sigma V19W4 hardened response-recovery plan

## Decision

V19W4 supersedes the unexecuted V19W3 launcher. It preserves V19W3's frozen
terminal workload rule and exact-binmap recovery implementation, but it cannot
run until the unchanged base V19W process exits and writes its final report.

The successor is necessary for two reasons:

1. the independent 2,880-cell snapshot found 86 earlier omissions spanning all
   four CCD IDs, while the original V19W2 evidence covered only CCD 3; and
2. V19W3 asserted that the base archive remained read-only but did not compare
   a complete protected-tree fingerprint before and after recovery.

V19W2B has now passed six additional real CIAO cases across CCD 0, 1 and 2.
V19W4 makes that result a mandatory parent and verifies the read-only claim
cryptographically.

## Terminal launch gate

V19W4 refuses to start while any process command contains
`run_sigma_v19w_full_response_production.py`. After that process exits, it
requires the frozen full-interval final report, exact base config and runner
hashes, the expected 5,082-cell manifest, and the final base product-index hash.

The recovery workload is then derived rather than guessed:

\[
\mathcal W_{\rm recover}
=\mathcal W_{5082}-\mathcal W_{\rm independently\ valid\ base}.
\]

A base checkpoint counts as valid only after task identity, exact event counts,
all cell gates, PHA histograms and links, ARF/RMF integrity, product sizes and
all SHA-256 hashes pass.

## Exact recovery method

Each missing or invalid cell receives one attempt in the separate
`/home/henry/sigma-v19w4-response-recovery/v100` scratch area. The method is the
unchanged V19W2 implementation:

- select pixels by equality with the frozen integer binmap label;
- materialize the exact source and background event subsets;
- choose the selected 0.5--7 keV event nearest the DETX/DETY centroid;
- transform that event through the unreprojected detector frame and require
  its CCD to equal the manifest CCD;
- generate source PHA, background PHA, ARF and RMF;
- use the commissioned zero-background construction where required; and
- run the independent checkpoint auditor before admitting the cell.

A valid base checkpoint always takes precedence. Recovery products are never
copied over, relinked into, or promoted inside the base archive.

## Stronger base immutability proof

Before recovery, V19W4 recursively hashes every file under the base archive's
`completed`, `failed_attempts`, `quarantine`, and `partial` roots. The digest
includes relative path, byte size and file content hash. It also records an
independent logical inventory of every valid and invalid base checkpoint.

After recovery and the complete unified audit, both fingerprints are repeated.
The pass condition is exact equality of:

- file counts by protected root;
- total bytes;
- relative paths;
- every file-content SHA-256;
- the valid-checkpoint inventory; and
- the invalid-checkpoint error inventory.

This converts “the code should not modify the base” into an observed invariant.

## Double unified audit

The first complete audit occurs while writing the unified product index. V19W4
then reopens that index and independently revalidates every one of its 5,082
cells and 20,328 products against the frozen manifest and checkpoint reports.

Only a full pass may report
`hardened_unified_5082_response_archive_passed_and_v19x_successor_may_be_frozen`.
The original V19X remains unauthorized because it assumes one response archive.
A later successor must consume each unified row's declared base-or-recovery
directory.

## Claim boundary

V19W4 is an implementation and integrity protocol. It does not combine a
spectrum, fit gas temperature or density, construct a causal source, open a
lensing or halo target, or select a long-wave action or constant.

## Status command

```powershell
wsl.exe -e bash -lc 'cd /mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification && /home/henry/miniforge3/bin/conda run --no-capture-output -n sigma-ciao-4.18 python scripts/run_sigma_v19w4_hardened_response_recovery.py --status-only'
```
