# Sigma V19W2B cross-detector response commissioning results

## Decision

The exact-binmap recovery implementation passed all six additional
commissioning cells. V19W2B closes the main coverage gap in V19W2: the original
five successful cells all came from CCD 3, whereas the independently audited
live omission set spans CCD 0, 1, 2 and 3.

The same frozen implementation now has direct CIAO evidence on every CCD in the
omission set, source counts from one to 346 when V19W2 and V19W2B are considered
together, zero and positive backgrounds, and eight Bullet observation contexts.
This authorizes freezing a hardened terminal recovery successor. It does not
authorize the original single-archive V19X spectrum-combination protocol.

## Why the added commissioning was necessary

The read-only 2,880-cell archive snapshot passed every checkpoint and product
hash gate. However, 86 manifest tasks were absent at or before the current
production frontier. Their implementation axes were broader than the retained
failed-attempt directory suggested:

| Snapshot property | Value |
|---|---:|
| Valid completed cells | 2,880 |
| Independently hashed product files | 11,520 |
| Audited product bytes | 19,057,351,680 |
| Earlier missing manifest tasks | 86 |
| Missing source-count range | 1--346 |
| Missing cells with zero background | 14 |
| Missing cells with positive background | 72 |
| Distinct missing ObsIDs | 8 |
| Distinct missing CCD IDs | 4 |

V19W2 had already covered CCD 3 and five failure/count classes. V19W2B selected
six source-only omissions to cover CCDs 0--2, the four observation contexts not
represented in V19W2, one-event/zero-background edges, and high-count cells.
No spectrum shape, temperature, gas state, lensing coordinate, halo target or
gravity result entered selection.

## Results

| Cell | Source events | Background events | Declared CCD | Recovered detector CCD | Four-product bytes |
|---|---:|---:|---:|---:|---:|
| `BULLET_bin190_obs4986_ccd0` | 177 | 601 | 0 | 0 | 7,416,000 |
| `BULLET_bin102_obs4986_ccd1` | 1 | 0 | 1 | 1 | 7,459,200 |
| `BULLET_bin25_obs5355_ccd2` | 56 | 35 | 2 | 2 | 7,401,600 |
| `BULLET_bin76_obs5356_ccd2` | 340 | 63 | 2 | 2 | 7,418,880 |
| `BULLET_bin253_obs5357_ccd0` | 1 | 0 | 0 | 0 | 7,459,200 |
| `BULLET_bin127_obs5357_ccd1` | 281 | 55 | 1 | 1 | 7,467,840 |

For every cell:

1. the CIAO binary mask equaled the frozen integer binmap label pixel for
   pixel;
2. the materialized source and background event subsets reproduced the frozen
   0.5--7 keV counts;
3. the selected event nearest the DETX/DETY centroid mapped back to the declared
   CCD in the unreprojected detector frame;
4. the source and background PHA channel histograms were exact;
5. the ARF and RMF were finite and nonempty;
6. `BACKFILE`, `ANCRFILE`, and `RESPFILE` links were exact;
7. blank-sky scaling or the declared zero-background construction passed; and
8. all four product sizes and SHA-256 hashes passed an independent checkpoint
   audit.

## Transparent implementation corrections

Two pre-execution guard bugs stopped before the commissioning scratch or a
response product existed: the first looked for a nonexistent base-config
scratch field; the second probed free space on a not-yet-created directory.
Those guards were corrected to use the explicit protected base path and the
existing `/home/henry` filesystem.

The next execution produced all six valid immutable checkpoints. The wrapper
then requested a nonexistent `detector_reference` JSON key after each
independent audit and falsely recorded six failures. The underlying reports
store the value at `response_position.detector_medoid`. Version 1.0.3 corrected
only that reporting path and reused the completed checkpoints. No cell,
response setting, count, scientific gate or product was changed after results
were seen.

## Claim boundary

This establishes recovery implementation coverage, not a scientific gas map or
gravity result. The terminal missing-cell count remains unknown until V19W
exits. The current 86 omissions are a point-in-time lower bound on work before
the production frontier, not the final workload.

No spectrum was combined or interpreted. No temperature, density, Mach state,
shock speed, long-wave parameter, lensing target, halo map or gravity formula
was opened or changed. The base V19W archive remained untouched.

## Reproduction

```powershell
wsl.exe -e bash -lc 'cd /mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification && /home/henry/miniforge3/bin/conda run --no-capture-output -n sigma-ciao-4.18 python scripts/run_sigma_v19w2b_cross_detector_response_commissioning.py'
python -m pytest tests/test_sigma_v19w2b_cross_detector_response_commissioning.py -q
```

The machine-readable report has SHA-256
`3a2ba737b2edf5d1a2668bf390dc8ffeeeb2296031f6bc37f8a5679d7794bf87`.
