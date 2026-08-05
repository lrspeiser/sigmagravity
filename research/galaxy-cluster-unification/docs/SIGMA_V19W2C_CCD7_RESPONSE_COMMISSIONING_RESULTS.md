# Sigma V19W2C CCD7 response commissioning results

## Decision

The exact-binmap recovery implementation passed all six preregistered CCD7
commissioning cells. This closes the detector boundary that V19W2 and V19W2B
had not exercised: a source detector that is entirely absent from the matching
blank-sky event geometry.

For each selected cell the materialized blank-sky subset contained exactly zero
events at all energies. The implementation therefore generated the weighted
source ARF/RMF without passing the incompatible blank-sky file to
`specextract`, constructed an exact zero-count background PHA, linked it to the
source PHA, and passed the independent product audit.

This authorizes freezing V19W5 as the terminal recovery successor. It does not
authorize recovery while the original base process is active, combine a
spectrum, or establish a gravity result.

## Why another commissioning was required

V19W2 commissioned five CCD3 cases. V19W2B then commissioned six cases on CCD0,
CCD1 and CCD2, which covered every detector present in its earlier 2,880-cell
snapshot. The later Abell 2146 production interval exposed CCD7, the fifth and
last detector ID in the 5,082-cell manifest.

The new read-only snapshot found:

| Snapshot property | Value |
|---|---:|
| Independently valid completed cells | 3,706 |
| Independently hashed product files | 14,824 |
| Audited product bytes | 24,523,954,560 |
| Retained failed-attempt directories | 510 |
| Exhausted cells | 255 |
| Exhausted CCD7 cells | 254 |
| Affected CCD7 observation contexts | 2 |

The repeated base error was not a spectral or physical failure. The response
reference correctly landed on CCD7 in the source observation, but CCD7 did not
exist in the blank-sky geometry for ObsIDs 10464 and 10888. The unchanged V19W2
path was designed for precisely the zero-event boundary but had never been run
on CCD7.

## Outcome-blind selection

Each observation has 128 frozen CCD7 manifest rows. Before running a corrected
case, V19W2C sorted those rows by source-band count and production index and
selected the minimum, lower median and maximum ranks. All six were independently
recorded as exhausted base failures. No response shape, temperature, gas state,
lensing coordinate, halo target, galaxy result or gravity parameter entered the
selection.

| Cell | Source events | Background events, 0.5--7 keV | Background events, all energy | Recovered CCD | Four-product bytes |
|---|---:|---:|---:|---:|---:|
| `ABELL2146_bin7_obs10464_ccd7` | 137 | 0 | 0 | 7 | 7,119,360 |
| `ABELL2146_bin76_obs10464_ccd7` | 226 | 0 | 0 | 7 | 7,119,360 |
| `ABELL2146_bin32_obs10464_ccd7` | 532 | 0 | 0 | 7 | 7,119,360 |
| `ABELL2146_bin44_obs10888_ccd7` | 22 | 0 | 0 | 7 | 7,116,480 |
| `ABELL2146_bin103_obs10888_ccd7` | 43 | 0 | 0 | 7 | 7,113,600 |
| `ABELL2146_bin32_obs10888_ccd7` | 152 | 0 | 0 | 7 | 7,116,480 |

Every cell passed all of the following:

1. the CIAO mask equaled the frozen integer binmap label pixel for pixel;
2. the materialized source and background histograms matched the manifest;
3. the source-event detector medoid mapped back to CCD7;
4. the source-only weighted ARF and RMF were finite and nonempty;
5. the exact zero-count background PHA and all PHA links were valid;
6. blank-sky scaling remained exact; and
7. all four product sizes and SHA-256 hashes passed an independent checkpoint
   audit.

## What this establishes

The base failure does not require changing the scientific regions, event
membership, detector location or response physics. It requires respecting the
fact that an absent background detector contributes zero background events,
instead of asking CIAO to generate a response from nonexistent background
geometry.

This is implementation evidence only. A six-cell pass does not prove all 256
manifest CCD7 cells will complete. The terminal V19W5 run remains one attempt
per missing cell and fails closed if any case does not satisfy the same gates.

## Reproduction

```powershell
wsl.exe -e bash -lc 'cd /mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification && /home/henry/miniforge3/bin/conda run --no-capture-output -n sigma-ciao-4.18 python scripts/run_sigma_v19w2c_ccd7_response_commissioning.py'
python -m pytest tests/test_sigma_v19w2c_ccd7_response_commissioning.py -q
```

The machine-readable report has SHA-256
`2bd039a45c934639b2bb0855b7db668ba4326369c42a9c97c47a355776301d82`.
