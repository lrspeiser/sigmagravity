# Sigma V19V concurrent response-pilot result

V19V **passed every frozen gate**.  Full 5,082-cell response production is now
technically authorized under the unchanged V19R method and V19U batching rules.

## Concurrent result

| Check | Result |
|---|---:|
| New response cells attempted | 4 |
| New response cells passed | 4 |
| Attempts needed per cell | 1 |
| Concurrent pilot wall time | 48.144 s |
| Observed maximum concurrency | 4 |
| Successful response throughput | 0.08308 cells/s |
| Projected 5,082-cell response time | 16.991 h |
| Median four-product size | 6,596,640 bytes |
| Projected full response archive | 33,524,124,480 bytes |

The full storage projection is 33.52 GB (31.22 GiB), close to V19U's
33.75-GB estimate from the commissioning cell.

## Cell-level transfer

| Cell | Source / background events | Elapsed | ARF positive bins | RMF nonzero elements | Result |
|---|---:|---:|---:|---:|---|
| Bullet bin 257, ObsID 5355, CCD 2 | 82 / 98 | 28.767 s | 1,070 | 531,523 | pass |
| Bullet bin 23, ObsID 5356, CCD 2 | 266 / 38 | 23.383 s | 1,070 | 533,158 | pass |
| Abell 2146 bin 75, ObsID 13020, CCD 0 | 153 / 57 | 27.926 s | 1,070 | 524,534 | pass |
| Abell 2146 bin 101, ObsID 13120, CCD 2 | 217 / 146 | 48.138 s | 1,070 | 532,414 | pass |

For every cell:

- the positive-exposure count, extracted source count and frozen manifest count
  agreed exactly;
- the background count agreed exactly;
- the response centroid mapped to the frozen CCD in both science and blank-sky
  geometry;
- source and background PHA channel histograms matched the selected events;
- the ARF was finite and positive and the RMF finite and nonzero;
- `BACKFILE`, `ANCRFILE` and `RESPFILE` links were present;
- the effective background scale matched the frozen value to better than one
  part per million;
- no retry or scientific exception was needed.

All four PHA/background/ARF/RMF sets, cell reports and extraction logs were
hashed and snapshotted.  The result establishes that the response construction
transfers across both clusters, count strata and CCDs under concurrent load. It
does **not** construct a temperature map or test Sigma Gravity; those require
combining every admitted observation/CCD contribution by adaptive region. No
temperature, lensing target or gravity parameter was opened or changed here.
