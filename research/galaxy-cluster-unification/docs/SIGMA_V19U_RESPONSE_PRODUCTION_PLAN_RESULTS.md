# Sigma V19U response-production plan result

V19U **passed every frozen planning gate** and authorizes only the four-cell
throughput pilot.  It does not yet authorize the full response-production run.

## Exact workload

| Quantity | Frozen result |
|---|---:|
| Bullet response cells | 3,812 |
| Abell 2146 response cells | 1,270 |
| Total response cells | 5,082 |
| Deterministic batches | 80 |
| Maximum cells per batch | 64 |
| Final batch cells | 26 |
| Maximum concurrent cells | 4 |

Every V19Q task occurs exactly once.  The task order is fixed by cluster,
observation, adaptive-region identifier and CCD; a response or temperature
outcome cannot affect its batch membership.

## Frozen throughput pilot

Two source-count quantiles were selected independently within each cluster
after the protocol was committed.  This gives four new cells:

| Cluster | Quantile | Region | ObsID | CCD | Source events | Background events |
|---|---:|---:|---:|---:|---:|---:|
| Bullet | 0.25 | 257 | 5355 | 2 | 82 | 98 |
| Bullet | 0.75 | 23 | 5356 | 2 | 266 | 38 |
| Abell 2146 | 0.25 | 75 | 13020 | 0 | 153 | 57 |
| Abell 2146 | 0.75 | 101 | 13120 | 2 | 217 | 146 |

The pilot passes only if all four cells complete under the unchanged V19R
response rules.  A failed cell is retained and reported; it cannot be silently
dropped.  One identical retry is allowed for an execution failure, but no
scientific input or fit setting may change.

## Resource bounds

The four calibrated V19R response products occupy 6,641,280 bytes for the
commissioning cell.  Applying that measured size to every cell projects
33.751 GB (31.433 GiB) for the response archive, or about 34.395 GB if every
grouped PHA were retained too.  V19U requires at least 84.377 GB free before a
full run; the frozen WSL capacity snapshot had 777.893 GB free.

The first response took approximately 42 seconds by operator wall-clock
observation.  That implies roughly 59.3 serial hours or an ideal four-worker
floor of 14.8 hours.  These are planning estimates, not measured throughput.
The pilot must replace them with concurrent timing and peak-storage evidence.

The response archive will remain in a hashed production store rather than
being committed wholesale to Git LFS.  Compact manifests, reports and derived
maps remain versioned.  No additional spectrum, thermodynamic map, lensing
target or gravity parameter was created or changed in V19U.
