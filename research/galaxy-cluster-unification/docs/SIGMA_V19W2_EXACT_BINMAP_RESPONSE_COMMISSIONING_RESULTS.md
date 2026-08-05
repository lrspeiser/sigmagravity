# Sigma V19W2 exact-binmap response commissioning results

## Decision

V19W2 passed all six aggregate gates under protocol 1.2.0. The corrected
response path is authorized for a separately frozen recovery of every task
missing from the final base V19W archive, after the still-running base process
exits. V19X is not authorized by this commissioning result.

The pass changes no region, binmap label, event assignment, energy interval,
response weighting, blank-sky scale, spectrum, temperature, lensing input or
gravity parameter. It does not test the long-wave Sigma hypothesis. It repairs
the observational path needed before that hypothesis can be tested against gas
state and raw cluster lensing.

## Exact results

| Cell | Failure class | Source 0.5--7 keV | Background 0.5--7 keV | Background all-energy | Derived reference CCD | Four-product bytes |
|---|---|---:|---:|---:|---:|---:|
| `BULLET_bin45_obs554_ccd3` | source polygon +1 | 85 | 17 | 38 | 3 | 7,413,120 |
| `BULLET_bin135_obs3184_ccd3` | background polygon +1 | 346 | 919 | 1,905 | 3 | 7,401,600 |
| `BULLET_bin232_obs4984_ccd3` | source and background +1 | 271 | 323 | 689 | 3 | 7,427,520 |
| `BULLET_bin154_obs4985_ccd3` | reprojected two-event centroid off CCD | 2 | 14 | 44 | 3 | 7,453,440 |
| `BULLET_bin36_obs4984_ccd3` | AF_UNIX path and zero blank-sky rows | 14 | 0 | 0 | 3 | 7,398,720 |

Every cell passed:

- the frozen V19U source/background event counts;
- the second count after exact-mask event materialization;
- exact source and background PHA channel histograms;
- finite positive ARF and finite nonzero RMF audits;
- PHA background, ARF and RMF link checks;
- effective blank-sky scale agreement within (10^{-6}); and
- response-reference placement on the manifest CCD.

## What the commissioning learned

The authoritative V19P/V19Q assignment is the integer V19M binmap, not DS9
polygon membership. A few polygon boundaries include one event belonging to an
adjacent bin. Exact binary masks restore the disjoint partition.

CIAO cannot directly emit a Boolean `dmimgcalc` image, so the mask writer uses
a WCS-preserving image of ones followed by `dmimgthresh` at the closed integer
interval `bin_id:bin_id`. NumPy independently verifies every output pixel.

Passing a pixel-mask virtual filter directly to `dmextract` propagates a
`MASK` image instead of the WMAP needed by weighted response generation. The
verified workaround first materializes the exact event subset with `dmcopy`,
then runs the unchanged weighted response settings on that physical subset.

Reprojected SKY coordinates are not safe detector calibration references for
very sparse cells. The final rule selects the actual 0.5--7 keV event nearest
the exact subset's DETX/DETY centroid, maps that detector medoid through the
observation's unreprojected source-excluded event frame, and requires the
result to return the declared CCD. This is deterministic calibration metadata,
not a fitted coordinate.

For an exactly empty blank-sky subset, source-only `specextract` creates the
weighted response, `dmextract` creates the valid zero-count background PHA,
and the ordinary `BACKFILE` and blank-sky scaling audits are then applied.

## Transparent failed-closed history

The commissioning did not hide unsuccessful invocations. Versions 1.0.0--1.0.2
stopped before a mask or response because of an absent Astropy dependency, one
transcribed manifest count, and an unsupported Boolean image. Version 1.0.3
proved the five exact mask counts but exposed the MASK/WMAP interaction.
Version 1.1.0 passed four cells and isolated the remaining detector-reference
problem. Each correction was frozen before its next scientific run and used a
new scratch root when failed response products existed.

## Integrity and claim boundary

- Config SHA-256: `df3041c54bf14fc8c7071229ef5facc4011be1e94645027ff5f37903e9d516d7`
- Runner SHA-256: `7bfa95bacc2d6904fb67262b1523cb58fab9cb492a0ccfcec15ae1ea41185810`
- Report SHA-256: `830f7d2cca5f84b2970a013c5cd89be9652d7628c107dbf5bbd5dfb966751efc`
- Scratch root: `/home/henry/sv19w2_v120`
- Base V19W archive modified: no
- Spectrum combined or fitted: no
- Temperature, density, Mach state or propagation speed inferred: no
- Lensing, halo or gravity payload opened: no
- Gravity formula or parameter changed: no

## Next authorized step

Allow the frozen base V19W process to exit normally. Then freeze a successor
that derives its workload from the final base report, runs the V19W2 correction
for every missing task, and independently audits all 5,082 manifest cells and
all four products per cell across the base and recovery archives. Only that
complete, hash-exact archive can authorize an updated V19X gas-combination
protocol.

## Reproduction

```powershell
python -m pytest tests/test_sigma_v19w2_exact_binmap_response_commissioning.py -q
wsl bash -lc '/home/henry/miniforge3/bin/conda run --no-capture-output -n sigma-ciao-4.18 python /mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification/scripts/run_sigma_v19w2_exact_binmap_response_commissioning.py'
```
