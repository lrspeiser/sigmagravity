# Sigma V19W3 full response recovery plan

## Purpose

V19W3 removes the delay between the terminal base V19W report and a corrected
response archive without touching the running process. Its launch gate requires
both base process IDs to be absent and the final base report plus its product
index to match the frozen config and runner hashes.

The final recovery list is deliberately not guessed at freeze. It is the 5,082
row V19U manifest minus every base checkpoint that passes an independent audit
of identity, manifest event counts, cell gates, PHA channel histograms, PHA
links, ARF/RMF contents, file sizes and SHA-256.

## Recovery path

Every absent or invalid checkpoint uses the fully commissioned V19W2 path:

1. create a WCS-preserving binary image for the exact V19M integer bin;
2. verify source and background 0.5--7 keV counts against the frozen manifest;
3. materialize both exact event subsets;
4. choose an actual source event nearest the DETX/DETY centroid and require its
   detector-frame ICRS conversion to return the declared CCD;
5. create the unchanged weighted source/background PHA, ARF and RMF products;
6. use the separately commissioned zero-background path where necessary; and
7. repeat every PHA, response, link, blank-sky scale, size and hash audit.

V19W3 allows one recovery attempt per missing cell. A failure remains retained
and closes the protocol rather than silently changing membership or response
settings.

## Unified archive

Products are not copied into the base archive. The final index points each
manifest task to either its independently valid base checkpoint or its V19W3
recovery checkpoint. A second complete audit must validate 5,082 unique cells,
20,328 products, and every recorded hash before a successor V19X configuration
may be frozen.

The original V19X configuration expects one complete base V19W archive and is
therefore not authorized by V19W3. A successor must consume the unified index
and validate both archive roots.

## Current state

At freeze, base V19W was still running and no final report, final missing-cell
count or V19W3 product existed. The runner's `--status-only` mode is read-only;
ordinary execution fails before creating the recovery root while the base
process or final-report gate is unsatisfied.

## Reproduction after the terminal gate

```powershell
wsl bash -lc '/home/henry/miniforge3/bin/conda run --no-capture-output -n sigma-ciao-4.18 python /mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification/scripts/run_sigma_v19w3_full_response_recovery.py'
```
