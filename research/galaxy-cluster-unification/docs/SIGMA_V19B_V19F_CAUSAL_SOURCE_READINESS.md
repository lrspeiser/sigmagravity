# Sigma v19B-v19F causal-source readiness checkpoint

## Outcome

The project is still pursuing the root equation that must predict apparent
halo scale, shape, displacement, and strength from baryons alone. The current
causal-assembly branch has now passed its source-data acquisition and detector
reduction gates on a new, target-blind mechanism-development pair: the Bullet
Cluster and Abell 2146.

This is a **source-readiness result**, not evidence that the mechanism works.
No replacement-cluster lensing coordinate, inferred halo map, strong-lensing
model, or gravity score has been opened.

## What is now available

| Stage | Bullet Cluster | Abell 2146 | Gate result |
|---|---:|---:|---|
| Secure member rows with quoted velocity uncertainty | 78 | 63 | pass |
| Published direct shock significance | 6.83 sigma | 6.50 sigma | pass |
| Published time-since-passage interval | 0.10-0.20 Gyr | 0.24-0.28 Gyr | usable only as an uncertainty prior |
| Frozen Chandra observations | 10 | 10 | pass |
| Reprocessed observations | 10 | 10 | pass |
| Flare-cleaned exposure | 561.128 ks | 418.013 ks | pass |
| Worst retained-exposure fraction | 0.855990 | 0.993911 | pass; gate was 0.50 |
| Point-source masks across observations | 874 detections | 532 detections | complete |
| Matched blank-sky products | 10 | 10 | pass |

The Chandra chain used CIAO 4.18.0, CIAO contrib 4.18.2, CALDB main
4.12.4, and ACIS background events 4.9.7. The inherited isolated environment
passed all 37 official smoke tests. Nineteen observations are VFAINT mode. The
older Bullet ObsID 554 is FAINT mode and was processed without pretending that
VFAINT cleaning was available.

The reprocessing stage produced 240 calibrated files totaling 1,223,566,899
bytes. The cleaning stage produced 460 files totaling 4,109,457,603 bytes in
the local reproducible scratch area. Every command, event header, input and
output hash, retained exposure, source-mask count, and blank-sky scaling value
is represented in the machine-readable reports.

## How this bears on halo size

The current hypothesis is not that a cluster's dark-matter halo radius is a
free number. It is that the apparent halo is the metric response to a measured
baryonic state containing more information than an instantaneous density map.
The candidate input categories are:

1. the present gas and galaxy mass distribution;
2. the relative position and velocity distribution of collisionless baryons;
3. the positions, normals, and propagation speeds of merger shocks;
4. the elapsed assembly interval and its projection uncertainty; and
5. the overlap geometry of the two baryonic substructures.

The output must be a field. Apparent halo radii such as `R50` and `R80`, its
centroid, ellipticity, and shear orientation must be measurements of that
predicted field rather than fitted cluster parameters.

The earlier thermal and member-stress experiments showed that baryonic state
contains some information about apparent **extent**, but not a transferable
amplitude, centroid, or shear direction. V19 asks whether a genuinely causal
assembly coordinate supplies the missing phase information. The new data make
that question testable; they do not answer it yet.

## What remains sealed and unknown

We do not yet know whether the causal variable is identifiable from these
data, whether it transfers between the two systems, or whether it improves on
baryons-only gravity. In particular, the following work must precede any
lensing comparison:

1. freeze absolute and relative astrometric registration;
2. freeze a common exposure/background-corrected source-map grid;
3. freeze an automated surface-brightness edge and shock-normal estimator;
4. freeze adaptive spectral regions and density/temperature uncertainties;
5. construct a projection and clock ensemble from source-side constraints;
6. define one target-blind causal source from those observables; and
7. hash that source and its universal transfer rule.

Only after step 7 may the Bullet and Abell 2146 lensing targets be opened. A
successful development transfer would still require a different four-cluster
validation and holdout sample, because Abell 2146 lacks a reported
spectroscopic lensed family and cannot satisfy the final raw-lensing gate.

## Decision boundary

The branch advances only if the same source construction transfers between
both clusters without cluster labels or cluster-specific amplitude, length,
orientation, or clock parameters. If the source is not identifiable, or if it
fails the frozen full-field amplitude, extent, centroid, and shear gates, the
failure is recorded without adding a fitted halo scale.

Machine-readable records:

- `results/sigma_v19b_replacement_cluster_screen/report.json`
- `data/raw/sigma_v19c_assembly_sources/provenance.json`
- `results/sigma_v19d_member_catalog_extraction/report.json`
- `results/sigma_v19e_chandra_acquisition/provenance.json`
- `results/sigma_v19f_chandra_repro/report.json`
- `results/sigma_v19f_chandra_cleaning/report.json`
