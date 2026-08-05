# Sigma V19BH blind-cluster admission protocol

## Decision

The next raw-lensing sample will be selected from a broad public source
universe, but no cluster is a holdout merely because HST images or a published
lens model exist.  V19BH freezes an outcome-blind admission sequence and a
balanced metadata shortlist.  It opens no multiple-image coordinate, parity,
critical curve, convergence map, shear map or Sigma residual.

The provisional shortlist contains four systems on the relaxed side of a
published X-ray morphology classification and four disturbed systems.  All
eight aliases were absent from tracked analysis-bearing project files at the
frozen pre-protocol commit.  Their status is still **not admitted**.  Each must
independently pass the family, uncertainty, baryon-completeness and comparator
gates before six cluster IDs can be frozen.

## Why this improves the cluster test

The Hubble Frontier Fields are excellent data, but they are not a fresh test
for this project.  All six occur in prior project analyses or inputs; MACS
J0416 in particular was used repeatedly for mechanism development.  Reusing
them as supposedly untouched evidence would make the final score circular.

The opposite error is to accept a fresh system with too little information.
PLCK G004.5-19.5 illustrates it: public RELICS products and a published model
exist, but the published strong-lensing constraint is one three-image family.
That is below the frozen minimum of three secure families and eight images and
cannot test image topology broadly.

The most useful public starting point is the SGAS archive.  It provides HST
imaging and conventional lens models for 37 clusters, with the accompanying
study reporting lensing constraints and spectroscopy.  A newer Chandra study
covers 28 SGAS lenses, has more than 1,000 X-ray counts inside `R500` for every
system, and classifies them from concentration, asymmetry, centroid shift and
X-ray-to-BCG separation.  That gives us a source-side way to require relaxed
and disturbed clusters before seeing whether Sigma succeeds.

## Frozen admission sequence

1. Audit aliases and previous project exposure using a frozen repository
   commit.
2. Establish source-side readiness from stars, hot gas, BCG, intracluster
   light, member galaxies, masks, PSFs and uncertainty ensembles.  A
   lens-derived mass map may not repair a missing baryon component.
3. Read only constraint counts and covariance metadata.  Require at least
   three secure families, one spectroscopic family, eight images and a stated
   positional-uncertainty rule; do not read their coordinates.
4. Select six systems spanning both sides of dynamical state, cool-core state,
   mass and merger projection.  Freeze the IDs, source manifests and sealed
   target-file hashes only after all source gates pass.
5. After one action and no more than five universal constants are frozen, open
   the raw coordinates once and score root recovery, multiplicity, topology,
   positional RMS and baryon-to-halo gap closure.

The metadata-only shortlist is intentionally larger than six.  A system can
fail readiness without forcing us to relax a scientific gate or choose a
replacement after seeing a target.

## Point of view beyond the headline tests

The same equation owes predictions in a strict order.

First come weak lensing and merging-cluster offsets.  Both use the same
quasistatic Weyl potential and resolved baryons as strong lensing.  If a model
gets a radial acceleration amplitude right but misses cross shear, critical
curve orientation or the direction of a merger offset, the spatial response
is wrong; another amplitude adjustment is not a repair.

Next come dwarf satellites, streams and dynamical friction.  The long-wave
idea predicts environmental dependence where host and satellite fields
overlap and may predict smooth changes in stream precession.  It does not
automatically explain compact stream gaps or substructure-lensing anomalies.
Those require the action to generate compact field structure.  Likewise,
dynamical friction requires a derived destination for orbital energy—baryonic
wakes, Sigma radiation or field excitation—not a verbal analogy to a dark
matter wake.

Cosmic growth and the CMB are last because they require a covariant background
and perturbation theory.  They are also decisive: a field that works only in
late-time galaxies but cannot preserve early gravitational potentials or
produce the observed growth history is not a replacement for dark matter.

Solar-System and relativistic tests remain mandatory hard vetoes, but they do
not choose the galaxy/cluster formula at this stage.

## Reproduction

```powershell
python scripts/check_sigma_v19bh_blind_cluster_admission.py
python -m pytest tests/test_sigma_v19bh_blind_cluster_admission.py -q
```

The machine-readable result is
`results/sigma_v19bh_blind_cluster_admission/report.json`.

## Public sources

- SGAS HST images and lens-model archive: <https://archive.stsci.edu/hlsp/sgas>
- Chandra strong-lens dynamical-state study: <https://arxiv.org/abs/2511.12707>
- RELICS reserve archive: <https://archive.stsci.edu/hlsp/relics>
