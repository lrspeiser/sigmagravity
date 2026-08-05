# Sigma V19L pre-science fixture failure

## Outcome

V19L stopped before reopening either cluster science array.  Its mandatory
curved-step fixture produced 64 significant discontinuity seeds but no retained
arc, so the injected-shock recovery gate failed.

This is a graph-topology implementation failure, not a source-data or gravity
result.  No V19L science seed, profile, arc, lensing target, gravity formula or
gravity parameter was evaluated.

## What V19L fixed

The quartic continuous null removed the smooth-field false positives that
survived V19K:

- uniform field: 0 passing seeds and 0 arcs;
- linear gradient: 0/1,752 passing seeds and 0 arcs;
- smooth radial beta profile: 0/236 passing seeds and 0 arcs; and
- injected circular density jump: 64/64 passing seeds.

The likelihood is therefore separating a local step from all three frozen
smooth controls.  The remaining failure occurs after likelihood fitting.

## Why the injected arc still failed

The frozen V19L graph considered 48 seed pairs within 40 kpc.  Every pair
passed the 30-degree normal and tangent tests, but only 22 straight segments
stayed completely inside the gap-closed candidate mask.  The resulting graph
fragmented into components whose two largest contained five seeds each.  The
unchanged geometry gate requires at least six nodes, so no component reached a
circle fit.

This exposes a category error in the revised topology rule.  A thin curved
pixel ridge is not faithfully represented by requiring the straight chord
between two sampled evidence points to remain inside that ridge.  The rule can
reject an unbroken curved path even when both endpoints and their orientations
are correct.

## Three-failure decision

The automatic-front lane has now failed in three materially different ways:

1. V19J's local contrast statistic promoted ordinary smooth contours to very
   long candidate fronts.
2. V19K's step likelihood fixed that statistic, but its seed-distance graph
   confused evidence spacing with the maximum empty ridge gap.
3. V19L fixed the smooth null and enlarged the sampling neighborhood, but its
   straight-path graph fragmented the known circular shock.

The project will not weaken the five-sigma score, six-node arc, 100-kpc length,
20-kpc empty-gap or masking gates.  It will also not introduce another tuned
seed radius.  The next causal X-ray measurement must change representation:
infer thermodynamic discontinuities from adjacent, independently fitted
spatial regions and their temperature/density posterior, rather than trying to
assemble a front from a thresholded one-pixel ridge.

The authoritative failure artifact is
`results/sigma_v19l_fixture_corrected_fronts/fixture_failure.json`.
