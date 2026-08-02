# P0682 spent multi-cluster deflection atlas

Frozen before metrics: 2026-08-02  
Verdict: numerical integrity **pass**; radial morphology **pass**; provisional constant amplitude **pass**; baryonic predictor **not advanced**

## Question

P0678 found that the compact-halo field missing from the RX J2129 absolute
baryonic calculation was broad, nearly radial, and aligned with the baryonic
field. P0682 asks whether that is an RX J2129 accident or a repeated property
of the six already-spent CLASH systems.

The test uses only deflection vectors. It does not differentiate a sampled
field, calculate convergence, fit a new gravity law, score new image roots, or
open the P0633/P0640 sealed outcomes.

## Construction

For each cluster, the frozen training-only `GR_plus_cluster_halo` NIE
parameters define a comparator deflection

\[
\boldsymbol\alpha_h(\boldsymbol\theta).
\]

The independently reconstructed Tian baryonic radial profile defines

\[
\boldsymbol\alpha_b(\boldsymbol\theta)
=\alpha_b(|\boldsymbol\theta|)\hat{\boldsymbol r}
\]

at the common reference source redshift `z=2`. Both are sampled at 6,912
points through the minimum-to-maximum observed strong-lens image radius of
each system. The halo field is split exactly into a radial-bin monopole and an
angular residual. The numerical reconstruction error is below `9.31e-18`.

The NIE field is a diagnostic target, not a unique dark-matter map. RXJ1347 is
a stress case because several fit parameters sit at bounds. It is excluded
from the primary constant/predictor calculation.

## System results

| Cluster | halo/baryon vector RMS | median radial halo/baryon | alignment cosine | angular RMS fraction | predictor target reliable? |
|---|---:|---:|---:|---:|---|
| MACS0329 | 12.63 | 11.64 | 0.960 | 0.153 | no |
| MACS0429 | 8.85 | 8.41 | 0.968 | 0.166 | yes |
| MACS1115 | 11.57 | 11.43 | 0.982 | 0.192 | no |
| MACS1931 | 7.16 | 6.83 | 0.965 | 0.259 | yes |
| RXJ1347 | 6.88 | 6.49 | 0.946 | 0.327 | no; boundary stress case |
| RXJ2129 | 6.11 | 6.12 | 0.984 | 0.188 | yes |

All five non-boundary systems pass the preregistered radial morphology rule:
alignment is at least `0.90` and the angular RMS fraction is no more than
`0.35`. Their median radial ratios have a geometric mean of `8.59`, a log
scatter of `0.127 dex`, and a multiplicative scatter of `1.34`. This passes
the frozen `0.20 dex` constant-amplitude gate.

This is a more specific clue than “add more lensing”: within the observed
annuli, the compact comparator mainly asks for a broad radial amplification of
the baryonic deflection. Angular structure remains important for precise image
positions, but it is not the dominant missing power.

## Predictor test

Six baryonic/global predictors were tested against the log median radial
ratio with one-cluster-out linear prediction. No predictor advances.

- Annulus pivot radius reaches the exact frozen rank threshold
  (`Spearman rho=0.80`) and a leave-one-out RMSE ratio of `0.719` relative to
  a constant.
- The baryonic enclosed-mass proxy gives the smallest leave-one-out ratio
  (`0.709`) but only `rho=0.50`.
- Only MACS0429, MACS1931, and RXJ2129 meet the target-reliability definition.
  The frozen minimum is four systems.

The radius clue is also not yet an admissible first-principles input: the
observed lens annulus depends on which background sources happened to be
available. A physical replacement should predict its scale from the baryonic
distribution, not from the target image locations.

## Why RXJ2129 is 6.1 here rather than 3.3 in P0678

The two numbers have different denominators and sampling domains. P0678 used
the P0674 registered 3D baryon map, whose reduced scalar-field RMS in its
strong-lens annulus was `2.449 arcsec`, and obtained halo/scalar `3.317`.
P0682 uses the spherical Tian baryonic profile used by the raw multi-cluster
runner; its RXJ2129 baryonic RMS is `1.176 arcsec` over the observed image
annulus, while the sampled halo RMS is `7.189 arcsec`, giving `6.113`.

This difference is evidence of baryonic-map uncertainty and representation
sensitivity, not a contradiction or a license to choose the preferred ratio.
The next law must be stress-tested against both baryonic reconstructions.

## Decision and next falsification

P0682 advances only two development clues:

1. the target missing field is robustly radial at first order; and
2. a constant dimensionless amplification near the tested cluster regime is
   a simpler starting hypothesis than a weakly supported object predictor.

It does **not** advance the value `8.59` as a universal gravity constant. The
next frozen spent-data stage should construct a smooth baryon-only radial law
that approaches Newton/GR at high acceleration, approaches an amplified
cluster branch without using lens-image radii, and obtains its transition and
effective extent from dimensionless baryonic field quantities. It must then:

1. reproduce the deflection-level atlas under both Tian and registered 3D
   baryon maps;
2. pass RXJ2129 image multiplicity, parity, critical-curve, and `<3 arcsec`
   spent-heldout gates;
3. remain stable across three solver resolutions and fixed baryon-mass
   sensitivities; and only then
4. open the frozen galaxy/cluster holdouts once under one global parameter
   vector, followed by Solar checks.

## Reproduction

```powershell
python scripts/run_p0682_spent_multicluster_deflection_atlas.py
python -m pytest tests/test_spent_deflection_atlas.py -q
```

Canonical artifacts are in
`results/p0682_spent_multicluster_deflection_atlas/`.

