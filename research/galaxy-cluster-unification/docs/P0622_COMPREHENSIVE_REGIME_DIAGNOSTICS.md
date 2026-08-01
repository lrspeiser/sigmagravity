# P0622 comprehensive regime diagnostics

## Outcome

The broad SigmaGravity validation pattern has now been copied into the current
galaxy-cluster formula program and strengthened into a regime-diagnostic suite.
It does not merely ask whether an aggregate score improves. It asks where the
formula changes sign, which object dominates a mean, which measured properties
track the failure, whether a lens root was lost, and what kind of evidence each
number actually represents.

The current construction is not promoted. The scalar parent remains reasonably
close to fixed RAR on galaxies, the Solar analytic proxies pass, and the angular
route has some real raw-lens response. It still misses the two central targets:
galaxy parity with fixed RAR and raw multi-cluster parity with the limited
compact-halo comparator.

| Test | Current formula | Comparator | Result |
|---|---:|---:|---|
| 131 SPARC galaxies, 968 untouched outer points | 12.571 km/s RMSE | fixed RAR 10.348 km/s | 1.215x worse |
| Shared +90-degree route, five fixed-geometry clusters | +1.685% mean | P0554 scalar | only 3/5 improve |
| Same route after omitting RXJ2129 | +0.046% mean | P0554 scalar | effectively neutral |
| Shared +90-degree route, five-cluster median | +0.116% | P0554 scalar | much smaller than mean |
| Frozen A383 full-refit transfer | 9.081 arcsec | P0554 9.097 arcsec | +0.174%, but inadequate absolute error |
| MS2137 full-refit transfer | 2/3 held-out roots | 3 required | not RMS-scoreable |
| Raw validation aggregate | 19.076 arcsec | compact halo 9.989 arcsec | 1.910x worse |
| Mercury analytic proxy | -1.730 mas/century | absolute margin 3.1 | pass |

## Formula under test

The scalar galaxy and radial cluster parent is P0554. The current angular
cluster layer is

\[
\boldsymbol\alpha_{\rm test}(\mathbf x)
=\boldsymbol\alpha_{0554}(r)
+\frac{Q^2}{1+\Delta_{80}}\,
\mathcal R_{90^\circ}
[\delta\boldsymbol\alpha_{\rm route}(\mathbf x)].
\]

Here:

- \(\Delta_{80}\) is the scalar excess at the radius containing 80% of the
  baryonic tracer weight.
- \(\Delta_{80}/(1+\Delta_{80})\) is the fraction of the excess assigned to
  the conservative route template.
- \(Q\) is the measured baryonic quadrupole asymmetry.
- \(Q^2/(1+\Delta_{80})\) sets the angular correction strength.
- The route width is \(0.23R_{80}\sqrt{1+Q^2}\).
- The return length is \(0.36R_{80}\).
- The whole route template is rotated by one shared +90-degree phase.
- No gravity parameter is fitted to an individual galaxy or cluster.

The angular term is defined to vanish for the project’s axisymmetric galaxy
translation and point-source Solar translation. That is a compatibility
property, not a successful galaxy or Solar observation of the angular term.

## What was copied and what was strengthened

The original SigmaGravity science suite supplied the broad structure:

- galaxy rotation curves;
- cluster lensing;
- Solar-System bounds;
- morphology splits;
- parameter sensitivity;
- holdout and comparator accounting.

P0622 adds safeguards that the older broad suite did not consistently enforce:

- Raw image positions are kept separate from GR/NFW-derived acceleration
  products.
- Root completeness is checked before any lens RMS is interpreted.
- Inherited null-regime scores are labeled rather than treated as new evidence.
- Galaxy interactions include sample size and a 2,000-draw bootstrap interval.
- Continuous correlations receive false-discovery-rate adjustment and a
  mass-controlled partial-rank check.
- Cluster averages receive leave-one-system-out influence analysis.
- A per-cluster best phase remains diagnostic and is forbidden as a promoted
  formula choice.
- Every scenario is labeled as raw observation, derived observation, analytic
  proxy, synthetic invariant, inherited result, or spent diagnostic.

## Galaxy regimes

### The important result is a bias reversal, not just a larger RMSE

| Galaxy condition | P0554 RMSE | Fixed-RAR RMSE | Ratio | Mean P0554 residual |
|---|---:|---:|---:|---:|
| All | 12.571 | 10.348 | 1.215 | -2.470 km/s |
| Dwarf mass | 9.350 | 6.315 | 1.481 | -6.038 km/s |
| Intermediate mass | 11.907 | 9.828 | 1.212 | -4.476 km/s |
| Giant mass | 15.254 | 12.808 | 1.191 | +4.673 km/s |
| Gas rich | 11.222 | 8.625 | 1.301 | -6.247 km/s |
| Gas poor | 12.140 | 9.628 | 1.261 | +1.001 km/s |
| Very deep, below 0.03 \(a_0\) | 16.249 | 10.239 | 1.587 | -12.100 km/s |
| Moderate inclination | 9.407 | 10.197 | 0.923 | -1.076 km/s |

The scalar formula is too weak in the deepest-acceleration and dwarf regimes,
then crosses to an overprediction in giant systems. This points to the scalar
amplitude or transition law. The angular route cannot repair it because that
layer is exactly null in the axisymmetric SPARC translation.

The gas-rich versus gas-poor contrast initially looks like a reason to add a
gas parameter. The continuous analysis warns against that conclusion. Gas
fraction correlates with mean velocity bias before controlling for mass
(Spearman \(\rho=-0.288\), \(p=8.6\times10^{-4}\)), but the partial relation
after controlling ranked baryonic mass falls to \(\rho=-0.063\), \(p=0.474\).
The raw gas signal is therefore largely a mass-family signal.

The cleanest remaining predictor of relative error is inclination: the
mass-controlled rank relation is \(\rho=0.254\) with adjusted
\(q=0.025\). This should first be treated as a possible observational or
deprojection systematic, not immediately as new gravity. If a physical formula
responds strongly to viewing angle in an axisymmetric rotation-curve test, it
needs a clear three-dimensional reason.

### Interaction bins

The interaction scan evaluates 32 bins containing at least five galaxies.
Some apparent winners are small and uncertain. For example, the best nominal
bin—seven intermediate-mass mixed disk/bulge galaxies—has a P0554/RAR ratio of
0.906, but its bootstrap interval is 0.575–1.776. It is not a secure win.

The clearest failure interaction is high-surface-brightness galaxies with
rising outer curves: 12 galaxies, P0554/RAR ratio 1.828, bootstrap interval
1.279–2.446, and mean residual -7.135 km/s. Observed outer-curve shape is a
diagnostic target and cannot be used as an input to a blind prediction, but it
shows exactly which velocity profile the scalar law fails to build.

## Cluster regimes

### One responsive cluster dominates the mean

| Cluster | \(Q\) | \(\Delta_{80}\) | +90-degree change | Outcome |
|---|---:|---:|---:|---|
| MACS0329 | 0.278 | 7.218 | +0.116% | weak improvement |
| MACS0429 | 0.222 | 7.010 | -0.320% | worse |
| MACS1115 | 0.374 | 7.122 | +0.614% | weak improvement |
| MACS1931 | 0.512 | 11.420 | -0.226% | worse |
| RXJ2129 | 0.294 | 10.258 | +8.241% | strong improvement |

The five-system mean is +1.685%, but the median is +0.116%. Removing RXJ2129
reduces the mean to +0.046%. The angular route therefore has a real response,
but the aggregate improvement is not broadly distributed.

Neither \(Q\), \(\Delta_{80}\), route strength, nor scalar baseline error
explains the sign in this five-system sample. No cluster correlation survives
multiple-testing correction. The largest descriptive relation is with the
pre-existing residual-vector alignment, not the route amplitude. This agrees
with the P0617 first-order result: support changes how much the field moves an
image, while angular alignment decides whether that movement approaches or
recedes from the observed location.

The formula can therefore derive an amplitude from baryons but cannot yet
derive the direction that makes that amplitude useful. A universal +90-degree
rotation is not that missing rule, and selecting each cluster’s best phase
would simply replace dark-matter object fits with a gravity phase fit.

### Topology remains a separate failure mode

All positive phases in the current five-system scan keep 18/18 held-out roots.
Negative phases lose one RXJ2129 root. In the bounded width-strength factorial,
only 1 of 27 variants is root-safe across all required systems. Small changes
can move a source across a caustic, so a lower finite residual after losing a
root is not an improvement.

MS2137 demonstrates the reporting rule: both scalar and routed refits recover
only 2 of 3 held-out images, so no comparative RMS conclusion is allowed.

## Parameter lessons

The new matrix combines 19 scalar, route, support, and phase coordinates.

- Screen exponent has the largest Solar response and can cross the Solar
  boundary while barely changing galaxies.
- Mass-dependent transition radius strongly changes galaxies and derived
  clusters, but it prefers conflicting directions in raw cluster groups and
  crosses Solar/root boundaries.
- Weak concentration leakage can change RXJ2129 roots and residuals far more
  than its galaxy score, showing that it is a topology lever.
- Photon response changes lensing while leaving galaxy dynamics and Mercury
  unchanged, but different raw clusters prefer different directions.
- Route strength and width can change roots rapidly; contrast saturation is
  nearly marginal in the tested range.
- Width, return length, and center-crossing rules change response magnitude but
  do not supply a universal beneficial sign.
- Angular phase is the largest recent route response, but it is
  cluster-dependent and dominated by RXJ2129.

No single coordinate is simultaneously dominant, Solar-safe, topology-safe,
and directionally consistent across galaxies and raw clusters.

## What the tests establish—and what they do not

The suite establishes that:

1. The P0554 scalar error has reproducible mass, acceleration, inclination, and
   rotation-shape structure.
2. The scalar bias reverses from dwarfs to giants.
3. The angular route response is heterogeneous across clusters.
4. RXJ2129 dominates the current mean cluster gain.
5. Root topology can change non-smoothly under small parameter and phase
   changes.
6. The current exact galaxy/Solar route nulls preserve compatibility.

It does not establish that:

- the inclination relation is new gravity rather than deprojection error;
- the observed rotation-curve shape can be used predictively;
- the five spent phase systems are a new validation sample;
- the derived CLASH acceleration products are theory-neutral raw lensing;
- the Solar proxies replace a multi-planet ephemeris likelihood;
- one compact-halo score represents all dark-matter solutions;
- the route formula is a covariant field theory.

## Next discriminating program

1. Develop a universal scalar correction on a development subset, freeze it,
   and require it to remove the dwarf-to-giant sign reversal on untouched
   galaxies. Inputs may include baryonic mass, acceleration, potential depth,
   and three-dimensional baryonic structure, but not observed outer velocity
   slope.
2. Derive an angular direction from baryons alone—gas/star centroid offset,
   external tidal axis, or resolved multipole orientation—and freeze it before
   image scoring.
3. Test at least five new clusters with complete scalar baselines. Require all
   held-out roots and a positive leave-one-cluster-out result after omitting the
   most responsive system.
4. Replace the Solar proxy layer with a joint multi-planet ephemeris
   likelihood.
5. Compare at matched flexibility: one universal formula against both compact
   halo controls and object-specific halo fits, with parameter counts shown.

## Reproduction

```powershell
python scripts/run_p0622_validation_suite.py
```

Use `--skip-build` to validate existing artifacts without rebuilding the
regime tables and figure. The runner executes 77 selected scientific checks
covering invariants, routing, morphology, Solar safety, P0554 sensitivities,
the P0612–P0620 transfer chain, and P0622’s new differential contracts.

The machine-readable result directory contains point predictions, 32 galaxy
regime scores, 32 galaxy interaction scores, 22 continuous correlation tests,
seven cluster summaries, the full phase-response matrix, leave-one-cluster-out
influence, a 19-row parameter-domain matrix, a provenance-labeled scenario
matrix, the JSON report, and the summary figure.
