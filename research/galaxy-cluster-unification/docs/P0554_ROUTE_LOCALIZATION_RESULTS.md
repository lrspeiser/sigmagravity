# P0554 route-localization results

## Outcome

Changing where the angular response is routed matters more consistently than
turning up its global amplitude, although the measured effect remains small.
A route aimed along the softened gravitational direction of neighboring
baryonic member galaxies outperformed the original route aimed at one global
light centroid.

The fixed-geometry screen compared 12 formulas: no route, global-center and
fixed-origin routes, three global/local mixtures, three local-neighbor
softening scales, and three symmetric bend angles. All used the topology-safe
eta = 0.30 amplitude and conserved the route-map integral.

The screen winner was the pure local-neighbor direction with 200 kpc
softening. Its four-transfer-cluster local-response score improved by 0.824%
versus no route, compared with 0.580% for the global-centroid route. It moved
all four transfer clusters in the favorable direction in that diagnostic.

## Local-neighbor equation

For member galaxy $i$, define the cataloged baryonic-neighbor vector

$$
\mathbf v_i=
\sum_{j\ne i}w_j
\frac{\mathbf x_j-\mathbf x_i}
{(|\mathbf x_j-\mathbf x_i|^2+s^2)^{3/2}},
\qquad
\hat{\mathbf d}_i=\frac{\mathbf v_i}{|\mathbf v_i|}.
$$

The tested endpoint is

$$
\mathbf y_i=\mathbf x_i+L_i\hat{\mathbf d}_i,
$$

with the inherited route length $L_i$, $s=200$ kpc, and the same normalized
source weights and extent gate as the earlier A0279 construction. The route
map is normalized before being converted into a curl-free deflection field,
and its circular monopole is removed.

This does not assert that gravity obeys an inverse-square member-to-member
propagation law. It asks whether the surrounding baryonic field gives a more
useful direction than a single cluster center.

## Exact-refit and global-root follow-up

The 200 kpc candidate was then refit from eight starts in each cluster and
searched globally across all 27 source families. The fair position comparison
uses the 15 primary-transfer families for which all three formulas assign every
published image.

| Formula | Equal-family RMS | Improvement vs no route |
|---|---:|---:|
| No angular route | 6.939 arcsec | 0.000% |
| Global-centroid route | 6.910 | +0.412% |
| Local-neighbor route | 6.887 | **+0.745%** |

The local route improves another 0.335% relative to the global route. It also
improves the exact observed-seed held-out RMS in all four transfer clusters.
It misses the frozen strong-survival gate because the primary assignment gain
is below 1%.

The topology outcome is unusually clean: no one of the 27 family root counts
changes between no route, the global route, and the local route. All three have
eight missing-multiplicity families, twelve exact families, one
demagnified-only-surplus family, six potentially observable-surplus families,
seven screened surplus roots, and 17/18 observed-seed held-out roots.

## Per-cluster lesson

| Cluster | Global route vs no route | Local route vs no route |
|---|---:|---:|
| MACS0329 | +0.936% | +1.296% |
| MACS0429 | -0.285% | -0.068% |
| MACS1115 | +0.014% | +0.030% |
| RXJ2129 | -1.088% | **+1.418%** |
| MACS1931, selection system | +0.468% | -1.282% |

The important change is RXJ2129: the global-center route has the wrong sign,
while the neighbor direction flips it to a positive response. MACS1931 moves
the wrong way, reinforcing that one center-return template is not universal.

## Cross-domain interpretation

All localization variants retain the same SPARC, radial CLASH, Mercury, and
Solar scores as the photon-softness radial parent. This follows from the
zero-monopole angular construction and is a preservation statement, not a new
fit.

The current evidence supports a modest universal statement: **the direction
of an angular gravitational response is more usefully tied to the surrounding
baryonic field than to one global center.** It does not establish the 200 kpc
number, the inverse-square exponent, or the existence of physical gravity
flow. Those are the next parameters to perturb.

## Limits

The candidate was selected and tested on spent clusters; member light omits
accepted registered gas and diffuse ICL; four of five geometry solutions touch
nuisance bounds; and the construction remains a projected phenomenology. No
formula is promoted.

## Reproduction

```powershell
python scripts/run_p0554_route_localization_screen.py
python scripts/run_p0554_route_localization_screen.py --postprocess-only
python scripts/run_p0554_local_neighbor_exact.py
python scripts/run_p0554_local_neighbor_exact.py --postprocess-only
python -m pytest tests/test_route_template.py tests/test_p0554_route_localization_screen.py tests/test_p0554_local_neighbor_exact.py -q
```

Machine-readable products are in `results/p0554_route_localization_screen/`
and `results/p0554_local_neighbor_exact/`.
