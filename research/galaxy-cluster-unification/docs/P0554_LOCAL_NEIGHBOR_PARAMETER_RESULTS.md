# P0554 local-neighbor parameter results

## Outcome

The data are more sensitive to **the physical scale over which nearby baryons
define the route direction** than to whether neighbor influence falls as
approximately $1/R$, $1/R^2$, or $1/R^3$. The inherited 200 kpc softening is
the aggregate optimum within the frozen grid, while every smaller and larger
tested scale is worse. That makes a finite cluster-scale coherence length the
clearest clue in this screen.

It is not yet a universal constant. RX J2129 prefers 100 kpc, MACS0429 prefers
250 kpc, and MACS0329 and MACS1115 prefer 200 kpc. No variant passes the frozen
rule for an exact nonlinear follow-up, so no formula is promoted.

## Formula varied

For member galaxy $i$, the generalized directional field is

$$
\mathbf v_i=\sum_{j\ne i} w_j^{p_w}
\frac{\mathbf x_j-\mathbf x_i}
{(|\mathbf x_j-\mathbf x_i|^2+s^2)^{(p_d+1)/2}},
\qquad
\hat{\mathbf d}_i=\frac{\mathbf v_i}{|\mathbf v_i|}.
$$

Here $s$ is a smoothing or coherence length, $p_d$ controls distance falloff,
and $p_w$ controls how strongly bright baryonic members dominate the direction.
At distances much larger than $s$, $p_d=1$ is a $1/R$ influence and $p_d=2$
is inverse-square. The parent is $(s,p_d,p_w)=(200\,\mathrm{kpc},2,1)$.

The test varied one coordinate at a time: ten smoothing scales from 50 to
1000 kpc, six falloff powers from 0.5 to 3, and six light-weight powers from
0.5 to 2. Duplicate parent settings were evaluated once, giving 20 formulas
on five clusters and 100 route fields.

## What moved the result

| Coordinate | Best tested value | Aggregate change vs parent | Clusters improved | RMS span across profile |
|---|---:|---:|---:|---:|
| Coherence length $s$ | 200 kpc, parent | 0.000% | parent | **0.1164 arcsec** |
| Light-weight power $p_w$ | 2.0, boundary | +0.337% | 2 of 4 | 0.1062 arcsec |
| Distance power $p_d$ | 2.5 | +0.053% | 2 of 4 | 0.0261 arcsec |

The apparent preference for $p_w=2$ is not universal: it gains 0.767% in
MACS0329 but worsens MACS1115 by 0.034%, and it sits at the edge of the tested
range. It may be bright-member dominance rather than a physical exponent.

The $1/R$ case is 0.103% worse than the inverse-square parent in the primary
aggregate. Moving to $p_d=2.5$ gains only 0.053%, with two clusters improving
and two worsening. The precise distance exponent is therefore the weakest of
the three tested levers.

## Cross-domain meaning

Every variation retains the parent SPARC outer error of 12.571 km/s, radial
CLASH error of 0.1964 dex, Mercury precession prediction of -1.730
milliarcseconds per century, and all Solar proxy passes. This is by
construction: the directional map has its circular monopole removed and is
absent in the point-mass Solar controls. These are preservation tests, not new
improvements in galaxies or the Solar System.

The strongest defensible observation is therefore narrow: a route direction
derived from baryons has a measurable cluster-scale correlation length, while
the current data do not select a universal $1/R^n$ propagation law. The next
physically useful input is a registered, independently measured map of all
baryons—member galaxies, hot gas, and diffuse intracluster light—to see whether
the differing preferred scales collapse to one value.

## Limits

This is a fixed-geometry, Jacobian-linearized sensitivity screen on spent
clusters. The catalogs omit accepted registered gas and diffuse ICL, four of
the earlier five exact geometry fits touched nuisance bounds, and small score
changes can reflect catalog selection or projected geometry. An exact root
test was deliberately not triggered because no candidate improved at least
0.2%, improved at least three of four primary clusters, and kept every cluster
within the 0.5% worsening limit.

## Reproduction

```powershell
python scripts/run_p0554_local_neighbor_parameter_screen.py
python scripts/run_p0554_local_neighbor_parameter_screen.py --postprocess-only
python -m pytest tests/test_route_template.py tests/test_p0554_local_neighbor_parameter_screen.py -q
```

Machine-readable products are in
`results/p0554_local_neighbor_parameter_screen/`.
