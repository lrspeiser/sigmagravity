# Sigma V19BD two-cluster directional-source uncertainty results

## Decision

V19BD **passed every frozen source-only gate**. It evaluated all 8,192 draws
for both collisionless member ensembles, reproduced the V19BA/V19BC aggregate
centroid diagnostics exactly, and quantified scale-free directional morphology
without reading gas, lensing, halo, or gravity targets.

No propagation length, coupling, phase, polarization, smoothing kernel, or
gravity formula was selected.

## Primary scale-free result

| Statistic | Bullet median (95% interval) | Abell median (95% interval) |
|---|---:|---:|
| Second-moment offset / light RMS radius | 0.1233 (0.0872–0.1531) | 0.1272 (0.0657–0.1811) |
| Opposite-current separation / light RMS radius | 0.4972 (0.4655–0.5293) | 0.6823 (0.5983–0.7887) |
| Current/light-major-axis `cos(2 Delta theta)` | 0.9983 (0.9805–1.0000) | 0.7135 (0.5930–0.8265) |
| Light-axis ellipticity | 0.4730 (0.4669–0.4796) | 0.7645 (0.7508–0.7805) |

The paired posterior difference for the normalized second-moment offset is

\[
 d_{\Pi,\rm Abell}-d_{\Pi,\rm Bullet}
 =0.0005\quad[-0.0673,\ 0.0716]\quad(95\%).
\]

Within these catalog models, both clusters therefore place the
velocity-dispersion-weighted centroid about 12% of one luminosity RMS radius
away from the luminosity centroid. That is the most transferable source-side
feature found here.

## What does not transfer as one template

The signed-current patterns are clearly different after size normalization:

\[
 d_{j,\rm Abell}-d_{j,\rm Bullet}
 =0.1870\quad[0.0969,\ 0.2912]\quad(95\%).
\]

Bullet's current-separation axis is almost exactly parallel to its luminosity
major axis. Abell is still preferentially aligned but substantially less so;
its paired alignment difference has median `-0.2824` and 95% interval
`[-0.4048,-0.1671]`.

This rejects the convenient interpretation that every merger can be assigned
one fixed current separation or orientation. A universal field equation would
have to accept the measured stress/current tensor as input and generate a
different response geometry for each baryonic state without introducing an
object-specific direction.

## Relation to the long-wavelength premise

The result identifies a possible source, not a force law. For a later
nonnegative long-wave operator `H_L`, the covariance-like quantity

\[
 D_L=H_L[\Pi_{\parallel\parallel}]
 -{H_L[j_\parallel]^2\over H_L[\rho_L]}
\]

has two desirable properties: it is determined by baryonic measurements and
it responds to incoherent multistream motion rather than an object label. The
common normalized second-moment displacement suggests that this part of the
source is not dominated by the posterior uncertainty of either catalog.

The differing current topology also shows why a scalar switch is inadequate.
If a long-wavelength gravitational mode survives, its tensor orientation must
come from the local filtered source state. A wavelength longer than a stellar
system can explain why its tidal variation is locally tiny, but wavelength
alone still says nothing about its amplitude, coupling, Solar screening, or
effect on photons.

## Limitations

- These are two deliberately selected merging clusters, not a cluster
  population or evidence for a galaxy–cluster transition.
- The intervals are conditional posterior uncertainty intervals, not
  frequentist detection significances.
- Bullet uses relative Bessel-I light while Abell uses relative HST F814W;
  amplitudes were intentionally not compared.
- Only line-of-sight velocity is measured. Transverse current and the complete
  three-dimensional stress tensor remain unknown.
- Source morphology has not been compared with a lensing residual.

## Reproducibility and next step

The unchanged frozen runner was executed twice. The 16,384-row morphology
table, 8,192-row paired comparison, and rendered figure were byte-for-byte
identical. Their hashes are recorded in
`results/sigma_v19bd_two_cluster_directional_source_uncertainty/reproducibility_audit.json`.

The collisionless source audit is now as far as it can responsibly proceed
without the gas state. V19W/V19X remain the next required evidence stage before
the covariant source-state operator and its universal constants can be frozen.
