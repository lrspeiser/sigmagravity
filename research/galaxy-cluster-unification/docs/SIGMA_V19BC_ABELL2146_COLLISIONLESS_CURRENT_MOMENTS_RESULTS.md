# Sigma V19BC Abell 2146 collisionless-current moment results

## Decision

V19BC **passed every frozen gate**. It projected all 516,096 rows of the V19BB
Abell 2146 member ensemble onto the frozen V19H tangent grid and produced the
same three unsmoothed collisionless moment maps used for the Bullet Cluster:
relative luminosity, signed line-of-sight current, and positive line-of-sight
second moment.

This is a measured-source result, not a gravity or lensing result. No lensing
coordinate, inferred dark-matter map, halo fit, response amplitude,
propagation length, mass-to-light ratio, transverse velocity, missing
luminosity, or gravity parameter entered the calculation.

## Frozen numerical gates

| Diagnostic | Result | Frozen gate |
|---|---:|---:|
| Ensemble draws | 8,192 | exactly 8,192 |
| Members per draw | 63 | exactly 63 |
| Rasterized rows | 516,096 | exactly 516,096 |
| Finite-luminosity members per draw | 51–58; median 56 | exactly frozen range 51–58 |
| Explicit missing-photometry rows | 55,428 | retained, never deposited |
| Maximum WCS round-trip error | `2.09e-10` arcsec | <= `1e-6` arcsec |
| Maximum per-draw deposition error | `1.75e-15` | <= `1e-12` |
| Minimum normalized pixel Cauchy–Schwarz margin | `-3.21e-16` | >= `-1e-12` |
| Minimum second-moment pixel | 0 | >= 0 |
| Minimum global beta variance | `8.09e-6` | >= `-1e-15` |
| Members outside the X-ray analysis mask | none | exactly none |

All positions, including missing-photometry states, had four in-grid
cloud-in-cell neighbors. Missing light remained an explicit measurement state
and contributed no invented moment.

## What the Abell source maps reveal

The ensemble-mean velocity moments are not scalar copies of the light map:

| Target-blind morphology diagnostic | Abell 2146 |
|---|---:|
| Second-moment to luminosity centroid offset | 8.602 arcsec |
| Positive to negative current centroid separation | 49.389 arcsec |
| Pixel correlation of luminosity and second moment | 0.4793 |
| Luminosity-weighted RMS line-of-sight speed | 962.50 ± 22.05 km/s |

The signed current separates oppositely moving member populations, while the
second moment moves the effective source centroid and is only moderately
correlated with light. Density alone therefore discards measured directional
and velocity-dispersion information in Abell just as it did in the Bullet
Cluster.

## Two-cluster source comparison

The same source definition now gives:

| Diagnostic | Bullet V19BA | Abell V19BC |
|---|---:|---:|
| Second-moment/light centroid offset | 12.603 arcsec | 8.602 arcsec |
| Opposite-current centroid separation | 51.634 arcsec | 49.389 arcsec |
| Luminosity/second-moment correlation | 0.5996 | 0.4793 |
| RMS line-of-sight speed | 998.24 ± 8.99 km/s | 962.50 ± 22.05 km/s |

The agreement in qualitative behavior is useful: two independently assembled
merger catalogs both contain a roughly 50-arcsec separation between opposite
line-of-sight current centroids, a displaced second-moment centroid, and a
second-moment map that is far from perfectly correlated with luminosity.

It is not yet evidence for modified gravity. The clusters have different
redshifts, member selection, photometric bands, completeness, and merger
viewing geometries. The angular similarities were not a preregistered equality
test, and the relative luminosity amplitudes cannot be compared as masses.

## Relevance to the long-wavelength hypothesis

For a future nonnegative operator `H_L`, both clusters now supply the projected
moments for

\[
 D_L(\boldsymbol x)=H_L[\Pi_{\parallel\parallel}](\boldsymbol x)
 -{H_L[j_\parallel](\boldsymbol x)^2\over H_L[\rho_L](\boldsymbol x)}.
\]

This source is small for coherent motion within a response neighborhood and
grows where differently directed currents overlap. It gives the proposed
long-wavelength gravitational mode something measured and directional to
respond to, rather than an arbitrary cosmic phase or object label.

The wavelength argument still supplies only scale separation: a mode much
longer than a stellar system can have negligible local tidal variation while
varying across a galaxy or cluster. It does not determine the coupling,
polarization, propagation law, or Solar-System screening of the Sun's own near
field. Those must emerge from one covariant action and pass the same matter and
photon tests.

## Reproducibility and next step

The unchanged frozen runner was executed twice. The 40.6 MB ten-HDU FITS map,
8,192-row diagnostics table, and rendered figure were byte-for-byte identical.
Their hashes are recorded in
`results/sigma_v19bc_abell2146_collisionless_current_moments/reproducibility_audit.json`.
The report itself is excluded because it contains an intentional timestamp.

The collisionless source prerequisite is now complete for both development
clusters. V19W/V19X must finish the independent gas thermodynamic source before
any directional long-wave operator is frozen or exposed to sealed lensing
targets.
