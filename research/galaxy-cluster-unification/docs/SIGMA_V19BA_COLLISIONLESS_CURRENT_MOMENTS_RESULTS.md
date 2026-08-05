# Sigma V19BA collisionless-current moment results

## Decision

V19BA **passed every frozen gate**.  It projected all 589,824 rows of the
V19AZ Bullet Cluster member ensemble onto the frozen V19H tangent grid and
produced unsmoothed maps of relative luminosity, signed line-of-sight current,
and the positive line-of-sight second moment.

This is a measured-source result, not a gravity or lensing result.  No
lensing coordinate, inferred dark-matter map, halo fit, response amplitude,
propagation length, mass-to-light ratio, transverse velocity, or gravity
parameter entered the calculation.

## Frozen numerical gates

| Diagnostic | Result | Frozen gate |
|---|---:|---:|
| Ensemble draws | 8,192 | exactly 8,192 |
| Members per draw | 72 | exactly 72 |
| Rasterized rows | 589,824 | exactly 589,824 |
| Maximum WCS round-trip error | `1.81e-10` arcsec | <= `1e-6` arcsec |
| Maximum per-draw deposition error | `2.67e-15` | <= `1e-12` |
| Minimum normalized pixel Cauchy-Schwarz margin | `-3.31e-16` | >= `-1e-12` |
| Minimum second-moment pixel | `1.78e-17` | >= 0 |
| Minimum global beta variance | `1.04e-5` | >= `-1e-15` |
| Members outside the X-ray analysis mask | `66` | exactly `66` |

All sources had four in-grid cloud-in-cell neighbors.  Member 66 was retained
on the common full grid rather than dropped because it falls outside the X-ray
analysis mask.

## What the source maps reveal

The ensemble-mean collisionless moments are not simply copies of the light
map:

| Target-blind morphology diagnostic | Result |
|---|---:|
| Second-moment to luminosity centroid offset | 12.603 arcsec |
| Positive to negative current centroid separation | 51.634 arcsec |
| Pixel correlation of luminosity and second moment | 0.5996 |
| Luminosity-weighted RMS line-of-sight speed | 998.24 +/- 8.99 km/s across draws |
| Fraction of relative luminosity inside the X-ray mask | 0.983590 |

The important result is qualitative but precise: reducing the member catalog
to one scalar luminosity density erases real directional information.  The
signed current separates oppositely moving subpopulations, and the second
moment emphasizes a spatial pattern that is only moderately correlated with
light.  A source term constructed from collisionless velocity covariance can
therefore differ from a density-only source without asking whether the object
is a galaxy or a cluster.

For a future nonnegative long-wave operator `H_L`, V19BA supplies the
sufficient projected moments for

\[
 D_L(\boldsymbol x)=H_L[\Pi_{\parallel\parallel}](\boldsymbol x)
 -{H_L[j_\parallel](\boldsymbol x)^2\over
 H_L[\rho_L](\boldsymbol x)}.
\]

This quantity vanishes for perfectly coherent motion within the response
neighborhood and grows when differently directed currents overlap.  It is a
concrete way to express the idea that a gravitational response can be nearly
unchanged inside a star system yet accumulate or reorganize over a much longer
wavelength.  V19BA deliberately does not choose `H_L`, the wavelength, or its
coupling.

## What this does not establish

V19BA does not show that gravity follows these maps.  The maps use relative
Bessel-I luminosity rather than absolute stellar mass, provide only one of the
three velocity components, and cover one cluster.  The 12.6-arcsec offset and
51.6-arcsec separation are diagnostics of the ensemble-mean source maps, not
lensing residuals or detection significances.

A successful theory still has to derive the propagation equation from one
covariant action, use one universal length and coupling, recover GR within the
Solar System, and predict raw lensing roots and galaxy rotation curves with
the same metric.  A literal free metric wave is not sufficient by itself; its
phase and polarization would be unexplained extra initial data.  The serious
next lane is a sourced long-wave tensor response whose source may include the
measured collisionless covariance.

## Reproducibility

The unchanged frozen runner was executed twice after the reporting-only JSON
serialization correction.  The 28.7 MB ten-HDU FITS map, 8,192-row global
diagnostics table, and rendered figure were byte-for-byte identical.  Their
hashes are recorded in
`results/sigma_v19ba_collisionless_current_moments/reproducibility_audit.json`.
The report itself is excluded because it contains an intentional timestamp.

The next source-side step remains the completion of V19W/V19X thermodynamic
maps.  Only after combining the independently measured gas and collisionless
source states should the universal long-wave operator be frozen and exposed
to raw lensing targets.
