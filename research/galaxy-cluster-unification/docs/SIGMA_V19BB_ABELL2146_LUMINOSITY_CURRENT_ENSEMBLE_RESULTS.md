# Sigma V19BB Abell 2146 luminosity-current ensemble results

## Decision

V19BB **passed every frozen source-only gate**. One additional astrometric
scatter of `0.45 arcsec`, shared by the complete published Abell 2146 member
table, explained the frozen catalog candidate offsets better than the
unchanged V19AA position model. The scale was selected without magnitude,
lensing, halo, gravity, subcluster, BCG, or spectral-type information.

This is a catalog calibration and source-uncertainty result, not a fitted
cluster gravity length. It does not test the long-wavelength hypothesis.

## Frozen numerical gates

| Diagnostic | Result | Frozen gate |
|---|---:|---:|
| Selected shared extra astrometric scatter | `0.45 arcsec` | interior grid point |
| Prior sensitivity selections | `0.45 arcsec` for all four priors | grid-index spread <= 3 |
| Held-out folds improving over `s=0` | 7 of 7 | at least 5 of 7 |
| Exact expected finite-F814W members per draw | 56.231 | at least 50 |
| Maximum posterior normalization error | `2.22e-16` | <= `1e-12` |
| Maximum sampled/exact state-frequency difference | 0.00758 | <= 0.03 |
| Ensemble draws | 8,192 | exactly 8,192 |
| Members per draw | 63 | exactly 63 |
| Ensemble rows | 516,096 | exactly 516,096 |
| Raw HSC source files hash-verified | 126 | exactly 126 |

The selected grid index did not change at counterpart priors `0.8`, `0.9`,
`0.95`, or `0.99`. Every held-out fold independently selected `0.45 arcsec`
on its training members and improved its predictive log evidence over the
zero-extra-scatter control by 17.80 to 30.47 log units.

## What was measured

The exact posterior makes a catalog candidate the top state for 59 members
and the explicit null state the top state for four. The ensemble expects
56.231 members with measured F814W luminosity and 6.769 with missing
photometry per draw. Missing values remain missing; no luminosity, stellar
mass, or transverse velocity was invented.

For each finite-luminosity state, V19BB records the relative F814W luminosity

\[
 \ell_i=10^{-0.4(A_{F814W,i}-20)},
\]

draws the quoted redshift uncertainty, recomputes the cluster median velocity,
and forms the line-of-sight current proxy `ell_i v_parallel`. This creates the
Abell counterpart to the Bullet Cluster's probabilistic collisionless source
ensemble under an explicit uncertainty model.

## Relevance to the long-wavelength idea

The user-proposed scale separation remains physically possible in the narrow
sense that a gravitational mode with wavelength much longer than a stellar
system is nearly constant across that system but can vary across a galaxy or
cluster. A representative `lambda_Sigma = 37.7 kpc` gives a 100-AU fractional
tidal phase scaling of roughly `1.7e-16`.

Wavelength alone, however, does not decide the amplitude, phase, polarization,
or source of the mode, and it does not screen the Sun's own ordinary near
field. The admissible new test is therefore not another isotropic radial
filter. It is a sourced, directional long-wave response whose phase and tensor
orientation are determined by measured baryonic density, current, stress, or
velocity covariance. V19BB supplies one of the two required collisionless
source ensembles for that test.

## Claim boundary and next step

The `0.45 arcsec` value is specific to the measurement uncertainty of one
published coordinate table. It is not a Sigma wavelength, universal gravity
constant, halo radius, or cluster-specific force parameter. The ensemble uses
relative light, not absolute stellar mass, and measures only line-of-sight
motion.

The immediate successor is to rasterize these draws on the frozen V19H Abell
grid using the V19BA luminosity/current/second-moment definitions. The V19W
instrument-response production must finish before V19X can build the matched
gas thermodynamic maps. Only then can one directional source-state operator be
frozen and evaluated against sealed lensing targets with the same constants
used for galaxies and Solar-System checks.

## Reproducibility

The unchanged frozen runner was executed twice. The calibration table, exact
state marginals, member summary, 19.1 MB compressed ensemble, and rendered
figure were byte-for-byte identical. Their hashes are recorded in
`results/sigma_v19bb_abell2146_luminosity_current_ensemble/reproducibility_audit.json`.
The report itself is excluded because it contains an intentional timestamp.
