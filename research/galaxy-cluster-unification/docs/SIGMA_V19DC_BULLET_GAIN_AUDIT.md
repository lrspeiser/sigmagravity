# Sigma V19DC Bullet gain audit

## Outcome

The frozen gain audit passes every registered gate for all nine primary
ObsIDs. All 3,483 original background cells and RMFs match their frozen sizes
and SHA-256 values, and their 1,024-channel EBOUNDS grids are exact.

| Result | Value |
|---|---:|
| Weakest line improvement | Delta Cash 573.46 |
| Largest window-induced centroid shift | 0.00571 keV |
| Gain slope range | 0.99822--1.00536 |
| Gain intercept range | -0.03846--0.02150 keV |
| Per-ObsID Fe-energy-equivalent velocity uncertainty | 407.7--852.7 km/s |
| Median per-ObsID equivalent uncertainty | 565.6 km/s |

The terminal report SHA-256 is
`ad077394036351bcd33f0eda9a623392109754521b5c07b72ccc52c935c01883`.
These are detector-nuisance measurements, not gas velocities. Because the
uncertainty is not negligible, the source fitter must propagate the nine
intercept/slope covariances with the same response weights used to combine the
source spectra. It may not treat gain as zero or add the median per-ObsID value
directly to every combined region.

## Frozen question

Before fitting a source Fe-K redshift, V19DC asks whether the nine primary ACIS
observations have a measurable detector energy offset or slope and how much
uncertainty that calibration adds at the observed Bullet Fe-K energy.

The audit uses only the original matched blank-sky background PHA and RMF
EBOUNDS for all 3,483 primary Bullet cells. It does not use the ASCA-scaled
combined background because that product does not preserve the union's raw
integer counts. No source PHA, source line, temperature, abundance, redshift or
velocity is opened.

The payload-blind execution plan passes with nine ObsIDs and 3,483 unique
cells. The frozen runner SHA-256 is
`ed41b7a833edc5702621746369a698458af4da91bfd9b7608fe704d37ba2a915`.

## Frozen model

For each ObsID, the non-overlapping blank-sky cell counts are summed channel by
channel after exact product-hash and 1,024-channel EBOUNDS checks. Ni K-alpha at
7.4782 keV and Au L-alpha at 9.7133 keV are each fit with an ungrouped Poisson
likelihood, a positive exponential-linear local continuum and a bin-integrated
Gaussian.

The runner uses the existing Windows Astropy/SciPy environment and resolves the
unchanged `/home/henry` response archive through the Ubuntu-24.04 WSL UNC
mount. A payload-blind attempt established that the CIAO environment itself
lacks SciPy; no detector array or fit was opened by that failed preflight.

The primary half-window is 0.30 keV. Fixed 0.25 and 0.35 keV windows must move
each centroid by no more than 0.015 keV. Each line must improve Cash by at least
25 over its continuum-only fit, and its Delta-Cash=1 profile interval must
close inside the frozen search range.

The two recorded centroids determine

\[
E_{\rm cal}=b+sE_{\rm recorded}.
\]

Their conservative profile variances are propagated by the exact Jacobian to
the full intercept/slope covariance. The primary source analysis will propagate
that covariance without shifting source data; a later gain-corrected branch
will test velocity-sign stability.

## Claim boundary

A V19DC pass authorizes the already-frozen Bullet source redshift fitter. The
gain-uncertainty fraction cannot pass until source redshift errors exist. This
stage cannot establish a gas current, open Abell 2146, choose a Sigma source or
alter a gravity equation.
