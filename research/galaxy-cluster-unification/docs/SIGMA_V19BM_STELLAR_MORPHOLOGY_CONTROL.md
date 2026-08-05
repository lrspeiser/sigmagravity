# Sigma V19BM stellar-morphology control

V19BM implements the remaining non-gas input to the frozen V19BL density
novelty test. It uses the first 4,096 authoritative member-catalog posterior
draws in each cluster, paired by sample number with the independent V19X4 gas
draws.

For every draw, finite member light is divided by its total within that cluster
and filter. Member coordinates are transformed through the frozen Chandra WCS
onto the exact V19X4 east/north physical grid, deposited with cloud-in-cell
weights, and convolved at 50 and 100 kpc. Unit light is conserved. The mean
light in every adaptive gas region is then converted to a within-draw
percentile rank.

Only that rank enters the density nuisance model. Absolute Bullet Bessel-I and
Abell 2146 F814W amplitudes are never compared, and the product is not called a
stellar-mass map. Equal sample numbers are deterministic Monte Carlo pairing,
not a claim of measured gas-star covariance.

The executor is frozen now but cannot run terminally until V19X4 produces its
three hash-bound common-grid branches. Its current preflight reads no observed
gas posterior, lensing, halo, action, or gravity payload.
