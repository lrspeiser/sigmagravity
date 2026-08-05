# Sigma V19BA collisionless-current moment plan

## Physical question

The directional long-wavelength idea needs a baryonic quantity that differs
between coherent flow and many crossing flows without identifying an object as
"a galaxy" or "a cluster."  The collisionless velocity covariance is such a
quantity.  In ordinary continuum mechanics it is the Reynolds or random-motion
stress.  Here we can measure one projected component directly from the Bullet
Cluster member redshifts.

V19BA does not yet modify gravity.  It converts the complete V19AZ uncertainty
ensemble into the three unsmoothed projected moments from which any later
long-wave line-of-sight covariance must be constructed.

## Frozen source moments

For member `i`, let `ell_i` be its relative Bessel-I luminosity and

\[
 \beta_{\parallel i}={v_{\parallel i}\over c}.
\]

Each V19AZ realization supplies its position and redshift draw.  On the common
V19H Bullet tangent grid, cloud-in-cell deposition gives

\[
 \rho_L(\boldsymbol x)=
 \sum_i\ell_i\delta_2(\boldsymbol x-\boldsymbol x_i),
\]

\[
 j_\parallel(\boldsymbol x)=
 \sum_i\ell_i\beta_{\parallel i}
 \delta_2(\boldsymbol x-\boldsymbol x_i),
\]

\[
 \Pi_{\parallel\parallel}(\boldsymbol x)=
 \sum_i\ell_i\beta_{\parallel i}^2
 \delta_2(\boldsymbol x-\boldsymbol x_i).
\]

The first is a relative luminosity density, the second is a signed projected
current, and the third is a positive projected second moment.  They are not an
absolute mass density or a complete stress tensor.

For a future nonnegative long-wave averaging operator `H_L`, these are the
sufficient moments for

\[
 D_L=H_L[\Pi_{\parallel\parallel}]
 -{H_L[j_\parallel]^2\over H_L[\rho_L]}.
\]

`D_L` is the local luminosity-weighted variance of the line-of-sight currents.
It vanishes when every contributing source has the same velocity and grows
when oppositely directed currents occupy the same long-wave neighborhood.
This is the precise version of the proposed "gravity vectors do not completely
sum until larger distances" intuition.  V19BA does not choose `H_L`, its scale,
or a gravitational coupling.

## Why this is not a repeat of closed formulas

- V17E tested gas thermal pressure alone.  V19BA measures collisionless member
  current moments.
- V4B propagated a stress made from the gravitational deflection vector and
  produced a zero-integral polarization redistribution.  V19BA measures a
  positive matter second moment before any gravitational response is applied.
- V4C and V11A tested isotropic and anisotropic scalar-memory laws.  V19BA does
  not apply either propagation law.
- P0657 diffused the already-computed gravitational field along its own field
  lines.  V19BA leaves the field untouched and constructs target-blind matter
  inputs.

Thus this protocol can reveal whether the needed measured source exists before
we risk inventing another carrier action.

## Grid and uncertainty

All 589,824 ensemble rows are deposited on the exact `626 x 626`, 0.984-arcsec
V19H Bullet grid.  Bilinear four-pixel cloud-in-cell deposition preserves each
draw's total luminosity, current, and second moment.  No smoothing kernel or
member-size profile is inserted.

The output FITS stores across-draw mean, population standard deviation, and all
pairwise covariances of the three channels.  The per-draw global table retains
the non-Gaussian realization-level totals.  The complete V19AZ ensemble remains
the authoritative distribution for any nonlinear downstream solve.

Every V19AZ position lies inside the full grid.  Member 66 lies outside the
nearest-pixel X-ray analysis mask in every draw.  It remains a real
spectroscopic member and is retained; the report will quantify rather than
hide this gas/member coverage mismatch.

## Decision gates

The protocol passes only if all parent hashes match; all 8,192 draws contain
exactly 72 rows; every source has four in-grid neighbors; WCS round trips are
accurate within `1e-6` arcsec; per-draw deposited moments are conserved within
`1e-12`; the pixelwise Cauchy-Schwarz margin
`rho_L Pi_parallel_parallel - j_parallel^2` is nonnegative within `1e-12`;
and the only member outside the analysis mask is 66.

No lensing, halo, gravity residual, propagation length, response amplitude,
absolute mass-to-light ratio, missing BRI, or transverse velocity may enter.
