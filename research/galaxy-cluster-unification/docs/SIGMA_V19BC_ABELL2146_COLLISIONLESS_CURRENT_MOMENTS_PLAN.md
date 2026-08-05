# Sigma V19BC Abell 2146 collisionless-current moment plan

## Purpose

V19BB produced a target-blind, candidate-marginalized Abell 2146 ensemble with
all 63 spectroscopic members in 8,192 draws. V19BC will apply the unchanged
V19BA projection rule so the Bullet Cluster and Abell 2146 have directly
comparable collisionless source moments before any long-wavelength response is
chosen.

This stage does not test gravity. It measures whether relative light and
line-of-sight velocity encode spatial structure beyond a scalar luminosity
map.

## Frozen construction

For every draw and every member having measured relative F814W luminosity
`ell_i`, define

\[
 \beta_{\parallel i}={v_{\parallel i}\over c},
\]

and deposit the three unsmoothed particle moments

\[
 \rho_L=\sum_i\ell_i\delta_2(\boldsymbol x-\boldsymbol x_i),\qquad
 j_\parallel=\sum_i\ell_i\beta_{\parallel i}
 \delta_2(\boldsymbol x-\boldsymbol x_i),
\]

\[
 \Pi_{\parallel\parallel}=\sum_i\ell_i\beta_{\parallel i}^2
 \delta_2(\boldsymbol x-\boldsymbol x_i).
\]

The deposition is the same conservative four-pixel cloud-in-cell operator
used in V19BA. The grid is the frozen 745 by 745 V19H Abell Chandra tangent
grid at 0.984 arcsec per pixel. A position-only audit established before this
freeze that all 516,096 sampled positions have four grid neighbors and none
falls outside the nearest-pixel X-ray analysis mask.

## Missing measurements

Every draw retains all 63 member rows. Between 51 and 58 rows have measured
F814W luminosity. A blank-luminosity row remains in the completeness and grid
audit but deposits no moment. No flux, stellar mass, or transverse velocity is
inferred.

This differs from dropping a galaxy: the report will retain finite and missing
counts for every draw, while the authoritative V19BB ensemble preserves the
full position/state posterior.

## Relevance to the long-wave hypothesis

For any later nonnegative propagation operator `H_L`, the three maps are
sufficient to construct the line-of-sight velocity-covariance source

\[
 D_L=H_L[\Pi_{\parallel\parallel}]
 -{H_L[j_\parallel]^2\over H_L[\rho_L]}.
\]

This is the part of the proposed long-wavelength idea that can be tied to
measured baryonic state. A mode whose wavelength exceeds a stellar system can
be nearly uniform locally, but its amplitude and direction must still be
sourced rather than inserted as a free sinusoid. V19BC supplies unsmoothed
source moments; it does not choose `H_L`, a wavelength, amplitude, phase, or
polarization.

## Failure discipline and next step

The stage fails closed if the parent hashes, row counts, finite-count range,
grid geometry, deposition conservation, positivity inequalities, or missing
data rules fail. The grid, member inventory, and map definition will not be
changed after seeing the moment morphology.

After a pass, the remaining prerequisite is the V19W/V19X gas thermodynamic
source. Only after the collisionless and gas states exist for both development
clusters will a directional source-state long-wave operator be frozen and
evaluated against sealed lensing targets.
