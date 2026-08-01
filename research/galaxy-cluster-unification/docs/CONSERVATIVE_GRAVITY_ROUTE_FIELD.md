# Conservative gravity-route field

## Purpose

This is a concrete mathematical version of the hypothesis that an apparent
dark-matter distribution is partly a map of where baryon-generated gravity
arrives, not where unseen matter resides. It separates two questions that the
data constrain differently:

1. how much field a baryonic source produces; and
2. where that field is expressed after propagation through the surrounding
   spacetime geometry.

The scalar potential/path law supplies the first quantity. A normalized route
kernel supplies the second.

## Forward field equation

Let `rho_b(x)` be the three-dimensional baryonic density. The safer form keeps
the ordinary baryonic field local and routes only the extra response above it:

\[
\rho_g(\mathbf y)=\rho_b(\mathbf y)+
\int d^3x\,[A(\mathbf x)-1]\rho_b(\mathbf x)
\left([1-f(\mathbf x)]\delta^3(\mathbf y-\mathbf x)
+f(\mathbf x)P(\mathbf y\mid\mathbf x)\right),
\]

where:

- `A(x)-1` is the scalar excess above Newtonian/weak-field GR;
- `f(x)` is the routed fraction; and
- `P(y|x)` is the probability density for field sourced at `x` to be expressed
  at `y`, with `integral P(y|x) d3y = 1`.

The potential then obeys the familiar-looking equation

\[
\nabla^2\Phi(\mathbf y)=4\pi G\rho_g(\mathbf y).
\]

This is nonlocal new physics, but it has useful safeguards. Normalization of
`P` prevents directional routing from inventing extra integrated field. Setting
`A -> 1` returns ordinary Newtonian gravity and the weak-field limit of GR even
if a route geometry exists; setting `f -> 0` keeps the excess at its source.
The previous potential/path
experiments provide candidate universal forms for `A`; the present task is to
infer and then compress `P` into a predictive law.

In a lens plane, the corresponding observable equation is

\[
\kappa_{\rm pred}(\boldsymbol\theta)=
\int d^2\theta'\,K(\boldsymbol\theta\mid\boldsymbol\theta')
\kappa_b(\boldsymbol\theta'),
\]

with a local-plus-routed kernel `K`. Convergence, shear, flexion, and multiple
image positions are all derived from the same predicted potential; light and
mass are not assigned independent arbitrary corrections.

## Inverse backtracking from apparent halo locations

For each cluster:

1. Build independent source maps for stars, BCG/ICL, hot gas, and cold gas,
   retaining their covariance and redshift uncertainty.
2. From public strong- plus weak-lensing chains, reconstruct samples of the
   lensing potential and its convergence/shear field. Call the non-baryonic
   residual an **arrival map**, not dark matter.
3. Discretize baryonic source pixels `i` and arrival pixels `j`. Solve for a
   nonnegative transport matrix `T_ij` whose row sums equal the routed field
   available from each baryonic source and whose column sums reproduce the
   arrival map.
4. Minimize a regularized cost such as

   \[
   \sum_{ij}T_{ij}\left[
   d_{ij}^2/L^2+\lambda_c C_{ij}+\lambda_\gamma G_{ij}
   \right]+\epsilon\sum_{ij}T_{ij}\log T_{ij}.
   \]

   Here distance discourages gratuitously long routes, `C_ij` penalizes paths
   inconsistent with the baryonic potential geometry, and `G_ij` penalizes
   endpoint directions inconsistent with measured lensing shear. The last term
   prevents a brittle one-pixel solution.
5. Backtrack each arrival pixel through `T_ij` to obtain a probability
   distribution over baryonic origins. Fit smooth curved streamlines between
   those endpoint distributions, with curvature regularization and projected
   depth marginalized rather than fixed.
6. Compress the reconstructed routes into a low-parameter universal kernel
   based only on pre-lensing baryonic invariants: potential depth, tidal tensor,
   density gradients, component type, and source separation.
7. Freeze that kernel and predict raw image positions, shear, flexion, and time
   delays in untouched clusters.

The transport solution is a discovery instrument, not the final law. A kernel
that merely memorizes every arrival map has no explanatory value.

## What existing observations can and cannot answer

Strong-lensing image positions tightly constrain potential gradients at sparse
locations. Weak shear constrains a broader two-dimensional tidal field. Flexion
is sensitive to small-scale gradients, and time delays probe the potential
itself. Multiple source redshifts provide limited depth discrimination.
Together they can test whether one baryon-derived route kernel predicts several
independent observables and clusters.

They cannot uniquely reveal a three-dimensional propagation path. Projection,
the mass-sheet degeneracy, line-of-sight structures, uncertain member depths,
and baryon-map systematics permit many route fields with similar two-dimensional
lensing. A defensible claim therefore requires posterior samples, multi-probe
predictions, radius- or family-level holdouts, and finally untouched clusters.

## Current empirical constraint

The simplest localized kernel—resolved member deflection minus its circular
average—has now failed predictive transfer on RX J2129. The routed fraction is
the strongest training variable, but the measured layout is not special under
angle randomization and one held-out nonlinear root fails for the zero-slip
parent. Radial dressing by the scalar dynamical or photon response has almost
no effect.

The next global, smooth member kernel passed a ten-cluster map-shape transfer,
but its unit-strength raw RX J2129 translation failed catastrophically. A
suppressed angular bridge, `s_theta=f_route^p`, selected `p=0.5` on RX J2129
training images and then worsened its held-out images. The post-hoc `p=2.5`
value was replayed without retuning on four other raw clusters. It recovered a
missing scalar-parent image root, but improved the three matched complete
systems by only 0.276%, retained a 19.160-arcsec absolute RMS, and was 1.886
times the compact-halo validation RMS.

The justified next step is therefore not another local amplitude grid. It is an
inverse transport reconstruction using complete gas, BCG/ICL, and member maps,
followed by compression into a route law and a locked forward prediction of
both strong- and weak-lensing observables.
