# Sigma v17F root-scale propagator

## Why this test exists

V17E asks a deliberately broad question: does independently measured gas
thermal stress contain transferable information about the missing cluster
lensing field? Its three fixed smoothing scales and ridge coefficients are
useful for source discovery, but they are not an elegant root equation.

V17F is the conditional reduction from that discovery test to one equation.
It was frozen before any regional temperature, thermal source map, v17E target
score, or v17F result existed. It runs only if every v17E gate passes. A failed
v17E source cannot be rescued by this smaller model family.

## The scale question in one equation

For either predeclared thermal source (q_b=q_{\rm total}) or
(q_b=q_{\rm contrast}), define one projected mediator:

\[
\boxed{
(1-L_\Sigma^2\nabla_\perp^2)s_\Sigma(\mathbf x)=q_b(\mathbf x)
}
\]

or, equivalently,

\[
\widehat s_\Sigma(\mathbf k)=
{\widehat q_b(\mathbf k)\over1+(kL_\Sigma)^2}.
\]

The one-metric lensing increment is

\[
\Delta\kappa_\Sigma=\beta_\Sigma s_\Sigma,
\]

with both shear components fixed by the same E-mode potential. There is one
coefficient for convergence and shear together; no photon multiplier or
external-shear term exists.

This separates two possible origins of an apparent halo size:

- The measured (q_b(\mathbf x)) carries an object-dependent source size,
  centroid, and geometry.
- The one universal (L_\Sigma) can broaden that source by a fixed physical
  propagation rule.

The amplitude (eta_\Sigma) cannot alter (R_{50}) or (R_{80}). Therefore,
matching those radii cannot be obtained merely by turning gravity up.

## Frozen comparison

The same source family and the same (L_\Sigma) are used in AS295 and PLCK
G287. For each transfer direction, only (eta_\Sigma) is learned from the
training cluster and then applied unchanged to the other cluster. One common
source/length pair is selected by symmetric leave-one-cluster-out error.

The declared length grid includes (L_\Sigma=0), which is the source-only
limit, and extends to 300 kpc. Selecting the 300 kpc endpoint fails
identifiability rather than licensing a longer search after seeing the result.
Both directional coefficients must be positive and agree within 0.15 dex.

V17F must retain every v17E scientific gate, remain within 5% of the flexible
v17E NRMSE, match (R_{50}) and (R_{80}) within 25% in both directions, and
change scored observables by no more than 2% when resolution doubles.

## How the result chooses the next action

| Result | Physical conclusion | Next action |
|---|---|---|
| (L_\Sigma=0) passes | Measured source extent is sufficient in this diagnostic | Derive a scale-free response; do not spend a constant on a range |
| One nonzero interior (L_\Sigma) passes | Source plus one universal correlation length explains the extent | Derive a finite-range covariant mediator and freeze the length before holdout |
| Amplitude passes, radii fail | Thermal stress correlates with strength, not halo size | Reject thermal stress as the scale-setting source |
| Radii pass, alignment/full field fails | Scale is informative but the carrier lacks directional structure | Retain the scale clue; require a dynamical tensor/vector orientation channel |
| V17E or V17F fails | Tested instantaneous gas-stress mechanism fails | Do not add lengths or exponents; move to collisionless stress or causal history |

## What a pass would not prove

The Helmholtz/Yukawa operator is established prior art, and this projected
formula is not a covariant theory. A pass would identify a compact scale
mechanism worth deriving. The next theory would still have to obtain the
source and response from one action, use one physical metric for matter and
light, conserve total stress, remain stable and luminal, screen in the Solar
System, reproduce galaxies, and pass untouched raw multiple-image tests.

The authoritative machine-readable freeze is
`configs/sigma_v17f_root_scale_propagator.json`.
