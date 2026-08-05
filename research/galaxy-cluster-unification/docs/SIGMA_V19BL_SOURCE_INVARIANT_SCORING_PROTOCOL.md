# Sigma V19BL source-invariant scoring protocol

## What is now fixed

V19BL turns the two source states admitted by V19BK into an exact, executable
mathematical test. It does not use lensing, a dark-matter reconstruction, or a
gravity formula.

The I4 candidate is a projected symmetric trace-free tensor made from the
gradients of gas density and entropy. Its two stored components behave like a
spin-two object: rotating the coordinate grid rotates the components twice as
fast, while the tensor amplitude, axial-angle differences, and joint novelty
score remain unchanged.

The I5 candidate is the squared sine of the projected angle between gas-density
and pressure gradients. Zero means locally parallel gradients; one means
perpendicular gradients. It is a scalar only. A 2D cross product points along
the line of sight, so this measurement cannot supply a preferred direction in
the image plane.

## Protections against false discoveries

Derivatives use centered finite differences on the common 10-kpc physical grid
after the already-frozen 50-kpc and 100-kpc smoothing. A region contributes
only when both gradients required by a candidate are detected at at least
three sigma using their full two-component posterior covariance. At least 32
regions must survive, which is safely above the 21 coefficients in the density
null.

The null model contains gas surface density, within-cluster stellar-light rank,
gas-density gradient, and two gas-density Hessian invariants. Its basis is
fixed at 21 terms. Analytic leave-one-region-out PRESS asks how much candidate
variance remains unpredictable. At least 20% must remain in at least 90% of
posterior draws. For I4 this is a joint two-component score, so rotating the
east/north axes cannot change the verdict.

Every candidate must survive both clusters, all three frozen
temperature-normalization dependence branches, both smoothing scales, and
250/350/500-kpc apertures. At least 90% of projection draws must keep amplitude
within 10%; I4 must also keep direction within 10 degrees. The 95% posterior
width of the I4 axis must be no more than 30 degrees. Removing one adaptive
region at a time must preserve the same tolerances in at least 90% of cases.

## What a later pass would mean

A pass would show that a particular projected gas state is stable, directional
where claimed, and not merely a nonlinear restatement of density or member
light in these two development clusters. It would authorize deriving the
least-field-content covariant action capable of sourcing that state.

It would not show that the state explains lensing, galaxy rotation, or any
other dark-matter-attributed observation. Those tests remain sealed until the
action fixes one physical metric and all universal constants.
