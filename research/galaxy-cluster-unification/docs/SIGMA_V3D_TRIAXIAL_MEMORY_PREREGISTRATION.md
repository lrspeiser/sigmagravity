# Sigma v3D triaxial-memory structural preregistration

## Question

Sigma v3C showed that a real isotropic wavelength filter cannot transform the
registered baryonic convergence and shear into the spent compact-halo maps.
Sigma v3D therefore asks a narrower question before another empirical fit:
can a nonlinear invariant of a baryon-forced, trace-free tidal memory remain
small for a compact nearly axisymmetric source while becoming large for a
distributed multi-component source of the same total baryonic mass?

This is a synthetic structural audit.  It opens no new galaxy or cluster data,
fits no observation, and cannot validate the theory.

## Frozen response and interaction envelope

The proposed dimensionless memory tensor obeys

\[
(1-L_\Sigma^2\Box_{\rm ret})\mathcal M_{ab}
=\mathcal S(g/a_\Sigma){\mathcal E_{ab}\over\mathcal E_*},
\qquad
\mathcal E_*={a_\Sigma\over L_\Sigma},
\]

where `E_ab` is the electric, spatial, symmetric trace-free Weyl tensor in a
physical timelike frame.  The frozen high-field screen is

\[
\mathcal S(x)={1\over1+x^4}.
\]

For a trace-free three-by-three matrix, define

\[
I_2=\operatorname{tr}(\mathcal M^2),\qquad
I_3=\operatorname{tr}(\mathcal M^3),
\]

\[
\mathcal D=I_2^3-6I_3^2,
\qquad
\mathcal V(\mathcal M)={\mathcal D\over(1+I_2)^3}.
\]

The provisional interaction envelope is

\[
\Gamma_{\rm int}={M_{\rm Pl}^2\over2}
\int d^4x\sqrt{-g}\,\lambda_\Sigma\mathcal E_*\mathcal V(\mathcal M).
\]

If completed, the theory would contain three universal constants:
`a_sigma`, `L_sigma`, and `lambda_sigma`.  It contains no object-specific
amplitude, center, orientation, scale, or initial profile.

This envelope is not yet a valid causal action.  A traditional single-copy
action containing a retarded inverse generally does not vary into purely
retarded equations.  Advancement from this audit only authorizes a subsequent
closed-time-path derivation or a healthy local constrained completion.

## Why this invariant was selected

For real symmetric trace-free `M`, the eigenvalue discriminant obeys

\[
\mathcal D\ge0.
\]

It is zero when two eigenvalues coincide and positive when all three are
distinct.  A softened isolated point or spherical source has the eigenvalue
pattern `(2a,-a,-a)` and therefore gives zero.  Tidal tensors from separated
sources are individually axisymmetric, but their tensor sum generically has
three distinct eigenvalues.  Consequently

\[
\mathcal V(\mathcal M_1+\mathcal M_2)
\ne
\mathcal V(\mathcal M_1)+\mathcal V(\mathcal M_2).
\]

The denominator bounds `V` between zero and one.  Since its Taylor expansion
begins at sixth order in `M`, it does not change the quadratic graviton
propagator around `M=0`.  This observation does not prove nonlinear stability.

## Frozen synthetic calculation

The static audit sets `G=a_sigma=L_sigma=1`.  It constructs a compact
bulge-plus-oblate-disk density and a nine-component distributed density with
the same total mass.  For each density it:

1. solves periodic Poisson gravity on a padded three-dimensional grid;
2. forms the trace-free Newtonian tidal Hessian;
3. multiplies it locally by `S(g/a_sigma)`;
4. applies the static Helmholtz memory `1/(1+L_sigma^2 k^2)` to every tensor
   component; and
5. integrates the bounded potential inside the central analysis volume.

Mass normalizations `0.3`, `1`, and `3` are frozen in advance.  The central
result is the median distributed-to-compact integrated-response ratio.  The
calculation is repeated on `65^3` and `81^3` grids.

## Frozen gates

The candidate advances only if every gate in
`configs/sigma_v3d_triaxial_memory_action_audit.json` passes.  The main physical
requirements are:

- the median distributed-to-compact response ratio is at least `10`;
- every frozen mass normalization has a ratio of at least `2`;
- the primary ratio changes by no more than `20%` at higher resolution;
- exact axisymmetric tensors give `|V| <= 1e-12`, while a fixed overlapping
  tensor fixture gives `V >= 1e-4`;
- the analytic variation matches trace-free finite differences to `2e-6`;
- rotation invariance and trace-free evolution pass their numerical gates; and
- `S(10^5) <= 1e-18`.

A structural pass is necessary but not sufficient.  It does not count as a
galaxy, cluster, Solar-System, causality, or stability pass.

## Prior-art boundary

The discriminant and its relationship to algebraically special Weyl tensors
are established.  The full relativistic speciality index was introduced for
curvature diagnostics by Baker and Campanelli, [*Making use of geometrical
invariants in black hole collisions*](https://arxiv.org/abs/gr-qc/0003031).
The static electric-Weyl discriminant used here is a restricted real
three-dimensional analogue, not a new curvature invariant.

Nonlocal gravitational response and entire or inverse-differential form
factors are also established research.  In particular, ordinary variation of
an explicitly nonlocal action can mix retarded and advanced kernels; see Zhang
et al., [*Acausality in Nonlocal Gravity Theory*](https://arxiv.org/abs/1601.03808).

The narrow project hypothesis is the screened, bounded constitutive use of the
discriminant to test baryonic component overlap under the project's universal
galaxy--cluster gates.  No originality claim is made until a broader literature
audit and a complete action exist.
