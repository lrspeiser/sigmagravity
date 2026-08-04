# Sigma v3E tidal-misalignment preregistration

## Question

Sigma v3C found that the missing cluster field is not a real isotropic function
of wavelength: baryonic and compact-halo convergence/shear maps have different
Fourier phases and orientations.  Sigma v3D then showed that unequal tidal
eigenvalues at one scale do not reliably distinguish compact and distributed
equal-mass sources over an extended field.

Sigma v3E asks a more direct question: does the *orientation* of the local
tidal tensor disagree with the orientation of the same baryonic tide averaged
over a universal physical scale?

This is a synthetic structural audit.  It opens no observational holdout and
cannot validate a theory.

## Frozen response

Let the dimensionless local electric-Weyl tide and its retarded memory be

\[
\widehat{\mathcal E}_{ab}={\mathcal E_{ab}\over\mathcal E_*},
\qquad
\mathcal E_*={a_\Sigma\over L_\Sigma},
\]

\[
(1-L_\Sigma^2\Box_{\rm ret})\mathcal M_{ab}
=\widehat{\mathcal E}_{ab}.
\]

Define their matrix commutator

\[
\mathcal C=[\widehat{\mathcal E},\mathcal M]
=\widehat{\mathcal E}\mathcal M
-\mathcal M\widehat{\mathcal E}.
\]

The frozen bounded potential is

\[
\boxed{
\mathcal V_{\rm mis}
=\mathcal S(g/a_\Sigma)
{\operatorname{tr}(\mathcal C^T\mathcal C)
\over
2(1+\operatorname{tr}\widehat{\mathcal E}^2)
(1+\operatorname{tr}\mathcal M^2)}
},
\qquad
\mathcal S(x)={1\over1+x^4}.
\]

The local screen multiplies the effect after the memory is constructed.  This
ordering is frozen because the v3D post-failure diagnostic showed that
screening the source before propagation and screening the local interaction
are physically inequivalent.  It is not selected from observational data.

The provisional interaction envelope is

\[
\Gamma_{\rm int}={M_{\rm Pl}^2\over2}
\int d^4x\sqrt{-g}\,
\lambda_\Sigma\mathcal E_*\mathcal V_{\rm mis}.
\]

Together with the Sigma-v1 galaxy sector, a completed theory would use three
universal constants: `a_sigma`, `L_sigma`, and `lambda_sigma`.  It has no
per-object field, amplitude, scale, center, ellipticity, shear, or orientation.

## Mathematical motivation

For symmetric tensors, the commutator is antisymmetric.  It vanishes exactly
when the tensors can be simultaneously diagonalized.  Thus an isolated
spherical source, whose local and smoothed tides have the same radial
eigenframe, gives zero even when the tidal amplitude is large.  Separated
baryonic components can rotate the eigenframe between the local and memory
scales and give a nonzero result.

The Böttcher--Wenzel inequality gives

\[
\|[A,B]\|_F^2\le2\|A\|_F^2\|B\|_F^2.
\]

Consequently `0 <= V_mis < 1` without numerical clipping.  Since the
commutator is quadratic in the weak fields and its squared norm is quartic,
the interaction does not modify the quadratic GR/Sigma-v1 propagator around a
zero-curvature background.  That statement does not establish nonlinear
stability.

## Frozen calculation and gates

The calculation reuses the already committed equal-mass compact and
distributed synthetic densities from v3D, verified by their configuration
hash.  It constructs the local trace-free Poisson Hessian, applies the
unscreened static Helmholtz memory to each component, evaluates the locally
screened commutator potential, and integrates it inside the same half-width-two
volume.  Mass normalizations are `0.3`, `1`, and `3`; grids are `65^3` and
`81^3`.

The candidate advances only if every gate in
`configs/sigma_v3e_tidal_misalignment_action_audit.json` passes.  The principal
physical gates are:

- median distributed-to-compact integrated response at least `10`;
- every mass normalization at least `2`;
- primary ratio changes by no more than `20%` at higher resolution;
- commuting tensors give `|V| <= 1e-12` and a fixed noncommuting fixture gives
  `V >= 1e-4`;
- random values remain in `[0,1]`, with rotation invariance to `1e-10`;
- analytic derivatives match trace-free finite differences to `3e-6`;
- quartic weak-field onset matches its asymptote to `1e-4`; and
- `S(10^5) <= 1e-18`.

## Health gates deliberately not claimed

The electric Weyl tensor requires a timelike frame.  A final theory must derive
that frame covariantly from a unique gravitational clock or constrained state;
choosing it per object would violate the goal.  Likewise, a retarded response
must come from a closed-time-path effective action or a healthy local
completion.  Ordinary variation of a single-copy retarded action is
insufficient.  The principal kinetic matrix on nonzero backgrounds remains
unknown.

No empirical solver will be built unless the structural gates pass, and no
observational holdout will be opened until all of these mathematical issues are
resolved.

## Prior-art boundary

The commutator bound is established matrix analysis, not new physics.  A
concise proof and history are given by Lu, [*Remarks on the Böttcher-Wenzel
Inequality*](https://arxiv.org/abs/1106.1827).  Electric/magnetic Weyl
decompositions and nonlocal curvature response are also established tools.

A targeted search did not identify this exact locally screened commutator of a
tidal tensor with its retarded scale-memory as a gravitational constitutive
term.  That absence is not proof of originality.  The only project-specific
claim at this stage is the frozen structural test and its result.
