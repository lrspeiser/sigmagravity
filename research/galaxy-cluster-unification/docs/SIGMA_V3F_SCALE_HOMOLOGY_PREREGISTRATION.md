# Sigma v3F scale-homology preregistration

## Question

Sigma v3E measured rotation between the local and scale-averaged tidal
eigenframes.  It raised the synthetic compact-versus-distributed median from
v3D's `1.1167` to `8.9789`, but narrowly missed the frozen factor-ten gate.
Commutators vanish whenever two symmetric tensors share eigenvectors, even if
their eigenvalue patterns change strongly with scale.

Sigma v3F tests the fuller statement: are the local and remembered tidal
tensors proportional as elements of the five-dimensional vector space of
symmetric trace-free tensors?

This is a synthetic structural audit.  It opens no observational holdout and
cannot validate a theory.

## Frozen response

Use the same dimensionless local electric-Weyl tide and unscreened retarded
memory as v3E,

\[
\widehat{\mathcal E}_{ab}={\mathcal E_{ab}\over\mathcal E_*},
\qquad
\mathcal E_*={a_\Sigma\over L_\Sigma},
\]

\[
(1-L_\Sigma^2\Box_{\rm ret})\mathcal M_{ab}
=\widehat{\mathcal E}_{ab}.
\]

Define

\[
I_E=\operatorname{tr}(\widehat{\mathcal E}^2),
\qquad
I_M=\operatorname{tr}(\mathcal M^2),
\qquad
J=\operatorname{tr}(\widehat{\mathcal E}\mathcal M),
\]

and the Gram determinant

\[
\mathcal G_{\rm hom}=I_E I_M-J^2.
\]

The frozen potential is

\[
\boxed{
\mathcal V_{\rm hom}
=\mathcal S(g/a_\Sigma)
{I_E I_M-J^2\over(1+I_E)(1+I_M)}
},
\qquad
\mathcal S(x)={1\over1+x^4}.
\]

As in v3E, the screen acts on the local interaction after the memory has been
constructed.  The provisional interaction envelope is

\[
\Gamma_{\rm int}={M_{\rm Pl}^2\over2}
\int d^4x\sqrt{-g}\,
\lambda_\Sigma\mathcal E_*\mathcal V_{\rm hom}.
\]

A completed model would use the same three universal constants
`a_sigma`, `L_sigma`, and `lambda_sigma`, with no object-specific gravity
parameters or hidden initial profile.

## Mathematical motivation

Cauchy--Schwarz in the STF Frobenius inner product gives

\[
J^2\le I_E I_M.
\]

Therefore `G_hom >= 0`, with equality exactly when the two tensors are linearly
dependent.  Also

\[
0\le\mathcal V_{\rm hom}<1.
\]

Unlike the v3E commutator, this response detects changes in eigenvalue ratios
even when the eigenvectors are unchanged.  Its numerator is quartic in weak
curvature, so it leaves the quadratic GR/Sigma-v1 propagator unchanged around
the zero-curvature background.  None of these algebraic properties proves
nonlinear stability.

## Frozen calculation and gates

The calculation reuses the exact hash-locked v3D compact and distributed
equal-mass densities.  It evaluates the local trace-free Poisson Hessian, the
unscreened static Helmholtz memory, the locally screened Gram potential, and
the integrated response within the same half-width-two volume.  Masses are
`0.3`, `1`, and `3`; grids are `65^3` and `81^3`.

The candidate advances only if every gate in
`configs/sigma_v3f_scale_homology_action_audit.json` passes:

- median distributed-to-compact response at least `10`;
- every mass normalization at least `2`;
- resolution change no more than `20%`;
- proportional tensors give `|V| <= 1e-12`, while a fixed nonproportional pair
  gives `V >= 1e-4`;
- random values remain in `[0,1]`, rotation invariance passes `1e-10`, and
  analytic derivatives match finite differences to `3e-6`;
- quartic onset matches its weak-field asymptote to `1e-4`; and
- `S(10^5) <= 1e-18`.

## Health and novelty boundary

Gram determinants, Cauchy--Schwarz, Weyl tides, and nonlocal memory operators
are established mathematics and physics tools.  Their use here is not claimed
as an original invariant.  A targeted formula search has not identified this
exact locally screened scale-homology term as a gravitational constitutive
interaction, but that is not proof of originality.

The same unresolved health requirements apply as in v3E: the timelike frame
must be derived covariantly without per-object freedom; the retarded state must
come from a causal closed-time-path or healthy local completion; and the full
principal kinetic matrix must be positive on Solar, galaxy, and cluster
backgrounds.  No empirical solver or holdout is authorized unless the
structural audit passes first.
