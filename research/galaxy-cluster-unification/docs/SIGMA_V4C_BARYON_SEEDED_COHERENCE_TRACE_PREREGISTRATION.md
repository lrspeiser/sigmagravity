# Sigma v4C baryon-seeded coherence-trace preregistration

## Physical question

Sigma v4B established that a lower-derivative vector-stress action can make a
broad correction, but its correction is the divergence of a bounded current.
It must integrate to zero and produced positive/negative spokes rather than
the required transferable convergence and shear.

Sigma v4C asks a physically distinct question: can measured baryons seed a
unique, broad **trace** response in regions where nearby gravitational vectors
do not add coherently? The state has no freely chosen profile, center,
orientation, or homogeneous solution. This is the cheapest source-level test
of that idea, not yet a fundamental theory.

## Frozen projected closure

Let

\[
H_L=(1-L_\Sigma^2\nabla^2)^{-1}
\]

denote the unique zero-padded decaying Helmholtz branch. From the total AQUAL
deflection vector \(a_i\), form its local-memory first and second moments,

\[
m_i=H_L[a_i],
\qquad
e=H_L[a_i a_i].
\]

The bounded directional-variance statistic is

\[
D={\max(e-m_i m_i,0)\over e},
\qquad 0\le D\le1,
\]

with `D=0` only where `e` is numerically zero. It is zero for a locally
coherent vector field and rises when a neighborhood contains cancelling or
misaligned gravitational directions. It uses the total field; no baryonic
components are segmented.

High-field suppression is

\[
A={\ell_\Sigma^2\over\ell_\Sigma^2+e}.
\]

The nonnegative observed-baryon seed and its constrained trace response are

\[
J=\max(\kappa_b,0) A D,
\qquad
\Sigma=H_L[J].
\]

The frozen source-level photon correction is

\[
\delta\kappa=\eta_\Sigma\Sigma,
\]

with its shear fixed by the E-mode Kaiser--Squires relation. A single
nonnegative `eta_sigma` is solved analytically for every pair
`(L_sigma,ell_sigma)`. The same three constants apply to AS295, PLCKG287,
convergence, and both shear channels.

Unlike v4B, this state has

\[
\int\Sigma\,d^2x=\int J\,d^2x\ge0.
\]

It can therefore carry broad trace curvature rather than only move curvature
between positive and negative regions. The cost is conceptual: this closure
must ultimately be derived from a one-metric constrained action or causal
effective action, or it would merely rename a halo-like response.

## Why the construction is not an object switch

`D` is a scalar computed from the total field and one universal memory
operator. An isolated coherent field, a disk, a bulge, a merging system, and a
cluster all pass through the same equations. The formula never receives a
galaxy/cluster label and never reads a member catalog, center, ellipticity, or
target-halo direction.

The baryonic factor prevents an independently existing response in a region
with no source. The fixed decaying Helmholtz branch forbids adding an arbitrary
homogeneous profile. These properties are necessary but not sufficient to
show that the eventual relativistic state is not hidden matter.

## Source-only bound selection

Before target scores, AQUAL deflections and Newtonian baryonic convergence
were inspected only to set broad numerical bounds. The 1st--99th percentile
deflections span roughly `7.38--34.66 kpc` in AS295 and
`8.57--49.23 kpc` in PLCKG287. A source-only grid over lengths
`10,30,60,120,240 kpc` and vector scales `5,15,50 kpc` produced comparable
baryon-weighted disorder ranges in the two clusters (`0.0326--0.8852` and
`0.0373--0.8767`). No target map or residual was read.

The frozen searches are therefore

\[
3.6458\le L_\Sigma\le300\ {m kpc},
\qquad
0.1\le\ell_\Sigma\le300\ {m kpc}.
\]

The same `L_sigma` controls both the coherence neighborhood and trace
propagation. A failed fit will not be rescued by adding a second scale.

## Required gates

All gates must pass:

1. Manufactured Helmholtz residual, uniform-field null, and rotation
   covariance are at most `1e-10` relative.
2. Scaling a manufactured vector by `1000` reduces its integrated seed to at
   most `1e-4` of the original high-field control.
3. The full trace-state integral equals the seed integral within `1e-10`
   fraction and the trace is nonnegative within `1e-10` of its RMS.
4. At least 50% of correction power lies at wavelengths of at least 50 kpc.
5. The unconstrained shared amplitude is positive and both nonlinear scales
   lie more than 1% of their logarithmic ranges from every bound.
6. Joint normalized Fourier RMSE is at most `0.500`.
7. Each cluster improves by at least 20%, and every convergence/shear channel
   improves.
8. Parameters trained on either cluster transfer to the other with normalized
   RMSE at most `0.800`.
9. Factor-three instead of factor-two padding changes joint RMSE by at most 5%.

Failure retires this exact closure. No exponent, object-dependent scale,
orientation, center, or free homogeneous state will be added afterward. No
untouched observation is opened.

## What a pass would permit

A pass would permit an action derivation, not an empirical claim. The next
stage would have to:

1. replace the lens-plane vector with a covariantly derived metric or
   nonmetricity acceleration relative to a physical clock/frame;
2. source the state from \(T_{\mu\nu}n^\mu n^\nu\), not convergence;
3. derive the memory operators from a causal in-in prescription or a
   constraint system with no free halo-like mode;
4. derive massive and photon response from one metric;
5. prove conservation, constraint count, positive modes, `c_T=c`, and the
   high-field PPN/Solar limit; and
6. express the result with at most five universal four-dimensional constants.

Until those steps succeed, v4C is only a falsifiable morphology hypothesis.
