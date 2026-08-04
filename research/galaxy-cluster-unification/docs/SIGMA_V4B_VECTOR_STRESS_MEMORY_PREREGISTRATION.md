# Sigma v4B vector-stress memory preregistration

## Why this is physically distinct

Sigma v4A varied a scalar made from the trace-free Hessian of the potential.
The resulting double divergence was mathematically correct but concentrated
the correction around rapid changes and removed only 0.794% of the missing
cluster-field power.

Sigma v4B implements the lower-derivative version of the original vector
interference idea. The gravitational field itself has a local directional
stress

\[
S_{ij}=u_i u_j-\frac12\delta_{ij}u^2,
\qquad
u_i=\frac{\partial_i\psi}{\ell_\Sigma}.
\]

Because `S` is quadratic in the **total** field, it automatically contains
cross terms between every baryonic contribution. It does not identify member
galaxies or apply a nonlinear law separately to a catalog of objects. A
compact coherent field and a field made from many interfering directions can
therefore differ without an object-class switch.

Variation now passes through one gradient of `psi`, so the field equation
contains one divergence rather than v4A's double divergence of a tidal
response. The frozen broad-power gate tests whether this actually cures the
edge-localization problem.

## Projected action

Let

\[
M=(1-L_\Sigma^2\nabla^2)^{-1}S
\]

and use the bounded misalignment scalar

\[
V(S,M)=
\frac{\|[S,M]\|_F^2}
{2(1+\|S\|_F^2)(1+\|M\|_F^2)}.
\]

The frozen projected functional is

\[
F[\psi]=\int d^2x\left[
-\frac12|\nabla\psi|^2-2\kappa_{\rm AQUAL}\psi
+\eta_\Sigma V(S,M)
\right].
\]

If `P_S` and `P_M` are the analytic tensor derivatives of `V`, self-adjoint
memory gives

\[
P=P_S+(1-L_\Sigma^2\nabla^2)^{-1}P_M.
\]

The chain rule through the quadratic stress is

\[
q_i=\frac{2}{\ell_\Sigma}P_{ij}u_j.
\]

Consequently

\[
R=-\partial_iq_i,
\qquad
\delta\kappa=-\frac{\eta_\Sigma}{2}R
=\frac{\eta_\Sigma}{2}\partial_iq_i.
\]

This correction sign follows from the displayed action and is frozen before
the maps are scored.

The interaction is naturally band-limited in field strength. For a very weak
field, `S` and `M` are quadratic in `u` and `V` begins at eighth order. For a
very strong common rescaling, the bounded scalar approaches an orientation
quantity and its force derivative decays. The Sigma-v1/AQUAL background is
therefore left to supply the galaxy low-acceleration law; v4B tests only the
additional geometry-sensitive channel.

## Covariant target and unresolved health questions

This projected functional is the weak target of a possible nonmetricity
completion. A spatial projection of a nonmetricity trace can reduce to the
gravitational vector, and a constrained STF memory can be required to obey
`(1-L_sigma^2 Delta)M=S`.

That is not yet an accepted four-dimensional action. A passing projected test
would still have to derive a covariant clock or baryonic energy frame, show
that the constrained memory has no independently specifiable halo-like state,
count its modes, prove positive kinetic/gradient matrices, retain luminal
tensor propagation, and derive both metric potentials. Failure of the cheap
map gate prevents those expensive claims.

## Frozen data and constants

The inherited source-only AQUAL deflections at source redshift two span about
7--50 lens-plane kpc over their 1st--99th percentiles. This was measured without
reading the target maps and fixes the broad search
`0.1 <= ell_sigma <= 300 kpc`. The memory range is
`3.6458 <= L_sigma <= 300 kpc`.

For every `(L_sigma,ell_sigma)`, one shared nonnegative `eta_sigma` is solved
analytically. The three quantities are shared by AS295, PLCKG287, convergence,
and both shear channels. There is no cluster center, direction, external
shear, homogeneous memory state, or object-specific constant.

## Gates

All gates must pass:

1. The local stress chain rule and the complete scalar-functional derivative
   agree with centered finite differences within `1e-6` relative error.
2. Full periodic source mean/RMS is at most `1e-10`, and the cropped source has
   at least 10% pixels of each sign.
3. At least 50% of correction power lies at wavelengths of 50 kpc or larger.
4. The unconstrained shared amplitude is positive and both nonlinear scales
   lie more than 1% of their logarithmic ranges from every bound.
5. Joint normalized Fourier RMSE is at most `0.500`.
6. Each cluster improves by at least 20%, and every cluster/channel improves.
7. Parameters trained on either cluster transfer to the other with normalized
   RMSE at most `0.800`.
8. Repeating the fit with factor-three instead of factor-two padding changes
   joint RMSE by at most 5%.

Failure retires this exact mechanism. The sign will not be flipped, and no
post-failure exponent, orientation, or cluster-specific scale will be added.
No untouched observation is opened in this stage.

