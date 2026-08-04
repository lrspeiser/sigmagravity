# Sigma v18 post-pressure gravitational-flux selection

## Decision

The next root equation will not add a fourth pressure screen. It will begin
from a gravitational-flux description that makes the effective halo an output
of the baryonic boundary-value problem. The additional spatial-state term is
not selected yet: the already-frozen v17C--v17F measurement chain must first
say whether resolved baryonic random stress actually carries transferable halo
extent and orientation information.

This is a mechanism-selection framework, not a claimed covariant theory.

## The first-principles bookkeeping

Introduce an outward gravitational displacement field
\(\mathbf D\) constrained by the observed baryons:

\[
\boxed{\nabla\cdot\mathbf D=4\pi G\rho_b.}
\]

Let \(W\) be the weak-field potential of the eventual one physical metric and
\(\mathbf g=-\nabla W\) the acceleration of slow matter. A first-order static
functional is

\[
\boxed{
\mathcal I_{\rm stat}=\int d^3x\left\{
{\mathcal H(\mathbf D,\nabla\mathbf D,\mathcal Z_b)
-\mathbf D\cdot\nabla W\over4\pi G}
-\rho_bW
\right\}.
}
\]

Varying \(W\) gives the Gauss constraint. Varying \(\mathbf D\) gives the
constitutive equation

\[
\nabla W={\delta\mathcal H\over\delta\mathbf D}.
\]

In the Newtonian limit \(\mathcal H=|\mathbf D|^2/2\), so
\(\mathbf g=-\mathbf D\). A modified theory changes the constitutive response,
not the baryonic charge.

The apparent halo that a conventional analysis would infer is then

\[
\boxed{
\rho_{\Sigma,\rm eff}
=-{1\over4\pi G}\nabla\cdot(\mathbf g+\mathbf D).
}
\]

This quantity is not an input and is not new matter. It is a diagnostic of
where the predicted physical-metric field differs from the baryonic Newtonian
flux.

## Where an apparent halo size can come from

If \(\mathcal H\) depends only on \(|\mathbf D|\), eliminating \(\mathbf D\)
returns published AQUAL. In isolated spherical symmetry,

\[
|\mathbf D|={GM_b\over r^2}
\]

and the universal acceleration \(a_\Sigma\) produces the familiar transition

\[
r_M=\sqrt{GM_b/a_\Sigma}.
\]

That is already MOND's scale mechanism; it is not a Sigma discovery and is
not sufficient for the cluster problem.

For extended, flattened, or multi-component systems, however,
\(\mathbf D(\mathbf x)\) is a vector boundary-value solution. Its transition
surfaces can overlap, bend, and become non-spherical. A spatial-state variable
\(\mathcal Z_b\), if independently justified, can make the constitutive
response depend on this continuous geometry without a galaxy/cluster label.
The effective halo's centroid, extent, and orientation must then be calculated
from \(\rho_{\Sigma,\rm eff}\), not assigned through \(r_s(M_b)\).

The v17O audit explains why this distinction matters. One shared relation
based on total mass or baryonic half-mass radius missed the inferred galaxy and
cluster scale radii by at least 0.354 dex. The equation must solve the spatial
field rather than turn one catalog scalar into a halo radius.

## Why a fixed elastic length is not automatically the answer

The minimal local flux-elasticity term would be

\[
\mathcal H_{\rm el}
={L_\Sigma^2\over2}s(|\mathbf D|/a_\Sigma)
(\partial_iD_j)(\partial_iD_j).
\]

For an isolated spherical exterior \(D=GM/r^2\), direct variation gives

\[
\delta g=-L_\Sigma^2{ds\over dD}{D^2\over r^2}.
\]

At \(r=r_M\), its relative importance scales as

\[
{\delta g\over g}\propto{L_\Sigma^2\over r_M^2}
\propto{L_\Sigma^2\over M_b}.
\]

Thus one fixed length preferentially changes low-mass systems; it is not a
universal substitute for every halo radius. V17F is allowed to retain exactly
one correlation length only if the blinded cross-cluster extent test selects
an interior value and passes all transfer, shear, power, and resolution gates.

## Evidence-constrained branch

Both clusters have passed the target-blind spatial extraction gate: AS295 has
29 frozen regions and PLCK G287 has 21. Their v17C spectra are being processed
under the frozen common response, background, model, and uncertainty rules.
No temperature or v17E lensing score existed when this decision tree was
frozen.

The result controls the next action:

1. If thermal stress fails any v17E gate, pressure and temperature cannot
   appear in \(\mathcal H\). The next measurement is the already-declared
   matched collisionless-stress test or a materially different causal state.
2. If v17E passes and v17F selects \(L_\Sigma=0\), derive a source-local
   spatial-state constitutive action and spend no range constant.
3. If v17E passes and v17F selects one nonzero interior length, derive a
   dynamical state carrier with that one universal correlation length. Pressure
   may source its state, but it may not directly charge the physical metric.
4. If extent passes while alignment or topology fails, retain only the scale
   clue and require a directional carrier. Do not fit orientation or shear.
5. If v17F fails or lands on its upper endpoint, do not extend the grid or add
   another length.

## Covariant action still required

The static functional is useful because it states exactly what must generate
the effective halo. It does not prove that a relativistic completion exists.
Before any holdout, the surviving branch must derive one covariant action that
has all of the following:

- matter minimally coupled to one physical metric;
- both massive-particle dynamics and photon lensing derived from that metric;
- a positive constitutive Hessian and bounded Hamiltonian;
- luminal tensor waves, stable propagation, and a well-posed initial-value
  problem;
- complete Solar suppression, including the spatial-state contribution;
- at most five universal constants and no per-object gravity parameters; and
- no direct pressure-only reciprocal metric or derivative-dependent matter
  metric.

## Prior-art boundary

The flux language and the \(\mathcal H(|D|)\) limit are Legendre-dual AQUAL.
QUMOND already performs nonlinear source routing. Refracted Gravity uses a
gravitational permittivity. Gravitational polarization is published in dipolar
dark-matter models, and universal kernels or vector carriers overlap nonlocal
gravity, Einstein-aether, and AeST. None of those broad ingredients is new.

A distinctive result would require the complete combination: an independently
measured baryonic dynamical state, a healthy gravitational—not dark-medium—
spatial carrier, one frozen metric law, emergent effective-halo geometry, raw
cluster topology, competitive galaxy predictions, and Solar consistency.

The authoritative target-blind freeze is
`configs/sigma_v18_post_pressure_flux_selection.json`.
