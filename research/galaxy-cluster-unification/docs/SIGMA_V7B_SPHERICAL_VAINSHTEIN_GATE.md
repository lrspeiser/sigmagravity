# Sigma v7B spherical Vainshtein gate

## Decision

The spherical Sigma v7B control is retired before observational data.  It shows
that ghost-free nonlinear screening can solve the *Solar-versus-large-scale*
problem of v7A, but it cannot by itself solve the *ordered galaxy versus
distributed cluster* problem.

The decisive reason is exact.  The spherical Vainshtein state depends on
`M/r^3`, or enclosed mean density.  Systems with the same `M/r^3` have the same
screening coordinate for every universal carrier range.  A protected
`10^11 M_sun`, 20-kpc archetype and a `10^14 M_sun`, 200-kpc strong-lensing
archetype are therefore mathematically indistinguishable to this screen.

There is also an independent amplitude ceiling.  In the one-metric-coupled
ghost-free bimetric family, the fully unscreened exterior enhances dynamics by
at most a factor `2` and light deflection by at most `1.5`.  That falls below the
factor-`3` conservative carrier target; the earlier empirical bridge required a
still larger diffuse-cluster response.

The result does **not** reject a full three-dimensional multi-source
Vainshtein solution.  Such a solution contains Hessian invariants and can, in
principle, distinguish a single coherent source from separated baryonic
components even when their enclosed mean densities agree.  That is the only
part of the spin-2 route that remains scientifically distinct from this failed
spherical control.

## Healthy nonlinear completion being tested

The complete ghost-free bimetric action has the established form

$$
\begin{aligned}
S={}&{M_g^2\over2}\int d^4x\sqrt{-g}\,R[g]
+{M_f^2\over2}\int d^4x\sqrt{-f}\,R[f]\\
&+m^2M_{\rm eff}^2\int d^4x\sqrt{-g}
\sum_{n=0}^4\beta_n e_n\!\left(\sqrt{g^{-1}f}\right)
+S_b[g,\psi_b].
\end{aligned}
$$

Only `g` is the physical matter metric.  The special elementary-symmetric
potential maintains the constraint that removes the Boulware--Deser mode.  The
result is one massless and one massive spin-2 field, with seven gravitational
degrees of freedom.  This is Hassan--Rosen/dRGT prior art, not a new Sigma
action ([Hassan & Rosen](https://arxiv.org/abs/1109.3515)).

The question for Sigma is whether this healthy carrier can supply the missing
orientation channel when joined to a separately derived low-acceleration trace
sector, without becoming a free halo state.

## Universal Vainshtein scaling

For a spherical source of mass `M`, the nonlinear radius is

$$
\boxed{
r_V=\left(r_S L_\Sigma^2\right)^{1/3}
=\left({2GM L_\Sigma^2\over c^2}\right)^{1/3}.
}
$$

Inside `r_V`, nonlinear helicity-zero interactions suppress the massive mode
and recover GR.  Outside `r_V` but inside the Compton range `L_Sigma`, the
additional spin-2 response is active.  This scaling and its application to
galaxy dynamics and cluster lensing were already studied by
[Platscher et al.](https://arxiv.org/abs/1809.05318); our spherical use is a
reproduction/control, not a novelty claim.

Rearranging the relation gives the mean density at transition:

$$
\boxed{
\bar\rho_V={3c^2\over8\pi G L_\Sigma^2}.
}
$$

Thus one universal range is equivalent to one universal mean-density threshold.
This automatically screens the Solar System because its enclosed density is
enormous.  It can also turn on a carrier at astrophysical distances without a
negative pole or an object-specific boundary condition.

## Exact no-label degeneracy

The dimensionless screening coordinate is

$$
{r\over r_V}
=\left({c^2r^3\over2GM L_\Sigma^2}\right)^{1/3}.
$$

Compare the deliberately label-free stress pair

$$
(M_A,r_A)=(10^{11}M_\odot,20\ {\rm kpc}),
$$

$$
(M_B,r_B)=(10^{14}M_\odot,200\ {\rm kpc}).
$$

They obey

$$
{M_A\over r_A^3}={M_B\over r_B^3}.
$$

Consequently

$$
{r_A\over r_{V,A}}={r_B\over r_{V,B}}
$$

for **every** `L_Sigma`.  The numerical scan covered 1 kpc through 10 Gpc and
found a maximum relative difference of `6.54e-16`.  Both systems cross at the
same carrier range,

$$
L_\Sigma=28.911\ {\rm Mpc}.
$$

No choice of the universal range can put the first system inside the screen and
the second outside.  Calling one a galaxy and the other a cluster would only
hide an object label inside the interpretation.

## Fixed maximum exterior amplitude

Let `theta` be the bimetric mixing angle and `s=sin^2(theta)`.  The linear
exterior potential has coefficients

$$
\alpha=(1-s)\left(1+{2s\over3}\right),
$$

$$
\beta={2s\over3}(1+2s).
$$

For distances outside the Vainshtein radius but well inside the carrier range,
the maximum dynamical factor is

$$
\max_\theta(\alpha+\beta)=2,
$$

while the corresponding maximum light-deflection factor is

$$
\boxed{
\max_\theta\left(\alpha+{3\beta\over4}\right)=1.5.
}
$$

The coefficients are non-negative throughout the healthy mixing interval.
Changing the graviton mass can decide *where* the response appears, but it
cannot raise this amplitude ceiling.  Adding a free normalization would leave
the frozen ghost-free bimetric family and become the kind of lensing multiplier
forbidden by the goal.

## What this teaches us

Vainshtein screening supplies one part of the desired physical picture:
positive gravity can strongly suppress an additional gravitational carrier in
dense regions and let it appear in diffuse regions.  The failure is that the
spherical approximation erases the information we care most about:

- whether mass is one coherent body or many separated components;
- the eigenvectors of the baryonic tidal Hessian;
- component overlap and interference; and
- where shear, folds, and caustics form.

This is the same distinction seen empirically when applying a nonlinear response
to individual components before summing produced more lensing roots.  A true
test must therefore solve the nonlinear Hessian equation before components are
combined.  A spherical `r_V` switch cannot serve as that test.

## Requirement for v7C

The next admissible calculation is the parameter-frozen, three-dimensional
decoupling-limit equation for the helicity-zero mode, schematically

$$
3\nabla^2\pi
+{1\over\Lambda_3^3}
\left[(\nabla^2\pi)^2-(\partial_i\partial_j\pi)^2\right]
=-{T_b\over M_{\rm Pl}},
$$

with coefficients fixed to a stable branch of the ghost-free potential.  Before
any observational score it must demonstrate:

1. an elliptic, unique static branch on nonspherical sources;
2. a positive perturbation kinetic matrix on that branch;
3. convergence to the analytic spherical solution;
4. no per-object transition width, center, orientation, or boundary state;
5. a response to separated components that is not reducible to total enclosed
   density; and
6. one derived matter/lensing metric, not a separate photon amplitude.

If this full solve reduces effectively to the spherical density threshold or
cannot exceed the fixed lensing ceiling, the positive spin-2 route is exhausted
for the present goal.

No observational arrays or raw holdouts were opened for v7B.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v7b_spherical_vainshtein.py
python -m pytest -q tests/test_sigma_v7_vainshtein.py
```

Machine-readable evidence is stored in
`results/sigma_v7b_spherical_vainshtein_gate/report.json`.
