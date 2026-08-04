# Sigma v17F action lift and one-metric obstruction

## Result

The v17F Helmholtz scale equation has a simple covariant, positive-energy
source-sector action. That action explains what a fitted propagation length
would mean physically. It does **not**, by itself, produce the linear Weyl
curvature assumed by the projected lensing diagnostic. This distinction is
fixed before any v17 thermal or lensing result.

## Covariant random-stress source

Let (u^\mu) be the baryonic Landau-frame timelike eigenvector and

\[
h_{\mu\nu}=g_{\mu\nu}+u_\mu u_\nu,
\qquad
S_{\mu\nu}=h_\mu{}^\alpha h_\nu{}^\beta T^b_{\alpha\beta}.
\]

Define the density-like random-stress scalar

\[
\boxed{
\mathcal J_T={\sqrt{S_{\mu\nu}S^{\mu\nu}}\over\sqrt3\,c^2}.
}
\]

For an isotropic fluid, (S_{\mu\nu}=p h_{\mu\nu}), so
(mathcal J_T=|p|/c^2). For nonrelativistic ideal gas,
(p=\rho k_BT/(\mu m_p)). Its line-of-sight projection therefore has the
same spatial dependence as the v17 gas proxy
(Sigma_g k_BT/(\mu m_pc^2)). The diagnostic's division by the lensing
critical surface density is only a map normalization; a fundamental action
cannot depend on source redshift or (Sigma_{\rm crit}).

This source also responds to collisionless velocity dispersion because that
appears in the spatial stress tensor. It does not contain a galaxy or cluster
label. Defining a smooth Landau frame through mergers and deriving this source
from the microscopic matter action remain nontrivial requirements.

## Minimal source/propagator action

After a field normalization, consider the source sector

\[
\boxed{
S_s=\int d^4x\sqrt{-g}\left[
-{L_\Sigma^2\over2}\nabla_\mu s\nabla^\mu s
-{s^2\over2}
+b_\Sigma s\mathcal J_T
\right].
}
\]

The dimensions can be carried by (b_\Sigma) and the normalization absorbed
into (s); only the normalization-invariant source coupling is physical once
the metric coupling is specified. Variation with respect to (s) gives

\[
\boxed{
(1-L_\Sigma^2\Box)s=b_\Sigma\mathcal J_T.
}
\]

In a quasistatic weak field, (Box\rightarrow\nabla^2), which is the v17F
Helmholtz equation before projection and amplitude normalization. For
(L_\Sigma>0), canonical normalization gives a positive kinetic term and

\[
m_\Sigma={1\over L_\Sigma},
\qquad
m_\Sigma c^2={\hbar c\over L_\Sigma}.
\]

Thus a common nonzero fitted length would be an inverse mediator mass, not a
separate halo radius. Object-to-object halo-size variation would still come
primarily from (mathcal J_T(\mathbf x)). At (L_\Sigma=0), (s) is an
auxiliary algebraic response (s=b_\Sigma\mathcal J_T), so no propagation
constant is justified.

The source coupling means baryonic matter and (s) exchange stress. General
covariance conserves their total stress, but universal free fall and a bounded
matter Hamiltonian must be derived; they do not follow merely from writing the
invariant.

## Why this is not yet the root gravity action

The projected v17F diagnostic assumes a response linear in (s):

\[
\Delta\kappa_\Sigma=\beta_\Sigma s,
\]

with the shear fixed by the same Weyl potential. The source action above does
not derive that relation.

If (s) is minimally coupled to the metric, it contributes through its own
stress-energy and the metric variation of (s\mathcal J_T). Since
(s\propto\mathcal J_T) at leading order, those contributions are quadratic
in the weak source, not the linear map used by v17F.

If all matter instead follows a purely conformal physical metric
(widetilde g_{\mu\nu}=A^2(s)g_{\mu\nu}), the direct scalar terms enter the
two weak-field potentials with opposite signs and cancel from

\[
W={\Phi+\Psi\over2}.
\]

That scalar can strengthen massive-particle dynamics without supplying the
needed direct lensing field. This is the standard scalar-tensor obstruction
already derived in `SIGMA_ONE_METRIC_ACTION_CONSTRAINTS.md`.

Therefore a v17F pass would establish a useful baryonic source and perhaps a
universal length, but not a complete theory. The next action must make the
mediator metric-active through one of the remaining routes:

1. sufficiently large, stable scalar stress backreaction;
2. one universal disformal physical metric; or
3. a healthy dynamical vector/tensor channel that creates Weyl anisotropic
   stress and carries orientation.

Each is established prior-art territory. The Sigma contribution could only be
the specific source and healthy completion that predicts galaxies and raw
cluster image topology with the same constants.

## Conditional decision

- If v17E fails, this action is not fitted or promoted; the measured thermal
  source did not carry enough transferable information.
- If v17E passes and v17F selects (L_\Sigma=0), derive an algebraic/source-
  local completion and remove the range constant.
- If a nonzero interior (L_\Sigma) passes, retain one inverse-mass scale and
  derive the full metric equations before opening a holdout.
- If size passes but shear alignment fails, retain the scale clue but require a
  directional carrier.
- If the full v17F gate fails, do not add a second range, running exponent, or
  cluster-specific length.

The provisional complete-theory budget remains at most five constants:
(a_\Sigma) for the galaxy transition, (L_\Sigma) only if required,
(b_\Sigma) for the stress source, and at most two healthy metric/carrier
couplings. None may be fitted per object.
