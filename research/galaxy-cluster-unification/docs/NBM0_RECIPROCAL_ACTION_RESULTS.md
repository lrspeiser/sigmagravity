# NBM0-A1 reciprocal action and finite model-space result

## Decision

The original linear NBM0 equation is now retired as a galaxy--cluster unifier
and retained as a reciprocal null model.  Completing the matter variation removes
its independent source parameter, and a radial-shape theorem rejects not only one
chosen Yukawa range but every nonnegative superposition of canonical massive
scalar exchanges around a finite baryonic source.

This conclusion required no astronomical parameter fit.  Of ten frozen response
families, seven fail structural gates, one is excluded because it recreates the
retired MOND/AQUAL galaxy branch, and two advance to action derivation:

1. a causal nonlinear nonlocal basin response; and
2. a self-gravitating basin phase whose own stress-energy participates in
   dynamics and lensing.

The machine-readable protocol is
`configs/nbm0_reciprocal_action_protocol.json`; the executable result is
`results/nbm0_action_space/report.json`.

## The reciprocal canonical action

The controlled action is

$$
S=\int d^4x\sqrt{-g}\left[
 {M_{\rm Pl}^2\over2}R
 -{M_{\rm Pl}^2\over2}\left((\nabla X)^2+L_X^{-2}X^2\right)
 -{M_{\rm Pl}^2c_U\over4}F^{(U)}_{\mu\nu}F_{(U)}^{\mu\nu}
 +\lambda(U_\mu U^\mu+1)
 \right]+S_m[\tilde g,\psi_m],
$$

with one physical metric,

$$
\tilde g_{\mu\nu}=e^{2\alpha X}
\left(g_{\mu\nu}+2\beta XU_\mu U_\nu\right).
$$

The Maxwell-aether term is only a weak-field control.  A preferred timelike
direction cannot be prescribed externally in a generally covariant completion,
but this truncation has not passed the full scalar/vector mode or preferred-frame
PPN gates.

Varying the *same* matter action that defines geodesics gives

$$
J_X={\sqrt{-\tilde g}\over2\sqrt{-g}}
\tilde T^{\mu\nu}{\partial\tilde g_{\mu\nu}\over\partial X},
\qquad
(\Box-L_X^{-2})X=-{J_X\over M_{\rm Pl}^2}.
$$

At $X=0$, $J_X=\alpha T+\beta T^{\mu\nu}U_\mu U_\nu$.  For cold matter,
$T\simeq-\rho c^2$ and $T^{\mu\nu}U_\mu U_\nu\simeq\rho c^2$, so

$$
(\nabla^2-L_X^{-2})X
=(\alpha-\beta){8\pi G\rho\over c^2}.
$$

Thus the source coefficient is not a separately fitted $\kappa_X$.  Defining
$d=\alpha-\beta$, a point mass produces

$$
X(r)=-{2dGM\over c^2r}e^{-r/L_X}.
$$

The phrase ``density-contrast source'' also needs refinement.  A local covariant
action does not simply know a manually subtracted $T_{\rm background}$.  The
homogeneous cosmological solution must be found first; its perturbation equation
then contains $\delta J_X$ on the slicing selected dynamically by $U^\mu$.

## What observations can identify

The weak-field physical metric still gives

$$
\Psi=U_N+c^2(\alpha-\beta)X,
\quad
\Phi=U_N-c^2\alpha X,
\quad
{\Phi+\Psi\over2}=U_N-{c^2\beta X\over2}.
$$

After substituting the reciprocal point-source solution, the observable
amplitudes are

$$
A_{\rm dyn}=2d^2,
\qquad
A_{\rm lens}=-\beta d,
\qquad
q={A_{\rm lens}\over A_{\rm dyn}}
=-{\beta\over2(\alpha-\beta)}.
$$

Consequently ideal overlapping dynamics and lensing identify only
$(A_{\rm dyn},L_X,q)$.  The transformation
$(\alpha,\beta,X)\mapsto(-\alpha,-\beta,-X)$ changes no weak-field observable.
The previously interesting $\alpha=\beta$ ``dynamics-blind'' metric limit is also
source-blind for nonrelativistic baryons: it generates neither isolated-object
dynamics nor isolated-object lensing at this order.

The synthetic injection used $A_{\rm dyn}=1.8$, $L_X=24$ kpc and $q=2.4$ over
$0.2<r/L_X<8$.  The solver recovered 1.8, 24.000 kpc and 2.4 with maximum
absolute log residual $6.6\times10^{-15}$ and Jacobian condition number 6.06.
This proves ideal identifiability of the reduced parameters; it does not replace
the failed real same-system data gate.

## Linear radial-shape theorem

Outside a finite point source, Newtonian gravity plus a canonical attractive
scalar has

$$
g(r)={GM\over r^2}E(r),
\qquad
E(r)=1+A(1+x)e^{-x},\quad x={r\over L_X},\quad A\ge0.
$$

Because

$$
{dE\over d\ln r}=-Ax^2e^{-x}\le0,
$$

the circular-speed slope obeys

$$
{d\ln v_c\over d\ln r}
=-{1\over2}+{1\over2}{d\ln E\over d\ln r}
\le-{1\over2}.
$$

Increasing the coupling makes the transition decline *more* steeply.  It never
creates a flat interval.  More generally, for a healthy nonnegative spectral
mixture,

$$
E(r)=1+\sum_i A_i(1+r/L_i)e^{-r/L_i},\qquad A_i\ge0,
$$

every term has the same sign of derivative, so the theorem survives any number
or continuum of positive-norm Yukawa modes.  The numerical scan used six ranges
from 0.01 to 1000 kpc and found a maximum speed slope of -0.500010.  This is a
sign/shape result, not a failure to try a large enough amplitude.

A linear fractional operator can evade the radial part.  In three spatial
dimensions, $(-\nabla^2)^p\Phi\propto\rho$ gives
$v_c^2\propto Mr^{2p-3}$.  The special $p=3/2$ case produces a logarithmic
potential and flat speed, but linear sourcing then predicts $v_c^4\propto M^2$
rather than the observed near-linear baryonic mass relation.  This repeats a
problem reported for other logarithmic/fractional galaxy potentials, not a new
solution.

## Frozen family decisions

| ID | Family | Decision | Decisive reason |
|---|---|---|---|
| A0 | Canonical conformal scalar | Reject | Scalar response cancels from the Weyl potential. |
| A1 | Disformal scalar with prescribed $U$ | Reject | No reciprocal equation for the preferred direction. |
| A2 | Canonical scalar plus dynamical aether | Null only | Positive Yukawa response is never flatter than Keplerian. |
| A3 | Massless canonical scalar | Reject | Constant Newtonian rescaling, no flat curve, no screening. |
| A4 | Positive spectral Yukawa continuum | Reject | Nonnegative spectral weight cannot turn the force on at large radius. |
| A5 | Linear fractional $p=3/2$ | Reject | Flat radial shape but $v^4\propto M^2$. |
| A6 | Nonlinear scale-invariant p-Laplacian | Exclude | This is the retired MOND/AQUAL galaxy mechanism. |
| A7 | Smooth external void basin | Reject for broad support | Uniform field cancels internally; leading regular tide has $g\propto r$. |
| A8 | Causal nonlinear nonlocal basin | Advance | Can in principle evade the positive-spectral theorem; action not yet known. |
| A9 | Self-gravitating basin phase | Advance | Can in principle give one extended source for dynamics and lensing; health not yet known. |

For A7, a smooth isotropic field around a galaxy center has a regular expansion
$X=X_0+X_2r^2/2+\cdots$.  Its constant is unobservable in internal dynamics, its
uniform gradient is a common acceleration, and its first isotropic relative force
is harmonic, $g\propto r$, giving $v\propto r$.  A sharp basin edge may create a
narrow transition, but the already-completed CF4 void-cage test found no robust
held-out directional or environmental signal.

## Prior-art boundary

The scalar--vector--tensor field content, dynamical unit timelike vectors,
conformal/disformal physical metrics, nonlocal form factors, and logarithmic
galaxy potentials all have prior art.  Relevant primary starting points include
[Bekenstein's TeVeS construction](https://arxiv.org/abs/astro-ph/0403694),
[the Einstein-aether wave analysis](https://arxiv.org/abs/gr-qc/0402005),
[the Einstein-aether status review](https://arxiv.org/abs/0801.1547), and a
[fractional-spacetime logarithmic-potential test](https://arxiv.org/abs/2112.13103).
The last of these likewise reports difficulty reproducing the Tully--Fisher
relation.

The project-specific contribution at this stage is narrower: the reciprocal
parameter reduction, the explicit lensing/dynamics sign map for this basin metric,
the preregistered same-system test, and the finite decision tree that prevents a
failed linear model from being rescued by arbitrary constants.

## Next derivations

A8 and A9 are research questions, not successful models.  Each now gets one
minimal action cycle:

- **A8:** localize a causal nonlinear nonlocal kernel with auxiliary fields,
  derive its quadratic pole/kinetic structure, and reject it before fitting if
  flat curves require a negative spectral weight, a MOND/AQUAL limit, or an
  object-dependent scale.
- **A9:** derive the smallest positive-energy basin condensate/nonminimal-coupling
  action and test whether an underdensity can repel while accumulated field energy
  around overdensities produces the same-sign dynamics and lensing.  Reject it if
  repulsion requires a ghost, if the stress only behaves as an unconstrained dark
  halo, or if Solar-System screening is absent.

No empirical gravity fit is authorized during either derivation.
