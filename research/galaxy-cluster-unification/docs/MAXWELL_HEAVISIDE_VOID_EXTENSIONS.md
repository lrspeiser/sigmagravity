# Maxwell-Heaviside void extensions: theory and test registry

Status: exploratory theory registry, recorded 2026-07-26 after the existing
SPARC, CF4, CLASH, and MaNGA results had been inspected. It is not a
preregistration for a discovery claim.

## Purpose and boundary

Maxwell-Heaviside gravity is useful here because a gravitational scalar
potential and three-vector potential can be combined into a four-potential,

$$
B_\mu=(\phi_g/c,\mathbf A_g),\qquad
H_{\mu\nu}=\nabla_\mu B_\nu-\nabla_\nu B_\mu.
$$

It predates general relativity and resembles the weak-field,
slow-motion gravitoelectromagnetic limit. It is not a full alternative that
reproduces all of general relativity. A fundamental vector-only theory has
problems with the sign of static attraction, positive field energy, universal
free fall, nonlinear self-gravity, and lensing. This registry therefore treats
the four-potential as an additional environment field or weak-field language,
not as a replacement for the spacetime metric.

The target is stricter than fitting a rotation curve. In the physical weak-field
metric

$$
d\tilde s^2=-(1+2\Psi/c^2)c^2dt^2
 +(1-2\Phi/c^2)d\mathbf x^2,
$$

the same theory must predict

$$
v_c^2(R)=R\,\partial_R\Psi,
\qquad
\boldsymbol\alpha=
\frac{1}{c^2}\int\nabla_\perp(\Phi+\Psi)\,d\ell.
$$

No extension may fit a separate lensing multiplier. Gravitational slip
$\Phi-\Psi$ must be derived.

## What "void charge" means

A void is not assumed to contain a new negative-mass material. Its baseline
source is the conserved cosmological density perturbation relative to a
declared background,

$$
\delta J^\mu=J^\mu-\bar J^\mu,
\qquad
\nabla_\mu J^\mu=0.
$$

For a static density perturbation, the Maxwell-Heaviside control becomes

$$
\nabla\cdot\mathbf E_g=-4\pi G\,\delta\rho.
$$

An underdensity has $\delta\rho<0$ and therefore produces an outward peculiar
field. This is ordinary void evacuation expressed in vector language. It does
not by itself modify internal galaxy dynamics: a uniform external field is
shared by the galaxy and its stars, while the measured CF4 tide is about five
orders of magnitude too small at the median.

The word *antigravity* below therefore has two carefully separated meanings:

1. a direct outward field sourced by a measured density deficit; and
2. a low-density environment changing the response to ordinary baryons, which
   can produce the additional inward acceleration required by galaxy dynamics
   and cluster lensing.

The second mechanism is the more promising one.

## Common covariant envelope

The extensions can be organized inside the schematic action

$$
S=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac{Z(X,Y)}{4}H_{\mu\nu}H^{\mu\nu}
-\frac{m_B^2(X)}{2}B_\mu B^\mu
+\mathcal L_X+\mathcal L_{\rm int}
\right]
+S_m[\tilde g_{\mu\nu},\psi_m],
$$

where $X$ is an optional environment field, $Y$ denotes allowed field
invariants, and $\tilde g_{\mu\nu}$ is the one physical metric followed by both
matter and light. The signs, functions, and degrees of freedom are not assumed
healthy merely because this expression can be written; each candidate needs a
Hamiltonian/stability analysis.

## Linear control plus nine modifications

### H0: conserved density-contrast control

Use the linear four-vector equation

$$
\nabla_\nu H^{\nu\mu}=-\frac{4\pi G}{c^2}\delta J^\mu.
$$

**Hole addressed:** gives the historical vector theory a cosmologically defined
void source rather than invented point voids or galaxy-pair midpoints.

**Prediction:** outward bulk flows and a tidal differential field with measured
void geometry. It does not produce a universal $1/R$ galactic acceleration.

**Status:** physical control, already disfavored as an explanation of internal
SPARC residuals by the zero-parameter CF4 tidal test. It remains useful for
bulk-flow and orientation checks.

### H1: enhanced void charge

Allow a single universal strength for the measured deficit,

$$
\nabla_\nu H^{\nu\mu}
=-\frac{4\pi G}{c^2}
\left(\delta J_b^\mu+\eta_v\delta J_v^\mu\right),
$$

where $\delta J_v^\mu$ is constructed from negative-density-contrast cells and
$\eta_v=1$ is H0.

**Hole addressed:** tests whether direct void repulsion merely has the wrong
amplitude.

**Hard problem:** the existing scale check implies a characteristic
$\eta_v\sim10^5$ would be required if the field remains linear. Such a value
would strongly alter void velocities, redshift-space distortions, and structure
growth. A large fitted value is a rejection signal, not evidence by itself.

**Distinct prediction:** lopsided and orientation-dependent galaxy residuals
aligned with the reconstructed external field.

### H2: potential-screened vector charge

Make the coupling weak in a deep baryonic well and active near a galaxy edge,

$$
q_{\rm eff}(\chi)=q_0S_\chi,
\qquad
S_\chi=\left[1+
\left(\frac{\chi}{\chi_t}\right)^n\right]^{-1},
\qquad
\chi=|\Phi_{\rm bar}+\Phi_{\rm host}|/c^2.
$$

The fifth-force contribution to a body is schematically

$$
\mathbf a_B=q_{\rm eff}(\chi)
(\mathbf E_g+\mathbf v\times\mathbf B_g).
$$

**Hole addressed:** a nearly uniform external field can now couple differently
to screened central matter and unscreened outer matter.

**Hard problem:** inserting $q_{\rm eff}$ only in the force law violates
energy-momentum conservation and the strong equivalence principle. It is a
phenomenological test only; H4 or H6 must generate the screening dynamically.

**Distinct prediction:** a transition surface, side-to-side asymmetry when the
void field is not disk-normal, and a correlation between transition radius and
independently measured host potential.

### H3: nonlinear gravitational permittivity

Replace the linear constitutive relation with a nonlinear vacuum response,

$$
\nabla_\nu\left[
\mathcal K_Y(Y,X)H^{\nu\mu}
\right]
=-\frac{4\pi G}{c^2}J_b^\mu.
$$

In the quasistatic limit,

$$
\nabla\cdot\left[
\mu\!\left(\frac{|\nabla\Psi|}{a_X},X\right)\nabla\Psi
\right]=4\pi G\rho_b.
$$

Choosing $\mu(x)\rightarrow x$ at low acceleration produces
$|\nabla\Psi|\propto1/R$ around an isolated finite mass and therefore a flat
outer rotation curve. The environment may change $a_X$, but the baryonic source
still determines the inward direction.

**Holes addressed:** correct galactic radial scaling and avoids requiring a
$10^5$ direct amplification of a tiny external tide.

**Prior-art boundary:** nonlinear Poisson/MOND behavior is established prior
art. The research question is whether one independently reconstructed void/host
field $X$ can set the response across galaxies and clusters without object
labels.

**Distinct prediction:** the RAR transition scale changes with $X$, while a
uniform external vector produces a smaller directional correction.

### H4: environment-dependent Maxwell-Proca range

Give the vector a positive, environment-dependent mass,

$$
\nabla_\nu H^{\nu\mu}-m_B^2(X)B^\mu
=-\frac{4\pi G}{c^2}J^\mu,
$$

with $m_B$ large in dense/deep environments and small in voids. The force range
$\lambda_B=1/m_B$ is then short in the Solar System and long on galactic or
intergalactic scales.

**Holes addressed:** local screening, finite interaction range, and the
longitudinal degree of freedom can be handled in a standard Proca framework.

**Hard problem:** a density-dependent mass must come from a dynamical field and
must not introduce ghosts or singular evolution. A long-range vector alone
still does not supply the observed lensing.

**Distinct prediction:** transition radii track both system potential and
environment, while the force cuts off beyond a measured correlation length.

### H5: two-channel MOG-like attraction and repulsion

Let the metric/tensor channel provide enhanced attraction and a healthy vector
provide a compensating short-range repulsion. A useful spherical benchmark is

$$
g(r)=-\frac{G_NM}{r^2}
\left[1+\alpha(X)
-\alpha(X)(1+\mu(X)r)e^{-\mu(X)r}\right].
$$

At small $r$ the two modifications cancel and recover Newtonian gravity. At
large $r$ the repulsive vector term decays, leaving stronger inward attraction.

**Holes addressed:** resolves the vector-attraction sign problem without a
negative kinetic energy and supplies an explicit screening shape.

**Prior-art boundary:** this is a scalar-tensor-vector/MOG benchmark, not a new
formula. The proposed new test is whether $\alpha$ and $\mu$ can be predicted by
the measured void/host field instead of being assigned from object mass or fit
per system.

**Hard problem:** a constant finite-range Yukawa correction is not
asymptotically flat; environmental running must predict the observed radial
range without becoming a flexible halo substitute.

### H6: disformal metric and derived lensing

Make matter and light follow one physical metric constructed from the Einstein
metric and vector,

$$
\tilde g_{\mu\nu}
=C(X,B^2)g_{\mu\nu}+D(X,B^2)B_\mu B_\nu.
$$

The field equations then determine both $\Psi$ and $\Phi$ and hence the slip

$$
\varpi(r)=\frac{\Phi(r)-\Psi(r)}{\Psi(r)}.
$$

**Hole addressed:** light no longer ignores the additional field, and the
galaxy-dynamics/lensing relation becomes a prediction.

**Hard rule:** $C$ and $D$ cannot be separately calibrated on CLASH after the
dynamical law is fit. A candidate advances only if the same constants predict
SPARC speeds, BCG dynamics, and CLASH lensing.

**Prior-art boundary:** disformal metric coupling and TeVeS-like lensing are
established. This is a required completion, not an originality claim.

### H7: dynamical scalar-vector-tensor environment field

Introduce a scalar $X$ that records the smoothed host/void environment and
changes the nonlinear vector/scalar response:

$$
(\Box-L_X^{-2})X=-\kappa_X(T-\bar T),
$$

$$
\nabla_\mu\left[
\mu(Y,X)\nabla^\mu\varphi
\right]=\kappa_\varphi T,
\qquad
a_X=a_0\exp(\beta X).
$$

The metric and four-vector provide the relativistic/lensing completion, while
$\varphi$ supplies an inward baryon-centered response. Voids influence the
strength or screening through $X$ rather than being treated as literal negative
objects.

**Holes addressed:** covariant environment variable, conservation, flat-curve
limit, host-potential dependence, and lensing completion can coexist in one
action.

**Connection to the current result:** the U0 logistic
$a_{\rm eff}(\chi)$ becomes a weak-field target that this model must derive,
with $X$ computed from baryons and large-scale structure rather than an
unobservable additive potential constant.

**Assessment:** best full-theory direction, but also the most important place
to control parameter count. Start with one range $L_X$, one environment
coupling $\beta$, and one fixed nonlinear response family.

### H8: generalized Einstein-Aether vector

Constrain a timelike vector $U^\mu U_\mu=-1$ and replace the Maxwell kinetic
term by a function of covariant derivative invariants,

$$
S_U\supset
\frac{M_{\rm Pl}^2}{2}\int d^4x\sqrt{-g}\,
a_0^2\mathcal F(\mathcal K,X),
$$

$$
\mathcal K=
c_1\nabla_\mu U_\nu\nabla^\mu U^\nu
+c_2(\nabla_\mu U^\mu)^2
+c_3\nabla_\mu U_\nu\nabla^\nu U^\mu
+c_4U^\mu U^\nu\nabla_\mu U_\alpha\nabla_\nu U^\alpha.
$$

**Holes addressed:** provides a covariant preferred time direction, nonlinear
MOND-like weak-field behavior, and metric lensing within a vector-tensor theory.

**Prior-art boundary:** generalized Einstein-Aether MOND is established. A
void-dependent $X$ is an extension to benchmark, not grounds for a broad new
theory claim.

**Hard problem:** gravitational-wave speed, preferred-frame PPN parameters,
Cherenkov constraints, and scalar/vector stability drastically restrict the
$c_i$ combinations.

### H9: retarded nonlocal environment memory

Allow the response to depend on the retarded large-scale density history,

$$
X=\Box_{\rm ret}^{-1}(T-\bar T),
\qquad
Z=Z(X),\quad m_B=m_B(X).
$$

This converts an absolute-potential-like idea into a defined covariant operator
with declared causal boundary conditions. A galaxy can then respond to an
extended host or void basin rather than only the density at one point.

**Holes addressed:** captures long-range environment and avoids using an
arbitrary constant offset in Newtonian potential.

**Assessment:** high-risk fallback. The retarded kernel, initial conditions,
localization, ghosts, and cosmological perturbations must be specified before
any fit. It should not advance unless local H7 and H8 completions fail for a
clear structural reason.

## Comparison and ranking

| Variant | Flat galaxy curves | Lensing completion | Solar-System path | Direct void direction | Current priority |
|---|---:|---:|---:|---:|---|
| H0 linear control | No | No | Trivial weak field | Yes | Retain as null |
| H1 enhanced deficit | No | No | Poor if large | Yes | Low; scale likely fatal |
| H2 screened charge | Not by itself | No | Possible | Yes | Directional diagnostic |
| H3 nonlinear response | Yes | Needs H6 | Yes in high-field limit | Environment modulation | High |
| H4 Proca range | Not by itself | Needs H6 | Yes | Yes | Medium ingredient |
| H5 MOG-like two channel | Finite-range approximation | Metric supplies it | Yes by cancellation | Through $X$ extension | Strong benchmark |
| H6 disformal metric | Inherits dynamics | Yes | Must be checked | Inherits source | Mandatory completion |
| H7 scalar-vector-tensor | Yes | Yes with H6-type metric | Screening required | Yes, plus inward baryon response | **Highest** |
| H8 generalized Aether | Yes for chosen $\mathcal F$ | Yes | Restricted parameter space | Through $X$ extension | High benchmark |
| H9 nonlocal memory | Possible | Possible | Kernel dependent | Yes | High risk |

The recommended complete candidates are combinations, not isolated rows:

1. **EV-SVT:** H3 + H4 + H6 + the minimal H7 environment equation.
2. **Environmental Aether:** H8 with one independently measured $X$ input.
3. **Environmental MOG control:** H5 with $\alpha(X)$ and $\mu(X)$ frozen from
   non-kinematic environment data.

## Ordered research program

1. Preserve H0 and the completed CF4 tide result as the zero-parameter null.
2. Test H2's directional signature using resolved two-dimensional H I velocity
   fields. Folded SPARC curves cannot identify it.
3. Derive the spherical and axisymmetric weak-field limits of EV-SVT. Require
   the low-acceleration solution to approach $1/R$ without inserting the RAR as
   the answer.
4. Map the derived weak-field coefficients to the already observed U0 target.
   The mapping is a consistency check, not a refit license.
5. Solve for both $\Phi$ and $\Psi$ and predict CLASH lensing with no
   lensing-only constant.
6. Use independently reconstructed gas, stellar, and satellite profiles to
   calculate host $X$ for the frozen E0-style BCG test.
7. Compare EV-SVT against H5 and H8 under identical system folds and parameter
   accounting.
8. Only after a candidate passes galaxies, BCGs, and clusters, calculate PPN
   parameters, mode speeds, ghost/gradient conditions, gravitational-wave
   propagation, and homogeneous cosmology. A failure in those checks rejects
   the candidate even if its rotation curves are good.

## Falsification rules

- A direct-void candidate fails if it needs an enhancement of order $10^5$
  while contradicting measured void outflows or predicts no directional
  residual.
- An environment candidate fails if its environment parameter changes sign
  across held-out folds or if an object-class label predicts better than the
  actual measured environment.
- A purported unified candidate fails if it needs independent dynamics and
  lensing multipliers.
- A screening candidate fails if it cannot recover GR in high-acceleration
  Solar-System and compact-object regimes.
- A vector theory fails if its kinetic matrix has a ghost, a gradient
  instability, an ill-posed Cauchy problem, or excluded propagation speeds.
- A model that fits only after choosing among several environment definitions
  on the scored sample remains exploratory and must be tested on untouched
  systems.

## Prior-art map

- Mashhoon, [Gravitoelectromagnetism: A Brief
  Review](https://arxiv.org/abs/gr-qc/0311030), reviews the relation between
  Maxwell-like gravity and the weak-field limit of general relativity.
- Bekenstein, [Relativistic gravitation theory for the MOND
  paradigm](https://arxiv.org/abs/astro-ph/0403694), is direct prior art for a
  dynamical metric, scalar, and four-vector predicting both dynamics and
  lensing.
- Zlosnik, Ferreira, and Starkman, [Modifying gravity with the Aether: an
  alternative to Dark Matter](https://arxiv.org/abs/astro-ph/0607411), is direct
  prior art for nonlinear vector-tensor gravity with MOND-like limits.
- Moffat, [Scalar-Tensor-Vector Gravity
  Theory](https://arxiv.org/abs/gr-qc/0506021), supplies the Maxwell-Proca and
  enhanced-attraction/repulsive-vector precedent represented by H5.
- Heisenberg, [Generalization of the Proca
  Action](https://arxiv.org/abs/1402.7026), develops vector derivative
  interactions with the degeneracy needed to avoid an extra ghost degree of
  freedom.
- Khoury and Weltman, [Chameleon
  cosmology](https://arxiv.org/abs/astro-ph/0309411), and Hinterbichler and
  Khoury, [Symmetron Fields](https://arxiv.org/abs/1001.4525), are prior art for
  environment-dependent screening.
- Deser and Woodard, [Nonlocal
  Cosmology](https://arxiv.org/abs/0706.2151), is prior art for inverse
  d'Alembertian environment/history dependence.

The potentially distinctive work here is the frozen use of observed void/host
fields to set a common response, followed by one galaxy-dynamics and
cluster-lensing prediction. It is not the invention of vector gravity,
screening, MOND-like constitutive laws, disformal metrics, or nonlocal gravity.
