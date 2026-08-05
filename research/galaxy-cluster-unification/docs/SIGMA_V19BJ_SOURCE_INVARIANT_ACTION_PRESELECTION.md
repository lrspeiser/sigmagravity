# Sigma V19BJ source-invariant/action preselection

## Decision frozen before the gas result

V19BJ freezes the question that the completed cluster thermodynamic maps are
allowed to answer.  It does **not** choose a new gravity equation.  After the
V19W4 unified response archive, V19X2 commissioning, all 494 regional fits and
the common-grid gas posterior pass, the source-only Bullet and Abell 2146 data
may test a small registered library of baryonic state variables.  Lensing,
halo maps and gravity residuals remain sealed during that selection.

The reason is simple.  Total baryonic density tells us where ordinary matter
is, but the same density can occur in a settled disk, a relaxed cluster or two
components crossing during a merger.  If Sigma Gravity is genuinely sensitive
to assembly state, that state has to be represented by a measurable covariant
quantity.  It cannot be the word "cluster," a fitted merger direction or a
switch inserted after seeing a lensing map.

## The declared baryon frame

Let the total baryonic rest-mass current be \(J_b^\mu\).  Where it is timelike,
define

\[
u^\mu={J_b^\mu\over\sqrt{-J_b^\nu J^b_\nu}},
\qquad
h_{\mu\nu}=g_{\mu\nu}+u_\mu u_\nu.
\]

This fixes the frame before a target is inspected.  In that frame, baryonic
stress-energy can be decomposed into density, pressure, energy or species
drift and trace-free stress.  The observed X-ray and member-galaxy maps are
only projected, noisy proxies for this four-dimensional state.  Projection,
astrometry and resolution therefore remain part of the posterior rather than
being hidden by a single best-looking map.

## Registered source features

| ID | Physical question | Kind | Eligible after gas? |
|---|---|---|---:|
| D0 | Does total density alone suffice? | scalar control | no |
| I1 | Does gas/star component overlap add information beyond total density? | scalar | yes |
| I2 | Does relative gas--collisionless motion supply an activation and direction? | vector plus norm | yes |
| I3 | Does anisotropic baryonic stress supply a transferable eigenframe? | trace-free tensor plus norm | yes |
| I4 | Do density and entropy gradients encode shock/assembly geometry? | trace-free gradient tensor | yes |
| I5 | Is pressure--density gradient misalignment identifiable in 3D? | pseudovector plus norm | yes |
| I6 | Is a causal material relaxation rate required? | future time-dependent control | no with present snapshots |

D0 is retained as a negative control.  A density-only constitutive rule falls
back into already published or closed AQUAL, QUMOND, refracted-gravity or
linear filtered-source territory.  I6 is scientifically useful but cannot be
claimed from one projected thermodynamic snapshot.

## Concrete pass/fail outcome

Each eligible feature must use the identical dimensionless definition in both
clusters.  It must be detected at at least 3 sigma, survive at least 90% of
allowed projection draws and 90% of leave-one-region-out trials, change by no
more than 10% in normalized amplitude and 10 degrees in orientation under the
resolution audit, and retain at least 20% cross-validated spatial variance
that total density alone cannot predict.  A directional feature must have a
95% axial-orientation width no larger than 30 degrees.

Advancement requires at least one scalar activation and one vector or tensor
direction to pass every gate in both clusters.  The same object, such as a
relative-current vector and its scalar norm, may provide both.  Equal
amplitudes are not required: the test asks whether one definition transfers,
not whether different mergers are identical.

If no feature passes, there is no authorized action to invent.  The next
measurement is a direct gas-velocity constraint or an independent merger
sample.  Thresholds cannot be relaxed, failed features cannot be combined
after inspection, and lensing cannot rescue an unidentifiable source.

## Manufactured-map algebra

The reusable projected-map implementation is in
`src/voidscreen/sigma_source_invariants.py`.  Six target-free tests establish
the algebra before any gas result is read:

- component overlap reaches zero for a pure component, one for equal
  components and is unchanged when gas and stars are exchanged;
- relative current is unchanged by a common velocity boost and rotates as a
  vector while its norm is rotation invariant;
- anisotropic stress is symmetrized, trace-free and rotation covariant;
- a manufactured density/entropy gradient recovers its known axial direction;
- normalized baroclinicity is zero for parallel gradients and unity for
  orthogonal gradients; and
- negative densities, zero normalization speeds and isotropic tensors without
  an axis fail closed.

These checks validate coordinate behavior and limiting cases only.  They do
not establish that an invariant is measurable in the clusters or that it
sources gravity.

## What the result can choose

V19BJ registers three action-placement classes, not three formulas:

1. A constrained composite response with no free halo-shaped initial state.
2. A causal dynamical response that carries energy and is forced by baryons.
3. A degenerate pure-metric nonlinear vertex whose GR quadratic propagator is
   unchanged.

A time-odd current or independently clocked lag would favor a genuinely
dynamical placement.  A time-even local tensor may be representable by a
lower-field-content constrained or pure-metric placement.  Mathematical
health—not lensing fit—then chooses between compatible placements: the full
action must conserve total stress-energy, have the correct degree-of-freedom
count, a bounded Hamiltonian, causal characteristics and one physical metric
for matter and light.

Closed Proca, aether, massive-spin-two, material-memory, linear Yukawa and
density-switch mechanisms cannot be revived under a new symbol.  After three
materially different action derivations fail the same gate, the mechanism is
reconsidered instead of gaining another parameter.

## Consequences beyond rotation curves and strong lensing

The source choice has immediate qualitative consequences that will later
become quantitative predictions:

- Weak-lensing shear and merger offsets must follow the same source tensor and
  Weyl potential as strong-lensing image roots.
- Component overlap or relative current predicts an environmental dependence
  for satellites; a density-only source does not provide that information.
- A smooth long mode can change stream precession but cannot imitate compact
  subhalos unless the derived field equations form stable compact structure.
- Replacing dark-matter dynamical friction requires a dynamical Sigma field
  with derived energy transfer; a static response cannot claim it.
- Growth and CMB effects remain unknown until a covariant action has a
  background and perturbation theory.

Solar and PPN constraints remain a later hard veto.  They are deliberately not
an optimization target in this galaxy/cluster source-selection stage.

## Reproduction

```powershell
python scripts/check_sigma_v19bj_source_invariant_action_preselection.py
python -m pytest tests/test_sigma_v19bj_source_invariant_action_preselection.py -q
```
