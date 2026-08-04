# Sigma v11B stress-free elastic-spacetime triad selection

> **Superseded gate result (2026-08-04):** on a tilted slice the strain square
> becomes quartic in a physical material velocity. Its Legendre Hessian crosses
> zero while material flow remains timelike. Exact v11B is retired before data.
> See [`SIGMA_V11B_TILTED_RANK_FALSIFICATION.md`](SIGMA_V11B_TILTED_RANK_FALSIFICATION.md).

## Decision

Sigma v11B passes a narrow theory-only architecture-selection gate. It models
the Sigma sector as three scalar “material coordinates” whose departures from
an internal Euclidean reference metric describe gravitational strain.

The construction has five useful properties at the selected background:

1. its unstrained vacuum has zero action, zero first variation, and therefore
   zero effective stress;
2. three positive phonon modes propagate—two shear modes with squared speed
   `3/11` and one longitudinal mode with squared speed `3/4`;
3. all three modes remain inside the physical metric cone in every direction;
4. scalar derivatives contain no Christoffel connection, avoiding v10D's
   anisotropic tensor-cone failure;
5. the kinetic and strain coefficients are constant rather than functions of
   another field's velocity, avoiding v11A's concavity failure.

Two thousand random propagation directions reproduce the exact spectrum to
`8.88e-16`. Only one new physical length, `L_Sigma`, is introduced, retaining
the five-constant limit.

This is not yet a viable theory. Nonlinear tilted rank, the complete metric
constraint algebra, graviton mass, weak `Psi/Phi`, Solar/PPN behavior,
cosmology, source uniqueness, and observational adequacy remain kill gates. No
astronomical product or holdout was opened.

## Physical idea

Ordinary elasticity separates two notions:

- coordinates used to label points in space;
- material labels attached to the medium.

If the geometry is unstrained, those labels can agree everywhere. Curvature
makes that impossible globally: no single Euclidean material map can match a
spatial metric with nonzero Riemann curvature at every point. The mismatch is
strain.

V11B tests whether the gravitational response attributed to a halo could
instead be the strain and relaxation of such a spacetime medium. The medium is
not supplied with an invisible density profile. Its reference state is
universal, and baryons affect it only by curving the one physical metric.

## Fields and symmetries

Retain the AeST metric `g_mn` and unit timelike aether `A^m`. Introduce three
spacetime scalars

$$
X^I(x),\qquad I=1,2,3,
$$

with an internal Euclidean metric `delta_IJ`. Internal translations and
rotations,

$$
X^I\rightarrow R^I{}_JX^J+c^I,
$$

do not change observables.

Define

$$
q^{\mu\nu}=g^{\mu\nu}+A^\mu A^\nu,
$$

$$
Q^I=A^\mu\nabla_\mu X^I,
$$

and the internal strain

$$
\boxed{
E^{IJ}=q^{\mu\nu}\nabla_\mu X^I\nabla_\nu X^J-\delta^{IJ}.
}
$$

Because each `X^I` is a scalar,

$$
\nabla_\mu X^I=\partial_\mu X^I.
$$

There is no connection term multiplying a nonzero carrier background.

## Selected action

Let

$$
E_{\rm TF}^{IJ}=E^{IJ}-{1\over3}\delta^{IJ}\operatorname{tr}E.
$$

The selected addition is

$$
\boxed{
\Delta\mathcal L_{11B}
={M_P^2\over L_\Sigma^2}
\left[
{1\over2}Q_IQ^I
-{s\over4}
\left(E_{\rm TF}:E_{\rm TF}+b(\operatorname{tr}E)^2\right)
\right].
}
$$

The aether supplies the preferred time derivative. The strain potential is a
sum of positive squares. The overall length `L_Sigma` controls the rigidity
relative to the Einstein-Hilbert metric term.

The flat reference state is

$$
g_{\mu\nu}=\eta_{\mu\nu},
\qquad A^\mu=(1,0,0,0),
\qquad X^I=x^I.
$$

It has

$$
Q^I=0,\qquad E^{IJ}=0.
$$

Thus both the new Lagrangian and its first variation vanish. Unlike an
ordinary covariant solid `F(B^IJ)` with nonzero first derivative, the healthy
time kinetic is supplied separately by the aether; the vacuum need not carry
a material energy density or pressure.

## Flat phonon spectrum

Write

$$
X^I=x^I+\pi^I.
$$

At linear order,

$$
Q^I=\dot\pi^I,
\qquad
E^{IJ}=\partial^I\pi^J+\partial^J\pi^I.
$$

For a transverse plane wave, `n dot pi=0`, the strain is trace free and gives

$$
c_{\rm shear}^2=s.
$$

For a longitudinal wave, `pi parallel n`,

$$
c_{\rm longitudinal}^2=s\left({4\over3}+2b\right).
$$

Reuse the already derived causal interior speeds

$$
c_{\rm shear}^2={3\over11},
\qquad
c_{\rm longitudinal}^2={3\over4}.
$$

They fix, rather than fit,

$$
\boxed{
b={1\over2}\left[{(3/4)\over(3/11)}-{4\over3}\right]
={17\over24}.
}
$$

The spectrum is therefore

$$
\boxed{
\left\{{3\over11},{3\over11},{3\over4}\right\}
}
$$

for every propagation direction. All kinetic coefficients and spatial
stiffnesses are positive, and all fronts are subluminal relative to the matter
metric.

## Metric tensor cone

The new action depends algebraically on the metric through `q^mn` but contains
no derivative of the metric. Consequently it cannot add a second-derivative
term to the TT metric equation. The high-frequency gravitational-wave front
remains the Einstein-Hilbert null cone:

$$
c_T=1.
$$

On the background `X^I=x^I`, a TT metric perturbation does enter `E^IJ` and
therefore receives an algebraic restoring term. This is a possible graviton
mass, not a changed front speed. It must satisfy gravitational-wave and Solar
bounds at a later gate; it is not being ignored.

## Why it can carry cluster geometry

The strain decomposes naturally into

- `tr(E)`, a volume/trace response that can contribute broad convergence;
- `E_TF`, a five-component shear response with principal axes;
- compatibility constraints tying the strain to one global material map.

Those compatibility constraints make separated baryonic components interact
nonlocally. Locally one can always choose coordinates that reduce a metric
perturbation, but one cannot globally transform away curvature. This is a
possible first-principles origin for the empirical clue

$$
\mathcal N\!\left(\sum_i\rho_i\right)
\ne\sum_i\mathcal N(\rho_i).
$$

Whether the resulting weak metric has the required sign and amplitude remains
unknown. A geometrically appealing strain variable is not sufficient; the
complete `Psi`, `Phi`, and Weyl equations must demonstrate it.

## State uniqueness

The universal asymptotic condition is an unstrained material frame,

$$
X^I\rightarrow x^I
$$

in the cosmological/local inertial reference state, modulo one global internal
translation and rotation. Those rigid transformations do not change `Q` or
`E` and are internal redundancies, not per-object halo parameters.

Static localized solutions use the decaying elastic displacement. Dynamical
solutions use retarded/no-incoming phonons. The nonlinear equations must still
prove that no regular source-free strained lump survives these rules.

## Constant count

The provisional physical constants are

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,L_\Sigma\}.
$$

The two phonon speeds and `b=17/24` are derived coefficients. No object center,
orientation, modulus, strain amplitude, or lensing coefficient may be fitted.

## Prior-art boundary

Three scalar material coordinates and internal Euclidean symmetry are
established relativistic-solid and spatial-condensation ideas. Relevant
primary references include
[Endlich, Nicolis, and Wang](https://arxiv.org/abs/1210.0569),
[Lin](https://arxiv.org/abs/1305.2069), and the gravitating-continuum EFT of
[Aoki et al.](https://arxiv.org/abs/2204.06672). Relativistic hyperelasticity
has a substantially older and broader formulation; see
[Beig and Schmidt](https://arxiv.org/abs/gr-qc/0211054).

V11B's project-specific hypothesis is the stress-free aether-time/strain-square
combination as a baryon-curvature response constrained by the existing Sigma
parameter budget. No novelty claim is made before an operator-level comparison
and survival of the nonlinear gates.

## Remaining kill gates

V11B may advance only through:

1. complete covariant variation, Hilbert stress, and Noether identity;
2. nonlinear tilted-aether Legendre rank for arbitrary finite strain and
   material velocity;
3. the complete metric-aether-AeST-triad constraint count and characteristics;
4. weak static equations for `Psi`, `Phi`, strain, and the Weyl potential;
5. proof that the response is not pure gauge and has the needed attractive
   sign;
6. graviton-mass, PPN, Mercury, Solar, FLRW, and cosmological-background tests;
7. a convergent 3D solver, followed only then by frozen galaxy and raw-cluster
   evidence splits.

Failure at any stage retires the exact action before data.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v11b_elastic_triad.py
python -m pytest -q tests/test_sigma_v11b_elastic_triad.py
```

Machine-readable evidence is in
`results/sigma_v11b_elastic_triad_selection/report.json`.
