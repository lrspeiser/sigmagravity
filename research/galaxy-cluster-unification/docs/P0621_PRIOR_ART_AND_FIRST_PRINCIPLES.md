# P0621: prior-art audit and first-principles explanation

## Bottom line

The current P0620 construction does **not** appear to be a renamed copy of one
published equation. A targeted comparison found no paper with this exact
combination:

\[
\boldsymbol\alpha_{\rm test}(\mathbf x)
=\boldsymbol\alpha_{0554}(r)
+{Q^2\over1+\Delta_{80}}
\,\mathcal R_{90}
\!\left[\delta\boldsymbol\alpha_{\rm route}(\mathbf x)\right].
\]

Its exact engineering recipe—P0554 radial enhancement, a self-routed fraction,
an `R80`-scaled endpoint kernel, a squared baryonic quadrupole amplitude, a
shared 90-degree template phase, annular monopole removal, and a curl-free
deflection reconstruction—may be original as a **phenomenological ansatz**.

The broader ideas are not new:

- an apparent or “phantom” mass distribution determined entirely by baryons is
  central to AQUAL and QUMOND;
- environmental redirection of gravitational field lines is the central picture
  of refracted gravity;
- potential-dependent enhancement on cluster scales is present in EMOND;
- displaced or unusually shaped apparent halos can arise in MOND/QUMOND;
- a quadrupolar lens correction with a selected phase can resemble ordinary
  ellipticity or external shear in conventional lens modeling; and
- treating the lensing response differently from the matter response is a
  gravitational-slip ansatz unless it is derived from a relativistic metric.

The honest claim is therefore not “a new theory of gravity.” It is:

> We tested a particular baryon-derived, conservative redistribution of an
> extra gravitational response. The exact construction was not found in the
> targeted literature, but it synthesizes several established modified-gravity
> ideas. Its angular response contains a potentially useful empirical clue,
> while its absolute cluster accuracy is not yet competitive.

This audit is a scientific prior-art search, not an exhaustive legal or patent
novelty opinion.

## What the current model actually is

### 1. The radial parent, P0554

For baryonic acceleration `g_b`, baryonic potential `Phi_b`, and radius `r`,
define

\[
\chi={\Phi_b\over c^2},\qquad
\ell_\Phi={\Phi_b\over g_b},\qquad
R_M=\sqrt{{G M_b\over a_0}},\qquad x={r\over R_M}.
\]

P0554 uses

\[
{g_{\rm dyn}\over g_b}
=1+q
\underbrace{{x^{0.75}\over1+(x/100)^{0.75}}}_{\text{accumulated path response}}
\underbrace{{1\over1+g_b/a_0}}_{\text{high-acceleration screen}}
\underbrace{\left({\ell_\Phi\over r}\right)^{0.25}}_{\text{extended-profile response}}
\underbrace{\left[1+\left({\chi\over2\times10^{-6}}\right)^{1.2}\right]}_{\text{potential-depth response}},
\]

with the universal fitted value `q = 1.2300686`. The diagnostic lensing response
is

\[
{g_{\rm lens}\over g_b}=1+1.75\left({g_{\rm dyn}\over g_b}-1\right).
\]

This is an empirical radial response law. The factor `1.75` is not yet derived
from a metric or from photon geodesics.

### 2. The angular route layer

At the radius containing 80 percent of the projected baryonic proxy weight,
define

\[
\Delta_{80}={\alpha_{0554}(R_{80})\over\alpha_b(R_{80})}-1,
\qquad
f_{\rm self}={\Delta_{80}\over1+\Delta_{80}}.
\]

The centroided projected baryonic quadrupole is

\[
Q=\sqrt{Q_{xx}^2+Q_{xy}^2},
\]

where

\[
Q_{xx}={\sum_i w_i(\Delta x_i^2-\Delta y_i^2)
\over\sum_i w_i(\Delta x_i^2+\Delta y_i^2)},\qquad
Q_{xy}={\sum_i w_i(2\Delta x_i\Delta y_i)
\over\sum_i w_i(\Delta x_i^2+\Delta y_i^2)}.
\]

Thus `Q = 0` for a rotationally symmetric projected source and approaches one
for a very elongated source. Squaring it makes the amplitude insensitive to
which end of an axis is labeled positive and makes the response fall rapidly
toward zero as symmetry is restored.

The code then performs the following operations:

1. Normalize the projected baryonic member-light proxy.
2. Move a fraction `f_self` of its template weight inward by `0.36 R80` and
   smooth it with width `0.23 R80 sqrt(1 + Q^2)`.
3. Rotate that **positive template map**, not the force vectors, by a shared
   90 degrees and renormalize it.
4. Use the P0554-minus-baryon radial excess as the carrier magnitude.
5. Remove the mean added convergence separately in every annulus.
6. Solve for a potential-derived, numerically curl-free deflection field.
7. Add that field with amplitude `Q^2/(1 + Delta80)`.

That implementation detail matters. The model does not literally turn every
gravity vector by 90 degrees. A literal rotation of a gradient field would
generally introduce curl and would not represent gravity derived from a scalar
potential. Our implementation rotates a source-like template first and derives
a conservative deflection afterward.

## Comparison with published ideas

| Published framework | Shared content | Important difference from P0620 |
|---|---|---|
| AQUAL / nonlinear Poisson MOND | The baryonic field enters a nonlinear field equation; an observer using Newtonian gravity can infer effective extra density. | AQUAL starts from a local field equation and action. P0620 starts from a fitted radial response plus a map-processing operator. |
| QUMOND | A baryon-derived modified source can be rewritten as “phantom” density, including geometry-dependent structure. | QUMOND's apparent density follows from one specified differential equation and boundary conditions. P0620 prescribes endpoint displacement, smoothing, phase, and annular subtraction. |
| Refracted gravity | Matter density changes an effective gravitational permittivity and redirects field lines; this can strengthen disk-plane or low-density-regime gravity. | Refracted-gravity directions follow locally from gradients of the permittivity. P0620 uses global quantities such as `R80`, an explicit route kernel, and a fitted shared phase. |
| Covariant refracted gravity | Supplies a scalar-tensor action whose weak-field limit resembles gravitational permittivity. | P0620 has no covariant action, stress-energy tensor, propagation law, or cosmological solution. |
| EMOND | Lets the MOND acceleration scale depend on potential depth, specifically to address clusters. | P0554's potential factor is conceptually adjacent, although its functional form and additional path factor differ. Potential dependence itself is not new. |
| TeVeS and other relativistic MOND completions | Show how modified galaxy dynamics and lensing can arise from a metric plus extra fields. | P0554 directly multiplies the new lensing channel by 1.75. It does not yet predict that ratio from two metric potentials. |
| Gravitational polarization / dipolar dark matter | Baryons can induce a polarized response whose effective density mimics MOND and need not trace baryons point by point. | This introduces a dynamical medium and a relativistic action. P0620 introduces no physical carrier or independent field dynamics. |
| MOND phantom-density offsets and QUMOND apparent halos | Apparent lensing peaks, negative-density regions, and halo asymmetries can be displaced from the baryonic distribution under geometry or an external field. | The general claim that apparent dark structure can be a transformed baryonic field is already published. P0620's particular transformation is different and has not yet reproduced those effects from a field equation. |
| Standard lens ellipticity and external shear | A quadrupolar correction with an adjustable orientation changes image positions and caustics. | A small gain from the 90-degree route phase can be ordinary missing lens geometry or line-of-sight structure. It is evidence for new gravity only if it beats matched conventional shear/multipole controls. |

### Closest mathematical precedent: QUMOND

QUMOND writes the modified potential in the schematic form

\[
\nabla^2\Phi
=\nabla\!\cdot\!\left[
\nu\!\left({|\nabla\Phi_N|\over a_0}\right)\nabla\Phi_N
\right].
\]

If the right-hand side is divided by `4 pi G`, it looks like ordinary baryons
plus a geometry-dependent “phantom” source. This is very close to the desired
statement that what is called a dark-matter map might instead show where the
baryonic gravitational field has been transformed.

P0620 is not QUMOND algebraically. Its source transformation depends on `R80`,
`Q`, fixed endpoint motion, smoothing, an imposed 90-degree phase, and
annulus-by-annulus zero-monopole removal. But until those operations are derived
from a deeper equation, QUMOND is the clearest proof that the broad idea is
already known and can be formulated more rigorously.

### Closest physical picture: refracted gravity

Refracted gravity modifies Poisson's equation schematically as

\[
\nabla\!\cdot\!\left[\epsilon(\rho_b)\nabla\Phi\right]
=4\pi G\rho_b.
\]

Just as an electric field changes direction across materials with different
permittivity, a gravitational field can change direction where the density and
therefore `epsilon` changes. This is the closest published version of the phrase
“baryonic gravity is redirected through other locations.”

The decisive difference is locality. A field equation tells every point what
to do from the fields and derivatives at that point. Our route operator knows a
cluster-wide radius, displaces template weight by a fixed fraction of that
radius, and rotates the completed map. That is a useful experimental recipe,
but not yet an underlying law of nature.

### The 90-degree result is especially vulnerable to rediscovery

A projected quadrupole is a spin-2 pattern: its physical axis is unchanged by a
180-degree turn. For an ideal quadrupole, a 90-degree template rotation largely
reverses the sign of the quadrupolar pattern. Conventional lens models already
use ellipticity, multipoles, and external shear with different amplitudes and
orientations.

Therefore the 90-degree result may be telling us something real about an
orthogonal baryonic response, or it may simply be a new parameterization of a
missing quadrupole. The present data do not distinguish those interpretations.
The required discriminator is a conventional control with the same flexibility
and a phase predicted before the lensing positions are scored.

## A first-principles explanation without the mythology

Imagine first that all we know is Newton's rule. Put down the ordinary matter,
compute its gravitational potential, and take the slope of that potential. The
slope is the acceleration. In general relativity the bookkeeping is richer—mass
and energy determine spacetime geometry, matter follows timelike geodesics, and
light follows null geodesics—but in a quiet, weak field the same potential-like
idea remains visible.

Now consider what an observation actually provides. We do not see a force field
directly. We see stars moving and images bending. We ask, “What distribution of
ordinary Newtonian mass would have made that happen?” The answer to that inverse
question is an *effective* density. It does not by itself tell us whether the
source was unseen matter or a different rule connecting visible matter to the
field.

That is the opening used by this project.

### The radial question: how much extra response is there?

The P0554 parent says that the local baryonic acceleration is not the whole
story. A weak field activates an additional response; a strong field suppresses
it. The response also depends on how much potential has accumulated along the
profile and on whether the mass is concentrated or extended.

This gives three useful limits:

- In the Solar System, `g_b` is large, so the extra term is screened.
- Farther out in a galaxy, `g_b` is small, so the extra term can matter.
- In a deep cluster potential, the potential-dependent factor can increase the
  response beyond what an acceleration-only rule would predict.

This is why P0554 can bridge more of the galaxy/cluster amplitude gap than a
plain radial MOND-like interpolation. It is also why it resembles EMOND and
other potential-dependent modified-gravity ideas.

### The angular question: where does that extra response appear on the sky?

A radial law knows only distance from a chosen center. A real cluster is not a
ball. It has a central galaxy, satellites, hot gas, infalling groups, and
foreground and background structure. Two clusters can require a similar total
deflection while requiring it in different directions.

The route layer asks whether the *shape* of the baryonic distribution can place
the extra response. `Q` is a compact measure of how far the map is from circular
symmetry. If the map is circular, it supplies no preferred axis and the angular
term vanishes. If it is elongated, the term grows. `R80` supplies a measured
system size so that the same dimensionless constants can be used on clusters of
different angular scale.

The annular subtraction then imposes a strict restriction: within every radius,
the angular layer merely moves the effective convergence from one direction to
another. It does not add net radial convergence there. This protects the scalar
parent from being secretly refitted by the angular term and helps keep the
deflection conservative.

That restriction also explains the results. Moving a little convergence around
can move caustics and rescue or destroy individual image roots. It can improve
the *shape* of a lens prediction. But it cannot supply the missing total
convergence. P0554 must already get the radial amount nearly right. It currently
does not.

### Why the present result is suggestive but weak

The shared 90-degree phase improved the mean fixed-geometry score on five
clusters by 1.685 percent, preserved all 18 tested roots, and gave RX J2129 an
8.241 percent improvement. Under a chronologically frozen full refit it changed
A383 from a small radial loss to a small gain.

But the full A383 improvement was only 0.174 percent and its error remained
9.081 arcseconds. On the broader raw validation, the model's 19.076-arcsecond
RMS was 1.91 times the 9.989-arcsecond compact-halo result. Only three of five
fixed-geometry systems improved. In galaxies the route layer is a defined
axisymmetric null, leaving P0554 at 12.592 km/s outer RMSE, 21.7 percent worse
than fixed RAR.

So the useful observation is narrow:

> Angular phase matters more than another small strength adjustment, and a
> baryon-derived angular correction can move exact lens roots without fitting
> a gravity parameter to each cluster.

The data do not yet show that the 90-degree rule is universal or physical.

## What a genuine first-principles completion would require

The missing step is not another coefficient. It is an equation that explains
why the coefficients and operators exist.

A viable completion should start from an action or covariant field equations,
for example a metric plus a new scalar, vector, tensor, or polarization field.
Its weak-field solution would then have to produce, rather than assume:

1. the low-acceleration screen;
2. the dependence on potential depth and extended mass profiles;
3. a finite nonlocal or propagation length tied to baryonic size;
4. a conservative anisotropic response tied to a measured baryonic tensor;
5. the observed matter-versus-light response; and
6. the Solar-System null without a special rule for each domain.

In a relativistic weak-field metric,

\[
ds^2\simeq -(1+2\Phi/c^2)c^2dt^2
+(1-2\Psi/c^2)d\mathbf x^2,
\]

slow matter is primarily sensitive to `Phi`, while lensing is sensitive to
`Phi + Psi`. P0554's photon multiplier should eventually become a prediction
for the relation between these metric potentials, not an independent constant.
That completion must also conserve energy and momentum, remain stable, avoid
unphysical negative-energy modes, propagate causally, and satisfy gravitational
wave and post-Newtonian constraints.

## The concrete work needed next

### Highest priority: test whether the angular clue is just ordinary shear

For each cluster, compare on identical raw image data and with matched parameter
count:

- P0554 alone;
- P0554 plus the frozen route rule;
- P0554 plus standard external shear;
- P0554 plus a generic zero-monopole quadrupole; and
- a conventional baryons-plus-halo control.

Use the same source-position treatment, covariance, optimization starts, and
holdout rule. If ordinary shear gives the same or better gain, the route phase
has not identified new gravity.

### Replace the shared 90-degree phase with a baryon-only prediction

Before viewing raw lens residuals, predict the direction from one independently
measured quantity:

- the hot-gas versus stellar centroid offset;
- the tidal axis from neighboring structure;
- the resolved stellar-plus-gas quadrupole; or
- the principal axis of the baryonic potential Hessian.

The test succeeds only if that frozen direction improves complete-root,
full-refit performance on new clusters. Choosing among phases after seeing the
lens result is discovery work, not confirmation.

### Use the full three-dimensional baryon budget

The current cluster angular proxy is dominated by HST member light. Clusters
also contain extended X-ray gas, intracluster light, the brightest cluster
galaxy, and line-of-sight structure. Any claim that baryons route gravity must
use those components, with uncertainties, because they define both the proposed
source and the conventional explanation.

### Repair absolute convergence

The angular term has zero annular monopole by construction, so it cannot close
the remaining radial mass gap. The scalar sector must approach halo-level raw
lens accuracy without per-cluster gravity tuning. It must simultaneously bring
the galaxy result to at least RAR/MOND-level accuracy. Until both occur, a
better caustic shape is not a unified explanation.

### Replace Solar proxies with a full relativistic calculation

The current symmetry and high-acceleration limits are encouraging, but the
final theory must predict the complete post-Newtonian parameters, planetary
ephemerides, light deflection, Shapiro delay, binary dynamics, and gravitational
wave propagation from the same action.

### Broaden beyond static galaxies and relaxed clusters

A defensible alternative to dark matter must eventually address weak lensing,
merging clusters, structure growth, the cosmic microwave background,
nucleosynthesis, and cosmological expansion. These are not optional polish;
they test whether the new field carries consistent energy and evolves through
time.

## Recommended terminology and claim boundary

For internal work, “gravity routing” remains a useful intuition. For external
scientific writing, use:

> **baryon-sourced conservative anisotropic effective-density ansatz**

Do not yet call it a field theory, a covariant theory, a solution to dark
matter, or evidence that gravity literally travels along the constructed
routes. A suitable present-tense claim is:

> A universal baryon-derived zero-monopole angular perturbation produced small,
> root-safe improvements in several raw cluster-lensing diagnostics, but did not
> reach conventional halo accuracy and did not improve galaxy rotation curves.

## Primary literature used in this comparison

- Bekenstein and Milgrom, [Does the missing mass problem signal the breakdown of
  Newtonian gravity?](https://ui.adsabs.harvard.edu/abs/1984ApJ...286....7B/abstract)
- Milgrom, [Quasi-linear formulation of MOND](https://arxiv.org/abs/0911.5464)
- Matsakos and Diaferio, [Dynamics of galaxies and clusters in refracted
  gravity](https://arxiv.org/abs/1603.04943)
- Sanna, Matsakos, and Diaferio, [Covariant formulation of refracted
  gravity](https://arxiv.org/abs/2109.11217)
- Hodson and Zhao, [Generalizing MOND to explain the missing mass in galaxy
  clusters](https://arxiv.org/abs/1701.03369)
- Bekenstein, [Relativistic gravitation theory for the MOND
  paradigm](https://arxiv.org/abs/astro-ph/0403694)
- Blanchet and Le Tiec, [Model of dark matter and dark energy based on
  gravitational polarization](https://arxiv.org/abs/0804.3518)
- Knebe et al., [On the separation between baryonic and dark matter: evidence
  for phantom dark matter?](https://arxiv.org/abs/0908.3480)
- Bilek, [Peculiar dark matter halos inferred from gravitational lensing as a
  manifestation of modified gravity](https://arxiv.org/abs/2408.02725)
- Famaey, Pizzuti, and Saltas, [On the nature of the missing mass of galaxy
  clusters in MOND: the view from gravitational
  lensing](https://arxiv.org/abs/2410.02612)
- Keeton, Kochanek, and Seljak, [Shear and ellipticity in gravitational
  lenses](https://arxiv.org/abs/astro-ph/9610163)
- Witt and Mao, [Probing the structure of lensing galaxies with quadruple
  lenses: the effect of external shear](https://arxiv.org/abs/astro-ph/9702021)

The search was completed on 2026-08-01 using title, abstract, equation, and
concept comparisons. No exact match to the complete P0620 operator was found.
