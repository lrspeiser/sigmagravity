# Prior-art audit and next-model test registry

Status: prospective model-development record, frozen 2026-07-26 before fitting
any potential-screened, boundary-layer, or void-wall-distance variant.

This is a scientific literature audit, not a patent search or a legal novelty
opinion. We will cite known mechanisms, call the present work a synthesis and
test protocol, and make no claim that a new physical theory exists unless it is
eventually defined by field equations and survives broader expert review.

## Bottom line on precedence

Most ingredients discussed so far have clear prior art:

| Ingredient | Prior-art status | Language permitted here |
|---|---|---|
| A low-acceleration law that approaches an added $1/R$ acceleration and flat outer rotation | Established in MOND/RAR phenomenology | Comparator or phenomenological reference, not our invention |
| The environment changing a galaxy's internal dynamics | Established in MOND external-field-effect work and screened modified gravity | Environmental test, not a new principle |
| Screening controlled by potential depth or density | Established in chameleon and symmetron theories | Generic potential/density-screening benchmark |
| A flattened screening surface and a rotation-curve upturn near a disk's screening radius | Directly studied in chameleon $f(R)$ simulations and SPARC fits | Direct prior art for the "effect appears at the galaxy edge" idea |
| Cosmic voids as relatively unscreened modified-gravity laboratories | Established | Motivation for a test, not a novelty claim |
| Watershed void centers, effective radii, and walls | Established in ZOBOV-derived catalogs | Standard geometry to adopt rather than galaxy-pair midpoints |
| Scalar, vector, or phase-based forces that reproduce galaxy phenomenology | Large existing literature | A broad theory class, not an originality claim |

The possibly distinctive contribution is narrower: a controlled comparison of
acceleration-screened, potential-screened, environment-shifted, and
void-wall-conditioned models under the same nuisance treatment and strict
galaxy-level validation, followed by directional and vertical falsification
tests. That is a test architecture and a particular synthesis. It is not yet a
new fundamental theory.

## Primary sources checked

- McGaugh, Lelli, and Schombert, [The Radial Acceleration Relation in
  Rotationally Supported Galaxies](https://arxiv.org/abs/1609.05917), establishes
  the tight relation between observed and baryonic acceleration in SPARC.
- Cabre et al., [Astrophysical Tests of Modified Gravity: A Screening Map of the
  Nearby Universe](https://arxiv.org/abs/1204.6046), explicitly separates
  self-screening from environmental screening using a 3-D nearby-universe map.
- Hinterbichler and Khoury, [Symmetron Fields: Screening Long-Range Forces
  Through Local Symmetry Restoration](https://arxiv.org/abs/1001.4525), is
  direct prior art for a low-density field becoming coupled while dense regions
  suppress it.
- Naik et al., [Imprints of Chameleon $f(R)$ Gravity on Galaxy Rotation
  Curves](https://arxiv.org/abs/1805.12221), predicts an oblate screening surface
  and rotation-curve/RAR upturns near a disk screening radius.
- Naik et al., [Constraints on Chameleon $f(R)$-Gravity from Galaxy Rotation
  Curves of the SPARC Sample](https://arxiv.org/abs/1905.13330), performs the
  direct SPARC search and reports no convincing evidence for the tested
  $f(R)$ models after halo-profile sensitivity is considered.
- Chae et al., [Testing the Strong Equivalence Principle II](https://arxiv.org/abs/2109.04745),
  relates fitted SPARC external-field effects to independently reconstructed
  large-scale structure. This is especially close prior art for connecting
  rotation curves to cosmic environment.
- Perico et al., [Cosmic voids in modified gravity
  scenarios](https://arxiv.org/abs/1905.12450), treats void profiles as tests of
  $f(R)$ and symmetron screening.
- Sutter et al., [A public void catalog from the SDSS DR7 Galaxy Redshift
  Surveys based on the watershed transform](https://arxiv.org/abs/1207.2524),
  supplies direct precedent for cataloged watershed voids rather than pairwise
  galaxy midpoints.
- Burrage, March, and Naik, [Accurate Computation of the Screening of Scalar
  Fifth Forces in Galaxies](https://arxiv.org/abs/2310.19955), shows why a binary
  screened/unscreened threshold is only a proxy and can overpredict a true
  scalar fifth-force signal.

This list is enough to prevent an accidental broad originality claim, but it is
not exhaustive. Any future paper draft must repeat the search in ADS, INSPIRE,
Crossref, and citation graphs, and should be reviewed by a modified-gravity
specialist.

## What the first experiment did and did not mean

The fitted added acceleration was

$$
g_{\rm add}=A_0e^{\beta\mathcal V}a_t
\left(\frac{g_{\rm bar}}{a_t}\right)^p S_g(g_{\rm bar}).
$$

For an outer point-mass baryonic field, $g_{\rm bar}\propto R^{-2}$, so $p=1/2$
makes $g_{\rm add}\propto1/R$. That $R$ is galactocentric radius. It is not
distance to a void. We never assigned a void an acceleration of
$-9.8\ \mathrm{m\,s^{-2}}$, and we never used a $1/d_{\rm void}$ force law.

The CF4 variable $\mathcal V=-\delta(\mathbf x_g)$ was the interpolated local
density contrast at the galaxy. It contained neither a void center nor a void
wall distance. The negative result therefore rejects the tested local-density
amplitude modulation; it does not test potential-controlled screening,
environment-controlled transition location, or void-wall geometry.

## Why galaxy midpoints are not void sources

A midpoint between two galaxies is only a geometric construction. It can fall
inside a filament, a group, an unobserved galaxy population, or a survey mask.
It is not evidence for a density minimum and will not be used as a void center.

For a catalog void, use all three of the following and keep them distinct:

1. The density minimum or a volume-weighted macrocenter $\mathbf x_v$.
2. The effective radius $R_{\rm eff}=(3V_v/4\pi)^{1/3}$, which summarizes volume
   but does not make an irregular watershed basin spherical.
3. The signed shortest distance $d_{\rm wall}$ to the actual watershed boundary.

The primary wall coordinate will be

$$
D_{\rm wall}=\begin{cases}
\min(d_{\rm wall}/R_{\rm eff},1), & \text{inside a catalog void},\\
0, & \text{otherwise}.
\end{cases}
$$

It is positive deeper inside a void and zero outside. A wall model cannot be fit
until the catalog or the complete frozen watershed algorithm, grid, density
cut, edge treatment, and hashes are committed.

If the source is instead the full density deficit, the distance is not one
center-to-galaxy number. The Newtonian peculiar field is an integral over every
cell:

$$
\mathbf g_\delta(\mathbf x)=G\bar\rho\int d^3x'\,\delta(\mathbf x')
\frac{\mathbf x'-\mathbf x}{|\mathbf x'-\mathbf x|^3}.
$$

The potential kernel is $1/d$ and the acceleration kernel is $1/d^2$. Inside an
ideal spherical top-hat underdensity the acceleration grows approximately
linearly from zero at the void center and is largest near the wall; it is not a
constant Earth-like $9.8\ \mathrm{m\,s^{-2}}$ acceleration.

## The constant-background problem

Making "default spacetime higher" by adding the same constant to a gravitational
potential changes no orbit, because forces depend on gradients. A nearly uniform
external acceleration also moves a galaxy and its contents together. In ordinary
gravity the first internal effect is the tidal tensor

$$
T_{ab}=\partial g_{\delta,a}/\partial x_b,\qquad
\Delta\mathbf g\simeq\mathbf T\,\mathbf r.
$$

At $R\sim10$ kpc, the natural smooth large-scale-structure scale is of order
$H_0^2R\sim10^{-15}\ \mathrm{m\,s^{-2}}$, before density factors, far below the
roughly $10^{-10}\ \mathrm{m\,s^{-2}}$ galactic acceleration scale. A successful
"background" interpretation therefore needs a nonlinear response, such as a
density- or potential-dependent scalar configuration. That idea overlaps
chameleon/symmetron screening and must make additional predictions.

## Frozen next models

All formulas use the same baryonic construction, nuisance priors, cuts, and
galaxy folds as the completed CF4 analysis. Let
$a_\star=1.2\times10^{-10}\ \mathrm{m\,s^{-2}}$ be a fixed normalization. The
machine-readable bounds are in `configs/next_model_registry.json`.

### A0: acceleration-screened reference

This is the already-tested environment-free formula and is retained only as a
reference:

$$
S_g=\left[1+\exp\left(\frac{\log_{10}g_{\rm bar}-\log_{10}a_t}{w_g}\right)\right]^{-1}.
$$

It is closely related to low-acceleration MOND/RAR phenomenology and carries no
originality claim.

### P0: self-potential-screened background

Construct a reproducible positive baryonic potential-depth proxy from the
rotation-curve mass model:

$$
|\Phi_{\rm bar}(R_j)|=\int_{R_j}^{R_{\max}}g_{\rm bar}(R)\,dR
+g_{\rm bar}(R_{\max})R_{\max}.
$$

The last term declares a Keplerian tail beyond the final measured radius. It is
an approximation to be checked on synthetic exponential disks, not a claim that
the galaxy is spherical. Define $\chi=|\Phi_{\rm bar}|/c^2$ and

$$
S_\Phi=\left[1+\exp\left(\frac{\log_{10}\chi-\log_{10}\chi_t}{w_\Phi}\right)\right]^{-1},
$$

$$
g_{\rm pred}=g_{\rm bar}+A_0a_\star
\left(\frac{g_{\rm bar}}{a_\star}\right)^pS_\Phi.
$$

P0 tests whether transition location follows the whole potential depth rather
than only local acceleration. Potential screening itself is prior art; the value
is the matched comparison.

### P1: environment shifts the screening transition

The failed CF4 model changed the force amplitude. P1 instead changes where the
galaxy unscreens:

$$
\log_{10}\chi_{t,i}=\log_{10}\chi_{t,0}+\zeta\mathcal V_i.
$$

$\mathcal V_i$ is the frozen standardized CF4 underdensity. The void-phase sign
prediction is $\zeta>0$: a more underdense environment activates the same outer
law at greater potential depth, hence farther inward. This is also adjacent to
environmental chameleon screening, so the project will describe it as a new test
of a different coupling structure, not a new mechanism.

P1 is evaluated only if P0 is not inferior to A0 and RAR in strict galaxy
cross-validation. The old amplitude-modulated model is retained as a negative
control.

### B1: boundary-layer diagnostic

To distinguish a force that merely exists outside the screened region from one
concentrated at its transition, add one signed global coefficient:

$$
g_{\rm B}=\kappa a_\star\frac{dS_\Phi}{d\ln R},
$$

$$
\frac{dS_\Phi}{d\ln R}=
\frac{S_\Phi(1-S_\Phi)}{w_\Phi\ln10}
\frac{R g_{\rm bar}}{|\Phi_{\rm bar}|}.
$$

The hypothesized inward boundary pressure predicts $\kappa>0$. A negative or
galaxy-dependent value counts against it. B1 is a midplane diagnostic, not a
field theory; it is fit only after P0 and is not combined with P1 unless each
one-parameter extension independently passes validation.

### W1: independently measured void-wall transition

The external catalog is now frozen to Malandrino et al., [A Bayesian catalog of
100 high-significance voids in the Local
Universe](https://arxiv.org/abs/2507.06866), repository commit
`bbbc34594d92eeef32897d67d291d54eb384be6e`. It is based on 50 posterior
large-scale-structure realizations and supplies individual $32^3$ Voronoi-cloud
shape fields, so this project does not invent galaxy midpoints or tune its own
void finder.

The primary boundary is the catalog authors' volume-preserving recommendation:
Voronoi overlap strictly greater than 0.37. SPARC ICRS positions are transformed
to the catalog's Cartesian box using its declared $h=0.681$ and box center
$(340.5,340.5,340.5)\ h^{-1}$ Mpc. If a galaxy lies in more than one cloud,
the primary assignment is the largest trilinearly interpolated overlap, with
catalog index breaking an exact tie. Signed wall distance is measured on that
cloud's thresholded grid with a Euclidean distance transform. One half voxel is
subtracted because the binary interface lies between cell centers; the field is
then trilinearly sampled and clipped at zero for an inside point. Distance is
divided by the published mean effective radius. No
rotation-curve value enters construction.

Replace $\mathcal V_i$ in P1 with $D_{\rm wall,i}$:

$$
\log_{10}\chi_{t,i}=\log_{10}\chi_{t,0}+\zeta_wD_{\rm wall,i}.
$$

The prediction is $\zeta_w>0$. A center-distance version is a declared
sensitivity analysis, not the primary score. Density minimum, macrocenter,
effective-radius, and wall-distance scores will not be tried as an uncorrected
menu.

### T0: physical tide and direction check

Compute $\mathbf g_\delta$ and $\mathbf T$ from the CF4 grid with no fitted force
normalization. Score the relative acceleration across a 10 kpc disk and compare
its sign and magnitude with the required residual. A direct external-field
model must use side-resolved 2-D velocity fields because its leading tidal
response generally contains orientation-dependent and quadrupolar structure
that a folded SPARC rotation curve can hide.

No phenomenological amplification parameter is allowed in T0. If the physical
tide misses by orders of magnitude, that is a failed ordinary-void-gravity
mechanism, not permission to rename an arbitrary multiplier as gravity.

## Ordered validation and stopping rules

1. Commit this registry before running P0, P1, B1, W1, or T0.
2. Verify potential integration and analytic boundary derivatives on synthetic
   point-mass and exponential-like curves. CPU and CUDA must agree.
3. Use SPARC only for development: identical radial splits, strict galaxy folds,
   paired galaxy bootstrap intervals, and the same nuisance treatment.
4. P0 advances only if its held-out-galaxy score is not worse than A0 and is
   competitive with fixed RAR. Prefer the simpler model on a tie.
5. P1 or B1 advances only if its paired 95% interval for change in held-out
   $\chi^2$ per point excludes zero in the improving direction and its predicted
   sign is stable in all folds. They are a two-hypothesis family; apply Holm's
   correction to confirmatory p-values.
6. W1 can run only from the frozen Malandrino catalog commit and boundary rule
   above. T0 is a physical-scale check and cannot be rescued by B1 or W1.
7. Because aggregate SPARC behavior has already informed these variants, final
   confirmation must use a non-overlapping sample. LITTLE THINGS is the first
   candidate because it provides homogeneous resolved dwarf-galaxy curves and
   2-D H I data. Objects overlapping SPARC must be removed. Survey-specific
   baryonic modeling must be validated before combining likelihoods.
8. No claim of a void mechanism is allowed from radial fits alone. It must also
   predict at least one of transition-radius/environment, side-to-side/orientation,
   or vertical-kinematic observables before those data are examined.

## Outcomes that would discriminate the ideas

| Observation | Acceleration screen | Potential screen | Threshold shift | Boundary layer | Direct void tide |
|---|---:|---:|---:|---:|---:|
| Same $g_{\rm bar}$ but different galaxy compactness | Same activation | Different activation | Different if environment also differs | Localized change at potential boundary | No universal prediction |
| Void environment changes overall outer amplitude | Possible only through added amplitude parameter | No | Not primarily; it moves the transition | Only near boundary | Direction and tensor dependent |
| Strong localized slope feature | No requirement | Smooth transition | Shifted smooth transition | Yes | Depends on geometry |
| Side-to-side or orientation dependence | No | No in P0 | No in scalar score P1 | No in the midplane B1 proxy | Yes generically |
| Vertical transition closer than radial transition | No 3-D prediction | Expected for an oblate potential screen | Environment moves it | Boundary signal follows the surface | Tensor dependent |

The last two rows are important: without them, several flexible radial laws can
fit the same one-dimensional rotation curve and cannot identify the physical
source.
