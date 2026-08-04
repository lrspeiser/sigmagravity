# Sigma halo-scale identifiability

## The question this memo makes testable

A dark-matter fit usually receives a halo amplitude, scale radius, centroid,
ellipticity, and sometimes subhalos as object-level parameters. Sigma Gravity
cannot take those quantities as inputs. It must predict the corresponding
field strength and spatial extent from measured baryons and no more than five
universal constants.

The immediate question is therefore not merely whether a baryon-only field has
enough amplitude. It is:

> Which measured baryonic invariants determine the size, concentration,
> centroid, and orientation of the field normally represented by a dark halo?

No result in this memo is called a dark-halo measurement. The v16/v17 spent
targets are conventional model reconstructions used for inverse diagnostics;
raw multiple-image positions remain the required observational test.

## Source scale versus propagation scale

Any candidate must distinguish two conceptually different origins of an
apparent halo size.

1. **Source-derived scale.** The baryonic source itself has an object-dependent
   extent and geometry. Examples are gas pressure, collisionless velocity
   dispersion, separated mass components, and their gradients.
2. **Universal propagation scale.** A field equation can redistribute that
   source over one universal correlation length $L_\Sigma$, or through a
   scale-free nonlinear response set by a universal acceleration
   $a_\Sigma$.

For a normalized positive source and a normalized isotropic convolution
kernel, second moments add schematically:

\[
R_{\rm field}^2\simeq R_{\rm source}^2+R_K^2.
\]

This immediately exposes a failure mode. If $R_K=L_\Sigma$ dominates every
object, the theory predicts nearly the same halo-like size everywhere. A
universal constant is allowed, but a universal *output size* is unlikely to
explain dwarfs, giant disks, groups, and clusters. Most of the variation must
therefore emerge from a measured source or a scale-free transition.

The familiar acceleration transition illustrates the second possibility. In
spherical symmetry,

\[
{G M_b(<r_t)\over r_t^2}\sim a_\Sigma
\quad\Longrightarrow\quad
r_t\sim\sqrt{{G M_b\over a_\Sigma}}.
\]

One universal acceleration then generates different radii for different
baryonic masses. This is MOND-like prior art and is not a Sigma novelty. It
also controls a radial monopole more naturally than the nonradial Hessian
structure required by cluster lensing.

By contrast, a linear finite-range equation,

\[
(\nabla^2-L_\Sigma^{-2})\sigma=-\beta_\Sigma J_b,
\]

has an asymptotic range fixed mainly by $L_\Sigma$. It can smooth a measured
source but cannot explain a broad population of halo radii by itself. Allowing
a separately fitted $L_\Sigma$ for each object would simply rename the halo
scale radius and is prohibited.

## Baryonic quantities and the halo properties they could determine

| Measured baryonic information | Possible field consequence | What it cannot determine alone |
|---|---|---|
| Enclosed baryonic mass $M_b(<r)$ | acceleration-transition radius and monopole strength | cluster shear orientation and multiple separated peaks |
| Surface/volume-density profile | concentration and local screening | causal history or velocity dispersion |
| Density gradients and Hessians | local orientation and sharpness | v15 showed static versions do not transfer sufficiently |
| Gas temperature and pressure | random-stress source extent and shock/merger structure | collisionless member stress and line-of-sight depth |
| Member velocity dispersion | multistream/collisionless random stress | unavailable for a matched two-cluster test today |
| Component separation and overlap | centroid shifts, ellipticity, and potential image multiplicity | required response amplitude without a field equation |
| Universal $a_\Sigma$ | mass-dependent transition scale | nonradial topology by itself |
| Universal $L_\Sigma$ | propagation/smoothing width | population-dependent size if it dominates the source |
| Universal coupling $\beta_\Sigma$ | amplitude | shape or scale; amplitude fitting cannot repair topology |

This table is a causal hypothesis inventory, not a license to insert every
quantity into a regression. A final equation must select its invariants through
one action.

## The v17 amplitude-independent halo-scale test

After the unchanged static response is trained on one cluster and transferred
to the other, define the remaining one-metric target triplet

\[
\Delta\mathbf F_{\rm req}
=(\Delta\kappa,\Delta\gamma_1,\Delta\gamma_2)
\]

and the transferred thermal prediction

\[
\Delta\mathbf F_{\rm pred}
=(\Delta\kappa_\Sigma,\Delta\gamma_{1\Sigma},
\Delta\gamma_{2\Sigma}).
\]

Both convergence and shear come from the same scalar-potential feature; no
lens-only coefficient is available. Inside the frozen analysis mask, define

\[
u(\mathbf x)=
(\Delta\kappa)^2+(\Delta\gamma_1)^2+(\Delta\gamma_2)^2.
\]

The field-energy centroid is computed separately for required and predicted
triplets. $R_{50}$ and $R_{80}$ are the smallest centroided radii enclosing
50% and 80% of $u$. Multiplying a triplet by any nonzero amplitude leaves
these radii unchanged. Position and orientation are still scored by full-field
NRMSE and shear alignment; recentering is used only to isolate size.

The scale diagnostic was frozen before either spent target was opened. It
passes only if both $R_{50}$ and $R_{80}$ are within 25% in both transfer
directions, with no cluster radius or normalization. Zero or nonfinite fields
fail rather than receiving an imputed radius. Doubling map resolution must
change the predicted radii by no more than 2%.

## How this constrains a root equation

A schematic target is

\[
\mathcal E_{\mu\nu}
[g,\Sigma;a_\Sigma,L_\Sigma,\epsilon,\beta_\Sigma]
={8\pi G\over c^4}T^b_{\mu\nu},
\]

with a dynamical Sigma equation

\[
\mathcal D[\Sigma,g;a_\Sigma,L_\Sigma,\epsilon]
=\beta_\Sigma\,\mathcal J[T_b,g].
\]

This is a requirements envelope, not a proposed action. A viable completion
must define $\mathcal E$, $\mathcal D$, and $\mathcal J$ by varying one
covariant action; derive the stress tensor and conservation law; and make the
same physical metric $g_{\mu\nu}$ govern matter and light.

The reserved roles of the four provisional constants are:

- $a_\Sigma$: a universal nonlinear acceleration/gradient threshold;
- $L_\Sigma$: a universal propagation or correlation length, if data require
  one;
- $\epsilon$: a bounded high/deep-field response parameter; and
- $\beta_\Sigma$: a universal coupling amplitude.

One of these should be removed if the equations do not need it; none may become
an object-level setting. The field must also approach GR in the Solar System,
keep (c_T=c), and possess positive kinetic and gradient energy.

## Decision tree fixed before the v17 target

1. If thermal stress fails transfer or halo-scale gates, do not tune a thermal
   radius. Reject the tested instantaneous gas-stress source.
2. If amplitude improves but $R_{50}/R_{80}$ fail, the source may correlate
   with strength but does not explain halo size. Do not repair this with a
   cluster-specific $L_\Sigma$.
3. If size passes but alignment/topology fails, the source has useful extent
   but lacks the tensor/nonlocal orientation channel required by lensing.
4. If every spent gate passes, derive the response from one healthy action and
   freeze its constants before any new cluster is opened.
5. Only raw held-out multiple-image roots, galaxy dynamics, and local/relativistic
   gates can turn the source clue into a defensible theory.

The established ingredients overlap MOND/AQUAL transition scaling, refracted
gravity, nonlocal kernels, scalar/vector-tensor gravity, and stress-coupled
gravity. Their formulas and primary sources are inventoried in
[`FORMULA_AND_PRIOR_ART_REGISTRY.md`](FORMULA_AND_PRIOR_ART_REGISTRY.md). A
genuine Sigma contribution would have to be the specific healthy combination
that predicts source-derived extent and one-metric shear—not a renamed member
of those published families.
