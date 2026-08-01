# Baryon-to-lensing-excess gravity-flow inversion

## Outcome

We now have a concrete way to ask where a putative baryon-generated gravity
flow would have to originate if it reappeared in the locations that standard
lens models normally describe with dark matter. The implementation was run on
ten RELICS clusters, 832 strict photometric member galaxies, 1,000 Lenstool
convergence realizations, and ten GLAFIC maps.

The inverse maps are reproducible between the two lens-model methods. Their
median path-density correlation is 0.969. They imply a typical minimum
projected source-to-excess distance of about 90 kpc and a 90th-percentile
distance of about 154--159 kpc. The brightest inferred origin of the strongest
excess peak is the same under Lenstool and GLAFIC in 8/10 clusters.

This is not evidence that gravity followed those paths. The real angular
galaxy layout beats radius-preserving randomized layouts in only 6/10 Lenstool
systems and 5/10 GLAFIC systems, with only one Lenstool system reaching the
32-shuffle permutation threshold and none under GLAFIC. Much of the inverse
structure can therefore be explained by the radial distributions alone.

## A field equation that can represent the idea

In the weak-field limit, let baryonic matter remain the only material source,
but allow a nonlocal metric response:

$$
\nabla^2\Phi(\mathbf y)=4\pi G\left[
\rho_b(\mathbf y)+q\int
\rho_b(\mathbf x)K_\theta(\mathbf y\mid\mathbf x,E_b)\,d^3x
\right].
$$

$K_\theta$ says where the field generated at $\mathbf x$ contributes again;
$E_b$ contains only baryonic geometry. The additional term is an effective
gravitational response, not a new material density. Its covariant completion
would have to be a causal, divergence-free nonlocal tensor $\Sigma_{\mu\nu}$:

$$
G_{\mu\nu}=8\pi G T^b_{\mu\nu}+\Sigma_{\mu\nu}[g,T_b],
\qquad \nabla^\mu\Sigma_{\mu\nu}=0.
$$

The present test identifies only the normalized spatial shape of $K$. It
cannot determine the absolute multiplier $q$. A valid theory must later derive
$q$, conserve the full stress-energy budget, and remain screened in the Solar
System.

This broad nonlocal-field form is related to existing nonlocal-gravity work;
it is not by itself a new theory. The distinctive part of this project is the
data-driven origin-to-destination inversion, its radial-angle controls, and the
requirement that the resulting baryonic kernel predict held-out raw lensing
data. See [Mashhoon's nonlocal-gravity formulation](https://arxiv.org/abs/1101.3752)
and the [RELICS product archive](https://archive.stsci.edu/hlsp/relics).

## How the backtracking works

For each cluster we construct:

- a source vector $b_i$ from strict member-galaxy F160W light at positions
  $\mathbf x_i$;
- a local baryonic template $B(\mathbf y)$, smoothed at 20 kpc;
- a normalized convergence map $\kappa(\mathbf y)$ from each lens model; and
- an operational excess map

$$
e(\mathbf y)=\left[\kappa(\mathbf y)-\hat a B(\mathbf y)\right]_+,
$$

where $\hat a$ is the non-negative least-squares projection of local light onto
the lensing shape. It removes a median 23.5% of the positive Lenstool shape and
27.6% of GLAFIC. This coefficient is a morphological nuisance value, not a
dark-matter fraction or stellar mass-to-light ratio.

We then solve

$$
P^*=\arg\min_{P\ge0}
\sum_{ij}P_{ij}\lVert\mathbf y_j-\mathbf x_i\rVert^2
+\epsilon\sum_{ij}P_{ij}(\ln P_{ij}-1),
$$

subject to

$$
\sum_jP_{ij}=b_i,
\qquad
\sum_iP_{ij}=e_j.
$$

The backtracked probability that excess pixel $j$ came from baryonic source
$i$ is

$$
\Pr(i\mid j)=\frac{P^*_{ij}}{e_j}.
$$

This is a minimum-projected-distance attribution. It does not prove the field
took the minimum route, but it converts a vague arc picture into source tables,
destination tables, route lengths, and testable direction statistics.

## What an actual arc could look like

The 2-D maps constrain endpoints but not the unseen height of an arc. A clear
family of compatible paths is

$$
\Gamma_{ij}(t)=\left(
(1-t)\mathbf x_i+t\mathbf y_j,
4h_{ij}t(1-t)
\right),\qquad0\le t\le1.
$$

The first two coordinates lie in the lens plane; the final coordinate is an
unobserved excursion. Every $h_{ij}$ produces the same projected endpoints.
The output therefore reports arc lengths for $h/d=0,0.25,0.5,1$ but does not
fit or claim a preferred height. Time delays, source-redshift tomography, or a
derived field action would be needed to identify it.

## Numerical results

| Quantity | Lenstool | GLAFIC |
|---|---:|---:|
| Median cluster-level path | 90.2 kpc | 90.5 kpc |
| Median cluster-level 90th percentile | 153.5 kpc | 159.4 kpc |
| Median inward-direction cosine | 0.209 | 0.182 |
| Median fraction ending inward | 0.508 | 0.483 |
| Real layout shorter than median radial shuffle | 6/10 | 5/10 |
| Systems with permutation $p\le0.05$ | 1/10 | 0/10 |

The route length is sensitive to the entropic smoothing scale, as it must be:
the median rises from about 83--85 kpc at 25 kpc smoothing to about 109 kpc at
100 kpc smoothing. The conclusion should therefore be “roughly one hundred
kiloparsecs,” not a new 90 kpc constant.

The two methods agree well in most systems but not all. Path-map correlations
range from 0.776 to 0.999. RXC J0600.1-2007 and RXC J0032.1+1808 are especially
method-sensitive, so their apparent arcs should not be used to build a law
until raw lens observables replace reconstructed convergence targets.

## The strongest new observation

The important signal is not a universal center return. Across all ten systems,
the fraction of transport ending inward rises sharply as the member-galaxy
distribution becomes more radially extended:

- $R_{50}$ versus inward fraction under Lenstool:
  $\rho=0.976$, FDR-adjusted $q=1.9\times10^{-4}$;
- $R_{50}/R_{80}$ versus inward fraction under GLAFIC:
  $\rho=0.939$, adjusted $q=0.0036$; and
- every leave-one-cluster-out jackknife keeps the same sign.

This relationship is partly geometric: sources farther from a central target
have more opportunity to travel inward. It nevertheless agrees with the
earlier independent observation within the same ten systems that a
center-return forward model works better when the baryonic distribution is
extended. It suggests a collective, extent-gated response rather than one
fixed arc attached to every galaxy.

A compact exploratory gate fitted to the inverse paths is

$$
C={R_{50}\over R_{80}},\qquad
s(C)={1\over1+\exp[-4.78(C-0.649)]}.
$$

Here $s$ is the inferred inward-return share. The joint same-data fit halves
the inward-fraction RMS error relative to a constant, and its leave-one-cluster-
out RMS is 0.121. This is a candidate generator, not a confirmed coefficient.

The associated scale-free kernel to test next is

$$
K(\mathbf y\mid\mathbf x)=
[1-s(C)]G_{\eta R_{80}}(\mathbf y-\mathbf x)
+s(C)\int p(\ell)G_{\eta R_{80}}
(\mathbf y-\mathbf x-\ell\hat{\mathbf c}_x)\,d\ell,
$$

with the exploratory geometric values

$$
\operatorname{median}(\ell)\simeq0.36R_{80},\qquad
\ell_{90}\simeq0.65R_{80},\qquad
\eta\simeq0.23.
$$

$\hat{\mathbf c}_x$ points toward the baryonic luminosity center. The first
term remains local when the cluster is too compact for a coherent return; the
second activates as the baryonic distribution becomes broadly extended.

## What the present observations answer

They answer:

1. which observed member galaxies are the least-cost origins of each apparent
   excess peak;
2. the projected distance and direction distribution required by that
   interpretation;
3. how sensitive those origins are to lens-model method and residual
   definition; and
4. whether specific galaxy angles add information beyond the radial profile.

They do not answer:

1. whether any gravity line physically followed an inferred route;
2. whether the apparent excess is actually dark matter, hot gas, lens-model
   prior structure, or a nonlocal field response;
3. the absolute strength $q$ of the response;
4. the line-of-sight path or arc height; or
5. whether one kernel also predicts galaxy rotation curves and Solar-System
   gravity.

## Next falsification

Freeze the extent-gated kernel above before opening another sample. On unused
RELICS, HFF, BUFFALO, or CANUCS clusters, construct complete baryonic maps from
stellar mass plus X-ray/SZ gas. Fit at most one universal amplitude $q$ on a
training set, then predict raw multiple-image positions, time delays, and weak
shear in held-out clusters. Compare against baryons-only GR, a smooth central
halo, and a standard dark-matter lens model with parameter counts disclosed.

Failure would mean the inverse paths were a descriptive re-expression of
radial lens morphology. Success would mean a baryonic nonlocal kernel predicted
where apparent excess lensing occurs without using that excess map to draw the
paths.

## Reproduction

```powershell
python scripts/run_gravity_flow_inverse.py
python scripts/analyze_gravity_flow_inverse_drivers.py
pytest tests/test_gravity_flow_inverse.py tests/test_gravity_flow_inverse_results.py
```

The complete tables and path maps are in `results/gravity_flow_inverse`.
