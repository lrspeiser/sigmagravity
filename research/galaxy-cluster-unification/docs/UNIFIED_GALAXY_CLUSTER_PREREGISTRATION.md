# Preregistered galaxy-dynamics and cluster-lensing test

Status: formulas and decision rules frozen before fitting, 2026-07-26.

## The narrower scientific target

It is not accurate to say that no theory addresses galaxies and cluster lensing.
The standard cosmological model does so with dark-matter halos, and modified
gravity programs including STVG and relativistic MOND have made claims in both
regimes. The harder target pursued here is narrower: **one baryon-linked law,
with no per-object dark halo and no lensing-only rescaling, must predict both
stellar circular speeds and cluster lensing acceleration.**

This project will not claim invention of a potential-dependent acceleration
scale. Zhao and Famaey's [EMOND](https://arxiv.org/abs/1207.6232) explicitly made
the MOND scale depend on acceleration and potential depth to bridge galaxies and
clusters. Later cluster tests found partial success but persistent problems
([Hodson & Zhao 2017](https://arxiv.org/abs/1701.03369)). U0 below is therefore a
direct prior-art control.

The specific U1 closure, grouped cross-validation, and paired comparison are a
new test within this project. That is not a broad originality claim: length- and
scale-dependent gravity has extensive prior art, and an eventual paper would
need a full ADS/INSPIRE citation audit.

## Why lensing requires more than a rotation-curve formula

Write the weak-field metric as

$$
ds^2=-(1+2\Psi/c^2)c^2dt^2+(1-2\Phi/c^2)d\mathbf{x}^2.
$$

Slow stars respond to $-\nabla\Psi$, while light responds to the lensing
combination $\Phi+\Psi$. A rotation curve alone constrains only $\Psi$. The
primary test fixes zero gravitational slip, $\Phi=\Psi$, so the same effective
field that predicts stellar speeds must also predict lensing. No slip parameter
will be fitted after inspecting CLASH. This is consistent with, though much less
precise than, recent CLASH-VLT kinematics+lensing constraints that include
$\eta=1$ within their uncertainties
([Pizzuti et al. 2025](https://arxiv.org/abs/2509.16317)).

For spherical clusters the prediction is

$$
M_{\rm lens,pred}(<r)=\frac{g_{\rm pred}(r)r^2}{G}.
$$

For a galaxy midplane point it is

$$
v_{\rm pred}(R)=\sqrt{R g_{\rm pred}(R)}.
$$

Thus one $g_{\rm pred}$ makes both observables.

## Data

The galaxy domain is the already frozen SPARC snapshot with the same objective
quality, inclination, and minimum-point cuts as the void tests. This first joint
test fixes the 3.6 micron stellar mass-to-light ratios to 0.5 for disks and 0.7
for bulges. It therefore tests predictive shape without using held-out velocities
to adjust per-galaxy nuisances.

The cluster domain is the official CDS table accompanying
[Tian et al. 2020](https://arxiv.org/abs/2001.08340): 84 radii in 20 CLASH
clusters. Its baryonic accelerations combine hot gas, the BCG, and other cluster
stars; its total accelerations come from joint strong-lensing, weak-lensing
shear, and magnification mass reconstructions. The authors find a cluster RAR
slope near $1/2$ but an acceleration scale about 17 times the galaxy value.

The CLASH points are a sharp first discriminator, not perfect truth. A newer
non-parametric reconstruction finds that the cluster offset depends strongly on
uncertain outer gas extrapolation
([Mistele et al. 2025](https://arxiv.org/abs/2506.13716)). We will therefore
report a sensitivity score with the Tian intrinsic scatter added, and treat a
future non-parametric baryon-profile reconstruction as required confirmation.

## Shared baryonic field quantities

With the potential zero fixed at infinity, construct

$$
|\Phi_{\rm bar}(r_j)|=\int_{r_j}^{r_{\max}}g_{\rm bar}(r)\,dr
+g_{\rm bar}(r_{\max})r_{\max}.
$$

The tail is the declared point-mass continuation. Define

$$
\chi=|\Phi_{\rm bar}|/c^2,
\qquad
\ell_{\rm bar}=|\Phi_{\rm bar}|/g_{\rm bar}.
$$

$\ell_{\rm bar}$ is the local scale length of the baryonic field. In a
point-mass exterior, $|\Phi|=GM/r$ and $g=GM/r^2$, hence
$\ell_{\rm bar}=r$. It operationalizes the proposed edge effect without using
an object-class label or a void distance.

Every model uses the same RAR-shaped closure

$$
g_{\rm pred}=\frac{g_{\rm bar}}
{1-\exp[-\sqrt{g_{\rm bar}/a_{\rm eff}}]}.
$$

This is a phenomenological quasistatic closure, not yet a covariant field
theory. In a nonspherical system an algebraic acceleration relation need not be
curl-free; passing this test would justify deriving an action, not replace that
derivation.

## Frozen models

### Fixed RAR

$a_{\rm eff}=a_0=1.2\times10^{-10}\,\mathrm{m\,s^{-2}}$, with no fitted
parameter.

### J0: one freely fitted universal acceleration scale

$a_{\rm eff}=a_{\rm joint}$ is shared by every galaxy and cluster. This checks
whether one compromise constant is enough.

### U0: EMOND-like potential control

$$
S_\chi=\left[1+\exp\left(-\frac{\log_{10}\chi-\log_{10}\chi_t}{w}\right)\right]^{-1},
$$

$$
a_{\rm eff}=a_0\exp[\ln(F)S_\chi].
$$

It has three global parameters: $F$, $\chi_t$, and $w$. This is labeled
EMOND-like and carries no originality claim.

### U1: baryonic coherence-length hypothesis

$$
a_{\rm eff}=a_0\left[1+\left(\frac{\ell_{\rm bar}}{\ell_c}\right)^q\right].
$$

It has two global parameters, $\ell_c$ and $q$. It predicts a smooth edge
departure based only on the baryonic field geometry. It is deliberately not
saturated in this first test, making extrapolation failure possible and the
hypothesis easier to falsify.

### Domain oracle

The galaxy scale stays fixed at $a_0$, while a separate cluster scale is fitted.
This is not a unified candidate. It estimates how much predictive accuracy is
available merely by labeling a system “cluster.”

Exact bounds are frozen in `configs/unified_model_registry.json`.

## Validation and stopping rule

System names are sorted, permuted once with seed 20260726, and assigned
round-robin to five folds. Entire galaxies and entire clusters are held out.
Global parameters fit the other four folds by the ordinary summed standardized
residual likelihood; there is no hand-chosen galaxy/cluster weight. Galaxy
residuals are in velocity with the existing 5 km/s floor. Cluster residuals are
in log acceleration and propagate the local $g_{\rm bar}$ error through the
model slope. The published 0.063-dex intrinsic scatter is a sensitivity analysis,
not the primary error.

A candidate advances only if all are true:

1. It lowers held-out cluster $\chi^2$ per point relative to fixed RAR.
2. Its held-out galaxy $\chi^2$ per point is no more than 5% worse than fixed
   RAR.
3. It lowers the equal-domain macro-average held-out $\chi^2$.
4. A paired bootstrap of complete galaxies and complete clusters supports the
   stated improvement; pointwise resampling is not allowed.

If neither U0 nor U1 passes, we stop tuning this SPARC+CLASH pair. The next move
would be a derived field equation or new data, not another unchecked formula.
