# Galaxy dynamics and cluster lensing: joint result

Status: completed 2026-07-26.

## Bottom line

The best tested idea is a **partial empirical bridge, not a unified theory**.
The U0 potential-screened law substantially improves held-out CLASH cluster
lensing while retaining nearly all of fixed RAR's held-out SPARC performance.
It passes the frozen relative advancement rule. However, its absolute CLASH
chi-square is still high, its fitted transition varies across folds, and it
underpredicts an untouched sample of 50 MaNGA brightest-cluster galaxies
(BCGs). The result justifies one sharper test using independently measured host
baryonic potentials; it does not justify claiming discovery.

It would also be incorrect to claim that no existing theory addresses both
galaxies and cluster lensing. Lambda-CDM does so with object-specific dark
matter halos, and modified-gravity programs have attempted both regimes. This
project's narrower target is one baryon-linked law with no per-object dark halo
and no lensing-only rescaling.

## The common prediction

The weak-field metric was fixed to zero gravitational slip,
$\Phi=\Psi$. Slow matter and light therefore see the same inferred acceleration
$g_{\rm pred}$; no separate lensing multiplier is fitted. Given a baryonic
profile,

$$
|\Phi_{\rm bar}(r_j)|=\int_{r_j}^{r_{\max}}g_{\rm bar}(r)\,dr
+g_{\rm bar}(r_{\max})r_{\max},
\qquad \chi=|\Phi_{\rm bar}|/c^2.
$$

The U0 closure is

$$
S_\chi=\left[1+\exp\left(-\frac{\log_{10}\chi-\log_{10}\chi_t}{w}\right)\right]^{-1},
$$

$$
a_{\rm eff}=a_0\exp[\ln(F)S_\chi],
\qquad
g_{\rm pred}=\frac{g_{\rm bar}}
{1-\exp[-\sqrt{g_{\rm bar}/a_{\rm eff}}]}.
$$

It predicts either observable without changing the field law:

$$
v_{\rm pred}(R)=\sqrt{R g_{\rm pred}(R)},
\qquad
M_{\rm lens,pred}(<r)=\frac{g_{\rm pred}(r)r^2}{G}.
$$

This spherical/no-slip construction is a falsifiable phenomenological mapping,
not a covariant field equation.

## Frozen whole-system validation

The development data contain 3,034 points in 131 SPARC galaxies and 84 points
in 20 CLASH clusters. Five folds hold out complete galaxies and complete
clusters. Fits use the ordinary summed standardized-residual likelihood, 16
deterministic starts, and no manually chosen domain weight. Paired uncertainty
uses 100,000 grouped bootstrap draws.

| Model | SPARC $\chi^2$/point | CLASH $\chi^2$/point | CLASH RMS (dex) | Equal-domain macro |
|---|---:|---:|---:|---:|
| Fixed galaxy RAR | 8.969 | 42.991 | 0.508 | 25.980 |
| J0: one fitted constant | 9.139 | 41.543 | 0.499 | 25.341 |
| **U0: potential screen** | **9.323** | **4.994** | **0.169** | **7.159** |
| U1: coherence length | 13.055 | 7.027 | 0.237 | 10.041 |
| Domain-labeled oracle | 8.969 | 1.648 | 0.111 | 5.309 |

Relative to fixed RAR, U0 changes the SPARC score by +0.354 per point
(95% grouped-bootstrap interval -0.209 to +0.945) and the CLASH score by
-37.996 (-46.464 to -29.567). Its equal-domain change is -18.821
(-23.108 to -14.548). The SPARC degradation is 3.94%, inside the frozen 5%
limit. J0 also clears the literal relative gate but leaves the cluster mismatch
almost intact; U0 is the only tested candidate with a large bridge effect.

Adding Tian et al.'s 0.063-dex intrinsic CLASH scatter lowers U0's cluster score
to 2.757 per point. That remains above one. Absolute scores are also affected by
the deliberately fixed SPARC mass-to-light ratios and simple error model, so the
clean conclusion is comparative: U0 is much better than fixed RAR on this joint
task, but is not statistically adequate yet.

The full-development U0 parameters, used without refitting in the BCG check,
are

| Parameter | Value |
|---|---:|
| $F$ | 29.9869 |
| $\chi_t$ | $3.2252\times10^{-6}$ |
| $w$ | 0.2136 dex |

Across held-out folds, $F$ ranges from 16.7 to the upper bound of 100 and $w$
ranges from its lower bound of 0.1 dex to 0.352 dex. This instability suggests
that the fit may be learning the gap between the galaxy and cluster samples
rather than a well-measured continuous transition.

## What failed

U1 used the baryonic coherence length $\ell=|\Phi_{\rm bar}|/g_{\rm bar}$ to
make the modification grow near system edges. It helps CLASH but worsens the
SPARC score by 45.6%, so it fails the galaxy-preservation rule. The strongest
pathology occurs where signed baryonic components nearly cancel: $g_{\rm bar}$
approaches zero while $|\Phi_{\rm bar}|$ stays finite, making $\ell$ singular.
That is a physical defect of this local ratio, not merely an optimizer issue.

## External BCG dynamics check

The frozen U0 law was next evaluated on one reported outer dynamical point for
each of 50 MaNGA BCGs. No BCG value entered the SPARC+CLASH fit. This is a
post-discovery external check, not a blind test, because the paper's aggregate
cluster-scale RAR result was already known.

Only the BCG's own baryons were used, with the declared point-mass tail
$|\Phi|=g_{\rm bar}r_{\rm last}$. U0 activates only weakly at the median BCG
potential and systematically underpredicts the observations.

| Model | $\chi^2$/point | RMS (dex) | Mean residual (dex) |
|---|---:|---:|---:|
| Fixed galaxy RAR | 9.962 | 0.327 | -0.293 |
| **Frozen U0** | **7.149** | **0.299** | **-0.258** |
| Cluster-scale RAR reference | 2.188 | 0.133 | +0.081 |

U0 improves on fixed galaxy RAR by -2.813 chi-square per point, with a paired
BCG-bootstrap interval of -3.475 to -2.195. It therefore generalizes in the
right direction, but it does not complete the galaxy-to-cluster bridge.

## Host-potential diagnostic and next decisive test

A post-hoc inverse calculation asks how much *additional* potential would make
the frozen U0 law reproduce each BCG point. This is not a prediction and was not
inserted into the score. Finite solutions exist for 47 of 50 BCGs. Their median
required host contribution is

$$
\Delta\chi_{\rm host}=3.08\times10^{-6},
\qquad
\sqrt{|\Phi_{\rm host}|}=526\ {m km\,s^{-1}},
$$

with a 10th--90th percentile range of 344--758 km/s. For scale only, the
extended baryonic potentials reconstructed from the much more massive CLASH
sample have a median equivalent speed of 808 km/s. The missing BCG scale is
therefore astrophysically plausible, but plausibility is not prediction.

The next candidate, E0, must keep the full-development U0 parameters frozen and
replace

$$
\chi_{\rm self}\longrightarrow
\chi_{\rm self}+\chi_{\rm host,bar},
$$

where $\chi_{\rm host,bar}$ is integrated from independently measured host gas,
BCG, and satellite-star profiles for the same system. There may be no constant
offset fitted from BCG dynamics and no cluster-class label. E0 must be tested on
a new or properly reconstructed sample, because the present BCG outcomes have
now been inspected. If independently measured host potentials do not remove
the negative BCG residual without damaging SPARC, the potential-screen bridge
should be rejected in this form.

Outer-gas systematics are an equally important falsifier: recent non-parametric
CLASH work finds that the apparent cluster RAR offset changes substantially with
gas extrapolation. E0 therefore needs explicit baryon-profile uncertainty, not
only acceleration error bars.

## Originality boundary

Potential-dependent MOND acceleration scales are established prior art,
especially EMOND. U0 is intentionally labeled an EMOND-like control and is not
claimed as our invention. The useful project contribution is narrower: one
no-slip predictive map, complete-system cross-validation across SPARC and
CLASH, an untouched intermediate-regime dynamical check, and a frozen next
falsifier. Any publication claim would still require a full literature and
covariant-theory audit.

## Reproduction artifacts

- `results/unified_cv/report.json`: exact held-out metrics and bootstrap results.
- `results/unified_cv/fold_parameters.csv`: fold-by-fold fitted parameters.
- `results/external_bcg/report.json`: untouched BCG scores.
- `results/bcg_host_potential_diagnostic/report.json`: inverse scale diagnostic.
- `configs/unified_model_registry.json`: frozen formulas, bounds, and seed.
- `configs/host_potential_E0_registry.json`: zero-fit next-test specification.
- `docs/UNIFIED_GALAXY_CLUSTER_PREREGISTRATION.md`: rules frozen before fitting.
- `docs/BCG_EXTERNAL_TEST_PLAN.md`: external score frozen before execution.
