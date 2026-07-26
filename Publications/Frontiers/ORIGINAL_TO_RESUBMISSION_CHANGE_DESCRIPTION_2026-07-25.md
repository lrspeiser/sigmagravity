# Description of Changes from the Original Submission to the Revised Manuscript

**Manuscript ID:** 1866133
**Original title:** *Σ-Gravity: Coherence-Dependent Gravitational Enhancement in Galaxies and Clusters*
**Revised title:** *Σ-Gravity: A Coherence-Motivated Empirical Response Tested in Galaxies and Clusters*

## Documents compared

This comparison uses the preserved original-submission family in `docs/sigmagravity_paper.pdf`, `docs/sigmagravity_paper.tex`, and `docs/sigmagravity_paper_journal.pdf`; the revised reviewer-continuity manuscript in `Publications/Frontiers/Reviewer_Continuity/`; and the original reports from Reviewer 1 and Reviewer 2 as reproduced in the point-by-point response files.

The repository contains several historical builds of the original paper. One build reports a 43% galaxy win rate and another reports 47%; Reviewer 2 explicitly commented on the 47% figure. This summary follows the substance of the reviewers' criticism and uses the final revised analysis values when describing the resubmission.

## Overall description

The revision preserves the central empirical structure of Σ-Gravity but narrows what the data are claimed to establish. The original manuscript combined an acceleration response, a kinematic-coherence interpretation, and a path-length scaling into a proposed unified galaxy-and-cluster framework. It then used that interpretation to make comparatively strong claims about cluster lensing, counterrotating galaxies, Solar-System consistency, conservation, and cross-domain validation.

The revised manuscript separates three levels that were previously blended together:

1. an empirical nonrelativistic response that can be tested on galaxy rotation curves;
2. coherence, source geometry, and path length as possible physical explanations for that response; and
3. a future fundamental, relativistic, or cosmological theory that is not supplied by the present work.

The first level is retained and tested more rigorously. The second is presented as a hypothesis. The third is explicitly outside the paper's demonstrated scope.

The revised paper does not replace the submitted cluster normalization with a newly fitted value, does not retune the core galaxy predictor to the new analyses, and does not incorporate exploratory directed-string, refracted-gravity, shell-transfer, or auxiliary-field formulas into the reported model. Those ideas remain separate research directions.

## What was preserved

The following key elements of the original work remain in the revised manuscript:

- The same acceleration scale,
  \[
  g^\dagger=\frac{cH_0}{4\sqrt{\pi}}.
  \]
- The same bounded acceleration function,
  \[
  h(g_N)=\sqrt{\frac{g^\dagger}{g_N}}\frac{g^\dagger}{g^\dagger+g_N}.
  \]
- The empirical enhancement structure, rewritten in identifiable form as
  \[
  \Sigma(g_N,B)=1+B h(g_N).
  \]
- The locked galaxy implementation with \(A_0=e^{1/(2\pi)}\), the fixed \(\sigma=20\ {\rm km\,s^{-1}}\) regulator, and the bounded fixed-point factor calculated from the predicted rather than observed velocity.
- Fixed stellar mass-to-light ratios and no fitted halo or response amplitude for each individual SPARC galaxy.
- The Fox cluster value \(B_{\rm Fox}=8.446\) as the submitted calibration baseline.
- Coherence and source organization as motivations for future testing.
- A direct comparison with MOND using the same baryonic inputs.

The important change is not the removal of these elements. It is the removal of conclusions that the present implementation cannot independently support.

## Scope simplification

The original manuscript ranged across SPARC, Milky Way stellar kinematics, Fox clusters, Solar-System behavior, dwarf satellites, high-redshift galaxies, wide binaries, merging clusters, a phase-coherence extension, and alternative field implementations. The revised submission concentrates the evidentiary argument on:

- the locked and ablated SPARC predictors;
- the actual-SPARC-scale-length sensitivity;
- the Fox calibration and no-refit CLASH profile check;
- the algebraic-versus-numerical QUMOND comparison;
- the matched counterrotation diagnostic; and
- explicit nonrelativistic, relativistic, and cosmological boundaries.

The broader exploratory tests remain available in the research repository, but they are not used to support the revised paper's main conclusion. This reduces the number of claims that depended on provisional assumptions and keeps the manuscript focused on analyses that directly answer the reviews.

## Shared concerns raised by both reviewers

### 1. The model was presented too much like a completed theory

**Original paper.** The manuscript described a “covariant coherence scalar,” a unified galaxy-to-cluster amplitude, geodesic motion in an enhanced potential, conservation by construction, Solar-System safety, and implications distinct from both MOND and ΛCDM. Although it acknowledged the absence of a first-principles derivation, the surrounding language still implied theory-level consistency.

**Revised paper.** The paper now calls Σ-Gravity an “alternative nonrelativistic phenomenological parameterization of selected missing-mass observations.” The title changes from “Coherence-Dependent Gravitational Enhancement” to “A Coherence-Motivated Empirical Response Tested” so that it identifies the model as phenomenological and does not imply successful performance in every test regime. The Abstract, Introduction, Discussion, and Conclusions distinguish the empirical response from any proposed physical mechanism.

**How this addresses the concern.** This preserves the empirical proposal while no longer claiming that coherence has been derived as a cause of modified gravity or that the model is already a complete gravitational theory.

**Location.** Revised title and Abstract; Sections 1, 2.1–2.3, 4.1–4.6, and 5.

**Remaining limitation.** The physical origin of \(B\) is still unknown. The revision says so rather than claiming to have solved it.

### 2. Coherence was not independently measured

**Original paper.** The response was written as \(\Sigma=1+A\mathcal C h(g_N)\), with \(\mathcal C\) described as a covariant coherence scalar. In practice, the galaxy value was calculated from the model-predicted velocity. The paper also stated that a radial window based on disk scale length produced equivalent results.

**Revised paper.** The observable response is expressed through the combined amplitude \(B=A\mathcal C\). The locked galaxy factor is explicitly labeled endogenous because it is computed from the predicted velocity. It is called a phenomenological regularizer motivated by rotational support, not an independent measurement of coherence. An operational phase-space estimator is provided only as a proposed future test.

**How this addresses the concern.** The revision no longer treats the successful behavior of the endogenous factor as proof that measured kinematic coherence changes gravity. It reports only the narrower ablation result that the bounded factor changes and modestly improves the locked predictor relative to its acceleration-only form.

**Location.** Sections 1, 2.1–2.2, 3.2, 4.2, and 5.

**Remaining limitation.** A genuine coherence test still requires independently measured streaming velocities and velocity-dispersion tensors, with held-out galaxies and no use of the target rotation curve to define the predictor.

### 3. Path length was not operationally defined or identifiable

**Original paper.** The amplitude law \(A(L)=A_0(L/L_0)^n\), with \(L_0=0.4\) kpc and \(n=0.27\), was presented as a principled connection between disks and clusters. Every Fox cluster was nevertheless assigned the same \(L=600\) kpc.

**Revised paper.** The canonical empirical law is written in terms of \(B\). Because \(L\) is constant within the Fox calibration sample, the paper states that those data identify one effective cluster amplitude rather than a path-length exponent. I could not supply a unique operational definition of \(L\) for a general baryonic distribution, so \(A(L)\) is removed from the predictive response and conclusions using it to connect disks and clusters are withdrawn. \(L_0\) and \(n\) are not used in the locked galaxy result. Path length and source concentration remain possible motivations, not established parts of the canonical predictor.

**How this addresses the concern.** The revision removes the appearance that a general three-dimensional path-length functional was derived or validated.

**Location.** Sections 2.1–2.2, 2.7, 4.3, and Table 2.

**Remaining limitation.** No unique path-length functional for an arbitrary three-dimensional baryonic distribution has been found.

### 4. The cluster analysis confused calibration with validation

**Original paper.** The exponent \(n\) was calibrated on 42 Fox clusters, while the resulting median predicted-to-observed mass ratio of 0.987 and repeated splits of the same catalog were used to support cluster success and cross-domain consistency.

**Revised paper.** The Fox systems are called the calibration sample, 0.987 is called an in-sample calibration result, and repeated Fox splits are called calibration-stability checks. A fixed name-normalization rule excludes the exact Fox overlaps MACS0416, MACS0717, and MACS1149 from Tian/CLASH. A separate no-refit check then freezes \(B_{\rm Fox}=8.446\), \(h(g_N)\), and \(g^\dagger\) and applies them to 73 radial measurements in 17 disjoint clusters. That external check gives a median predicted-to-observed ratio of 1.318; radius-bin medians are 1.170, 1.228, 1.613, and 1.961 at 100, 200, 400, and 600 kpc. The weighted log-residual slope is +0.162 dex per dex, with a 5,000-draw cluster-bootstrap 95% interval of [+0.115, +0.217].

**How this addresses the concern.** Calibration and external evaluation are now visibly separate. The unfavorable no-refit result is reported, and the revised paper does not fit \(B\approx5.2\) or another replacement amplitude to restore the cluster claim.

**Location.** Sections 2.7, 3.5–3.6, 4.1, and Figure 2.

**Remaining limitation.** The CLASH exercise is a profile-level no-refit check, not a complete likelihood using independently reconstructed gas, stellar, intracluster-light, and lensing covariance for every system.

### 5. The cluster baryon prescription was too simplified

**Original paper.** The baryonic mass inside 200 kpc was approximated as \(0.4\times0.15M_{500}\). Although sensitivity was mentioned, the resulting calibrated mass ratio was still used to support a strong cluster conclusion.

**Revised paper.** The same expression is retained solely to reproduce the submitted Fox calibration and is explicitly labeled a simplified baryon-concentration proxy. The paper states that changing the 0.4 factor by 25% changes the predicted ratio by about 30%, so the calibrated amplitude cannot be separated from the baryon assumption.

**How this addresses the concern.** The revision does not present the proxy as a precision reconstruction of cluster baryons.

**Location.** Sections 2.7 and 3.5; Table 2.

**Remaining limitation.** The 42 Fox systems have not all been reanalyzed with measured radial gas, brightest-cluster-galaxy, satellite, and intracluster-light profiles. This is an important outstanding analysis rather than a problem hidden by the revised wording.

### 6. The counterrotation result lacked valid controls

**Original paper.** Sixty-three counterrotators were compared with more than 10,000 normal galaxies. The reported 44% lower JAM/NFW-inferred dark-matter fraction and \(p<0.01\) were described as a confirmed prediction, and the paper stated too strongly that ΛCDM predicted no difference.

**Revised paper.** Repeat MaNGA observations are first reduced to one row per physical galaxy. Sixty-two counterrotators are then matched to 310 unique controls on stellar mass, physical size, Sérsic index, axis ratio, inclination, redshift, and JAM fit quality. Greedy nearest-neighbor matching uses standardized covariates, hardest-case-first ordering, five controls per case, no replacement, and no caliper. The maximum post-match absolute standardized mean difference is 0.066. The matched difference in \(f_{\rm DM}(<R_e)\) is −0.0081 with a matched-set bootstrap 95% interval of [−0.0577, +0.0453], which is consistent with zero. The “confirmed prediction” and “ΛCDM predicts no difference” statements are removed.

**How this addresses the concern.** The manuscript now reports the balanced null result rather than preserving the original association as evidence.

**Location.** Sections 2.8, 3.7, 4.7, and Figure 4.

**Remaining limitation.** The available outcome is still derived from JAM/NFW modeling and is not a direct observable. A decisive test requires common forward modeling of MaNGA velocity and dispersion maps. Environment and merger history were not available in the local matching catalog.

### 7. Statistical conclusions were overstated

**Original paper.** The 47% win rate was highlighted as a key result even though MOND won the complementary 53%, and repeated radial measurements and unmatched populations supported stronger claims than their dependence structure allowed.

**Revised paper.** The primary sample contains 164 galaxies and 2,745 disk-dominated radial points. Mean per-galaxy RMS is 16.366 km s\(^{-1}\) for Σ-Gravity and 16.056 km s\(^{-1}\) for the tested MOND prescription. The mean paired difference is +0.309 km s\(^{-1}\), with a galaxy-bootstrap 95% interval of [−0.040, +0.659]. Σ-Gravity has lower RMS for 71 of 164 galaxies, with no ties and exact two-sided binomial \(p=0.101\) under the null \(p=0.5\). The paper therefore states that the comparison does not resolve a statistically significant aggregate difference; it does not claim a win, equivalence, or noninferiority.

The revision also adds:

- 20,000 galaxy-level bootstrap resamples;
- a paired sign-flip test;
- an exact object-level binomial comparison;
- observational-error-weighted sensitivity;
- 20%, 30%, and 40% bulge-threshold checks plus all 171 usable galaxies;
- an 81-combination frozen nuisance grid;
- cluster-grouped residual summaries; and
- galaxy-level matching and bootstrap uncertainty for counterrotation.

**How this addresses the concern.** The statistical unit is now the galaxy or cluster rather than each radius, and effect sizes with uncertainty determine the wording.

**Location.** Sections 2.5, 3.1–3.7, Table 1, and Figures 1–4.

**Remaining limitation.** Reviewer 2 marked statistical consultation as required. No external statistical consultation occurred or is promised. The response instead asks that the bounded claims be evaluated on the object-level methods, frozen code, residuals, sensitivity analyses, and reproducible outputs now supplied. If consultation is treated as a mandatory procedural condition, that condition remains unfulfilled.

### 8. Claims throughout the paper were too strong

| Original claim or implication | Revised statement |
|---|---|
| “Coherence-Dependent Gravitational Enhancement” | “A Coherence-Motivated Empirical Response Tested” |
| Coherence is the operative physical scalar | The data identify \(B\); coherence is one possible explanation |
| Unified path-length relationship | One effective Fox amplitude; no identified 3D path functional |
| Cluster lensing success or prediction | Fox in-sample calibration conditional on baryons and lensing closure |
| Fox holdouts provide independent validation | Fox splits measure within-catalog calibration stability |
| SPARC “win rate” | No statistically significant aggregate difference resolved; no equivalence claim |
| Counterrotation confirms the theory | Matched secondary diagnostic consistent with zero |
| Solar-System constraints are satisfied | High-acceleration sanity check only; PPN behavior is undetermined |
| Lensing and dynamics use the same derived potential | Equality is an empirical closure; gravitational slip is undetermined |
| Alternative to dark matter or ΛCDM generally | Galaxy-scale empirical alternative; not a replacement cosmology |

## Reviewer 1-specific changes

### 9. Demonstration of the claimed covariant reduction

**Reviewer 1 concern.** The original manuscript gave a covariant expression and stated that it reduced to the observational coherence equation without demonstrating the reduction.

**Change.** The claimed reduction is removed. The revised paper gives only a schematic second-moment motivation and explicitly lists the frame, anisotropy, inclination, and endogeneity problems. It does not claim a covariant coherence field.

**Resolution status.** Addressed by withdrawing the unsupported derivation, not by supplying a full covariant theory.

### 10. Actual SPARC photometric scale lengths

**Reviewer 1 concern.** The paper should use the actual SPARC scale lengths rather than an \(N/3\) radius proxy.

**Change.** The revised paper first corrects the model description: the locked production predictor uses the bounded fixed-point factor and does not use \(R_d\). It then adds a separate zero-parameter test using each catalog scale length:

\[
W(r,R_d)=\frac{r}{R_d/(2\pi)+r},\qquad B_{R_d}=A_0W(r,R_d).
\]

The catalog-\(R_d\) candidate gives mean RMS 16.513 km s\(^{-1}\), compared with 16.366 km s\(^{-1}\) for the locked predictor. The paired difference is +0.147 km s\(^{-1}\), with 95% interval [−0.088, +0.385]. The actual \(R_d\) assignments do not beat 2,000 random reassignments (\(p=0.711\)) or a common median \(R_d=2.3\) kpc, which gives RMS 16.413 km s\(^{-1}\).

**Resolution status.** Fully tested for the specific window requested. The result is negative and is reported as such. It does not rule out every possible geometry dependence.

### 11. The algebraic QUMOND approximation

**Reviewer 1 concern.** The paper used \(g_{\rm eff}\simeq\Sigma g_N\) without quantifying its error for disks.

**Change.** The revised paper labels the relation an approximation and compares it with three-dimensional periodic-FFT QUMOND solutions for axisymmetric analytic reconstructions representative of F574-2, UGC05716, and NGC3741. The check uses spatially constant \(B=1\), a \(65^3\) grid with box half-width \(8R_d\), and reports radial acceleration differences over \(0.75\le r/R_d\le5\). Median absolute acceleration differences are 5.19%, 4.88%, and 3.96%; local maxima are 20.54%, 18.30%, and 7.73%. A \(49^3\)-to-\(65^3\) check gives 0.72% median and 3.67% maximum acceleration change.

**Resolution status.** The approximation error is quantified. A full exact-field SPARC likelihood using observed gas and bulge maps remains future work.

### 12. Solar-System reasoning

**Reviewer 1 concern.** Setting the Sun's coherence to zero was imposed phenomenologically and did not follow from the field equations.

**Change.** The compact-source \(\mathcal C=0\) argument and the claim of satisfying the Cassini bound are removed. The revised paper states only that \(h(g_N)\) is small at Solar-System accelerations and explicitly says this is not a PPN calculation.

**Resolution status.** Addressed by removing the unsupported conclusion. Actual PPN, equivalence-principle, and compact-source behavior remain unresolved.

### 13. Lensing and gravitational slip

**Reviewer 1 concern.** Equality between lensing and dynamical mass was assumed rather than derived from a relativistic theory.

**Change.** The revised paper writes the weak-field metric, distinguishes the potential governing nonrelativistic dynamics from the sum of potentials governing lensing, and states that the present nonrelativistic action does not determine the slip η = Φ/Ψ. Cluster comparisons are described as empirical mass comparisons conditional on a closure assumption.

**Resolution status.** The assumption is now transparent; it is not theoretically derived.

### 14. Figure and table quality

**Reviewer 1 concern.** Figure and table quality was inadequate.

**Change.** The main results are reorganized into four regenerated vector figures: paired SPARC statistics and nuisance sensitivity; Fox calibration versus no-refit CLASH evaluation; algebraic-versus-numerical QUMOND error; and matched counterrotation balance and effect interval. Two tables explicitly identify dataset roles and parameter status. High-resolution PNG alternatives accompany the vector files.

**Resolution status.** Addressed in the revised submission package.

## Reviewer 2-specific changes

### 15. A nonrelativistic action and the boundary of the derivation

**Reviewer 2 concern.** The response was inserted ad hoc without an action, mechanism, or theoretical reason for its functional forms.

**Change.** The revision provides an action-based QUMOND embedding for an independently specified \(B\). With

\[
z=\frac{|\nabla\Phi_N|^2}{(g^\dagger)^2},\qquad
Q(z,B)=z+4B\left[z^{1/4}-\arctan(z^{1/4})\right],
\]

the derivative satisfies \(Q_z=1+B h(g_N)\), and variation produces the nonrelativistic field equations. The paper states exactly where this construction stops: it embeds the empirical response but does not derive \(B\), coherence, a relativistic completion, or photon propagation. Noether conservation is claimed only for a closed action in which every dynamical field is varied.

**Resolution status.** The absence of any action has been addressed. Reviewer 2's deeper request for a first-principles physical origin remains unresolved and is acknowledged rather than overstated.

The Discussion also adds a deliberately limited comparison with published Refracted Gravity work. That literature is cited as an example of how density and source geometry can enter a more developed field theory, but the revised paper explicitly states that it neither derives nor validates the Σ-Gravity response. It is future theoretical context, not evidence for the present model.

### 16. Parameter identifiability

**Reviewer 2 concern.** The relationship among \(n\), \(L_0\), \(A_0\), and the SPARC result was unclear, leaving hidden parameter freedom.

**Change.** The revised paper derives the deep-acceleration relation

\[
V^4\rightarrow B^2GM_b g^\dagger,
\]

showing that disk normalization constrains \(B^2g^\dagger\), not \(B\) and \(g^\dagger\) independently. It states that a constant \(L\) across Fox clusters cannot identify \(n\), and that \(L_0\), \(n\), and \(R_d\) do not enter the locked galaxy result. Table 2 distinguishes fixed choices, calibrated quantities, sensitivities, and assumptions.

**Resolution status.** Addressed at the empirical-identifiability level. The constants are not claimed to have unique first-principles derivations.

### 17. Cosmological context

**Reviewer 2 concern.** A model presented as an alternative to dark matter must address the CMB, primordial abundances, and large-scale structure.

**Change.** The revised Introduction explicitly acknowledges the cosmological evidence supporting ΛCDM. The Discussion and Conclusions state that no cosmological extension is developed and list the required future domains: the CMB spectrum, primordial abundances, perturbation growth, nonlinear structure formation, and cluster abundance evolution. “Alternative” is limited to a galaxy-scale empirical choice among this response, MOND, or a fitted halo.

**Resolution status.** The paper no longer makes a cosmological replacement claim. It does not solve the cosmological problem.

### 18. Statistical consultation

**Reviewer 2 concern.** The checklist indicated that a statistician should evaluate the study.

**Change.** The statistical design has been substantially upgraded, including independent-unit resampling, paired intervals and tests, frozen nuisance analyses, sample-threshold checks, matching diagnostics, and machine-readable residuals and split definitions.

**Resolution status.** The methodological substance is addressed through reproducible object-level analyses, but the request for an external statistician is not fulfilled. The response says this directly and does not promise a consultation.

## Concerns that were answered by limitation rather than solved

Several reviewer requests cannot honestly be described as fully solved within this revision:

1. **First-principles mechanism:** the auxiliary QUMOND action reproduces the response for prescribed, spatially constant \(B\), but it is not an action for the endogenous galaxy prescription and no dynamical equation derives \(B\) from matter organization.
2. **Relativistic completion:** gravitational slip, photon propagation, PPN parameters, gravitational waves, and compact-source behavior remain undetermined.
3. **Universal cluster law:** the external CLASH result reveals radius-dependent biased transfer of the fixed Fox amplitude under the current assumptions.
4. **Cluster baryon reconstruction:** the Fox calibration still uses a simplified aperture-mass proxy.
5. **Direct counterrotation test:** the matched catalog result is null and still uses a model-derived dark-matter fraction; direct map-level forward modeling remains to be done.
6. **Measured coherence:** the locked factor is endogenous, not an independent phase-space observable.
7. **General path length:** no unique functional for arbitrary three-dimensional baryonic distributions is supplied.
8. **Cosmology:** no CMB, primordial-abundance, or structure-formation calculation is attempted.
9. **External statistical review:** the methods were improved and made reproducible, but no independent statistician was retained; the response does not claim otherwise.

These are not omitted from the revised paper. They define its scope and the next decisive tests. Revised Section 4.7, “Staged future investigations,” now orders them into conditional steps with explicit advancement criteria: independently measured coherence must improve held-out galaxy predictions; a cluster law must transfer without refitting under frozen baryon and covariance models; a physical $B$ field must arise from a closed, stable action; and relativistic or cosmological development remains contingent on those earlier tests. The reviewer responses reproduce the same distinction between completed revisions and future investigations.

## Concise submission description

The revised manuscript retains the original empirical acceleration response and locked galaxy predictor but substantially narrows their interpretation. Coherence and path length are now presented as possible physical motivations rather than established dependencies, and \(A(L)\) is removed from the predictive response because no operational three-dimensional path functional was established. The Fox cluster result is relabeled as an in-sample calibration under a simplified baryon and lensing closure, and a new no-refit CLASH profile check reveals a cluster-bootstrap-supported radial overprediction. The SPARC comparison is reanalyzed with galaxy-level paired uncertainty, error weighting, sample-threshold sensitivity, and a frozen nuisance grid; it does not resolve a statistically significant aggregate difference from the tested MOND prescription, and no equivalence claim is made. Actual SPARC photometric scale lengths are tested without refitting and do not outperform permutation or fixed-scale controls. The algebraic disk approximation is compared with numerical fixed-\(B\) QUMOND solutions. The original counterrotation confirmation is replaced by a balanced matched-control analysis consistent with zero. Unsupported covariance, conservation, Solar-System, gravitational-slip, and cosmological claims are removed or explicitly bounded. An auxiliary nonrelativistic QUMOND action reproduces the response for prescribed, spatially constant \(B\), while the origin and dynamics of \(B\) remain an open first-principles problem.

## Bottom line

The revised paper's defensible contribution is narrower but stronger: it presents a fixed-form galaxy response for which the present SPARC comparison does not resolve a statistically significant aggregate difference from the tested MOND prescription, quantifies the contribution and limitations of its bounded galaxy factor, supplies an auxiliary fixed-\(B\) nonrelativistic action construction, and reports rather than conceals the failure of the submitted cluster amplitude to transfer without radial bias. It does not claim equivalence, that coherence has been demonstrated, that cluster lensing is solved, that counterrotation confirms the model, or that Σ-Gravity replaces ΛCDM cosmology.
