# Response to Reviewer 1

**Manuscript ID:** 1866133  
**Manuscript title:** *Σ-Gravity: A Coherence-Motivated Empirical Response Tested in Galaxies and Clusters*

I thank the Reviewer for the constructive report. I agree with the central concern that the submitted manuscript stated several theoretical and observational conclusions more strongly than the evidence allowed. I have retained the empirical response while revising the relevant passages to distinguish fixed assumptions, calibration, no-refit evaluation, and unresolved theoretical questions. I revised the title from “Coherence-Dependent Gravitational Enhancement” to “A Coherence-Motivated Empirical Response Tested” so that coherence is identified as the proposed interpretation and galaxies and clusters are identified as test regimes, without implying that the present data establish a physical dependence or successful performance in both regimes.

I apologize for the time taken to provide this response. I used the interval to complete the analyses reported in this revision and to make the manuscript changes needed to reflect the reviewers' feedback accurately.

The most important changes are:

- Σ-Gravity is now presented as a nonrelativistic phenomenological alternative for selected missing-mass observations, not as a completed fundamental or cosmological theory.
- The title now describes a coherence-motivated empirical response tested in galaxies and clusters rather than an established coherence dependence or a successful application in both regimes.
- Coherence and path length are hypotheses rather than established causes of the response.
- The Fox cluster result is explicitly labeled an illustrative calibration and is no longer presented as a positive cluster result in the Abstract.
- A no-refit CLASH profile check is reported separately and reveals radial bias.
- The manuscript now explains the provenance and non-uniqueness of \(h(g_N)\), \(g^\dagger\), \(A_0\), \(F\), and \(\sigma\), and defines exactly what “locked” means.
- The action section now states explicitly that the fixed-\(B\) action is not an action for the endogenous galaxy prescription.
- The actual SPARC photometric scale lengths are tested in a zero-parameter radial-window sensitivity with permutation and fixed-value controls.
- The algebraic disk calculation is compared with numerical axisymmetric QUMOND solutions.
- The counterrotation claim is replaced by a balanced matched-control analysis whose interval includes zero.
- Lensing, Solar-System, equivalence-principle, and cosmological claims are limited to what the present equations calculate.

For clarity, the responses below distinguish work completed in this revision from investigations that remain open. I do not present an open item as resolved by relabeling it “future work”; where the requested calculation could not be completed within the present nonrelativistic framework, I state the limitation and the observation or derivation that would be required to resolve it.

## 1. Theoretical basis of coherence, path length, covariance, and conservation

**Reviewer comment.** The coherence scalar, path length \(L\), and \(A(L)\) are phenomenological; the manuscript does not justify its claims about covariance, conservation, or equivalence principles.

**Response.** I agree. The revision separates the observable response \(B\) from its possible interpretation. The data identify \(B=A\mathcal C\), not \(A\) and a physical coherence field separately. The implemented galaxy factor is described as an endogenous phenomenological regularizer because it uses the model-predicted velocity.

I also added a nonrelativistic QUMOND action for an independently prescribed, spatially constant \(B\). The final text states explicitly that Equation (10) reproduces the prescribed fixed-\(B\) response only; it is not an action for the endogenous rotational-support prescription in Equation (6), and it does not establish the conservation or dynamical consistency of that prescription. It does not derive \(B\). If \(B\) is inferred from observed or predicted kinematics, a closed theory would have to vary it as an independent field with its own source, kinetic, and backreaction terms before Noether-conservation claims could be made.

The final manuscript also gives the provenance of the fixed empirical choices. The square-root part of \(h(g_N)\) supplies the deep-acceleration scaling, while its rational gate suppresses the response at high acceleration; this constrains the asymptotes but does not make the exact transition unique. The geometric factor in \(g^\dagger\), the mode-counting motivation for \(A_0\), and the characteristic-dispersion motivation for \(\sigma=20\ {\rm km\,s^{-1}}\) are now labeled heuristics rather than derivations.

**Before.** The submitted manuscript presented a covariant coherence scalar and suggested that the construction supported conservation and equivalence-principle conclusions.

**After.** Those claims are withdrawn. The title now says “A Coherence-Motivated Empirical Response Tested” rather than “Coherence-Dependent Gravitational Enhancement,” and coherence is described as a possible physical motivation requiring independent phase-space data and its own dynamical field equation.

**Location.** Revised manuscript p. 1, title, Abstract, and Sec. I.C; pp. 2–3, Secs. II.B–II.D and Eqs. (4)–(11); pp. 3–4, Sec. II.F; pp. 7–8, Sec. V.B; and p. 8, Sec. V.D.

**Remaining limitation.** No covariant coherence field, relativistic completion, or first-principles source equation for \(B\) is claimed.

## 2. Claimed reduction of the covariant scalar

**Reviewer comment.** The manuscript states that the covariant expression reduces to the observational coherence equation but does not demonstrate the reduction.

**Response.** I agree and have removed that claimed reduction. The revised text gives only a schematic second-moment motivation for the bounded galaxy factor and lists the frame, anisotropy, inclination, and endogeneity limitations.

**Before.** A general covariant reduction was asserted.

**After.** No covariant reduction is claimed; the factor is explicitly phenomenological.

**Location.** Revised manuscript pp. 2–3, Sec. II.C, “Galaxy Implementation and Status of Coherence,” especially Eqs. (5)–(6) and the paragraphs immediately following them.

## 3. Fox clusters are calibration, not prediction

**Reviewer comment.** The exponent was calibrated on the same 42 clusters used to quote the median ratio 0.987; random holdouts are not independent validation.

**Response.** I agree. The 42 Fox clusters are now labeled an illustrative calibration sample, and 0.987 is called an in-sample calibration result. Repeated Fox splits are described only as calibration-stability checks. The final Abstract no longer presents the near-unity Fox ratio as a positive cluster result; it states instead that the simplified calibration is secondary and that its normalization fails to transfer without radial bias.

To answer the request for a no-refit check, I used the 20 Tian/CLASH systems as the starting catalog and applied an objective name-normalization rule: uppercase alphanumeric identifiers with punctuation removed. This excludes the three systems also present in Fox---MACS0416, MACS0717, and MACS1149---leaving 73 radial measurements in 17 disjoint clusters. I froze the cluster calculation at \(B=8.45\), \(g^\dagger=9.60\times10^{-11}\ {\rm m\,s^{-2}}\), and the stated response \(h(g_{\rm bar})\). The median predicted-to-observed ratio is 1.318. The radius-bin medians are 1.170, 1.228, 1.613, and 1.961 at 100, 200, 400, and 600 kpc. A weighted residual trend is \(+0.162\) dex per decade in radius, with a 95% cluster-bootstrap interval \([+0.115,+0.217]\). The Fox-calibrated amplitude therefore does not transfer successfully to these radial data and provides no support for a universal fixed cluster-amplitude law under the present assumptions. I report this unfavorable transfer result rather than recalibrating the manuscript to it.

**Before.** The Fox ratio and within-catalog holdouts were described as prediction or validation.

**After.** The Fox calculation is visibly secondary and illustrative, calibration and no-refit evaluation are separated, and no replacement cluster amplitude or formula is introduced.

**Location.** Revised manuscript p. 4, Sec. III.C; p. 6, Sec. IV.C; Table I on p. 5; and Figure 2 on p. 7.

**Remaining limitation.** The CLASH analysis is a no-refit profile check, not a complete joint baryon/lensing likelihood on a new survey. A definitive validation still requires independently measured gas, stellar, intracluster-light, and lensing profiles with covariance fixed in advance.

## 4. Simplified cluster baryonic masses

**Reviewer comment.** The universal baryon fraction and fixed 0.4 concentration factor are oversimplified and materially affect the main cluster result.

**Response.** I agree. I was not able to complete the requested component-by-component baryonic reconstruction for the 42 Fox systems, and I do not present the Fox exercise as satisfying that request. Equation (15) is now labeled a simplified baryon-concentration prescription used only to reproduce the submitted calibration. The manuscript reports that changing the 0.4 factor by 25% changes the predicted mass ratio by approximately 30%. Because measured gas, brightest-cluster-galaxy, satellite, and intracluster-light profiles were not available for all 42 systems, I removed this exercise as a principal positive test and retained it only to document the submitted calibration and motivate the independent no-refit transfer check. It is now described as illustrative in the Abstract, Results, Discussion, Conclusions, Table I, Table II, and Figure 2.

**Before.** The baryon proxy supported a precision cluster-success claim.

**After.** The Fox result is explicitly conditional on the proxy, is not used as precision evidence for a universal law, and is not treated as a principal positive test of the framework.

**Location.** Revised manuscript p. 4, Sec. III.C and Eq. (15); p. 6, Sec. IV.C; and Table II on p. 5.

**Remaining limitation.** I have not reconstructed gas, brightest-cluster-galaxy, satellite, and intracluster-light profiles for all 42 Fox systems. The text identifies this as required future work.

The revised title describes the response as “Tested in Galaxies and Clusters.” This retains the scope of the prespecified cross-system transfer test while avoiding an implication that the cluster application succeeded. The Introduction and Abstract make the unfavorable cluster result explicit.

## 5. No operational definition of path length \(L\)

**Reviewer comment.** Characteristic values are assigned by system class, but \(L\) cannot be calculated uniquely from a general three-dimensional baryonic distribution.

**Response.** I agree. Because every Fox cluster received the same \(L=600\) kpc, the sample identifies one effective amplitude, not the exponent \(n\) independently. The reported \(n=0.27\) is therefore only a historical reparameterization of the Fox-calibrated amplitude for the adopted reference constants; it is not used in the canonical response or claimed as an independently measured law. Because the available data do not identify a general path-length functional, I removed the relation from the canonical response rather than replacing it with a more flexible fitted law.

**Before.** \(A(L)\) was presented as a cross-scale law connecting disks and clusters.

**After.** I could not supply a unique operational definition of \(L\) from a general baryonic distribution. I therefore removed \(A(L)\) from the predictive response and removed conclusions that used it to connect disks and clusters. Path length is retained only as a possible motivation for future work; no substitute formula is fitted in this revision.

**Location.** Revised manuscript pp. 2–3, Sec. II.C; pp. 3–4, Sec. II.F; Table II on p. 5; and pp. 7–8, Sec. V.B.

## 6. Actual SPARC photometric scale lengths

**Reviewer comment.** The analysis should use actual SPARC photometric scale lengths.

**Response.** I agree that the earlier wording was not sufficient. My first code audit established that the locked production predictor uses the fixed-point factor \(F(V_\Sigma)\) and does not use \(R_d\). I initially corrected the documentation by separating \(R_d\), \(L_0\), and \(n\) from the canonical equation. I then performed the literal no-refit test requested by the Reviewer, using each galaxy's catalog photometric scale length in

\[
W(r,R_d)=\frac{r}{R_d/(2\pi)+r},\qquad B_{R_d}=A_0W(r,R_d),
\]

with the same baryonic inputs, the same 164-galaxy sample, fixed \(A_0\), and no fitted parameter. This candidate replaces the endogenous factor rather than multiplying it, so it isolates the proposed scale-length dependence without adding a new degree of freedom.

The mean galaxy RMS is \(16.513\ {\rm km\,s^{-1}}\) for the catalog-scale-length candidate and \(16.366\ {\rm km\,s^{-1}}\) for the locked predictor. The paired difference is \(+0.147\ {\rm km\,s^{-1}}\), with a galaxy-bootstrap 95% interval \([-0.088,+0.385]\ {\rm km\,s^{-1}}\). This interval does not resolve a statistically significant paired difference; it is not an equivalence test. The catalog window does not improve the mean result.

I also tested whether the actual galaxy-to-galaxy assignments contain information. Across 2,000 permutations of \(R_d\), the actual assignment has one-sided \(p=0.711\) for outperforming a random assignment. Fixing every galaxy to the sample-median \(R_d=2.3\) kpc gives a lower mean RMS of \(16.413\ {\rm km\,s^{-1}}\). On all 171 usable galaxies and 3,373 valid points, the locked and catalog-window mean RMS values are 17.415 and 17.578 km s\(^{-1}\), respectively. These controls do not support the measured photometric scale length as the controlling variable in this particular window.

**Before.** The manuscript discussed a photometric/path-length window as if it entered the locked prediction.

**After.** The manuscript preserves Equation (6) as the canonical predictor, defines and reports the catalog-\(R_d\) candidate as a separate structural sensitivity, and states that \(L_0\) and \(n\) are not used for the galaxy result. The negative controls are reported so the result does not imply support for a measured scale-length mechanism.

The window \(W=r/[R_d/(2\pi)+r]\) is the literal scale-length form already documented with the submitted research model; it was not chosen after reviewing these results, and I did not screen alternative windows in this reviewer-requested test.

**Location.** Revised manuscript pp. 2–3, Sec. II.C; p. 4, Secs. III.A–III.B; p. 5, Tables I–II; p. 6, Sec. IV.B; and pp. 7–8, Sec. V.B. The [archived scale-length outputs](https://github.com/lrspeiser/sigmagravity/tree/main/Publications/Frontiers/analysis/sparc_scale_length) provide the machine-readable audit.

**Interpretation.** The test shows that radial suppression can improve on the acceleration-only ablation, but the actual catalog \(R_d\) assignments do not add detectable information through this window. I therefore retain source geometry as a future physical hypothesis without incorporating this unsuccessful candidate into the core formula.

## 7. Algebraic field approximation

**Reviewer comment.** The algebraic field approximation requires justification.

**Response.** I agree. The relation \(g_{\rm eff}\simeq\Sigma g_N\) is now explicitly labeled an approximation. I compare it with numerical three-dimensional QUMOND solutions for analytic axisymmetric disk reconstructions representative of F574-2, UGC05716, and NGC3741. The numerical check fixes \(B=1\), solves the two Poisson equations with a periodic FFT on a \(65^3\) grid in a box extending to \(8R_d\), uses an exponential--\(\mathrm{sech}^2\) disk of scale height \(0.2R_d\), and compares radial accelerations over \(0.75\le r/R_d\le5\). A \(49^3\)-to-\(65^3\) resolution comparison gives a median discrepancy of 0.72% and maximum discrepancy of 3.67% over the reported range.

Median absolute acceleration differences are 5.19%, 4.88%, and 3.96%, with local maxima of 20.54%, 18.30%, and 7.73%, respectively. These are fixed-\(B\) field-geometry checks, not solutions of the endogenous Equation (6) model.

**Before.** The algebraic relation was used without a quantified nonspherical-field error.

**After.** The error is quantified and treated as a geometry-dependent model systematic.

**Location.** Revised manuscript p. 3, Sec. II.E and Eq. (12); p. 6, Sec. IV.D; and Figure 3 on p. 8.

**Remaining limitation.** The three solutions use analytic reconstructions rather than full observed gas and bulge maps.

## 8. Counterrotation controls and the ΛCDM statement

**Reviewer comment.** The unmatched populations may differ in mass, morphology, size, inclination, environment, merger history, and data quality; a matched control or multivariate analysis is needed. The statement that ΛCDM predicts no difference is too strong.

**Response.** I agree. I first retained one MaNGA observation per physical galaxy, choosing the lowest JAM \(\chi^2\) where duplicate observations existed. I then matched 62 counterrotators to 310 unique controls on stellar mass, physical size, Sérsic index, axis ratio, inclination, redshift, and JAM fit quality. Matching used greedy nearest neighbors on jointly standardized covariates, processed the hardest-to-match case first, selected five controls per case without replacement, and used no caliper. Eligibility required complete matching and outcome fields, a nonnegative quality flag, and positive \(R_e\); no additional common-support trimming was imposed, and all 62 eligible cases were matched. The maximum post-match absolute standardized mean difference is 0.066, below the prespecified 0.1 threshold.

The matched difference in JAM/NFW-derived \(f_{\rm DM}(<R_e)\) is \(-0.0081\), with bootstrap 95% interval \([-0.0577,+0.0453]\). The bootstrap resamples the 62 complete matched sets and retains each five-control mean. The result is consistent with zero. I removed “confirmed prediction,” the unmatched headline \(p\)-value, and the claim that ΛCDM predicts no difference.

**Before.** A large unmatched association was described as confirmation.

**After.** The matched null result is reported as a secondary diagnostic, and counterrotation is retained only as a proposed direct future test.

**Location.** Revised manuscript p. 4, Sec. III.D; p. 6, Sec. IV.E and Eq. (18); and Figure 4 on p. 8.

**Remaining limitation.** Environment and merger-history covariates were not available in the local catalog. A direct test requires common forward modeling of the MaNGA velocity and dispersion maps.

## 9. Solar-System discussion

**Reviewer comment.** Assigning the Sun \(\mathcal C=0\) is imposed rather than derived.

**Response.** I agree and removed that assignment. The revision states only that the empirical acceleration function is small at high Newtonian acceleration and that this is not a PPN calculation.

**Before.** The manuscript claimed Solar-System compatibility from an imposed coherence value.

**After.** Solar-System, compact-source, equivalence-principle, and PPN behavior are explicitly undetermined.

**Location.** Revised manuscript pp. 3–4, Sec. II.F, final paragraph; and p. 8, Sec. V.D.

## 10. Lensing and gravitational slip

**Reviewer comment.** Equality of lensing and dynamical mass was assumed by setting gravitational slip to zero; a modified-gravity theory must derive the two metric potentials.

**Response.** I agree. The revised manuscript writes the weak-field metric and states that nonrelativistic dynamics constrains \(\Psi\), while lensing depends on \(\Phi+\Psi\). The present action does not determine \(\eta=\Phi/\Psi\).

**Before.** The cluster comparison was described as a lensing prediction.

**After.** It is an empirical mass comparison conditional on a lensing closure assumption.

**Location.** Revised manuscript pp. 3–4, Sec. II.F; Table II on p. 5; and p. 8, Sec. V.D, including Eq. (19).

## 11. Overstated claims

**Reviewer comment.** Phrases such as “successfully predicts cluster lensing masses,” “confirmed prediction,” and “supports the framework’s validity” exceed the evidence.

**Response.** I agree and removed them.

| Submitted wording | Revised wording |
|---|---|
| “successfully predicts cluster lensing masses” | “calibrates the Fox aperture-mass ratio under a simplified baryon prescription” |
| “independent validation” from Fox holdouts | “within-catalog calibration stability” |
| “confirmed prediction” from counterrotation | “matched secondary diagnostic consistent with zero” |
| “supports the framework’s validity” | “motivates further prespecified testing” |
| “alternative to dark matter” without qualification | “galaxy-scale phenomenological alternative; not a replacement cosmology” |

**Location.** Revised manuscript p. 1, Abstract and Sec. I.C; pp. 5–8, Secs. IV.A–IV.E and Figures 1–4; pp. 6–9, Sec. V; and p. 9, Sec. VI.

## 12. Figures and tables

**Reviewer comment.** Figure and table quality was not satisfactory.

**Response.** The revision replaces the main comparison graphics with four vector figures generated from frozen machine-readable outputs. Figure 1 shows the paired SPARC analysis and nuisance grid; Figure 2 visually separates Fox calibration from the no-refit CLASH check; Figure 3 shows the algebraic-field error; and Figure 4 shows counterrotation balance and the matched interval. Tables I and II distinguish dataset roles and parameter/assumption status.

The figures use distinct colors and marker shapes, remain interpretable in grayscale, and are also supplied as 360-dpi PNG files.

**Location.** Revised manuscript Tables I–II on p. 5; Figures 1–2 on p. 7; and Figures 3–4 on p. 8.

## 13. Statistical methods

**Reviewer comment.** The statistical methods were not valid or correctly applied.

**Response.** The revised analysis treats galaxies or clusters, not repeated radii, as independent resampling units. For SPARC, it reports a paired effect size, 20,000 galaxy-bootstrap interval, paired sign-flip test, and exact object-level binomial comparison. There were no tied galaxy-level RMS values, and the exact binomial null is \(p=0.5\). The mean Σ-minus-MOND RMS contrast is \(+0.309\ {\rm km\,s^{-1}}\), with 95% interval \([-0.040,+0.659]\). This analysis does not resolve a statistically significant aggregate difference and does not establish equivalence or noninferiority.

The revision also:

- freezes an 81-combination nuisance grid rather than selecting assumptions after the result;
- repeats the SPARC comparison with observational-error weighting, giving a paired mean of \(+0.193\ {\rm km\,s^{-1}}\) and bootstrap 95% interval \([-0.177,+0.564]\), which likewise does not resolve a significant aggregate difference;
- repeats the sample construction at 20%, 30%, and 40% bulge thresholds and on all 171 usable galaxies; the mean Σ-minus-MOND contrast remains positive in all four samples;
- tests the actual photometric scale-length assignments against 2,000 galaxy-level permutations and a fixed-median negative control;
- separates cluster calibration from no-refit evaluation;
- groups radial cluster measurements by system;
- reports matched covariate balance and a galaxy-bootstrap interval for counterrotation; and
- preserves split definitions and residuals in machine-readable files.

**Before.** Point-level and unmatched comparisons supported stronger claims than their sampling structure allowed.

**After.** Object-level paired statistics and explicit uncertainty intervals determine the wording.

**Location.** Revised manuscript p. 4, Sec. III.B; pp. 5–6, Secs. IV.A–IV.C and IV.E; Table I on p. 5; Figures 1–2 on p. 7; and Figure 4 on p. 8.

## Defined future investigations

The revised manuscript now consolidates the unresolved work in Sec. V.E, “Staged Future Investigations.” The sequence is designed to prevent a more flexible formula from substituting for an independent test:

1. **Measured coherence.** Forward-model MaNGA velocity and dispersion maps and evaluate an independently measured phase-space estimator on complete held-out galaxies. An acceleration-plus-coherence model advances only if it improves on the acceleration-only response by a prespecified margin with a stable sign across morphology and mass.
2. **Cluster transfer.** Assemble a disjoint sample with radial X-ray gas, brightest-cluster-galaxy, satellite, intracluster-light, and lensing covariance fixed before inference. A cluster response is called predictive only if it transfers without refitting across systems and radii.
3. **Path length and source geometry.** Derive a density-, concentration-, or geometry-sourced field without system-class switches or per-object parameters, then compare it with constant \(B\) on held-out systems. The unsuccessful SPARC \(R_d\) window is not promoted.
4. **Exact field solutions.** Replace the algebraic disk approximation with axisymmetric or three-dimensional solutions using observed gas, stellar, and bulge maps and propagate the resulting field-solution uncertainty through the galaxy likelihood.
5. **Fundamental and relativistic completion.** Promote \(B\) to an independently varied field with explicit kinetic and source terms, test stability and conservation in a closed action, and only then derive the two metric potentials, photon propagation, PPN limits, equivalence-principle behavior, and gravitational-wave propagation.
6. **Confirmatory inference.** Freeze samples, covariance models, and decision thresholds before evaluation; preserve public code, machine-readable residuals, and complete split definitions for independent replication.

These items are proposed tests with failure conditions, not additional results claimed by the present manuscript.

**Location.** Revised manuscript pp. 8–9, Sec. V.E, “Staged Future Investigations.”

## Closing response

The revision does not change the core empirical response to obtain a more favorable result. Instead, it narrows the interpretation:

- the galaxy comparison does not resolve a statistically significant aggregate difference from the tested MOND prescription under common baryonic assumptions, and no equivalence claim is made;
- the Fox cluster result is calibration;
- the external radial cluster check limits universality;
- the matched counterrotation result does not confirm coherence; and
- the tested photometric scale-length window does not outperform its controls, so coherence, path length, and source organization remain hypotheses requiring further work.

Coherence remains the principal physical motivation for the bounded factor. The internal ablation supplies an initial reason to test that hypothesis, but the response now makes clear that only independently measured phase-space data and held-out prediction can determine whether coherence is the cause.

I thank the Reviewer for identifying the changes needed to separate an interesting phenomenological finding from conclusions that the current theory and data cannot yet support.
