# Σ-Gravity: A Coherence-Motivated Empirical Response Tested in Galaxies and Clusters

**Article type:** Original Research  
**Journal/section:** *Frontiers in Astronomy and Space Sciences* — Extragalactic Astronomy  
**Manuscript ID:** 1866133

**Leonard Speiser¹\***  
¹ Horizon 3, Independent Research, Los Altos, CA, United States  
\* Correspondence: leonard@horizon3.net  
ORCID: [0009-0008-8797-2457](https://orcid.org/0009-0008-8797-2457)

**Keywords:** galaxy rotation curves; radial acceleration relation; modified gravity; dark matter; galaxy clusters; gravitational lensing; SPARC; phenomenology

## Abstract

The dynamics of galaxies and galaxy clusters exceed the predictions obtained from directly observed baryonic matter, a discrepancy conventionally modeled with nonbaryonic dark matter or, at galaxy scales, phenomenological modifications such as modified Newtonian dynamics (MOND). We investigate Σ-Gravity as an additional nonrelativistic phenomenological approach to selected missing-mass observations, not as a disproof of dark matter or a cosmological replacement for ΛCDM. Its empirical response is \(\Sigma(g_N,B)=1+B h(g_N)\), where \(h(g_N)=\sqrt{g^\dagger/g_N}\,g^\dagger/(g^\dagger+g_N)\) and \(g^\dagger=cH_0/(4\sqrt{\pi})\); the locked galaxy implementation supplements this acceleration dependence with a bounded factor motivated by rotational support. For 164 SPARC galaxies after excluding bulge-dominated radial measurements, using fixed stellar mass-to-light ratios and no per-galaxy halo fitting, the mean velocity root-mean-square residual is \(16.366\ {\rm km\,s^{-1}}\) for Σ-Gravity and \(16.056\ {\rm km\,s^{-1}}\) for the MOND prescription tested with the same baryonic inputs. The paired analysis does not resolve a statistically significant difference between these prescriptions under the tested assumptions. Relative to the acceleration-only form, the bounded factor improves the mean RMS by \(0.517\ {\rm km\,s^{-1}}\), with a galaxy-bootstrap 95% interval of \([0.242,0.795]\ {\rm km\,s^{-1}}\). Thus the factor makes a measurable contribution within the locked predictor, although its endogenous construction does not establish independently measured coherence. The cluster exercise is secondary and illustrative: 42 Fox et al. strong-lensing systems determine one effective amplitude under a simplified baryon proxy, while a no-refit check using 73 radial measurements in 17 CLASH clusters not present in the Fox calibration sample gives a median predicted-to-observed ratio of 1.32 and reveals a radius-dependent bias. The Fox normalization therefore does not transfer successfully as a universal cluster law under the present assumptions. An auxiliary QUMOND action reproduces the response for independently prescribed, spatially constant \(B\), but not the endogenous galaxy prescription. Together, these results identify a fixed-form galaxy response whose aggregate difference from the tested MOND prescription is unresolved and expose a falsifiable limitation of its fixed cluster amplitude. Kinematic coherence remains the principal motivating hypothesis for the physical origin of \(B\), with independently measured phase-space order providing a defined next test; it is not established by the present analysis. No relativistic or cosmological extension is developed here.

## 1 Introduction

Galaxy rotation curves, galaxy-cluster dynamics, gravitational lensing, the cosmic microwave background, and large-scale structure collectively motivate the standard ΛCDM framework (Planck Collaboration, 2020). Within that framework, nonbaryonic cold dark matter supplies the dominant gravitating mass in galaxies and clusters and plays a central role in cosmological structure formation. The microscopic identity of dark matter nevertheless remains unknown, and individual galaxy analyses commonly require system-specific halo parameters. These facts do not negate the cosmological evidence for dark matter, but they motivate tests of compact empirical relationships between baryonic structure and the observed gravitational discrepancy.

MOND provides the best-known alternative phenomenology at galaxy scales (Milgrom, 1983; Sanders and McGaugh, 2002; Famaey and McGaugh, 2012). It relates the effective and Newtonian accelerations through a universal acceleration scale and organizes many disk-galaxy observations, including the radial acceleration relation (McGaugh et al., 2016). Residual mass discrepancies in galaxy clusters and the need for additional structure in relativistic and cosmological realizations motivate the investigation of other phenomenological responses.

Σ-Gravity is evaluated here as an additional phenomenological description of selected missing-mass observations alongside dark-matter and MOND descriptions. Its galaxy implementation combines a fixed acceleration response with a bounded factor motivated by rotational support and uses no per-galaxy response amplitude. For a disk rotation curve, this prescription can be compared directly with a MOND interpolation function or a per-galaxy dark-matter halo fit. The term “alternative” refers only to this empirical galaxy-scale choice; it does not imply that dark matter has been disproved or that its cosmological role has been replaced. Relativistic and cosmological extensions lie outside the present scope.

The observations analyzed here identify the combined response amplitude \(B\), rather than independently separating an amplitude and a physical coherence variable. The implemented galaxy factor is calculated from the model-predicted velocity rather than independent phase-space measurements, and assigning every cluster the same nominal path length identifies one cluster amplitude rather than a length exponent. Coherence remains the principal physical hypothesis motivating the bounded factor, but it is not an established ingredient of the empirical law. The present calculation is an initial test of whether such a factor contributes within the predictor; future work must determine whether independently measured phase-space coherence, source geometry, matter currents, or another property of baryonic organization predicts \(B\) on held-out systems.

This study defines the empirical response and its identifiable parameters, measures the contribution of the bounded galaxy factor through an acceleration-only ablation, gives an auxiliary fixed-\(B\) QUMOND construction, compares the locked galaxy predictor with MOND under common baryonic assumptions, and distinguishes cluster calibration from a no-refit external transfer-failure test. The cluster analysis is retained to expose a limitation, not as a successful application. These analyses preserve a falsifiable phenomenology while specifying what a deeper theory would need to derive.

## 2 Materials and methods

### 2.1 Canonical empirical response and identifiability

Let \(\Phi_N\) be the Newtonian potential of the baryonic density \(\rho_b\),

\[
\nabla^2\Phi_N=4\pi G\rho_b,
\tag{1}
\]

and define \(g_N=|\nabla\Phi_N|\). The canonical empirical enhancement is

\[
\Sigma(g_N,B)=1+B\,h(g_N),
\tag{2}
\]

where

\[
h(g_N)=\sqrt{\frac{g^\dagger}{g_N}}
\frac{g^\dagger}{g^\dagger+g_N},
\qquad
g^\dagger=\frac{cH_0}{4\sqrt{\pi}}
\simeq9.60\times10^{-11}\ {\rm m\,s^{-2}} .
\tag{3}
\]

The numerical value adopts \(H_0=70\ {\rm km\,s^{-1}\,Mpc^{-1}}\).

For \(g_N\ll g^\dagger\), \(h\sim\sqrt{g^\dagger/g_N}\); for \(g_N\gg g^\dagger\), \(h\rightarrow0\). The exact transition function is a design choice rather than a first-principles result: the square-root factor supplies the deep-acceleration scaling associated with flat rotation curves and the baryonic Tully–Fisher relation, while the rational gate suppresses the response at high acceleration. The scale \(g^\dagger\sim cH_0\) places the transition near the empirical galaxy acceleration scale; the geometric factor \(4\sqrt{\pi}\) is retained as a fixed phenomenological choice, not an independently derived constant. This study tests the function and its limiting behavior but does not claim that it is unique among nearby transition laws. In the deep-acceleration point-mass limit,

\[
V^4\rightarrow B^2GM_b g^\dagger .
\tag{4}
\]

The baryonic Tully–Fisher normalization therefore constrains \(B^2g^\dagger\), not \(B\) and \(g^\dagger\) independently. The observations considered here identify only the combined amplitude \(B\); separate interpretations in terms of \(B=A\mathcal C\) are therefore not assumed, and possible meanings of \(A\) and \(\mathcal C\) are deferred to the Discussion.

### 2.2 Galaxy implementation

The locked galaxy implementation uses the bounded fixed-point factor

\[
F(V)=\frac{V^2}{V^2+\sigma^2},
\qquad \sigma=20\ {\rm km\,s^{-1}},
\tag{5}
\]

with

\[
B_{\rm gal}=A_0F(V_\Sigma),\qquad
V_\Sigma=V_{\rm bar}\sqrt{1+B_{\rm gal}h(g_N)},\qquad
A_0=e^{1/(2\pi)} .
\tag{6}
\]

The form \(F=V^2/(V^2+\sigma^2)\) is the simplest bounded ratio suggested by an ordered velocity relative to a second velocity moment. The normalization \(A_0=e^{1/(2\pi)}\) is motivated by a mode-counting heuristic, and \(\sigma=20\ {\rm km\,s^{-1}}\) is adopted as a characteristic cold-disk dispersion scale. These motivations do not constitute first-principles derivations, and the parameters are treated here as fixed model choices.

The formula was developed with SPARC as its principal galaxy benchmark. In this paper, “locked” means that Equations (2), (3), (5), and (6) and their constants define the central configuration, that this configuration was not selected from the sensitivity grid, and that no per-galaxy response parameter is fitted. SPARC is therefore a benchmark evaluation sample, not a prospectively registered or external validation sample. The scale-length, nuisance, ablation, and weighting calculations are sensitivity tests of the same fixed predictor.

The iteration uses \(V_\Sigma\), not the observed velocity, and therefore does not directly fit the observed outcome. However, \(F(V_\Sigma)\) is endogenous to the prediction rather than an independently measured phase-space quantity. It is treated here as a phenomenological regularizer, not as a measured covariant coherence scalar. Writing \(x=V_\Sigma^2\), \(b=V_{\rm bar}^2\), and \(K=A_0h(g_N)\) reduces the fixed point to \(x^2+[\sigma^2-b(1+K)]x-b\sigma^2=0\). The product of its roots is negative, so there is exactly one positive physical solution. Initialization at \(V_{\rm bar}\) reached the \(10^{-6}\ {\rm km\,s^{-1}}\) tolerance for all 2,745 retained points within 26 iterations; no alternative branch is selected by initialization.

The archived production SPARC predictions were generated by the fixed-point Equation (6). The \(R_d\)-based window remained unused in that production path despite earlier documentation describing it as the galaxy predictor. Equation (6) is therefore retained as the locked implementation, and the catalog-\(R_d\) expression is evaluated separately without refitting. As that no-refit structural sensitivity, we test the literal scale-length candidate

\[
W(r,R_d)=\frac{r}{R_d/(2\pi)+r},
\qquad B_{R_d}=A_0W(r,R_d),
\]

with the catalog value of \(R_d\) assigned to each galaxy and all other baryonic and model choices held fixed. This is the previously specified rational window from the submitted model's \(R_d/(2\pi)\) construction; no alternate windows were screened in this sensitivity calculation. The candidate replaces \(F(V_\Sigma)\), is not multiplied by it, and is not used to alter the central Equation (6) result. The reference length \(L_0\) and exponent \(n\) do not enter either galaxy calculation.

A second velocity moment can schematically be decomposed in a stationary axisymmetric system as \(\langle v^2\rangle\simeq v_{\rm rot}^2+\sigma^2\). The ratio \(v_{\rm rot}^2/\langle v^2\rangle\) motivates a bounded measure of rotational support, but it is not a derivation of Equation (5). The decomposition is frame dependent; anisotropic systems require the dispersion tensor; and the implemented velocity is predicted rather than measured. No unique covariant reduction is claimed.

### 2.3 Auxiliary fixed-\(B\) QUMOND action

An auxiliary action can be written in the QUMOND framework (Milgrom, 2010) when \(B\) is independently prescribed and spatially constant. Define

\[
z=\frac{|\nabla\Phi_N|^2}{(g^\dagger)^2}
\tag{7}
\]

and

\[
Q(z,B)=z+4B\left[z^{1/4}-\arctan\!\left(z^{1/4}\right)\right].
\tag{8}
\]

Then

\[
Q_z=1+B\frac{z^{-1/4}}{1+\sqrt z}
=1+B h(g_N).
\tag{9}
\]

The gravitational action density

\[
\mathcal L_g=-\frac{1}{8\pi G}
\left[2\nabla\Phi\cdot\nabla\Phi_N-(g^\dagger)^2Q(z,B)\right],
\tag{10}
\]

together with the matter coupling \(-\rho_b\Phi\), gives by variation

\[
\nabla^2\Phi_N=4\pi G\rho_b
\tag{11}
\]

and

\[
\nabla^2\Phi=\nabla\cdot\left[Q_z\nabla\Phi_N\right].
\tag{12}
\]

This auxiliary construction reproduces the nonrelativistic response for prescribed, spatially constant \(B\). Equation (10) is not an action for the endogenous rotational-support prescription in Equation (6), does not derive \(B\), and does not establish conservation or dynamical consistency for that prescription. If \(B\) is promoted to a dynamical field, its own kinetic and source terms, Euler–Lagrange equation, and backreaction are required. Noether conservation follows for a closed action with the relevant symmetries and all dynamical fields varied; it cannot be inferred by inserting a value of \(B\) from observed or predicted kinematics after variation.

### 2.4 Algebraic rotation-curve approximation

The rotation-curve calculation uses

\[
g_{\rm eff}\simeq\Sigma g_N,\qquad
V_\Sigma\simeq V_{\rm bar}\sqrt{\Sigma}.
\tag{13}
\]

This relation is not exact for a general nonspherical mass distribution because Equation (12) contains a curl-field contribution. We compared the algebraic acceleration with numerical solutions of the fixed-\(B\) QUMOND equation for analytic disk reconstructions representative of F574-2, UGC05716, and NGC3741. The diagnostic uses spatially constant \(B=1\); it does not recompute the endogenous factor from the resulting velocity and therefore tests the geometry error of the prescribed response rather than the full locked galaxy model. A three-dimensional periodic FFT solver uses a \(65^3\) grid, a box half-width of eight disk scale lengths, and an exponential--\({\rm sech}^2\) disk with scale height \(0.2R_d\); comparison radii are \(0.75\)--\(5R_d\). A \(49^3\)-to-\(65^3\) check gives 0.72% median and 3.67% maximum acceleration change. Median absolute acceleration differences between the algebraic and numerical solutions are 5.19%, 4.88%, and 3.96%, respectively; maximum local differences are 20.54%, 18.30%, and 7.73%. These reconstructed models do not include the full observed gas and bulge maps, so the comparison is an approximation-error diagnostic rather than a validation of exact SPARC geometries.

### 2.5 SPARC sample and statistics

SPARC provides near-infrared photometry and resolved H I/Hα rotation curves for 175 disk galaxies (Lelli et al., 2016). The local archive contains 171 usable rotation-curve files. Baryonic velocities are constructed with fixed mass-to-light ratios

\[
\Upsilon_{\rm disk}=0.5,\qquad
\Upsilon_{\rm bulge}=0.7,
\tag{14}
\]

and no galaxy-specific halo or response amplitude is fitted. Because the circular-speed approximation is least secure where a spheroidal bulge dominates, a radial point is excluded from the primary disk analysis when the scaled bulge contribution is at least 30% of \(V_{\rm bar}^2\). Galaxies with fewer than three remaining points are excluded. The locked sample contains 164 galaxies and 2,745 disk-dominated radial points. Sensitivity checks repeat the analysis at 20% and 40% thresholds and with all 171 usable galaxies and all 3,373 valid radial points.

For each galaxy, we calculate an unweighted velocity root-mean-square (RMS) residual for Σ-Gravity and for MOND. This primary statistic gives each retained radius equal weight within a galaxy and each galaxy equal weight in the sample. In the secondary uncertainty-weighted calculation, \(w_{ij}=1/\sigma_{V,ij}^2\) and the within-galaxy statistic is \([\sum_j w_{ij}(V_{{\rm model},ij}-V_{{\rm obs},ij})^2/\sum_j w_{ij}]^{1/2}\); these galaxy-level values are then averaged with equal weight across galaxies. Galaxies, not radial measurements, are the independent resampling units. Uncertainty intervals use 20,000 galaxy bootstrap resamples. We use an exact binomial comparison of the fraction of galaxies for which each model has lower RMS and a paired sign-flip test of the mean RMS difference.

For the scale-length candidate, catalog \(R_d\) values are used without fitting. Two negative controls test whether the galaxy-to-galaxy assignments carry information: 2,000 permutations of \(R_d\) among the 164 galaxies and a common \(R_d\) fixed to the sample median. These calculations test this particular radial window, not every possible dependence on baryonic geometry.

The frozen nuisance grid contains 81 combinations of common disk and bulge mass-to-light ratios, a global distance scale, a global inclination offset, and the \(A_0\) normalization. A separate diagnostic varies \(\sigma\) from 10 to \(50\ {\rm km\,s^{-1}}\). These are sensitivity analyses, not fitted alternatives.

### 2.6 MOND comparison

MOND is evaluated with the same baryonic inputs using

\[
\nu(x)=\frac{1}{1-e^{-\sqrt{x}}},
\qquad x=\frac{g_N}{a_0},
\qquad a_0=1.2\times10^{-10}\ {\rm m\,s^{-2}} .
\tag{15}
\]

The comparison is between two specified galaxy-scale phenomenological prescriptions. It is not a comparison of complete relativistic or cosmological models.

### 2.7 Cluster calibration and no-refit profile check

The Fox et al. (2022) catalog relates strong-lensing strength to cluster properties. The sample used here contains 42 clusters with spectroscopic redshifts and \(M_{500}>2\times10^{14}M_\odot\). The baryon approximation is

\[
M_b(<200\ {\rm kpc})=0.4\times0.15\,M_{500}.
\tag{16}
\]

The factor 0.4 is a simplified baryon-concentration prescription rather than a measured gas-plus-stellar profile and provides a reproducible baseline for the Fox calibration. The calibrated cluster amplitude is \(B_{\rm Fox}=8.446\). Writing this amplitude through a power law with \(L=600\) kpc for every cluster does not identify a path-length exponent because \(L\) does not vary within the sample. In particular, the previously associated value \(n=0.27\) is only a reparameterization of this single calibrated amplitude for adopted reference constants; it is not used in the canonical response and is not independently measured by the cluster data. Repeated train/test splits within Fox refit the same catalog and are described only as calibration-stability checks.

The no-refit radial check uses the Tian et al. (2020) CLASH radial-acceleration catalog. Cluster identifiers were case-normalized, stripped of non-alphanumeric characters, standardized for the documented Abell and MACS aliases, and compared for exact equality with a Fox calibration name. This procedure excluded MACS0416, MACS0717, and MACS1149, leaving 73 measurements in 17 clusters. The Fox-calibrated \(B_{\rm Fox}=8.446\), \(h(g_N)\), and \(g^\dagger\) are fixed before comparison. For the trend in \(r_i=\log_{10}g_{{\rm pred},i}-\log_{10}g_{{\rm tot},i}\), the propagated uncertainty is \(\sigma_{r,i}^2=\sigma_{\log g_{\rm tot},i}^2+[s_i\sigma_{\log g_{\rm bar},i}]^2\), where \(s_i=d\log_{10}g_{\rm pred}/d\log_{10}g_{\rm bar}\), and the regression uses weights \(1/\sigma_{r,i}^2\). Repeated radii are retained within cluster, and complete clusters—not individual radii—are the resampling units in 5,000 bootstrap draws. Refitting this catalog is reported only as a diagnostic of catalog dependence.

### 2.8 Counterrotation diagnostic

The counterrotator catalog is based on the MaNGA sample of Bevacqua et al. (2022), and the secondary dynamical quantities are drawn from MaNGA DynPop (Zhu et al., 2023). Repeat observations are reduced to one row per galaxy by retaining the lowest-JAM-\(\chi^2\) entry. Sixty-two counterrotating systems with complete fields are matched to five unique controls each on stellar mass, physical size, Sérsic index, axis ratio, inclination, redshift, and JAM fit quality. The greedy Euclidean nearest-neighbor procedure standardizes covariates on the combined pool. Before matching, each counterrotator is assigned its minimum standardized Euclidean distance to any eligible control; cases are processed once in descending order of that initial minimum distance. Five controls are then selected per case without replacement and without a caliper, and all 62 eligible cases are retained. Balance was assessed with the absolute standardized mean difference (SMD), using 0.1 as the stated balance threshold. Bootstrap samples resample the 62 complete matched sets while retaining each five-control mean.

The compared outcome, \(f_{\rm DM}(<R_e)\), is inferred within a Newtonian JAM/NFW model. It is not a direct observable and is retained only as a secondary diagnostic. A direct test would require common forward modeling of MaNGA velocity and dispersion maps for Newtonian, acceleration-only, and acceleration-plus-order models, with complete galaxies held out.

### 2.9 Dataset roles and claim boundaries

Table 1 identifies which results are calibration, evaluation, or sensitivity analyses. Radial measurements or spaxels within the same object are not treated as independent galaxies or clusters. The nonrelativistic action does not determine photon propagation, gravitational slip, post-Newtonian parameters, gravitational-wave propagation, or cosmological perturbations. Comparisons with lensing masses therefore require an empirical closure and are not described as derived lensing predictions.

## 3 Results

### 3.1 SPARC comparison

For 164 galaxies and 2,745 disk-dominated radial points, the mean galaxy RMS is \(16.366\ {\rm km\,s^{-1}}\) for Σ-Gravity and \(16.056\ {\rm km\,s^{-1}}\) for MOND. The corresponding pooled point-level root-mean-square errors are \(20.952\) and \(20.363\ {\rm km\,s^{-1}}\).

Define the paired galaxy contrast

\[
\Delta_i={\rm RMS}_{\Sigma,i}-{\rm RMS}_{{\rm MOND},i}.
\tag{17}
\]

Its galaxy-weighted mean is \(+0.309\ {\rm km\,s^{-1}}\), with bootstrap 95% interval \([-0.040,+0.659]\ {\rm km\,s^{-1}}\). Σ-Gravity has lower RMS for 71 of 164 galaxies, or 43.3%, with bootstrap interval 36.0%–50.6%. There are no exact ties; the two-sided exact binomial test uses the null that either prescription has probability 0.5 of a strictly lower galaxy RMS and gives \(p=0.101\). The paired sign-flip \(p\)-value is 0.088. The analysis therefore does not resolve a statistically significant difference between the prescriptions under the tested assumptions; no equivalence or noninferiority claim is made.

The conclusion is sensitive to common nuisance choices. The central configuration remains the primary result and was not selected from the 81-case grid. Across those frozen combinations, including common changes in \(A_0\), the mean paired contrast ranges from \(-3.286\) to \(+2.779\ {\rm km\,s^{-1}}\) and retains the central sign in 64.2% of cases (Figure 1). Changing \(\sigma\) from the central 20 km s\(^{-1}\) to 10, 30, and 50 km s\(^{-1}\) gives mean Σ-Gravity RMS values of 16.695, 16.347, and 17.600 km s\(^{-1}\), respectively, compared with 16.366 km s\(^{-1}\) at 20 km s\(^{-1}\). The 30 km s\(^{-1}\) value is only 0.019 km s\(^{-1}\) lower than the central choice, and no alternative is selected. All four \(\sigma\) choices retain a positive mean Σ-minus-MOND contrast. These sensitivities show that the fixed choices are not uniquely identified and are a reason not to overinterpret the small difference between the prescriptions.

Weighting each galaxy's radial residuals by its reported velocity uncertainties gives mean RMS values of \(15.847\ {\rm km\,s^{-1}}\) for Σ-Gravity and \(15.654\ {\rm km\,s^{-1}}\) for MOND. The paired mean is \(+0.193\ {\rm km\,s^{-1}}\), with bootstrap 95% interval \([-0.177,+0.564]\ {\rm km\,s^{-1}}\), and likewise does not resolve a statistically significant difference. The 20%, 30%, and 40% bulge thresholds and the all-valid-points sample retain a positive mean Σ-minus-MOND contrast; full results are given in Supplement Section S4.

### 3.2 Endogenous-factor ablation

Replacing Equation (6) with the acceleration-only form \(B=A_0\) gives a mean galaxy RMS of \(16.882\ {\rm km\,s^{-1}}\). The locked endogenous factor therefore improves the mean RMS by \(0.517\ {\rm km\,s^{-1}}\), with bootstrap 95% interval \(0.242\)–\(0.795\ {\rm km\,s^{-1}}\). This establishes that the factor changes the predictor; it does not establish an independent coherence effect because \(F\) is calculated from \(V_\Sigma\). The correct interpretation is an internal ablation of the phenomenological equation.

### 3.3 Photometric scale-length hypothesis

Using the measured SPARC scale length in \(B_{R_d}=A_0W(r,R_d)\) gives a mean galaxy RMS of \(16.513\ {\rm km\,s^{-1}}\), compared with \(16.366\ {\rm km\,s^{-1}}\) for the locked predictor. The paired scale-length-minus-locked difference is \(+0.147\ {\rm km\,s^{-1}}\), with bootstrap 95% interval \([-0.088,+0.385]\ {\rm km\,s^{-1}}\). The same candidate improves on the acceleration-only ablation by \(0.370\ {\rm km\,s^{-1}}\), with interval \([0.111,0.620]\ {\rm km\,s^{-1}}\), but is worse than the tested MOND prescription by \(0.456\ {\rm km\,s^{-1}}\), with interval \([0.075,0.843]\ {\rm km\,s^{-1}}\).

The catalog assignments do not outperform random reassignment: their mean RMS lies within the 2,000-permutation distribution, with one-sided \(p=0.711\). A fixed median \(R_d=2.3\) kpc gives \(16.413\ {\rm km\,s^{-1}}\), also slightly better than the galaxy-specific catalog assignments. Thus radial suppression can improve on the acceleration-only response, but this test does not support the measured photometric scale length as the controlling variable in the particular window above. Equation (6) is therefore retained as the locked predictor, while source geometry remains a testable physical hypothesis rather than an established component.

### 3.4 Algebraic approximation error

The numerical QUMOND reconstructions show that the algebraic relation in Equation (13) has a radius-dependent error (Figure 3). The median absolute discrepancy is approximately 4%–5% in the three representative disks, while the largest inner-radius deviation reaches 20.5%. This uncertainty is comparable to or larger than the aggregate RMS separation between Σ-Gravity and MOND for some galaxies. It should be incorporated into future object-level likelihoods rather than treated as an exact identity.

### 3.5 Illustrative Fox cluster calibration

Equation (16) is used to calibrate the common cluster response on the 42 Fox systems, giving a median predicted-to-observed mass ratio of 0.987 and a scatter of 0.132 dex (Figure 2, left). Because the amplitude and simplified baryon proxy are inseparable in this calculation, the exercise is retained only as an illustrative in-sample calibration. Repeated Fox holdouts measure stability under resampling of the same catalog; they do not constitute independent validation.

The baryon prescription is materially influential. Varying the 0.4 concentration factor by 25% changes the predicted mass ratio by approximately 30%. The calibrated amplitude cannot therefore be separated from the assumed baryonic aperture mass.

### 3.6 No-refit CLASH profile check

With \(B_{\rm Fox}=8.446\) held fixed, the 73 measurements in 17 CLASH clusters not present in the Fox calibration sample give a median predicted-to-observed ratio of 1.318 and an RMS residual of 0.188 dex. Disjoint-sample median ratios are 1.17 at 100 kpc, 1.23 at 200 kpc, 1.61 at 400 kpc, and 1.96 at 600 kpc (Figure 2, right). The weighted log-residual slope is \(0.162\) dex per dex in radius, with a 5,000-draw cluster-bootstrap 95% interval \([0.115,0.217]\). Thus the Fox-calibrated amplitude increasingly overpredicts the profile at larger radius under the present baryonic and lensing assumptions.

This no-refit result does not identify a replacement amplitude or a revised cluster formula. It shows that the Fox-calibrated amplitude does not transfer successfully to the CLASH radial data and provides no support for a universal fixed cluster-amplitude law under the present assumptions.

### 3.7 Counterrotation

Matching reduces the maximum absolute SMD from 0.684 to 0.066, below the 0.1 balance threshold (Figure 4). The matched mean difference is

\[
\Delta f_{\rm DM}
=f_{{\rm DM},{\rm counter}}-f_{{\rm DM},{\rm control}}
=-0.0081 ,
\tag{18}
\]

with matched-set bootstrap 95% interval \([-0.0577,+0.0453]\). The result is consistent with zero and does not reproduce the large unmatched association. Because \(f_{\rm DM}\) is JAM/NFW-derived, the matched result is not a direct test of Σ-Gravity. Counterrotation remains a proposed future discriminant rather than a confirmed prediction.

## 4 Discussion

### 4.1 A bounded-factor empirical response

Σ-Gravity is an additional empirical response law for selected missing-mass observations. At galaxy scale, it can be evaluated alongside a fitted halo or a MOND interpolation function using a specified baryon-dependent prescription. The locked predictor has a numerically similar mean RMS to the tested MOND prescription, although the analysis neither establishes equivalence nor resolves a statistically significant difference. No individual response amplitude is fitted to each galaxy. That is an interesting empirical result, but it is not evidence that dark matter is absent or that the model supersedes MOND.

At cluster scale, the current implementation is weaker. The Fox exercise is retained only as an illustrative calibration under a simple baryon prescription, not as central positive evidence. The no-refit CLASH profiles show that this normalization does not transfer successfully and provides no support for a universal fixed cluster-amplitude law under the present assumptions. The Fox value is therefore retained only as a calibration baseline; no replacement cluster formula is introduced.

### 4.2 Coherence and source organization as hypotheses

The empirical response does not identify its physical mechanism. The principal hypothesis motivating the present construction is that the effective response depends on the organization or dynamical state of the baryonic source. Kinematic coherence, or another correlated property of organized matter, could contribute to a dynamical order parameter. The bootstrap-supported internal improvement of the bounded factor over the acceleration-only ablation supports continued investigation of this possibility within the model, while the endogenous construction prevents it from being interpreted as a detection of physical coherence.

For independently measured phase-space data, one operational estimator would be

\[
\mathcal C_{\rm kin}=
\frac{|\bar{\mathbf v}_{\rm stream}|^2}
{|\bar{\mathbf v}_{\rm stream}|^2+
{\rm tr}\,\boldsymbol{\sigma}^2}
\tag{19}
\]

in the system barycentric frame. This quantity would be small for a dispersion-supported cluster, whereas the Fox calibration uses a single effective response rather than an independently measured \(\mathcal C_{\rm kin}\). Therefore one measured coherence definition does not currently explain both disks and clusters. The hypothesis should advance only if independently measured phase-space order improves held-out predictions relative to an acceleration-only model, with a stable sign across mass and morphology.

### 4.3 Path length and source concentration

No unique operational definition of \(L\) for a general three-dimensional baryonic distribution is specified here. Consequently, \(A(L)\) is not used as a predictive cross-system relation. Because every Fox cluster receives \(L=600\) kpc, that sample identifies only an effective cluster amplitude, not the exponent \(n\); the value \(n=0.27\) is therefore not part of the canonical response or an independently measured result. The direct SPARC scale-length sensitivity adds a narrower result: the previously specified radial window benefits from suppression relative to the acceleration-only response, but the actual catalog \(R_d\) assignments carry no detectable advantage over permuted or fixed values. Path length and source concentration are retained only as possible motivations for future work. More general density- or geometry-sourced fields remain distinct hypotheses that require held-out tests.

### 4.4 Relation to Refracted Gravity

A related but distinct future direction is Refracted Gravity, in which a density-dependent gravitational permittivity modifies the Poisson equation and can redirect field lines in nonspherical sources (Cesare et al., 2020, 2022; Sanna et al., 2023). This literature shows how density and source geometry might enter a more developed theory, but it does not derive or validate the empirical Σ response. Any connection would require a separate theoretical and observational study.

### 4.5 Lensing, Solar-System, and relativistic limitations

In a general weak-field metric,

\[
ds^2=-\left(1+\frac{2\Psi}{c^2}\right)c^2dt^2
+\left(1-\frac{2\Phi_{\rm L}}{c^2}\right)d\mathbf x^2,
\tag{20}
\]

Here, the nonrelativistic potential \(\Phi\) in the action and field equation, Equations (10) and (12), corresponds to the dynamical metric potential \(\Psi\); \(\Phi_{\rm L}\) denotes the independent spatial metric potential relevant to lensing. Nonrelativistic dynamics primarily constrains \(\Psi\), while lensing depends on \(\Phi_{\rm L}+\Psi\). The gravitational slip \(\eta=\Phi_{\rm L}/\Psi\) is not determined by Equations (10)–(12). Equality between a phenomenological dynamical mass and a lensing mass is therefore a closure assumption, not a derived result.

At Solar-System accelerations, \(h(g_N)\) is small. This is a useful anomalous-acceleration sanity check, but it is not a calculation of the parameterized post-Newtonian parameter \(\gamma\), and the assumption that a compact source has no additional response has not been derived. Equivalence-principle, post-Newtonian, and gravitational-wave constraints remain open pending a relativistic completion.

### 4.6 Cosmological scope

No cosmological extension is developed here, and the present response law should not be interpreted as an alternative cosmology. Any such extension would need to address the cosmic microwave background spectrum, primordial abundances, perturbation growth, nonlinear structure formation, and cluster-abundance evolution that form central parts of the ΛCDM evidence base.

### 4.7 Staged future investigations

The open limitations define a staged research program in which additional claims depend on independent observables rather than additional formula flexibility. First, MaNGA velocity and dispersion maps should be forward-modeled for counterrotators and balanced controls under common baryonic and nuisance assumptions, with complete galaxies held out. An acceleration-plus-coherence model should advance only if an independently measured phase-space estimator improves galaxy-grouped held-out performance over the acceleration-only response by a prespecified margin and retains a stable sign across morphology and mass.

Second, a disjoint cluster sample should combine radial X-ray gas, brightest-cluster-galaxy, satellite, and intracluster-light profiles with the full lensing covariance fixed before any response amplitude is inferred. A cluster law should be called predictive only if it transfers without refitting across clusters and radii. Candidate density-, concentration-, or geometry-sourced fields should be specified without system-class switches or per-object parameters and compared with a constant-\(B\) baseline on complete held-out systems.

Third, the algebraic disk approximation should be replaced in the likelihood by exact axisymmetric or three-dimensional field solutions using observed gas, stellar, and bulge maps. A first-principles nonrelativistic extension would promote \(B\) to an independent field with explicit kinetic and source terms, vary all fields in a closed action, and test stability and conservation before interpreting \(B\) as coherence or path length. A relativistic completion would then need to derive both metric potentials, photon propagation, post-Newtonian behavior, equivalence-principle limits, and gravitational-wave propagation. Cosmological background and perturbation calculations would be warranted only if these preceding tests succeed. Confirmatory analyses should use frozen samples and covariance models, object-grouped validation, public code and machine-readable outputs, and independent replication.

## 5 Conclusions

Σ-Gravity provides a compact acceleration-based phenomenology that can be considered alongside dark-matter and MOND descriptions of selected missing-mass observations. On a locked sample of 164 SPARC galaxies, the paired analysis does not resolve a statistically significant rotation-curve difference from the MOND prescription tested with the same fixed baryonic assumptions and without fitting an individual halo or response amplitude to each galaxy. This is not an equivalence or noninferiority result. Within the Σ-Gravity predictor, the bounded rotational-support-motivated factor improves the mean RMS over the acceleration-only form by \(0.517\ {\rm km\,s^{-1}}\), with a 95% interval of \([0.242,0.795]\ {\rm km\,s^{-1}}\). This is a statistically supported internal contribution and provides an initial reason to test the coherence hypothesis with independent kinematics, although its endogenous construction does not establish independently measured coherence.

The Fox cluster result is an illustrative in-sample calibration conditional on a simplified baryon model and an assumed relation to lensing mass, not a principal positive test of the framework. A no-refit CLASH profile check reveals an increasing radial overprediction. The calibrated \(B=8.45\) is therefore not established as a universal cluster value, and a path-length interpretation remains a hypothesis.

An auxiliary nonrelativistic QUMOND action reproduces the response for independently prescribed, spatially constant \(B\), but it is not an action for the endogenous galaxy prescription, and the physical origin and dynamics of that amplitude remain unknown. Coherence or another property of baryonic organization is a possible explanation worth testing; it is not established by the current data. The matched counterrotation result is consistent with zero and is reported as a negative secondary diagnostic.

Together, these results document the performance of a fixed-form galaxy response with no per-galaxy amplitude, an auxiliary fixed-\(B\) QUMOND construction, and an external cluster test that reveals radius-dependent biased transfer of the Fox-calibrated amplitude under the present assumptions. They motivate further prespecified testing of Σ-Gravity as a galaxy-scale phenomenological alternative, without claiming to invalidate dark matter or supersede MOND. The next evidentiary step is not a more flexible response law, but a preregistered test using independently measured phase-space order and cluster baryon profiles on held-out systems. Those tests can determine whether coherence or another property of baryonic organization has causal explanatory value, whether the response reflects other new physics, or whether it is a compact description of correlated astrophysical structure; relativistic and cosmological development remains contingent on their outcome.

The measured SPARC scale lengths do not improve the tested radial-window extension over the locked predictor or its negative controls. This negative result preserves the empirical core while narrowing the claim: coherence, scale length, and source organization are candidate explanations for \(B\), not observationally established dependencies.

## Data availability statement

SPARC data are publicly available at [astroweb.cwru.edu/SPARC](http://astroweb.cwru.edu/SPARC/). The CLASH radial-acceleration catalog is available through VizieR as J/ApJ/896/70. The Fox cluster table used for calibration, frozen residuals, split definitions, matched samples, parameter diagnostics, and figure-generation code are provided in the [public repository](https://github.com/lrspeiser/sigmagravity/tree/main/Publications/Frontiers) and Supplementary Material.

## Funding

The author declares that no financial support was received for the research, authorship, and/or publication of this article.

## Conflict of interest

The author declares that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

## Acknowledgments

The author thanks Emmanuel N. Saridakis, Rafael Ferraro, and Tiberiu Harko for earlier discussions concerning theoretical consistency and modified-gravity frameworks.

## References

Bevacqua, D., Cappellari, M., and Pellegrini, S. (2022). SDSS-IV MaNGA: integral-field kinematics and stellar population of a sample of galaxies with counter-rotating stellar discs selected from about 4000 galaxies. *Mon. Not. R. Astron. Soc.* **511**, 139–157. doi: [10.1093/mnras/stab3732](https://doi.org/10.1093/mnras/stab3732)

Cesare, V., Diaferio, A., Matsakos, T., and Angus, G. (2020). Dynamics of DiskMass Survey galaxies in refracted gravity. *Astron. Astrophys.* **637**, A70. doi: [10.1051/0004-6361/201935950](https://doi.org/10.1051/0004-6361/201935950)

Cesare, V., Diaferio, A., and Matsakos, T. (2022). The dynamics of three nearby E0 galaxies in refracted gravity. *Astron. Astrophys.* **657**, A133. doi: [10.1051/0004-6361/202140651](https://doi.org/10.1051/0004-6361/202140651)

Famaey, B., and McGaugh, S. S. (2012). Modified Newtonian dynamics (MOND): observational phenomenology and relativistic extensions. *Living Rev. Relativ.* **15**, 10. doi: [10.12942/lrr-2012-10](https://doi.org/10.12942/lrr-2012-10)

Fox, C., Mahler, G., Sharon, K., and Remolina González, J. D. (2022). The strongest cluster lenses: an analysis of the relation between strong gravitational lensing strength and the physical properties of galaxy clusters. *Astrophys. J.* **928**, 87. doi: [10.3847/1538-4357/ac5024](https://doi.org/10.3847/1538-4357/ac5024)

Lelli, F., McGaugh, S. S., and Schombert, J. M. (2016). SPARC: mass models for 175 disk galaxies with Spitzer photometry and accurate rotation curves. *Astron. J.* **152**, 157. doi: [10.3847/0004-6256/152/6/157](https://doi.org/10.3847/0004-6256/152/6/157)

McGaugh, S. S., Lelli, F., and Schombert, J. M. (2016). Radial acceleration relation in rotationally supported galaxies. *Phys. Rev. Lett.* **117**, 201101. doi: [10.1103/PhysRevLett.117.201101](https://doi.org/10.1103/PhysRevLett.117.201101)

Milgrom, M. (1983). A modification of the Newtonian dynamics as a possible alternative to the hidden mass hypothesis. *Astrophys. J.* **270**, 365–370. doi: [10.1086/161130](https://doi.org/10.1086/161130)

Milgrom, M. (2010). Quasi-linear formulation of MOND. *Mon. Not. R. Astron. Soc.* **403**, 886–895. doi: [10.1111/j.1365-2966.2009.16184.x](https://doi.org/10.1111/j.1365-2966.2009.16184.x)

Planck Collaboration (2020). Planck 2018 results. VI. Cosmological parameters. *Astron. Astrophys.* **641**, A6. doi: [10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910)

Sanders, R. H., and McGaugh, S. S. (2002). Modified Newtonian dynamics as an alternative to dark matter. *Annu. Rev. Astron. Astrophys.* **40**, 263–317. doi: [10.1146/annurev.astro.40.060401.093923](https://doi.org/10.1146/annurev.astro.40.060401.093923)

Sanna, A. P., Matsakos, T., and Diaferio, A. (2023). Covariant formulation of refracted gravity. *Astron. Astrophys.* **674**, A209. doi: [10.1051/0004-6361/202243553](https://doi.org/10.1051/0004-6361/202243553)

Tian, Y., Umetsu, K., Ko, C.-M., Donahue, M., and Chiu, I.-N. (2020). The radial acceleration relation in CLASH galaxy clusters. *Astrophys. J.* **896**, 70. doi: [10.3847/1538-4357/ab8e3d](https://doi.org/10.3847/1538-4357/ab8e3d)

Zhu, K., Lu, S., Cappellari, M., Li, R., Mao, S., and Gao, L. (2023). MaNGA DynPop–I. Quality-assessed stellar dynamical modelling from integral-field spectroscopy of 10K nearby galaxies: a catalogue of masses, mass-to-light ratios, density profiles, and dark matter. *Mon. Not. R. Astron. Soc.* **522**, 6326–6353. doi: [10.1093/mnras/stad1299](https://doi.org/10.1093/mnras/stad1299)

## Tables

### Table 1. Role of each dataset and analysis

| Dataset or analysis | Role | Independent unit | Determines a model parameter? |
|---|---|---|---|
| SPARC disk sample | Benchmark evaluation | Galaxy | No per-galaxy parameter |
| SPARC photometric scale-length test | Secondary structural-hypothesis sensitivity | Galaxy | No; catalog values are used without fitting |
| Fox clusters | Illustrative in-sample calibration | Cluster | Yes, one effective cluster amplitude |
| Repeated Fox splits | Calibration-stability diagnostic | Held-out cluster within Fox | Refits on each training subset |
| Tian/CLASH profiles outside the Fox calibration sample | No-refit external profile check | Cluster, with radii grouped | No |
| Matched MaNGA/JAM catalog | Secondary counterrotation diagnostic | Counterrotator/control set | No |
| Numerical QUMOND disks | Algebraic-approximation diagnostic | Reconstructed galaxy model | No |

### Table 2. Parameter and assumption accounting

| Quantity | Role | Status | Principal limitation |
|---|---|---|---|
| \(g^\dagger\) | Acceleration scale in \(h(g_N)\) | Fixed model choice | BTFR constrains \(B^2g^\dagger\), not each factor |
| \(A_0=e^{1/(2\pi)}\) | Galaxy normalization before \(F(V_\Sigma)\) | Fixed model choice | Not uniquely derived from the data |
| \(\sigma=20\ {\rm km\,s^{-1}}\) | Regulator in \(F(V_\Sigma)\) | Fixed | Endogenous; sensitivity tested from 10–50 km s\(^{-1}\) |
| \(\Upsilon_{\rm disk},\Upsilon_{\rm bulge}\) | Stellar baryonic contributions | Fixed astrophysical assumptions | Change relative Σ/MOND performance |
| \(B_{\rm Fox}=8.446\) | Cluster response | Illustrative calibration | Does not transfer as a universal radial amplitude |
| \(0.4\times0.15M_{500}\) | Fox baryon mass inside 200 kpc | Approximation | Material amplitude sensitivity |
| Lensing closure | Relates response to lensing target | Assumed | Gravitational slip is undetermined |
| \(R_d\) | Photometric scale length | Tested only in a secondary no-refit window | Catalog assignments do not outperform permutation or fixed-median controls |
| \(L_0,n\) | Historical path-length parameterization | Hypothesis only; not used in the canonical response | No operational 3D path-length functional |

## Figure captions

**Figure 1. Locked SPARC comparison and nuisance sensitivity.** Left: per-galaxy velocity RMS for Σ-Gravity and the tested MOND prescription, with SPARC quality classes shown separately. Center: distribution of the paired contrast \({\rm RMS}_\Sigma-{\rm RMS}_{\rm MOND}\); the mean is \(+0.309\ {\rm km\,s^{-1}}\) and its 95% galaxy-bootstrap interval includes zero. Right: mean contrast for the 81 frozen nuisance combinations. Negative values favor Σ-Gravity and positive values favor MOND. Neither the primary paired interval nor this sensitivity grid establishes a statistically significant difference or equivalence.

![Figure 1](figures/figure_1_sparc_paired.png)

**Figure 2. Illustrative cluster calibration and no-refit radial evaluation.** Left: predicted and observed 200-kpc aperture masses for the 42 Fox clusters under the simplified baryon proxy and calibrated \(B_{\rm Fox}\). This panel is an illustrative in-sample calibration, not validation. Right: predicted-to-observed acceleration ratios for 73 Tian/CLASH radial measurements in 17 clusters after the fixed name-normalization rule excludes MACS0416, MACS0717, and MACS1149, with \(B_{\rm Fox}\) frozen. Large markers show disjoint-sample radius-bin medians. The systematic increase with radius shows that the fixed Fox-calibrated amplitude does not transfer without bias under the present baryonic and lensing assumptions.

![Figure 2](figures/figure_2_cluster_roles.png)

**Figure 3. Algebraic approximation error in representative axisymmetric disk reconstructions.** Fractional acceleration difference between the algebraic relation in Equation (13) and the numerical fixed-\(B\) QUMOND solution for analytic reconstructions representative of F574-2, NGC3741, and UGC05716. The comparison quantifies a geometry-dependent approximation error; it is not a fit to the locked endogenous prescription or the full observed gas and bulge maps.

![Figure 3](figures/figure_3_qumond_approximation.png)

**Figure 4. Matched counterrotation diagnostic.** Left: absolute standardized mean differences before and after matching counterrotators to controls. All post-match values are below the stated 0.1 balance threshold. Right: matched difference in the JAM/NFW-derived \(f_{\rm DM}(<R_e)\), with matched-set bootstrap 95% interval. The interval includes zero. Because the outcome is model derived, this is a secondary catalog diagnostic rather than a direct test of Σ-Gravity.

![Figure 4](figures/figure_4_counterrotation_matched.png)
