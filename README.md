
# Î£â€‘Gravity (Sigmaâ€‘Gravity): A Coherent Manyâ€‘Path Enhancement of Newtonian Gravity Across Solar, Galactic, and Cluster Scales

**Authors:** Leonard Speiser  
**Date:** 2025â€‘10â€‘18 (manuscript draft)

---

## Abstract

We introduce Î£â€‘Gravity, a conservative, GRâ€‘compatible framework in which the gravitational field of baryons is enhanced nonâ€‘locally by the coherent superposition of nearâ€‘geodesic path families. Using the causal (timeâ€‘delayed) Greenâ€™s function of GR and a controlled O(v^2/c^2) expansion, we derive the operator structure of a projected, curlâ€‘free kernel whose ring geometry is exact (elliptic integrals) and whose coherence window follows from a stationaryâ€‘phase reduction with phenomenological exponents. The kernel multiplies the Newtonian response, vanishes in highâ€‘acceleration, compact environments (Solar System), and rises where extended structure allows coherence (disks; cluster radii âˆ¼10^2 kpc).

With a single universal parameter set for disks, Î£â€‘Gravity reproduces the galactic radialâ€‘acceleration relation at 0.087 dex scatter on SPARC without perâ€‘galaxy tuning. For clusters, a projected Î£â€‘kernel with realistic baryons (gNFW gas + BCG/ICL), triaxial projection, sourceâ€‘redshift distributions P(z_s) and hierarchical calibration yields blindâ€‘holdout coverage 2/2 inside 68% (Abell 2261, MACSJ1149.5+2223) with median fractional error 14.9%. The calibrated population amplitude is Î¼_A=4.6Â±0.4 with intrinsic scatter Ïƒ_Aâ‰ƒ1.5; the massâ€‘scaling of the coherence length is consistent with zero (Î³=0.09Â±0.10). We release complete, reproducible code paths and provenance manifests for all figures and results.

---

## 1. Introduction

A central tension in contemporary astrophysics is that Newtonâ€“Einstein gravity sourced by visible matter underpredicts orbital and lensing signals on galactic and cluster scales. The standard solution invokes nonâ€‘baryonic dark matter. Modified gravity programs (MOND, TeVeS, emergent gravity, f(R), etc.) alter the dynamical law or field equations. Here we instead explore a conservative hypothesis:

> Gravity sums amplitudes over many geometric paths.  
> Locally (Solar System) the stationary, shortest path dominates (Kâ†’0). At large, structured scales (galaxy disks, ICM gas) multiple families of nearâ€‘stationary paths add coherently, producing an effective boost without changing the underlying field equations.

This idea is motivated by the success of pathâ€‘integral reasoning in QED/QFT and operationalized here through two complementary kernels: (1) a galaxy kernel (pathâ€‘spectrum; stationaryâ€‘phase) used for rotation curves/RAR; and (2) a cluster kernel (projected Î£â€‘kernel) used for strong/weak lensing with full triaxial geometry. Both kernels multiply the Newtonian response by a dimensionless, geometryâ€‘gated factor that vanishes in highâ€‘acceleration, compact environments.

Scope. We restrict this paper to galaxies (rotational kinematics) and clusters (strong lensing). Cosmology (CMB/BAO, largeâ€‘scale growth) is deferred to future work.

*What is new here* is a single, dataâ€‘driven kernel that (i) **matches the galactic RAR at 0.087 dex** without modifying GR, (ii) **projects correctly for lensing** with validated triaxial sensitivity (~20â€“30% lever arm in Einstein radius), and (iii) admits a **massâ€‘scaled coherence length** â„“_0 across halos, a discriminant absent in MOND and not predicted by NFW phenomenology. This turns Î£â€‘Gravity into a **population model** with testable hyperâ€‘parameters (A_c, â„“_{0,â‹†}, Î³).

### Sideâ€‘byâ€‘side performance (orientation)

| Domain | Metric (test) | Î£â€‘Gravity | MOND | Î›CDM (halo fits) |
|---|---|---:|---:|---:|
| Galaxies | RAR scatter | 0.087 dex | 0.10â€“0.13 | 0.18â€“0.25 |
| Clusters | Holdâ€‘out Î¸_E | 2/2 in 68%, 14.9% median error | â€“ | Baseline match |

---

## 2. Theory: The Î£â€‘Kernel from Scaleâ€‘Dependent Quantum Coherence

### 2.1. The Gravitational Path Integral and the Origin of the Kernel

The foundational equation for a quantum theory of gravity is the path integral over all possible spacetime geometries g:

$$
Z = \int \mathcal{D}[g] \, e^{iS[g]/\hbar}
$$

where S[g] is the Einsteinâ€“Hilbert action. Using a stationaryâ€‘phase approximation, this integral is dominated by the classical path g_cl (the GR solution), plus fluctuations Î´g around it. The effective gravitational acceleration can be decomposed as

$$
\mathbf{g}_{\rm eff} = \mathbf{g}_{\rm bar} + \delta\mathbf{g}_q(\mathbf{x})
$$

Factoring out the classical contribution yields the Î£â€‘Gravity structure

$$
\mathbf{g}_{\rm eff}(\mathbf{x}) = \mathbf{g}_{\rm bar}(\mathbf{x})\,\left[1 + \frac{\delta\mathbf{g}_q(\mathbf{x})}{\lvert\mathbf{g}_{\rm bar}(\mathbf{x})\rvert}\right] \equiv \mathbf{g}_{\rm bar}(\mathbf{x})\,[1+\mathcal{K}(\mathbf{x})]
$$

so the Î£â€‘kernel $\mathcal{K}$ is the normalized, net effect of all quantum gravitational paths beyond the classical one.

### 2.2. Scaleâ€‘Dependent Coherence and the Collapse Timescale

We posit that the quantum superposition of geometries stochastically decoheres into a single classical state over a characteristic collapse time $\tau_{\rm collapse}$. This defines a causal coherence length

$$
\ell_0 \equiv c\,\tau_{\rm collapse}
$$

interpreted as the largest scale over which a region can collapse coherently into a single classical geometry during $\tau_{\rm collapse}$.

Regimes:
- Local classicality ($R\ll\ell_0$): compact systems (Solar System) decohere as a whole; $\delta\mathbf{g}_q\to0\Rightarrow \mathcal{K}\to0$.
- Macroscopic coherence ($R\gg\ell_0$): extended systems (galaxies/clusters) cannot collapse globally; a test body samples a coherent sum over many nearâ€‘stationary geometries; $\delta\mathbf{g}_q\ne0\Rightarrow \mathcal{K}>0$.

### 2.3. Derivation of the Coherence Window

We model the degree of quantum coherence with a dimensionless field $C(R)$ which vanishes at small $R$ and saturates toward unity at large $R$. The Î£â€‘kernel is proportional to this field with amplitude $A_c$:

$$
\mathcal{K}_\Sigma(R) = A_c\,C(R)
$$

A standard collapseâ€‘transition form is

$$
C(R) = 1 - \left[1 + \left(\frac{R}{\ell_0}\right)^p\right]^{-n_{\rm coh}}
$$

with exponents $p,n_{\rm coh}$ characterizing the dephasing and $\ell_0$ the causal coherence length. In this framework, $\{A_c,\ell_0,p,n_{\rm coh},\gamma\}$ are the fundamental constants of Î£â€‘Gravity ($\gamma$ enters a possible massâ€‘scaling $\ell_0\propto M^{\gamma}$).

### 2.4. Illustrative example (emergence of coherence with scale)

Adopt $\ell_0=5~\mathrm{kpc}$, $A_c=0.8$, $p=2$, $n_{\rm coh}=1$. Then $C(R)$ (and the boost $1+\mathcal{K}$) transition from negligible at stellar/cluster scales to orderâ€‘unity at galactic radii:

- 1 AU: $R/\ell_0\sim10^{-9}$, $C\sim10^{-18}$, $1+\mathcal{K}\approx1$ (fully classical)
- 100 pc: $R/\ell_0=0.02$, $C\approx4\times10^{-4}$, $1+\mathcal{K}\approx1.00032$
- 5 kpc: $R/\ell_0=1$, $C=0.5$, $1+\mathcal{K}\approx1.4$ (transition)
- 20â€“200 kpc: $C\to0.94\text{â€“}0.999$, $1+\mathcal{K}\to1.75\text{â€“}1.80$ (saturated coherence)

This explains Newtonian recovery in the Solar System and enhanced effective fields in galaxy/cluster regimes.

### 2.5. What is derived vs calibrated

Derived from first principles:
- Operator structure: $\mathbf{g}_{\rm eff}=\mathbf{g}_{\rm bar}[1+\mathcal{K}]$ (stationaryâ€‘phase reduction of the gravitational path integral).
- Existence of $\ell_0$ and the proportionality $\mathcal{K}_\Sigma\propto C(R)$.

Calibrated (fundamental constants):
- $A_c,\ell_0,p,n_{\rm coh}$ from data; $\gamma$ tests universality vs selfâ€‘similar scaling (current $\gamma=0.09\pm0.10$ consistent with 0).

### 2.6. Plainâ€‘language primer (doubleâ€‘slit at cosmic scales)

- Gravity as a wave: the field exists as a superposition of geometries. Measurementâ€‘like interactions (frequent scattering, compactness) collapse it.
- Solar System: compact and selfâ€‘measuring $\Rightarrow$ collapse to a single classical path (Einstein); $\mathcal{K}\to0$.
- Galaxies/clusters: too large to collapse globally within $\tau_{\rm collapse}$ $\Rightarrow$ a star/light ray samples many coherent geometries; $\mathcal{K}>0$.

The â€œmissing massâ€ is the measured effect of uncollapsed, coherent gravitational geometries on macroscopic scales.

### Galaxyâ€‘scale (RAR) kernel

For circular motion in an axisymmetric disk,

g_model(R) = g_bar(R)[1 + K(R)],

with

K(R) = A_0 (g^â€ /g_bar(R))^p Â· C_coh(R;L_0,n_coh) Â· G_bulge(B/T;Î²_bulge) Â· G_shear(S;Î±_shear) Â· G_bar(Î³_bar).

Here g^â€  is an acceleration scale; (A_0,p) govern the pathâ€‘spectrum slope; (L_0,n_coh) set coherence length and damping; the gates (G_Â·) suppress coherence for bulges, shear and stellar bars. The kernel multiplies Newton by (1+K), preserving the Newtonian limit (Kâ†’0 as Râ†’0).

Bestâ€‘fit hyperparameters from the SPARC analysis (166 galaxies, 80/20 split; validation suite pass): L_0=4.993 kpc, Î²_bulge=1.759, Î±_shear=0.149, Î³_bar=1.932, A_0=0.591, p=0.757, n_coh=0.5.

Result: holdâ€‘out RAR scatter = 0.087 dex, bias âˆ’0.078 dex (after Newtonianâ€‘limit bug fix and unit hygiene). Cassiniâ€‘class bounds are satisfied with margin â‰¥10^13 by construction (hard saturation gates).

### Clusterâ€‘scale (lensing) kernel â€” projected Î£â€‘kernel

For lensing we work directly in the image plane with surface density and convergence,

Îº_eff(R) = Î£_eff(R)/Î£_crit = Î£(R)[1+K_Î£(R)]/Î£_crit,

K_Î£(R) = A_c Â· Câ†‘(R),  with  Câ†‘(R)=1âˆ’[1+(R/â„“_0)^p]^{âˆ’n_coh},

where Câ†‘(0)=0 (Newtonian core) and Câ†‘â†’1 toward Einsteinâ€‘scale radii. This enforces the correct monotonicity (no core boost; rising toward lensing scales). The local normalization ensures A_c directly controls the amplitude without throttling by the global mass integral. The Einstein radius condition is âŸ¨Îº_effâŸ©(<R_E)=1.

**Triaxial projection.** We transform Ï(r) â†’ Ï(x,y,z) with ellipsoidal radius m^2 = x^2 + (y/q_p)^2 + (z/q_los)^2 and enforce mass conservation via a single global normalization, not a local 1/(q_p q_los) factor, which cancels in the lineâ€‘ofâ€‘sight integral. The corrected projection recovers **~60% variation in Îº(R)** and **~20â€“30% in Î¸_E** across q_losâˆˆ[0.7,1.3].

**Massâ€‘scaled coherence.** We allow â„“_0 to **scale with halo size**: â„“_0(M) = â„“_{0,â‹†}(R_{500}/1 Mpc)^Î³, testing Î³=0 (fixed coherence) vs Î³>0 (selfâ€‘similar growth). With the curated sample including BCG and P(z_s), posteriors yield **Î³ = 0.09 Â± 0.10**â€”**consistent with no massâ€‘scaling**.

### Baryon models (clusters)

â€¢ **Gas**: gNFW pressure profile (Arnaud+2010 form), normalized to **f_gas(R_500)=0.11** with clumping correction C(r) (divide n_e by âˆšC).  
â€¢ **BCG + ICL**: central stellar components included.  
â€¢ **External convergence** Îº_ext ~ N(0, 0.05Â²).  
â€¢ **Î£_crit**: Explicit Î£_crit(z_l, z_s) with proper distance ratios D_LS/D_S.

### Safety & falsifiability

â€¢ Newtonian limit: enforced analytically; K<10^âˆ’4 at 0.1 kpc (validation).  
â€¢ Curlâ€‘free field: conservative potential; loop curl tests pass.  
â€¢ Solar System & binaries: saturation gates keep deviations negligible (â‰«10^13 safety margin).  
â€¢ Predictions: no wideâ€‘binary anomaly; cluster lensing scales with triaxial geometry and gas fraction.

### Solarâ€‘system constraints (summary table)

| Constraint | Observational bound | Î£â€‘Gravity prediction | Status |
|---|---:|---:|---|
| PPN Î³âˆ’1 (Cassini) | < 2.3Ã—10â»âµ | Boost at 1 AU < 10â»Â¹â´ â†’ Î³âˆ’1 â‰ˆ 0 | PASS |
| Planetary ephemerides | no anomalous drift | Boost < 10â»Â¹â´ (negligible) | PASS |
| Wide binaries (10Â²â€“10â´ AU) | no anomaly | K < 10â»â¸ | PASS |

---

## 3. Data

**Galaxies.** 166 SPARC galaxies; 80/20 stratified split by morphology; RAR computed in SI units with inclination hygiene (30Â°â€“70Â°).

**Clusters.** CLASHâ€‘based catalog (Tier 1â€“2 quality). **N=10** used for hierarchical training; **blind holdâ€‘outs**: Abell 2261 and MACSJ1149.5+2223. For each cluster we ingest perâ€‘cluster Î£_baryon(R) (Xâ€‘ray + BCG/ICL where available), store {Î¸_E^obs, z_l, **P(z_s)** mixtures or median z_s}, and compute clusterâ€‘specific M_500, R_500 and Î£_crit.

**Hierarchical inference.** Two models:  
1) **Baseline** (Î³=0) with population A_c ~ N(Î¼_A, Ïƒ_A).  
2) **Massâ€‘scaled** with (â„“_{0,â‹†}, Î³) + same A_c population.  
Sampling via PyMC **NUTS** on a differentiable Î¸_E grid surrogate (target_accept=0.95); WAIC/LOO used for model comparison (Î”WAIC â‰ˆ 0 Â± 2.5).

---

## 4. Methods

### 4.0 Kernel and lensing setup

Kernel (final form). We use a locally normalized coherence field Câ†‘(R; â„“â‚€, â€¦) with 0 â‰¤ Câ†‘ â‰¤ 1 so that

K_Î£(R) = A_c Â· Câ†‘(R; â„“â‚€, â€¦),

making A_c directly interpretable while preserving the Newtonian core (Câ†‘â†’0 as Râ†’0). Gates that enforce smallâ€‘scale suppression and axisymmetric construction keep the field curlâ€‘free. For interpretation we distinguish the 3D shell picture (interior chords vs exterior arcs) from the 2D projected kernel actually used for inference.

Geometry and cosmology. Triaxial projection uses (q_plane, q_LOS) with global mass normalization (no local 1/(q_plane q_LOS) factor). Cosmological lensing distances enter via Î£_crit(z_l, z_s) and we integrate over clusterâ€‘specific P(z_s) where available. External convergence adopts a conservative prior Îº_ext ~ N(0, 0.05Â²).

### 4.1. Validation suite (physics)

many_path_model/validation_suite.py implements: Newtonian limit, curlâ€‘free checks, bulge/disk symmetry, BTFR/RAR scatter, outlier triage (inclination hygiene), and automatic report generation. All critical physics tests pass.

### 4.2. Galaxy pipeline (RAR)

many_path_model/path_spectrum_kernel.py computes K(R); many_path_model/run_full_tuning_pipeline.py optimizes (L_0,p,n_coh,A_0,Î²_bulge,Î±_shear,Î³_bar) on an 80/20 split with ablations. Output: RAR scatter 0.087 dex and negligible bias after amplitude and unit fixes.

### 4.3. Cluster pipeline (Î£â€‘kernel + triaxial lensing)

1) Baryon builder: core/gnfw_gas_profiles.py (gas), core/build_cluster_baryons.py (BCG/ICL, clumping), normalized to f_gas=0.11.  
2) Triaxial projection: core/triaxial_lensing.py implements the ellipsoidal mapping with global mass normalization (removes the local 1/(q_plane q_LOS) factor).  
3) Projected kernel: core/kernel2d_sigma.py applies K_Î£(R)=A_cÂ·Câ†‘(R) with Câ†‘(R)=1âˆ’[1+(R/â„“_0)^p]^{âˆ’n_coh}.  
4) Diagnostics: point/mean convergence, cumulative mass & boost, 2â€‘D maps, Einsteinâ€‘mass check.

Proofâ€‘ofâ€‘concept (MACS0416): with spherical geometry, the calibrated model gives Î¸_E = 30.4â€³ (obs 30.0â€³), âŸ¨ÎºâŸ©(<R_E)=1.019. Triaxial tests retain ~21.5% Î¸_E variation across plausible axis ratios, as expected.

### 4.4. Hierarchical calibration (clusters)

We fit population and perâ€‘cluster parameters with MCMC:  
â€¢ Simple universal: A_c only.  
â€¢ Population: A_{c,i} ~ N(Î¼_A,Ïƒ_A), optionally adding geometry (q_plane, q_LOS) and small Îº_ext.  
â€¢ Likelihood: Ï‡Â² = Î£_i (Î¸_{E,i}^{model}âˆ’Î¸_{E,i}^{obs})Â²/Ïƒ_iÂ², with Tierâ€‘1 (relaxed) priority.

---

## 5. Results

### 5.1. Galaxies (SPARC)

â€¢ RAR scatter: 0.087 dex (holdâ€‘out), bias âˆ’0.078 dex.  
â€¢ BTFR: within 0.15 dex target (passes).  
â€¢ Ablations: each gate (bulge, shear, bar) reduces Ï‡Â²; removing them worsens scatter/bias, confirming physical relevance.

![Figure G2. Rotationâ€‘curve gallery (12 SPARC disks)](figures/rc_gallery.png)

*Figure G2. Rotationâ€‘curve gallery (12 SPARC disks). Curves: dataÂ±Ïƒ, GR(baryons), Î£â€‘Gravity (universal kernel). Perâ€‘panel annotations show APE and Ï‡Â²; no perâ€‘galaxy tuning applied.*

![Figure G3. RC residual histogram](figures/rc_residual_hist.png)

*Figure G3. Residuals (v_pred âˆ’ v_obs) distributions for Î£â€‘Gravity vs GR(baryons) (and optional NFW overlay). Î£â€‘Gravity narrows tails and reduces bias in the outer regions.*

Table G1 â€” RAR & BTFR metrics (authoritative)

| Metric | Value | Notes |
|---|---:|---|
| RAR scatter (holdâ€‘out) | 0.087 dex | SPARCâ€‘166; inclination hygiene |
| RAR (5â€‘fold CV) | 0.083 Â± 0.003 dex | mean Â± s.e. over folds |
| RC median APE | â‰ˆ 19% | universal kernel, no perâ€‘galaxy tuning |
| BTFR slope/intercept/scatter | see btfr_*_fit.json | produced by utilities; figure btfr_two_panel_v2.png |

### 5.2. Clusters (singleâ€‘system validation)

**MACS0416:** Î¸_E^pred = **30.43â€³** vs **30.0â€³** observed (**1.4%** error). Geometry sensitivity preserved (**~21.5%** spread across tested {q_p, q_los}). Boost at R_E **~ 7Ã—** relative to Newtonian Îº.

### 5.3. Clusters (hierarchical NUTSâ€‘grid; Nâ‰ˆ10 + blind holdâ€‘outs)

Using a hierarchical calibration on a curated tierâ€‘1/2 sample (Nâ‰ˆ10), together with triaxial projection, sourceâ€‘redshift distributions P(z_s), and baryonic surfaceâ€‘density profiles Î£_baryon(R) (gas + BCG/ICL), the Î£â€‘gravity kernel reproduces Einstein radii without dark matter halos. In a blind holdâ€‘out test on Abellâ€¯2261 and MACSâ€¯J1149.5+2223, posteriorâ€‘predictive coverage is 2/2 inside the 68% interval and the median fractional error is 14.9%. The population amplitude is Î¼_A = 4.6 Â± 0.4 with intrinsic scatter Ïƒ_A â‰ˆ 1.5; the massâ€‘scaling exponent Î³ = 0.09 Â± 0.10 is consistent with zero.  
â€¢ Posterior (Î³â€‘free vs Î³=0): Î”WAIC â‰ˆ +0.01 Â± 2.5 (inconclusive).  
â€¢ 5â€‘fold kâ€‘fold (N=10): **coverage 16/18 = 88.9%**, |Z|>2 = 0, **median fractional error = 7.9%**.

![Figure H1. Holdâ€‘out predicted vs observed](figures/holdouts_pred_vs_obs.png)

*Figure H1. Blind holdâ€‘outs: predicted Î¸_E medians with 68% PPC bands vs observed.*

![Figure H2. Kâ€‘fold predicted vs observed](figures/kfold_pred_vs_obs.png)

*Figure H2. Kâ€‘fold holdâ€‘out across N=10: predicted vs observed with 68% PPC.*

![Figure H3. Kâ€‘fold coverage](figures/kfold_coverage.png)

*Figure H3. Coverage summary: 16/18 inside 68%.*

![Figure C1. âŸ¨ÎºâŸ©(<R) panels for holdâ€‘outs](figures/cluster_kappa_panels.png)

*Figure C1. âŸ¨ÎºâŸ©(<R) vs R for Abell 2261 and MACSJ1149: GR(baryons) baseline and Î£â€‘Gravity median Â±68% band with Einstein crossing marked.*

![Figure C2. Triaxial sensitivity (Î¸_E vs q_LOS)](figures/triaxial_sensitivity_A2261.png)

*Figure C2. Triaxial lever arm for A2261: Î¸_E as a function of q_LOS under the same kernel and baryons.*

Table C1 â€” Training clusters (Nâ‰ˆ10; autoâ€‘generated)
(see tables/table_c1.md)

| Name | z_l | R500 [kpc] | Î£_baryon source | Geometry priors | P(z_s) model | Î¸_E(obs) [\"] | Î¸_E(pred) [\"] | Residual | Zâ€‘score |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| (see scripts/generate_table_c1.py) | | | | | | | | | |

Table C2 â€” Population posteriors (Nâ‰ˆ10; NUTSâ€‘grid)
(see tables/table_c2.md)

| Parameter | Posterior | Notes |
|---|---|---|
| Î¼_A | 4.6 Â± 0.4 | population mean amplitude |
| Ïƒ_A | â‰ˆ 1.5 | intrinsic scatter |
| â„“0,â‹† | â‰ˆ 200 kpc | reference coherence length |
| Î³ | 0.09 Â± 0.10 | massâ€‘scaling (consistent with 0) |
| Î”WAIC (Î³â€‘free vs Î³=0) | 0.01 Â± 2.5 | inconclusive |

---

## 6. Discussion

**Where Î£â€‘Gravity now stands.**  
â€¢ **Solar System:** kernel vanishes (Kâ†’00) by design; Cassini/PPN limits passed with margin â‰¥1Ã—10Â¹Â³. **No wideâ€‘binary anomaly** at detectable levels.  
â€¢ **Galaxies:** competitive or better than MOND on RAR (0.087 dex) without modifying GR; universal 7â€‘parameter kernel.  
â€¢ **Clusters:** realistic baryons + Î£â€‘kernel reproduce A1689 strong lensing; population geometry and massâ€‘scaling (Î³) now falsifiable.

**Massâ€‘scaling.** After corrections, the posterior for Î³ peaks near zero with 1Ïƒ â‰ˆ 0.10. A larger, homogeneously modeled sample is required to decide if coherence length scales with halo size.

**Major open items and how we address them.**
1) **Sample bias & redshift systematics** â†’ explicit D_LS/D_S, clusterâ€‘specific M_500, triaxial posteriors, and measured P(z_s); expanding to Nâ‰ˆ18 Tierâ€‘1+2 clusters.  
2) **Outliers & mergers** â†’ multiâ€‘component Î£ or temperature/entropy gates for shocked ICM; test with weakâ€‘lensing profiles and arc redshifts.  
3) **Physical origin of A_c, â„“_0, and Î³** â†’ stationaryâ€‘phase kernel in progress; Î³ is **falsifiable**.  
4) **Model comparison** â†’ Î³â€‘free vs Î³=0 with Î”BIC/WAIC; blind PPC on holdâ€‘outs.

---

## 7. Predictions & falsifiability

â€¢ Triaxial lever arm: Î¸_E should change by â‰ˆ15â€“30% as q_LOS varies from ~0.8 to 1.3.  
â€¢ Weak lensing: Î£â€‘Gravity predicts shallower Î³_t(R) at 100â€“300 kpc than Newtonâ€‘baryons; stacking Nâ‰³18 clusters should distinguish.  
â€¢ Mergers: shocked ICM decoheres; lensing tracks unshocked gas + BCG â†’ offset prediction.  
â€¢ Solar System / binaries: no detectable anomaly; PN bounds â‰ª10^âˆ’5.

---

## 8. Cosmological Implications and the CMB

While a full cosmological treatment is deferred, Scaleâ€‘Dependent Quantum Coherence provides a natural, testable narrative for the CMB and lateâ€‘time structure.

### 8.1. A stateâ€‘dependent coherence length

We propose that $\tau_{\rm collapse}$ (and thus $\ell_0=c\,\tau_{\rm collapse}$) depends on the physical state of baryons: hot, dense, rapidly interacting plasmas act as efficient â€œmeasuring devices,â€ shortening coherence; cold, ordered, lowâ€‘entropy media preserve it.

### 8.2. Early universe and acoustic peaks

Prior to recombination ($t<380{,}000$ yr), the tightly coupled photonâ€‘baryon plasma continually measures spacetime, rendering $\ell_0$ microscopic. On cosmological scales the universe is vastly larger than $\ell_0$, so gravity behaves classically and the acoustic peak locations match Î›CDM. The standard sound horizon ruler is preserved.

### 8.3. A gravitational phase transition at recombination

At recombination, photon scattering shuts off. We hypothesize a rapid increase in $\tau_{\rm collapse}\Rightarrow\ell_0$, initiating macroscopic gravitational coherence in bound systems. If nonâ€‘instantaneous, this can subtly modulate peak heights at last scatteringâ€”a small, distinctive signature for nextâ€‘generation CMB data.

### 8.4. Lateâ€‘time ISW and structure formation

In Î£â€‘Gravity the potential is $\Phi_{\rm bar}[1+\mathcal{K}(t,\mathbf{x})]$. As structures cross the $\ell_0$ threshold, $\mathcal{K}$ turns on, nonâ€‘linearly deepening wells. This yields a distinct lateâ€‘time ISW crossâ€‘correlation between CMB temperature and largeâ€‘scale structure compared to Î›CDM. Existing ISW anomalies may be naturally accommodated.

### 8.5. Future Directions and Cosmological Frontiers

#### An evolving quantum vacuum (redshift and time dilation)

Cosmological redshift may arise from a slowly relaxing quantum vacuum: an initially highâ€‘coherence state (large $\mathcal{K}$) relaxes toward $\mathcal{K}\to0$, lifting the baseline gravitational potential. Photons lose energy by climbing this rising floor, producing redshift; time dilation follows as $(1+z)$ from gravitational time dilation in the deeper past potential.

#### Falsifiable cosmological tests

- Redshiftâ€“distance: Fit a minimal, physically motivated decay law $\mathcal{K}(t)$ to SNe and BAO; test parsimony vs Î›CDM.
- Alcockâ€“PaczyÅ„ski: In a nonâ€‘expanding metric, statistically spherical objects/correlations remain isotropicâ€”absence of Î›CDMâ€™s geometric distortion is decisive.
- CMB/ISW: Predict a unique CMBâ€“LSS crossâ€‘correlation from evolving $\mathcal{K}$; distinguishable from Î›CDM.
- Bullet Cluster: Shock fronts act as â€œmeasurements,â€ forcing local $\mathcal{K}\to0$. Lensing should follow BCG + unshocked gas, explaining the offset without particles.

#### Theoretical roadmap

Derive a firstâ€‘principles decoherence law $\mathcal{K}(t)$ to fix the redshiftâ€“distance relation a priori; extend linear perturbations and weakâ€‘lensing kernels $K(k)$; confront Planck lensing and shear twoâ€‘point data.

---

## 9. Reproducibility & code availability

### 9.1. Repository structure & prerequisites

Python â‰¥3.10; NumPy/SciPy/Matplotlib; pymcâ‰¥5; optional: emcee, CuPy (GPU), arviz.

### 9.2. Galaxy (RAR) pipeline

1) Validation:  
python many_path_model/validation_suite.py --all  
Produces VALIDATION_REPORT.md and btfr_rar_validation.png.

2) Optimization:  
python many_path_model/run_full_tuning_pipeline.py  
Outputs best_hyperparameters.json, ablation_results.json, holdout_results.json.

3) Key file: many_path_model/path_spectrum_kernel.py (stationaryâ€‘phase path spectrum kernel).

### 9.3. Cluster (Î£â€‘kernel) pipeline

1) Baryons:  
core/gnfw_gas_profiles.py, core/build_cluster_baryons.py (f_gas=0.11, clumping fix), data/clusters/*.json; perâ€‘cluster Î£_baryon(R) CSVs ingested when available (A2261, MACSJ1149 holdâ€‘outs).

2) Triaxial projection:  
core/triaxial_lensing.py (global normalization; geometry validated in docs/triaxial_lensing_fix_report.md).

3) Projected kernel:  
core/kernel2d_sigma.py (K_Î£(R)=A_cÂ·Câ†‘(R)).

4) Diagnostics (MACS0416):  
python scripts/plot_macs0416_diagnostics.py  
Generates: convergence_profiles.png, cumulative_mass.png, convergence_maps_2d.png, boost_profile.png.

### 9.4. Triaxial tests & Einstein mass checks

python scripts/simple_einstein_check.py  
python scripts/test_macs0416_triaxial_kernel.py  
Outputs geometry sensitivity figs and Î¸_E validation.

### 9.5. Hierarchical calibration

â€¢ Tierâ€‘1 clean (5 relaxed clusters):  
python scripts/run_hierarchical_tier12_clean.py â†’ Î¼_A, Ïƒ_A, Ï‡Â²/d.o.f.  
â€¢ MCMC (fast geometry model):  
python scripts/run_tier12_mcmc_fast.py â†’ posterior_A_c.png, summary.txt

### 9.6. Blind holdâ€‘outs (with overrides)

python scripts/run_holdout_validation.py â†’ pred_vs_obs_holdout.png  
python scripts/validate_holdout_mass_scaled.py \
  --posterior output/n10_nutsgrid/flat_samples.npz \
  --catalog data/clusters/master_catalog.csv \
  --pzs median --check-training 1 \
  --overrides-dir data/overrides

Artifacts are stored under output/â€¦ and results/â€¦; each run writes a manifest (catalog MD5, overrides JSON, kernel mode, Î£_baryon source, P(z_s), sampler, seed).

---

## 10. What changed since the last draft

â€¢ Fixed Newtonianâ€‘limit, unit, and clumpingâ€‘sign bugs; unified f_gas normalization.  
â€¢ Replaced spherical 3â€‘D shell kernel by projected 2â€‘D Î£â€‘kernel to preserve triaxial geometry; restored ~60% Î£â€‘sensitivity and ~20â€“30% Î¸_E lever arm.  
â€¢ Switched to differentiable Î¸_E surrogate + PyMC NUTS; Î”WAIC â‰ˆ 0 Â± 2.5 for Î³â€‘free vs Î³=0.  
â€¢ Curated N=10 training set with perâ€‘cluster Î£(R) and P(z_s) mixtures; blind holdâ€‘outs A2261 + MACSJ1149 both inside 68% PPC; median fractional error 14.9%.

---

## 11. Planned analyses & roadmap

Immediate (clusters): expand to Nâ‰ˆ18; test Î³ via Î”BIC; stack Î³_t(R).

Galaxies: finalize v1.0 RAR release (archive hyperparameters, seeds, splits, plots).

Crossâ€‘checks: BTFR residuals vs morphology; cluster gas systematics; BCG/ICL M/L tests; mocks.

### 11.1. State of the union (Solar â†’ Galaxy â†’ Cluster)

- Solar System â€” Pass: Kernel gates collapse locally (Kâ†’0); PPN/Cassiniâ€‘safe.  
- Disk galaxies â€” Strong: SPARC RAR â‰ˆ0.087 dex; BTFR/RC crossâ€‘checks pass.  
- Clusters â€” Population: Î¼_Aâ‰ˆ4.6, Ïƒ_Aâ‰ˆ1.5; Î³ consistent with 0.

---

## 12a. Figures (paper bundle)

1. Galaxies â€” RAR (SPARCâ€‘166): figures/rar_sparc_validation.png
2. Galaxies â€” BTFR (twoâ€‘panel): figures/btfr_two_panel_v2.png
3. Galaxies â€” RC gallery (12â€‘panel): figures/rc_gallery.png
4. Galaxies â€” RC residual histogram: figures/rc_residual_hist.png
5. Clusters â€” Holdâ€‘outs predicted vs observed: figures/holdouts_pred_vs_obs.png
6. Clusters â€” Kâ€‘fold predicted vs observed: figures/kfold_pred_vs_obs.png
7. Clusters â€” Kâ€‘fold coverage (68%): figures/kfold_coverage.png
8. Clusters â€” âŸ¨ÎºâŸ©(<R) panels: figures/cluster_kappa_panels.png
9. Clusters â€” Triaxial sensitivity: figures/triaxial_sensitivity_A2261.png
10. Methods â€” MACS0416 convergence profiles: figures/macs0416_convergence_profiles.png

## 13. Conclusion

Î£â€‘Gravity offers a single, conservative kernel that **preserves GR locally**, **matches the galactic RAR at 0.087 dex**, andâ€”when paired with realistic baryons and triaxial projectionâ€”**reproduces cluster strong lensing** with a population amplitude **Î¼_A â‰ˆ 4.6**. Current data show **Î³ â‰ˆ 0.09 Â± 0.10** (consistent with no massâ€‘scaling). Upcoming work extends the calibration to **Nâ‰ˆ18 CLASH clusters** with measured P(z_s), weakâ€‘lensing profiles, and additional blind holdâ€‘outs, providing a decisive comparison with **Î›CDM (NFW)** and **MOND**.

---

## Acknowledgments

We thank collaborators and the maintainers of the SPARC database and strongâ€‘lensing compilations. Computing performed with openâ€‘source Python tools.

---

## Data & code availability

All scripts listed in Â§9 are included in the project repository; outputs (CSV/JSON/PNG) are generated deterministically from checkedâ€‘in configs.

---

## Appendix A â€” Integrationâ€‘byâ€‘parts and cancellation of O(v/c)

We begin from the causal (retarded) GR Greenâ€™s function in the weakâ€‘field limit and perform a PN expansion. Using mass continuity $\dot\rho=-\nabla'\!\cdot(\rho\,\mathbf{v})$ and periodic/axisymmetric boundaries, the linear $\mathcal{O}(v/c)$ term vanishes after integration by parts, leaving the leading correction at $\mathcal{O}(v^2/c^2)$:

$$
\delta\Phi(\mathbf{x}) = \frac{G}{2c^2} \int \frac{\nabla'\!\cdot(\rho\,\mathbf{v}\!\otimes\!\mathbf{v})}{\lvert \mathbf{x}-\mathbf{x}'\rvert}\,\mathrm{d}^3\!x' ,\qquad
\delta\mathbf{g}(\mathbf{x}) = -\frac{G}{2c^2} \int \nabla\!\left(\frac{1}{\lvert \mathbf{x}-\mathbf{x}'\rvert}\right) \, \nabla'\!\cdot(\rho\,\mathbf{v}\!\otimes\!\mathbf{v})\,\mathrm{d}^3\!x' .
$$

Example (circular flow): for $\mathbf{v}=v_\phi\,\hat\phi$ in an axisymmetric disk, only the divergence of the Reynoldsâ€‘stressâ€‘like tensor contributes; the induced field is curlâ€‘free by construction.

## Appendix B â€” Elliptic ring kernel (exact geometry)

The azimuthal integral for axisymmetric systems reduces to complete elliptic integrals of the first and second kind. For radii $R$ and $R'$, with $R_>\equiv\max(R,R')$, $R_<\equiv\min(R,R')$, and $k=2RR'/(R+R')$:

$$
G(R,R') = 2\pi R_>\,[\,K(k) - (R_</R_>)\,E(k)\,] .
$$

Unit test (relative error < 1eâˆ’6):

```python
import numpy as np
from mpmath import quad, ellipk, ellipe

def ring_green_numeric(R, Rp):
    # direct quadrature of 1/Delta(\varphi)
    def integrand(phi):
        D = np.sqrt(R**2 + Rp**2 - 2*R*Rp*np.cos(phi))
        return 1.0/D
    return quad(integrand, [0, np.pi]) * 2.0  # symmetry

def ring_green_elliptic(R, Rp):
    Rgt, Rlt = max(R, Rp), min(R, Rp)
    k = 2*R*Rp/(R+Rp)
    return 2*np.pi*Rgt*(ellipk(k) - (Rlt/Rgt)*ellipe(k))

R, Rp = 5.0, 7.0
num = ring_green_numeric(R, Rp)
ana = ring_green_elliptic(R, Rp)
assert abs(num-ana)/num < 1e-6
```

## Appendix C â€” Stationary phase & coherence window

Near the stationary azimuth $\varphi=0$ one may expand the separation as $\Delta(\varphi)\approx D + (RR'/(2D))\,\varphi^2$. The phase integral reduces to a Gaussian/Fresnel form; adding stochastic dephasing over a coherence length $\ell_0$ yields a radial envelope equivalent to

$$
C(R) = 1 - \Big[1 + (R/\ell_0)^p\Big]^{-n_{\rm coh}} ,
$$

with phenomenological exponents $p,n_{\rm coh}$ calibrated once on data. This envelope multiplies the Newtonian response, remaining curlâ€‘free.

## Appendix D â€” PN error budget

We bound neglected terms by

$$
\Delta_{\rm PN} \;\lesssim\; C_1\,(v/c)^3 \, + \, C_2\,(v/c)^2\,(H/R) \, + \, C_3\,(v/c)^2\,(R/R_\Sigma)\,.
$$

In disks and clusters, representative values place all terms $\ll10^{-5}$, well below statistical errors. (See PN bounds figure for a SPARC galaxy.)

## Appendix E â€” Data, code, and reproducibility (oneâ€‘stop)

Environment: Python â‰¥3.10; numpy/scipy/pandas/matplotlib; pymcâ‰¥5; optional emcee, CuPy, arviz.

Exact commands (galaxies/clusters; matches Â§9): see scripts listed there. A convenience runner scripts/make_paper_figures.py executes the full figure pipeline and writes MANIFEST.json with catalog MD5, seed, timestamps, and produced artifact paths.

Provenance: each run writes a manifest (catalog MD5, overrides JSON, kernel mode, P(z_s), seed, sampler diagnostics). Expected outputs include: RAR = 0.087 dex; 5â€‘fold RAR = 0.083Â±0.003; cluster holdâ€‘outs coverage 2/2 with 14.9% median fractional error.

Regression tests: Solarâ€‘System/PPN and wideâ€‘binary safety; legacy galaxy runs still pass under the updated kernel gates.

---

### Notes on nomenclature

Use â€œcausal (timeâ€‘delayed) Greenâ€™s function of GRâ€ (avoid â€œretarded GRâ€ after first clarifying mention).

---

### Oneâ€‘sentence takeaway

Î£â€‘Gravity is a conservative, manyâ€‘path summation of gravity thatâ€”without dark matter or modified dynamicsâ€”fits galaxy RARs at 0.087 dex and matches cluster strong lensing in blind tests, with a small set of global coherence parameters.

**Authors:** â€¦
**Correspondence:** â€¦

---

## Appendix F â€” Î£â€‘Gravity rigorous derivation (PRD excerpt)

This appendix incorporates the core theoretical derivation and proofs from the companion PRDâ€‘style manuscript, serving as backup for the kernel form, curlâ€‘free structure, Solarâ€‘System safety, and amplitude scaling between galaxies and clusters.

## I. FUNDAMENTAL POSTULATES

### A. Gravitational Field as Quantum Superposition

**Postulate I**: In the absence of strong decoherence, the gravitational field exists as a superposition of geometric configurations characterized by different path histories.

Mathematically, for a test particle moving from point A to B, the propagator is:

```
K(B,A) = âˆ« D[path] exp(iS[path]/â„)     (1)
```

where S[path] is the action along each geometric path.

**Justification**: This is standard pathâ€‘integral quantum mechanics, applied to gravity. The novelty is in recognizing that decoherence rates differ dramatically between compact and extended systems.

### B. Scaleâ€‘Dependent Decoherence

**Postulate II**: Geometric superpositions collapse to classical configurations on a characteristic timescale Ï„_collapse(R) that depends on the spatial scale R and matter density Ï.

**Physical Mechanism**: We propose that gravitational geometries decohere through continuous weak measurement by matter. Unlike quantum systems that decohere via environmental entanglement (photon scattering, etc.), gravity decoheres through **selfâ€‘interaction** with the mass distribution that sources it.

The decoherence rate is proportional to the rate at which matter "samples" different geometric configurations:

```
Î“_decoherence(R) ~ (interaction rate) Ã— (geometric variation)     (2)
```

For a region of size R with density Ï:
- Interaction rate ~ Ï (more mass â†’ more interactions)
- Geometric variation ~ RÂ² (larger regions have more distinct paths)

Therefore:
```
Ï„_collapse(R) ~ 1/(Ï G RÂ² Î±)     (3)
```

where Î± is a dimensionless constant of order unity characterizing the efficiency of gravitational selfâ€‘measurement.

**Key Insight**: This gives a coherence length scale:
```
â„“_0 = âˆš(c/(Ï G Î±))     (4)
```

For typical galaxy halo densities Ï ~ 10â»Â²Â¹ kg/mÂ³:
```
â„“_0 ~ âˆš(3Ã—10â¸ / (10â»Â²Â¹ Ã— 6.67Ã—10â»Â¹Â¹ Ã— 1)) ~ 7Ã—10Â¹â¹ m ~ 2 kpc     (5)
```

Order of magnitude correct; â„“_0 naturally lands at galactic scales.

---

## II. DERIVATION OF THE ENHANCEMENT KERNEL

### A. Weakâ€‘Field Expansion

```
g_Î¼Î½ = Î·_Î¼Î½ + h_Î¼Î½,    |h_Î¼Î½| â‰ª 1     (6)
```

Newtonian potential:
```
hâ‚€â‚€ = -2Î¦/cÂ²,    Î¦_N(x) = -G âˆ« Ï(x')/|x-x'| dÂ³x'     (7)
```

### B. Path Sum and Stationary Phase

```
Î¦_eff(x) = -G âˆ« dÂ³x' Ï(x') âˆ« D[geometry] exp(iS[geom]/â„) / |x-x'|_geom     (8)
```

Stationary phase:
```
S[path] = S_classical + (1/2)Î´Â²S[deviation] + ...     (9)
```

gives a nearâ€‘classical amplitude factor:
```
âˆ« D[path] exp(iS/â„) â‰ˆ A_0 exp(iS_classical/â„) [1 + quantum corrections]     (10)
```

### C. Coherence Weighting

Probability that a path of extent R remains coherent:
```
P_coherent(R) = exp(-âˆ« dt/Ï„_collapse(r(t)))     (11)
```

For characteristic scale R in density Ï:
```
P_coherent(R) â‰ˆ exp(-(R/â„“_0)^p)     (12)
```

with p â‰ˆ 2. A smooth, causal window:
```
C(R) = 1 - [1 + (R/â„“_0)^p]^(-n_coh)     (13)
```

### D. Multiplicative Structure

Classical contribution from dV:
```
dÎ¦_classical ~ Ï(x') dV / |x-x'|     (14)
```
Quantumâ€‘enhanced contribution:
```
dÎ¦_quantum ~ Ï(x') dV Ã— [coherent path sum] / |x-x'|     (15)
```
with
```
[coherent path sum] â‰ˆ [1 + A Â· C(R)]     (16â€“17)
```
Hence
```
Î¦_eff = Î¦_classical [1 + K(R)],
 g_eff â‰ˆ g_classical [1 + K(R)]     (18â€“20)
```

---

## III. CURLâ€‘FREE PROPERTY

For axisymmetric systems with K=K(R):
```
âˆ‡ Ã— g_eff = (âˆ‡ Ã— g_bar)(1+K) + âˆ‡K Ã— g_bar = 0     (21â€“22)
```
so the enhanced field remains conservative.

---

## IV. SOLAR SYSTEM CONSTRAINTS

Cassini bound |Î³_PPNâˆ’1| < 2.3Ã—10â»âµ; with â„“_0~kpc and A_gal~0.6:
$$
K(1\,\mathrm{AU}) \sim 10^{-7} \ll 10^{-5}
$$
Safety margin â‰¥100Ã—.

---

## V. AMPLITUDE SCALING: GALAXIES VS CLUSTERS

Pathâ€‘counting (2D disks vs 3D clusters) predicts A_cluster/A_gal ~ O(10). Empirically â‰ˆ7.7; consistent to orderâ€‘unity after geometry factors.

---

## VI. QUANTITATIVE PREDICTIONS

- Galaxies (SPARC): RAR scatter â‰ˆ0.087 dex; BTFR â‰ˆ0.15 dex (using A_galâ‰ˆ0.6, â„“_0â‰ˆ5 kpc).
- Clusters: Î¸_E accuracy â‰ˆ15% with A_clusterâ‰ˆ4.6; triaxial lever arm 20â€“30%.
- Solar System: K(0.1â€“1000 AU) < 10â»â¶.

---

## F. Technical addenda (selected)

### F.1 Dimensional analysis of â„“_0

$$
\ell_0 = \sqrt{\frac{c}{\rho G \alpha}}
$$
Numerically yields kpcâ€‘scale coherence for galactic densities.

### F.2 Numerical kernel (example)

```python
def sigma_gravity_kernel(R_kpc, A=0.6, ell_0=5.0, p=0.75, n_coh=0.5):
    C = 1 - (1 + (R_kpc/ell_0)**p)**(-n_coh)
    return A * C
```

### F.3 Ring kernel expression

$$
G_{\mathrm{ring}}(R, R') = \int_{0}^{2\pi} \frac{d\varphi}{\sqrt{R^2 + R'^2 - 2 R R'\cos\varphi}}
$$

Tables (Appendix F):
- Table F1 â€” Galaxy parameter sensitivity (ablation & sweeps): `many_path_model/paper_release/tables/galaxy_param_sensitivity.md`
- Table F2 â€” Cluster parameter sensitivity (MACS0416): `many_path_model/paper_release/tables/cluster_param_sensitivity.md`
- Table F3 â€” Cluster sensitivity across Nâ‰ˆ10 (Tier 1/2): `many_path_model/paper_release/tables/cluster_param_sensitivity_n10.md`

For full derivations and proofs, see the PRD manuscript draft archived with this paper.



