# Supplementary Material

## S1. Scope and reproducibility contract

This Supplement accompanies “Σ-Gravity: Coherence-Motivated Gravitational Enhancement in Galaxies and Clusters.” It documents the equations, data roles, exclusions, grouped uncertainty procedures, statistical checks, and software regression status used in the manuscript.

All repository-relative files cited below are available in the [public Σ-Gravity repository](https://github.com/lrspeiser/sigmagravity). The paths identify the corresponding uploaded machine-readable files and analysis code.

The primary analyses use the model defined in Equations (1)–(15) of the main manuscript. A secondary, no-refit photometric scale-length window is evaluated as a structural hypothesis; it is not promoted to the canonical response or used to retune the headline result.

No result in this Supplement changes a calibration into validation. The roles are fixed as follows:

- Fox clusters: calibration under the simplified baryon proxy.
- Repeated Fox splits: within-catalog calibration stability.
- Non-overlapping Tian/CLASH profiles: no-refit radial evaluation.
- Matched MaNGA catalog: secondary counterrotation diagnostic.
- Numerical disks: algebraic-approximation diagnostic.
- SPARC photometric scale lengths: secondary no-refit structural sensitivity with permutation and fixed-value controls.

## S2. Canonical response, dimensions, and asymptotes

The empirical response is

\[
\Sigma(g_N,B)=1+B\,h(g_N),
\qquad
h(g_N)=\sqrt{\frac{g^\dagger}{g_N}}
\frac{g^\dagger}{g^\dagger+g_N}.
\tag{S1}
\]

The quantities \(\Sigma\), \(B\), and \(h\) are dimensionless; \(g_N\) and \(g^\dagger\) have dimensions of acceleration. For \(x=g_N/g^\dagger\),

\[
h(x)=\frac{x^{-1/2}}{1+x}.
\tag{S2}
\]

The asymptotic forms are

\[
h(x)=x^{-1/2}+\mathcal O(x^{1/2})
\quad (x\ll1),
\tag{S3}
\]

and

\[
h(x)=x^{-3/2}+\mathcal O(x^{-5/2})
\quad (x\gg1).
\tag{S4}
\]

For a point baryonic mass \(M_b\), \(g_N=GM_b/r^2\). In the low-acceleration limit,

\[
g_{\rm eff}\simeq B\sqrt{g^\dagger g_N},
\qquad
V^2=r g_{\rm eff}\simeq B\sqrt{GM_b g^\dagger},
\tag{S5}
\]

so

\[
V^4=B^2GM_b g^\dagger.
\tag{S6}
\]

Thus a baryonic Tully–Fisher normalization identifies \(B^2g^\dagger\). It cannot identify a separate amplitude \(A\), coherence \(\mathcal C\), and acceleration scale when the response contains only their product.

### S2.1 Jacobian/identifiability summary

| Dataset/limit | Directly constrains | Does not separately constrain |
|---|---|---|
| Low-acceleration disk normalization | \(B_{\rm gal}^2g^\dagger\) | \(B_{\rm gal}\) and \(g^\dagger\) |
| Galaxy fixed-point response | \(A_0F(V_\Sigma)\) through predicted velocities | A measured coherence field |
| Fox endpoints with common \(L\) | One effective \(B_{\rm Fox}\) | \(n\), \(L_0\), and normalization independently |
| CLASH radial profiles | \(B_{\rm obs}(r)\) conditional on \(g_{\rm bar}\) | A universal source equation |
| Lensing aperture masses | An effective mass ratio under closure | Gravitational slip or photon geodesics |

The canonical galaxy model does not use \(R_d\), \(L_0\), or \(n\). The catalog \(R_d\) values are evaluated separately in Section S4.5 through an explicitly defined radial window; \(L_0\) and \(n\) are not used to obtain any galaxy result.

## S3. Nonrelativistic action variation

For independently specified \(B\), define

\[
z=\frac{|\nabla\Phi_N|^2}{(g^\dagger)^2}
\tag{S7}
\]

and

\[
Q(z,B)=z+4B\left[z^{1/4}-\arctan(z^{1/4})\right].
\tag{S8}
\]

Let \(u=z^{1/4}\). Since

\[
\frac{d}{dz}\left[u-\arctan u\right]
=\frac{1}{4}z^{-3/4}\left(1-\frac{1}{1+\sqrt z}\right)
=\frac{1}{4}\frac{z^{-1/4}}{1+\sqrt z},
\tag{S9}
\]

we obtain

\[
Q_z=1+B\frac{z^{-1/4}}{1+\sqrt z}
=1+B h(g_N).
\tag{S10}
\]

The action is

\[
S=\int dt\,d^3x\left\{
-\frac{1}{8\pi G}
\left[2\nabla\Phi\cdot\nabla\Phi_N-(g^\dagger)^2Q(z,B)\right]
-\rho_b\Phi+\mathcal L_{\rm matter}
\right\}.
\tag{S11}
\]

Variation with respect to \(\Phi\), followed by integration by parts and vanishing boundary variations, gives

\[
\nabla^2\Phi_N=4\pi G\rho_b.
\tag{S12}
\]

Variation with respect to \(\Phi_N\) gives

\[
\nabla^2\Phi=\nabla\cdot\left[Q_z\nabla\Phi_N\right].
\tag{S13}
\]

These equations reproduce the prescribed response only for independently supplied, spatially constant \(B\). They are not an action or a dynamical-consistency result for the endogenous galaxy prescription. If \(B=B(\mathbf x,t)\) is a field, Equation (S11) is incomplete unless its kinetic, source, and backreaction terms are specified and it is varied. If \(B\) is calculated from predicted or observed velocities after solving the field equations, conservation cannot be claimed from the fixed-\(B\) action alone.

Unit tests differentiate \(Q\) numerically and symbolically, test the dimensions of every public function, and verify the limits in Equations (S3)–(S4).

## S4. SPARC sample construction and statistics

### S4.1 Inputs and exclusions

The local archive includes 171 usable files from the 175-galaxy SPARC catalog. At each radius,

\[
V_{\rm bar}^2
=V_{\rm gas}|V_{\rm gas}|
+0.5V_{\rm disk}|V_{\rm disk}|
+0.7V_{\rm bulge}|V_{\rm bulge}|.
\tag{S14}
\]

Signed products preserve the convention of the SPARC component files. The Newtonian acceleration is \(g_N=V_{\rm bar}^2/r\) after unit conversion. The algebraic calculation treats the retained velocity as circular motion in the disk plane, so the primary analysis classifies a point as disk dominated when the scaled bulge term is below 30% of \(V_{\rm bar}^2\). This threshold reduces application of the disk approximation in strongly spheroid-dominated regions; it is not selected by optimizing either model's residual. At least three retained points are required. The final sample contains 164 galaxies and 2,745 radial measurements.

The production galaxy factor is obtained by fixed-point iteration:

1. initialize \(V_\Sigma=V_{\rm bar}\);
2. calculate \(F=V_\Sigma^2/(V_\Sigma^2+\sigma^2)\);
3. set \(B=A_0F\);
4. update \(V_\Sigma=V_{\rm bar}\sqrt{1+B h(g_N)}\);
5. repeat to the numerical tolerance used in the repository implementation.

Observed velocities are not used in this iteration. They enter only when the residual is calculated.

The fixed point is mathematically single valued. With \(x=V_\Sigma^2\), \(b=V_{\rm bar}^2\), \(s=\sigma^2\), and \(K=A_0h(g_N)\), Equation (6) becomes

\[
x^2+[s-b(1+K)]x-bs=0.
\]

For positive \(b\) and \(s\), the product of the two roots is \(-bs<0\), so exactly one root is positive. Initialization at \(V_{\rm bar}\) converges monotonically to that root and cannot select a second positive branch. The implementation stops when the largest velocity update is below \(10^{-6}\ {\rm km\,s^{-1}}\), with a limit of 50 iterations. All 2,745 primary points converged; the maximum was 26 iterations and the median galaxy-level maximum was 11 iterations.

### S4.2 Primary paired statistics

For galaxy \(i\) with \(m_i\) retained radii,

\[
{\rm RMS}_{k,i}
=\left[\frac{1}{m_i}\sum_{j=1}^{m_i}
\left(V_{k,ij}-V_{{\rm obs},ij}\right)^2\right]^{1/2},
\tag{S15}
\]

where \(k\in\{\Sigma,{\rm MOND}\}\). The primary contrast is

\[
\Delta_i={\rm RMS}_{\Sigma,i}-{\rm RMS}_{{\rm MOND},i}.
\tag{S16}
\]

Twenty thousand bootstrap samples draw 164 galaxies with replacement and recompute the mean contrast and the fraction with \(\Delta_i<0\). Radial points are not independently resampled. The sign-flip test multiplies each \(\Delta_i\) by an independent random sign under the paired null. There are no exact RMS ties. The exact binomial test uses the null that either prescription has probability 0.5 of a strictly lower galaxy RMS and compares the observed count with \({\rm Binomial}(164,0.5)\).

The unweighted per-galaxy RMS is primary because it gives every retained radius equal influence within a galaxy and every galaxy equal influence in the sample. As a secondary sensitivity, the reported velocity uncertainty \(s_{ij}\) defines

\[
{\rm RMS}^{(w)}_{k,i}
=\left[
\frac{\sum_j (V_{k,ij}-V_{{\rm obs},ij})^2/s_{ij}^2}
{\sum_j 1/s_{ij}^2}
\right]^{1/2}.
\]

The weighted mean RMS values are 15.8473 km s\(^{-1}\) for Σ-Gravity and 15.6544 km s\(^{-1}\) for MOND. Their paired mean difference is +0.1930 km s\(^{-1}\), with galaxy-bootstrap 95% interval [−0.1773, +0.5637] km s\(^{-1}\); the paired sign-flip \(p\)-value is 0.3132. The weighted analysis likewise does not resolve a statistically significant difference and is not an equivalence test.

Primary results:

| Quantity | Result |
|---|---:|
| Mean Σ RMS | 16.3657 km s\(^{-1}\) |
| Mean MOND RMS | 16.0563 km s\(^{-1}\) |
| Mean paired contrast | +0.3094 km s\(^{-1}\) |
| Bootstrap 95% interval | [−0.0395, +0.6586] km s\(^{-1}\) |
| Σ lower-RMS count | 71/164 |
| Σ lower-RMS fraction | 43.29% |
| Fraction bootstrap interval | [35.98%, 50.61%] |
| Exact binomial \(p\) | 0.1008 |
| Paired sign-flip \(p\) | 0.0884 |

### S4.3 Quality strata

| SPARC quality flag | Galaxies | Mean \(\Delta_i\) | Σ lower-RMS fraction |
|---|---:|---:|---:|
| 1 (high) | 95 | +0.786 km s\(^{-1}\) | 34.7% |
| 2 (medium) | 57 | −0.238 km s\(^{-1}\) | 52.6% |
| 3 (low) | 12 | −0.865 km s\(^{-1}\) | 66.7% |

The strata were not used for tuning. Their variation cautions against a superiority claim and does not establish equivalence.

### S4.4 Nuisance grid and ablation

The submitted central configuration remains the primary result and was not selected from the grid. The 81 frozen combinations vary common stellar mass-to-light assumptions, distance scale, inclination offset, and \(A_0\). The mean paired contrast ranges from −3.286 to +2.779 km s\(^{-1}\), and 64.2% of combinations retain the sign of the central result. This grid is descriptive; it does not marginalize over a calibrated astrophysical likelihood.

The acceleration-only ablation \(B=A_0\) gives mean RMS \(16.8823\ {\rm km\,s^{-1}}\). The full locked model improves the mean by \(0.5166\ {\rm km\,s^{-1}}\), with galaxy-bootstrap 95% interval [0.2420, 0.7945] km s\(^{-1}\). Because the factor uses \(V_\Sigma\), this is an equation ablation, not evidence for measured coherence.

The primary 30% bulge threshold is bracketed by stricter and looser cuts and by a calculation retaining all valid points:

| Sample | Galaxies | Points | Mean Σ RMS | Mean MOND RMS | Mean Σ−MOND RMS |
|---|---:|---:|---:|---:|---:|
| 20% bulge threshold | 155 | 2,511 | 15.4900 | 15.3794 | +0.1106 |
| 30% primary threshold | 164 | 2,745 | 16.3657 | 16.0563 | +0.3094 |
| 40% bulge threshold | 166 | 2,850 | 16.6182 | 16.1768 | +0.4414 |
| All valid points | 171 | 3,373 | 17.4155 | 17.1540 | +0.2615 |

The mean contrast remains positive in each sample, while its magnitude varies. This sensitivity shows that the 164-galaxy selection is not solely responsible for the sign, but it neither establishes a significant aggregate difference nor equivalence.

Machine-readable inputs and results:

- `research/sparc_statistical_validation/results/decision.json`
- `research/sparc_statistical_validation/results/galaxy_metrics.csv`
- `research/sparc_statistical_validation/results/nuisance_grid.csv`
- `research/sparc_statistical_validation/results/quality_strata.csv`

### S4.5 Photometric scale-length hypothesis

The secondary structural candidate uses

\[
W(r,R_d)=\frac{r}{R_d/(2\pi)+r},
\qquad
V_{R_d}=V_{\rm bar}
\sqrt{1+A_0W(r,R_d)h(g_N)}.
\]

The SPARC catalog value of \(R_d\) is used for each galaxy. The rational window and the scale \(R_d/(2\pi)\) are inherited literally from the submitted manuscript; they were not selected from alternative windows after inspecting the revision results, and no alternate form was screened in this reviewer-directed test. No parameter is fitted, and the candidate replaces rather than multiplies the endogenous factor \(F(V_\Sigma)\). It therefore tests whether this specific submitted photometric scale supplies the radial suppression already seen to be useful in the fixed-point ablation.

| Comparison on 164 galaxies | Mean RMS difference | Galaxy-bootstrap 95% interval |
|---|---:|---:|
| Scale-length candidate minus locked Equation (6) | +0.1468 km s\(^{-1}\) | [−0.0881, +0.3854] |
| Scale-length candidate minus acceleration-only | −0.3697 km s\(^{-1}\) | [−0.6197, −0.1108] |
| Scale-length candidate minus MOND | +0.4562 km s\(^{-1}\) | [+0.0753, +0.8433] |

The mean scale-length-candidate RMS is 16.5126 km s\(^{-1}\). In 2,000 galaxy-level permutations of the catalog \(R_d\) assignments, the random-assignment mean is 16.4601 km s\(^{-1}\), with central 95% interval [16.2720, 16.6370] km s\(^{-1}\). The one-sided probability that a permuted assignment performs at least as well as the actual assignment is 0.7111. A common \(R_d\) fixed to the sample median, 2.3 kpc, gives 16.4125 km s\(^{-1}\). With all 171 valid galaxies, the locked, scale-length, and MOND mean RMS values are 17.4155, 17.5776, and 17.1540 km s\(^{-1}\), respectively.

These controls show that radial suppression is beneficial relative to the acceleration-only response, but the galaxy-specific measured \(R_d\) values do not add detectable information through this window. The result rejects this particular scale-length implementation as a replacement for Equation (6); it does not reject all density-, concentration-, or geometry-dependent source fields.

Machine-readable outputs:

- `Publications/Frontiers/analysis/sparc_scale_length/summary.json`
- `Publications/Frontiers/analysis/sparc_scale_length/per_galaxy_primary.csv`
- `Publications/Frontiers/analysis/sparc_scale_length/rdisk_permutation.csv`
- `Publications/Frontiers/analysis/sparc_scale_length/bulge_threshold_sensitivity.csv`

## S5. Algebraic versus numerical QUMOND

The algebraic estimate \(\mathbf g\simeq Q_z\mathbf g_N\) is exact in spherical symmetry but not for a general disk. The diagnostic solves

\[
\nabla^2\Phi=\nabla\cdot[Q_z\nabla\Phi_N]
\tag{S17}
\]

on a three-dimensional periodic FFT grid for axisymmetric analytic baryonic reconstructions representative of three SPARC systems. It maps the Σ response through \(Q_z=1+B h(g_N)\) with spatially constant \(B=1\). \(B\) is not recomputed from the numerical circular velocity, so this diagnostic tests the prescribed fixed-\(B\) response and not the endogenous galaxy prescription. The density is an exponential--\({\rm sech}^2\) disk with scale height \(0.2R_d\); the box half-width is \(8R_d\), the primary grid is \(65^3\), and the comparison spans \(0.75\le r/R_d\le5\). A \(49^3\)-to-\(65^3\) check for UGC05716 gives a median absolute acceleration change of 0.72% and a maximum of 3.67%. The reported percentages compare radial acceleration, not circular velocity.

| Reconstruction | Median \(|g_{\rm alg}-g_{\rm num}|/g_{\rm num}\) | Maximum |
|---|---:|---:|
| F574-2 | 5.19% | 20.54% |
| UGC05716 | 4.88% | 18.30% |
| NGC3741 | 3.96% | 7.73% |

These are not full reconstructions of the observed gas, disk, and bulge maps. Their role is to quantify the scale of the algebraic approximation error and to prevent the algebraic relation from being described as the exact field solution.

Machine-readable output:

- [`research/reviewer_derivation_audit/results/qumond_axisymmetric_residuals.csv`](https://github.com/lrspeiser/sigmagravity/blob/main/research/reviewer_derivation_audit/results/qumond_axisymmetric_residuals.csv)
- [`research/reviewer_derivation_audit/results/qumond_axisymmetric_summary.json`](https://github.com/lrspeiser/sigmagravity/blob/main/research/reviewer_derivation_audit/results/qumond_axisymmetric_summary.json)
- [`research/reviewer_derivation_audit/results/qumond_grid_convergence.csv`](https://github.com/lrspeiser/sigmagravity/blob/main/research/reviewer_derivation_audit/results/qumond_grid_convergence.csv)

## S6. Cluster calibration, profile evaluation, and uncertainty

### S6.1 Fox calibration

The Fox table contains the selected 42 clusters, \(M_{500}\), and 200-kpc strong-lensing aperture masses. The simplified baryon proxy is

\[
M_b(<200{\rm\,kpc})=0.06M_{500}.
\tag{S18}
\]

For each cluster,

\[
g_{\rm bar}=\frac{GM_b}{r^2},
\qquad
M_{\Sigma}=\left[1+B_{\rm Fox}h(g_{\rm bar})\right]M_b,
\tag{S19}
\]

with \(r=200\) kpc. The calibrated amplitude is \(B_{\rm Fox}=8.4463\). The median \(M_\Sigma/M_{\rm SL}=0.987\), with 0.132 dex scatter.

The amplitude and the baryon proxy are not independently identified. Perturbing the concentration factor 0.4 by ±25% changes the predicted-to-observed ratio by approximately 30%. The Fox scatter is therefore not a full error budget.

Machine-readable input/output:

- `data/clusters/fox2022_sigma_results.csv`
- `research/reviewer_derivation_audit/results/fox_parameter_audit.json`

### S6.2 Tian/CLASH no-refit evaluation

The public catalog provides cluster name, radius, \(\log g_{\rm bar}\), \(\log g_{\rm tot}\), and quoted standard errors. The exclusion rule lowercases names, removes non-alphanumeric characters, maps full Abell names to `A` aliases and full MACS names to their four-digit aliases, and then requires exact equality with a Fox calibration name. It excludes MACS0416, MACS0717, and MACS1149. The resulting disjoint set contains 17 clusters and 73 measurements.

For each point, the no-refit prediction uses \(B=8.4463\). Log residuals are

\[
\epsilon_{ij}
=\log_{10}g_{{\rm pred},ij}-\log_{10}g_{{\rm tot},ij}.
\tag{S20}
\]

Quoted uncertainties in both accelerations are retained in the machine table. Aggregate descriptive values are not interpreted as 73 independent clusters. The radial trend fits \(\epsilon\) against \(\log_{10}(r/200\ {\rm kpc})\), weights by the propagated residual variance, and obtains its interval from 5,000 bootstrap samples of the 17 complete clusters. The submitted Fox calibration fixes \(B=8.4463\), \(h(g_N)\), and \(g^\dagger\); no CLASH quantity is used to alter them.

| Radius | Median \(g_{\rm pred}/g_{\rm tot}\) |
|---:|---:|
| 100 kpc | 1.170 |
| 200 kpc | 1.228 |
| 400 kpc | 1.613 |
| 600 kpc | 1.961 |

Across all 73 measurements, the median ratio is 1.3175 and the RMS log residual is 0.1883 dex. The weighted radial slope is 0.1619 dex per dex, with cluster-bootstrap 95% interval [0.1146, 0.2168].

No replacement amplitude or radial response is fitted in this no-refit analysis. The external catalog is used only to test whether the Fox calibration transfers without adjustment.

Machine-readable output:

- `research/reviewer_derivation_audit/results/tian_submitted_residuals.csv`
- `research/reviewer_derivation_audit/results/tian_cluster_audit.json`

## S7. Counterrotation matching

The counterrotator list contains 66 cataloged systems. Repeat MaNGA observations are reduced to one row per physical galaxy by retaining the entry with the lowest JAM \(\chi^2\), and 62 counterrotators then have complete fields for matching. Covariates are standardized on the combined eligible pool. A greedy Euclidean nearest-neighbor procedure processes the case with the largest nearest-control distance first and selects five controls per case without replacement and without a caliper. Completeness of matching and outcome fields, nonnegative quality flag, and positive \(R_e\) define eligibility; all 62 eligible cases remain. The result contains 310 matches and 310 unique control galaxies.

Matched covariates are:

1. \(\log M_\star\);
2. \(\log R_e\);
3. Sérsic \(n\);
4. axis ratio;
5. inclination;
6. redshift; and
7. JAM fit quality.

The maximum absolute SMD declines from 0.684 before matching to 0.066 after matching. The secondary outcome is

\[
\Delta f_{\rm DM}=-0.008061,
\tag{S21}
\]

with matched-set bootstrap 95% interval [−0.057668, +0.045266]. Each of 5,000 bootstrap draws resamples the 62 complete case sets and retains the mean of that case's five controls.

Environment, merger indicators, map-level data-quality covariates, point-spread functions, and velocity/dispersion covariances are not present in the local catalog. The result is therefore not promoted to a direct theory test. The required primary experiment is a common map-level forward model with complete-galaxy cross-validation.

Machine-readable output:

- `research/reviewer_derivation_audit/results/counterrotation_matched_controls.csv`
- `research/reviewer_derivation_audit/results/counterrotation_smd_before.csv`
- `research/reviewer_derivation_audit/results/counterrotation_smd_after.csv`
- `research/reviewer_derivation_audit/results/counterrotation_readiness.json`

## S8. Software regression and commands

### S8.1 Controlled manuscript suite

From the repository root:

```powershell
python -m pytest -q research/reviewer_derivation_audit/tests `
  research/sparc_statistical_validation/tests `
  Publications/Frontiers/scripts/test_sparc_scale_length_sensitivity.py
```

Result: **27 passed, 0 failed, 0 errors**.

### S8.2 Main reproducible analyses

```powershell
python research/reviewer_derivation_audit/run_sprint.py
python research/sparc_statistical_validation/run_validation.py
python "Publications/Frontiers/scripts/run_sparc_scale_length_sensitivity.py"
python "Publications/Frontiers/scripts/generate_revision_figures.py"
```

## S9. Frozen figure sources

| Figure | Frozen source |
|---|---|
| Main Figure 1 | SPARC `decision.json`, `galaxy_metrics.csv`, nuisance grid |
| Main Figure 2 | Fox calibration table; Tian no-refit residuals |
| Main Figure 3 | QUMOND axisymmetric validation CSV |
| Main Figure 4 | Counterrotation SMD files and readiness JSON |

The figure script performs plotting only. It does not refit amplitudes, rematch galaxies, or change sample membership.
