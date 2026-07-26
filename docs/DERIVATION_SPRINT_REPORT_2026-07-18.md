# Bounded Σ-Gravity Derivation Sprint Report

**Date:** 2026-07-18

**Companion plan:**
[`REVIEWER_REVISION_PLAN_2026-07-18.md`](REVIEWER_REVISION_PLAN_2026-07-18.md)

**Research package:**
[`research/derivation_audit`](../research/derivation_audit/README.md)

## Executive verdict

The sprint finds a mathematically honest nonrelativistic action for the
canonical acceleration response, but it does **not** derive or independently
validate either the coherence factor or the path-length law.

| Gate | Verdict | Evidence |
|---|---:|---|
| Closed QUMOND action with the submitted response | **GO** | The analytic derivative is exact; the numerical derivative agrees to (6.8\times10^{-8}) relative error. |
| (B=A C) from independently measured coherence | **NO-GO** | SPARC uses predicted velocity, clusters give (C_{\rm kin}\simeq0) rather than the submitted (C=1), and direct MaNGA maps are absent. |
| Fixed class-level path length (A(L)) | **NO-GO** | The submitted cluster amplitude is biased on disjoint CLASH profiles and a simple action-derived field does not outperform constant (B) in held-out clusters. |
| Full relativistic, lensing, PPN, or cosmological completion | Not tested | Explicitly outside this sprint. |
| Honest acceleration-based phenomenology | **GO** | The empirical response can be retained if (B) is labeled an empirical order/amplitude field and claims are moderated. |

The resulting recommendation is therefore:

> Do not present the current coherence and path-length relations as derived
> physics. Retain the QUMOND action as a legitimate nonrelativistic embedding,
> report (B) as empirical, and revise the manuscript around calibration,
> failed external checks, and specific future tests.

The submitted manuscript and production regression scripts were not changed
during this sprint.

## Reproduction

From the repository root:

```powershell
python research/derivation_audit/run_sprint.py
python -m pytest research/derivation_audit/tests -q
```

The final run completed all stages and the test suite reports `12 passed`.
The environment is frozen in
[`environment.json`](../research/derivation_audit/results/environment.json),
and downloaded file hashes and source URLs are in the two dataset manifests:

- [Tian et al. 2020 manifest](../research/derivation_audit/data/tian2020/manifest.json)
- [Mistele et al. 2025 manifest](../research/derivation_audit/data/mistele2025/manifest.json)

Random operations use seed `20260718`. Cluster validation uses fixed
leave-one-cluster-out splits recorded in
[`cluster_split_definitions.csv`](../research/derivation_audit/results/cluster_split_definitions.csv).

## 1. Canonical baseline and code audit

The research baseline removes unused window and scale-length terms and writes
the observable response through the only combination that enters it:

\[
B\equiv A C,
\qquad
g=g_N\,[1+B h(g_N)],
\]

\[
h(g_N)=\sqrt{\frac{g^\dagger}{g_N}}
\frac{g^\dagger}{g^\dagger+g_N}.
\]

Other theoretical branches in the repository are treated as exploratory and
are not allowed to redefine this submitted baseline.

The production-code audit confirms:

- SPARC coherence is evaluated inside a fixed-point loop from the
  model-predicted velocity, so it is not an independently measured predictor.
- The baseline rotation-curve function accepts `R_d` but does not use it.
- The cluster calculation omits a coherence factor, which is equivalent to
  assigning every cluster (C=1), and fixes (L=600\) kpc.
- Cluster baryonic mass is approximated by
  (M_b(<200\,{\rm kpc})=0.4\times0.15\,M_{500}).
- A coherence-window function remains in the code but is not part of the
  submitted SPARC baseline.

All 171 SPARC galaxies already have catalog photometric scale lengths. The
production heuristic has correlation 0.543 with the catalog value and a
median heuristic/catalog ratio of 1.735. The row-level comparison is frozen in
[`sparc_scale_length_audit.csv`](../research/derivation_audit/results/sparc_scale_length_audit.csv).

## 2. Honest nonrelativistic action

### 2.1 Closed QUMOND potential

Starting from action-based
[QUMOND](https://academic.oup.com/mnras/article/403/2/886/1184577), define

\[
z=\frac{|\nabla\Phi_N|^2}{(g^\dagger)^2},
\]

\[
Q(z,B)=z+4B\left[z^{1/4}-\arctan(z^{1/4})\right].
\]

Let (u=z^{1/4}). Then

\[
\frac{\partial}{\partial z}
4\left(u-\arctan u\right)
=\frac{z^{-1/4}}{1+\sqrt z},
\]

so

\[
Q_z=1+B\frac{z^{-1/4}}{1+\sqrt z}
=1+B h(g_N)
\]

exactly. The implementation uses a small-(u) series for
(u-\arctan u) to avoid catastrophic cancellation.

### 2.2 Field equations by variation

Use the nonrelativistic action density

\[
\mathcal L_g=-\frac{1}{8\pi G}
\left[2\nabla\Phi\!\cdot\!\nabla\Phi_N
-(g^\dagger)^2Q(z,B)\right],
\]

together with matter coupling (-\rho\Phi) and an order-field term
(\mathcal L_B). Variation with respect to (Phi) gives

\[
\nabla^2\Phi_N=4\pi G\rho.
\]

Variation with respect to (Phi_N) gives

\[
\nabla^2\Phi=\nabla\!\cdot
\left[Q_z\nabla\Phi_N\right].
\]

This is an action derivation of the QUMOND field equations, not a
reconstruction of a target enhancement.

If (B) is dynamical, its variation cannot be omitted. For a diagnostic
order-field Lagrangian with global stiffness (kappa), correlation length
(ell), density source (s_\rho), and coherence source (C_{\rm kin}), the
static equation has the form

\[
B-\ell^2\nabla^2B
=\beta s_\rho+\gamma C_{\rm kin}+\eta Q_B,
\]

where

\[
Q_B=4\left[z^{1/4}-\arctan(z^{1/4})\right]
\]

and (eta) absorbs ((g^\dagger)^2/(8\pi G\kappa)) and the numerical source
normalization. The (Q_B) term is mandatory when the order field and QUMOND
response belong to the same action. Both the density-only equation and this
coupled equation were tested below.

### 2.3 Conservation statement

Noether energy, momentum, and angular-momentum conservation follows when the
complete action is translation- and rotation-invariant, (B) has its own
dynamics, and all fields are varied. It does not follow if (B) or (C) is
inserted from observed velocities or from the model-predicted velocity. The
current production baseline therefore cannot use this action to claim full
conservation.

This action remains nonrelativistic. It does not derive gravitational slip,
lensing, PPN parameters, a Solar-System screening rule, the CMB, or structure
formation.

## 3. Deep limit and identifiability

For (g_N\ll g^\dagger),

\[
g\simeq B\sqrt{g^\dagger g_N}.
\]

For a spherical point mass, (g_N=GM_b/r^2), so circular motion gives

\[
V^4=B^2GM_b g^\dagger.
\]

The baryonic Tully-Fisher relation therefore constrains
(B^2g^\dagger), not (B) and (g^\dagger) independently.

The Jacobian audit gives:

| Observable/data design | Parameters written separately | Rank | Consequence |
|---|---|---:|---|
| Deep BTFR | (\log B,\log g^\dagger) | 1 of 2 | Only (B^2g^\dagger) is identified. |
| Fox clusters with one (L=600\) kpc | (\log A_0,n,\log L_0) | 1 of 3 | (n) is one fitted cluster amplitude in different notation. |
| Hypothetical systems with varied (L) | (\log A_0,n,\log L_0) | 2 of 3 | (A_0/L_0) normalization remains degenerate without an external anchor. |
| Dispersion-supported clusters with (C_{\rm kin}=0) | density and coherence couplings | 1 of 2 | Cluster data cannot identify the coherence coupling. |

The exact matrices are in
[`identifiability_audit.json`](../research/derivation_audit/results/identifiability_audit.json).

## 4. Independent CLASH cluster test

### 4.1 Frozen dataset and uncertainty treatment

The primary external check uses the public
[Tian et al. CLASH catalog](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/896/70?format=html&tex=true),
which contains 84 radial measurements in 20 clusters. Residual uncertainties
include both the reported (g_{\rm tot}) uncertainty and the (g_{\rm bar})
uncertainty propagated through the local model slope. All validation folds
hold out complete clusters.

Three objects are obvious overlaps with the Fox calibration sample:
`MACS0416`, `MACS0717`, and `MACS1149`. The strict no-refit result therefore
uses 73 measurements in 17 disjoint clusters.

### 4.2 Submitted fixed-amplitude result

The submitted values imply

\[
B_{\rm cluster}=A_0(600/0.4)^{0.27}=8.4463.
\]

On the disjoint profiles:

| Metric | Result |
|---|---:|
| Median predicted/observed acceleration | 1.318 |
| Mean residual | +0.141 dex |
| RMS residual | 0.188 dex |
| Median absolute residual | 0.122 dex |

The radial bias is systematic: the median predicted/observed ratio rises from
1.183 at 100 kpc to 1.339 at 200 kpc, 1.662 at 400 kpc, and 1.976 at 600 kpc.
The median positive empirical field

\[
B_{\rm obs}(r)=
\frac{g_{\rm tot}/g_{\rm bar}-1}{h(g_{\rm bar})}
\]

falls correspondingly from 6.93 at 100 kpc to 4.01 at 600 kpc.

This is an external failure of the submitted constant class-level path length,
not a successful prediction.

### 4.3 Calibration and radial diagnostic

Refitting all 84 points gives constant (B=5.200), equivalent to
(n=0.204) at the same fixed lengths. A 500-draw cluster bootstrap gives

\[
B=5.202^{+0.410}_{-0.398}\quad(95\%\ interval:
4.804\text{--}5.612).
\]

A diagnostic radial law

\[
B(r)=B_{200}(r/200\,{\rm kpc})^m
\]

gives (B_{200}=5.378) and (m=-0.161). The cluster-bootstrap 95% interval
for (m) is ([-0.214,-0.116]).

| Model | Parameters | Full-sample RMS | Leave-one-cluster-out RMS |
|---|---:|---:|---:|
| Constant (B) | 1 | 0.121 dex | 0.123 dex |
| Radial (B(r)), diagnostic | 2 | 0.098 dex | 0.102 dex |

The radial improvement survives cluster holdout, but it is a diagnostic fit,
not an independent prediction or a derivation. It says the missing variable is
more plausibly local/radial than a fixed system-class length.

### 4.4 Action-derived field candidates

The baryonic source was reconstructed without (g_{\rm tot}) from

\[
M_b(r)=\frac{g_{\rm bar}r^2}{G},
\qquad
\rho_b(r)=\frac{1}{4\pi r^2}\frac{dM_b}{dr}.
\]

All field parameters are global and each validation fold holds out a complete
cluster.

| Field model | Full-sample RMS | Leave-one-cluster-out RMS |
|---|---:|---:|
| Constant empirical (B) | 0.121 dex | 0.123 dex |
| (B-\ell^2\nabla^2B=\beta\rho_b) | 0.326 dex | 0.342 dex |
| Full action source (\beta\rho_b+\eta Q_B) | 0.163 dex | 0.177 dex |
| Coherence-only with cluster (C_{\rm kin}=0) | 0.884 dex | Not identifiable |
| Density plus cluster coherence | Same as density-only | Coherence column has zero rank |

The density-only fit prefers a very broad (ell\simeq600\) kpc and still
performs poorly. Adding the mandatory QUMOND (Q_B) backreaction is an
important theoretical consistency improvement, but it also fails the held-out
gate. Its additional parameters do not outperform constant (B).

The density reconstruction is based on only 3--5 radii per cluster and its
outer continuation is a diagnostic boundary choice. Thus this result does not
prove that every possible auxiliary-field theory fails. It does meet the
predeclared sprint rule: no tested action-derived field beats constant (B),
so the current (A(L)) derivation should be abandoned rather than promoted.

All residuals, fits, and fold parameters are in:

- [`tian_fox_frozen_residuals.csv`](../research/derivation_audit/results/tian_fox_frozen_residuals.csv)
- [`tian_loco_residuals.csv`](../research/derivation_audit/results/tian_loco_residuals.csv)
- [`cluster_bootstrap_posteriors.csv`](../research/derivation_audit/results/cluster_bootstrap_posteriors.csv)
- [`density_field_loco_residuals.csv`](../research/derivation_audit/results/density_field_loco_residuals.csv)
- [`coupled_action_loco_residuals.csv`](../research/derivation_audit/results/coupled_action_loco_residuals.csv)

## 5. Mistele covariance reconstruction check

The public covariance-bearing
[Mistele et al. profiles](https://zenodo.org/records/15476959) were used as a
second reconstruction check. After restricting to positive enclosed-mass bins
and overlapping radial ranges, the comparison contains 24 measurements in 14
clusters; all 14 use the supplied enclosed-mass correlation matrix.

Relative to the Tian (g_{\rm tot}) values, the reconstructed Mistele values
have mean difference +0.055 dex, RMS difference 0.136 dex, and median absolute
difference 0.118 dex.

This is not an independent validation because both products use CLASH lensing
inputs. It does show that catalog/reconstruction choice is material at roughly
the same scale as the constant-(B) residuals. The covariance-aware diagnostics
are in
[`mistele_cluster_covariance_diagnostics.csv`](../research/derivation_audit/results/mistele_cluster_covariance_diagnostics.csv).

## 6. Exact versus algebraic QUMOND for disks

A three-dimensional padded FFT solver was used to solve both QUMOND Poisson
equations for axisymmetric exponential-(\mathrm{sech}^2) disks. Three SPARC
galaxies were selected at low, median, and high catalog surface-density proxy.
The reconstructions use the actual catalog (R_{\rm disk}), catalog luminosity
with (M/L=0.5), and (1.33M_{\rm HI}). They are not exact SPARC gas/bulge
density maps.

| Representative | Surface-density class | Median algebraic error | Maximum algebraic error |
|---|---|---:|---:|
| F574-2 | Low | 5.2% | 20.5% |
| UGC05716 | Median | 4.9% | 18.3% |
| NGC3741 | High | 4.0% | 7.7% |

For the median case, changing the grid from (49^3) to (65^3) changes the
exact acceleration by 0.7% median and 3.7% maximum over the tested radii. The
algebraic-versus-field discrepancy is therefore larger than the observed
grid-convergence difference.

The algebraic approximation is usable as a several-percent approximation in
these reconstructions, but it is not exact and reaches about 20% locally. A
revision should either solve the field equation for the tested mass maps or
include this geometry-dependent systematic. Results are in
[`qumond_axisymmetric_residuals.csv`](../research/derivation_audit/results/qumond_axisymmetric_residuals.csv)
and
[`qumond_grid_convergence.csv`](../research/derivation_audit/results/qumond_grid_convergence.csv).

## 7. Independent coherence gate

The implemented independent estimator is

\[
C_{\rm kin}=\frac{|\bar{\mathbf v}_{\rm stream}|^2}
{|\bar{\mathbf v}_{\rm stream}|^2+\operatorname{tr}\boldsymbol\sigma^2},
\]

evaluated in the system barycentric frame. The phase-space implementation uses
signed azimuthal streaming about an independently supplied or angular-momentum
axis, so balanced counterrotation cancels in the ordered numerator. Unit tests
confirm (C\simeq1) for a cold co-rotating disk and (C\simeq0) for equal
counterrotation.

### 7.1 Cluster conflict

For a dispersion-supported cluster with negligible mean streaming,
(C_{\rm kin}\simeq0). If (B=A C), this removes the enhancement and gives a
Newtonian-limit RMS of 0.884 dex on CLASH. The submitted cluster analysis
instead assigns the equivalent of (C=1). This is a direct regime conflict;
the same operational definition cannot currently support both galaxies and
clusters.

### 7.2 Galaxy grouped comparison

The planned acceleration-only versus acceleration-plus-coherence grouped
cross-validation could not be run honestly:

- SPARC in this repository lacks independent spatially resolved stellar
  streaming and dispersion measurements and currently derives (C) from the
  predicted rotation speed.
- The MaNGA file in the repository is a JAM summary catalog, not the velocity,
  dispersion, mask, and inverse-variance MAPS required to evaluate
  (C_{\rm kin}) and forward-model the observations.

Consequently there is no held-out improvement estimate, no two-standard-error
test, and no stable cross-morphology effect. The independent coherence gate
fails rather than substituting a target-derived proxy.

## 8. Counterrotation audit

The available Bevacqua catalog contains 66 distinct listed IDs; 62 have
complete matching fields in the local MaNGA JAM catalog. Each was matched to
five controls using stellar mass, physical size, Sérsic index, axis ratio,
inclination, redshift, and JAM fit quality. Neither (f_{\rm DM}) nor another
outcome/coherence proxy was used for matching.

The 310 matches use 307 unique controls. The largest post-match absolute
standardized mean difference is 0.071, below the prespecified 0.1 balance
threshold.

As a secondary analysis only, the matched JAM/NFW (f_{\rm DM}(<R_e))
difference is

\[
\Delta f_{\rm DM}=f_{\rm DM,CR}-f_{\rm DM,control}=-0.0069,
\]

with cluster/pair bootstrap 95% interval ([-0.0567,+0.0466]). The interval
includes zero and does not reproduce a large suppression after available
covariate matching.

This is not the primary test because JAM/NFW (f_{\rm DM}) is model-derived.
The direct gate remains failed because DAP MAPS, environment measures, and
merger indicators are absent. The exact required object manifest is frozen in
[`counterrotation_required_map_manifest.csv`](../research/derivation_audit/results/counterrotation_required_map_manifest.csv).

The manuscript should not call the current counterrotation result a confirmed
prediction. At most, it can motivate collection and forward modeling of the
missing maps.

## 9. Opportunities found and rejected

The audit did find three genuine derivation opportunities:

1. **Closed QUMOND action — retained.** The exact (Q(z,B)) replaces the
   circular scalar-field reconstruction with an honest nonrelativistic action.
2. **A local radial order field — empirically motivated.** The negative radial
   exponent improves held-out CLASH residuals, so a local field is a better
   research direction than a fixed class-level path length.
3. **Mandatory (Q_B) backreaction — tested and rejected for now.** Including
   it closes the variation of (B), but the simple global coupled field remains
   worse than constant (B) out of sample.

No derivation in the existing repository supplies the missing independent
coherence source, fixes the cluster coherence conflict, or produces the
observed (B_{\rm obs}(r)) without empirical global fitting. Existing
current-current, holonomy, mode-counting, and scalar-field branches should
therefore remain explicitly exploratory.

## 10. Final decision and manuscript handoff

The machine-readable verdict is
[`decision_gates.json`](../research/derivation_audit/results/decision_gates.json).

The reviewer-revision phase should now proceed with these constraints:

1. Present the model as acceleration-based phenomenology with empirical (B),
   not as a derived coherence/path-length theory.
2. Add the nonrelativistic QUMOND action and its precise conservation
   conditions, while explicitly excluding relativistic and lensing claims.
3. Replace the same-sample cluster success claim with the disjoint CLASH
   failure and the calibrated (B\simeq5.2) result, clearly labeled as
   calibration.
4. Treat the radial (B(r)) law only as a new diagnostic research direction.
5. Use actual SPARC scale lengths and quantify exact-versus-algebraic QUMOND
   systematics.
6. Remove the counterrotation confirmation claim; report the matched secondary
   null result and specify the missing direct-map experiment.
7. Do not claim Solar-System safety, lensing closure, stress-energy
   conservation, equivalence-principle compliance, or cosmological viability
   from this nonrelativistic sprint.

This is a **NO-GO** for the submitted derivation story and a **GO** for a
narrower, transparent phenomenological paper with a real QUMOND action scaffold.

## Follow-up concentration-field thread

After this verdict, a separate baryon-only inverse-distance hypothesis was
tested without modifying the frozen conclusions above. Its first-stage
galaxy/cluster screen passed, but the full cross-regime stage found a BTFR-floor
and environmental-potential conflict. It is therefore no-go as a universal law
in the tested form. See the
[`CONCENTRATION_FIELD_EXPERIMENT_2026-07-18.md`](CONCENTRATION_FIELD_EXPERIMENT_2026-07-18.md)
first-stage report and the final
[`CONCENTRATION_FIELD_FULL_REGRESSION_2026-07-18.md`](CONCENTRATION_FIELD_FULL_REGRESSION_2026-07-18.md)
regression report.
