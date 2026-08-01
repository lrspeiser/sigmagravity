# MOND/dark-matter formula sweep

## Decision

The strongest current phenomenological candidate is

\[
\boxed{
\frac{g}{g_{\rm bar}}
=E_{\rm RAR}(g_{\rm bar})
+[1-w(C)]^2\left(\frac{1}{\epsilon(\rho_b)}-1\right)
}
\]

with

\[
E_{\rm RAR}=\frac{1}{1-\exp[-\sqrt{g_{\rm bar}/g_\dagger}]},
\qquad g_\dagger=1.2\times10^{-10}\ {\rm m\,s^{-2}},
\]

\[
w(C)=3C^2-2C^3,
\]

and

\[
\epsilon(\rho)=\epsilon_0+(1-\epsilon_0)
\left[1+\exp\left(-2Q\ln\frac{\rho}{\rho_c}\right)\right]^{-1}.
\]

The held-out bridge fit gives

\[
\epsilon_0=0.1295,\qquad
\log_{10}\!\left(\frac{\rho_c}{\mathrm{g\,cm^{-3}}}\right)=-23.785,
\qquad Q=0.4272.
\]

This candidate advances as an **empirical bridge**, not as a theory. It is
close to fixed RAR on SPARC and approaches the public CLASH error scale, but it
inherits the successful galaxy endpoint from RAR, uses an exploratory squared
coherence gate, and has not predicted raw lensing with a relativistic metric.

## Direct comparison

All BCG/cluster scores are five-fold held-out-system results. The objective
weights the BCG and cluster domains equally and each system equally within its
domain.

| Model | BCG RMSE (dex) | CLASH RMSE (dex) | Equal-domain RMSE (dex) | CLASH diagonal-error RMS | CLASH radial residual slope |
|---|---:|---:|---:|---:|---:|
| Density-only RG | 0.0934 | 0.1556 | 0.1284 | 1.497 | +0.050 |
| Cluster-scale RAR, \(g_\ddagger=2.02\times10^{-9}\) | 0.1321 | 0.1083 | 0.1208 | 1.164 | -0.009 |
| **RAR + squared coherence-gated RG** | **0.0911** | **0.1387** | **0.1174** | **1.365** | **+0.040** |
| Flexible quadratic \((g_{\rm bar},\rho)\) diagnostic | 0.0854 | 0.1006 | 0.0933 | 1.200 | -0.021 |

The candidate improves equal-domain RMSE over density-only RG by 0.0110 dex.
The paired system bootstrap interval for candidate minus RG is -0.0173 to
-0.0050 dex in 20,000 draws. All five parameter fits are interior to the
declared bounds.

The SPARC comparison uses the same 131 galaxies, 3,034 radial measurements,
and 968 untouched outer points as the original comparator run. Candidate
parameters are transferred without refitting; nuisance parameters come from
the fixed-RAR fit.

| SPARC radial holdout | Outer RMSE (km/s) | \(\chi^2/N\) |
|---|---:|---:|
| Fixed galaxy RAR | 10.68 | 5.10 |
| **Candidate, primary local-force coherence/density model** | **11.18** | **5.61** |
| Candidate, all 27 declared density geometries | 10.82--11.73 | 5.22--6.35 |
| Per-galaxy NFW, original inner-fit/outer-prediction test | 17.09 | 14.02 |
| Bridge-fitted fixed Sigma amplitude | 42.11 | 85.32 |

Thus the candidate is within 4.7% of fixed RAR in the primary SPARC transfer
and within 9.9% over every density sensitivity. It is better than the local
NFW radial-extrapolation control, but that does not mean it beats dark matter:
the NFW control fits two halo parameters per galaxy to the inner curve and its
poor extrapolation is not a general test of LCDM.

The CLASH comparison has the opposite limitation. The reported total
accelerations were obtained from spherical NFW deprojections of lensing mass
summaries. NFW therefore has zero residual by construction. The meaningful
number for the candidate is its error-normalized RMS of 1.365: it is near, but
not inside, one public diagonal standard deviation on average. Missing radial
covariance prevents a proper NFW likelihood comparison.

## What was tried when formulas failed

The experiments were deliberately sequential, with a new protocol saved before
each new score.

1. Eight direct Sigma/RG variants were tested: moving the RG threshold with
   acceleration or potential, changing the minimum permittivity with
   acceleration, and combining Sigma/RG susceptibilities additively, in
   quadrature, multiplicatively, or through a density gate. Additive Sigma+RG
   won the bridge at 0.0957 dex, but its transferred \(B=5.35\) gave 42.11 km/s
   on SPARC and was rejected.
2. Fixed RAR was combined with RG additively, in quadrature, and as a product.
   The product reached 0.1014 dex on the bridge but hit a fold boundary. Across
   every declared SPARC density geometry, the ungated additive and product laws
   gave at best 74.25 and 115.09 km/s, respectively.
3. A potential gate used the separation between typical SPARC and cluster
   values of \(g_{\rm bar}r/c^2\). A gate loose enough to fit clusters remained
   active for disks; a predictor-fixed protective gate preserved disks better
   but degraded CLASH RMSE to 0.2307 dex and generated a +0.157 radial slope.
   Galaxy and outer-cluster potential ranges overlap too much for one local
   threshold.
4. A linear coherence gate retained RAR for rotating systems and RG for BCGs
   and clusters. It passed the bridge, but global bulge-fraction proxies leaked
   too much RG response into a small subset of SPARC galaxies, raising RMSE to
   19.24--28.67 km/s across sensitivities.
5. Replacing global B/T with the local bulge force fraction improved the linear
   gate to 16.37 km/s but still failed.
6. Squaring the low-coherence gate retained roughly 95% of the RG channel for a
   typical MaNGA BCG with \(C\simeq0.095\), while suppressing it from 10.4% to
   1.1% at \(C=0.8\). This is the first variant that passes both the bridge and
   the SPARC transfer under a nontrivial disk/bulge mapping.

## Coherence definitions used

- MaNGA BCG: \(C=\mathrm{clip}(\lambda_{R_e},0,1)\), a measured projected
  ordered-motion proxy.
- CLASH cluster: \(C=0\), the pressure-supported spherical endpoint.
- SPARC at radius \(r\):

\[
C(r)=1-\frac{\Upsilon_b V_b^2(r)}
{|V_g(r)|^2+\Upsilon_d V_d^2(r)+\Upsilon_b V_b^2(r)}.
\]

The SPARC mapping is constructed from published component rotation curves and
fitted mass-to-light nuisance values without using the observed residual at the
point. It directly addresses the plane-versus-bulge concern: a globally bulged
galaxy can still be locally disk-dominated at its outer measured radius.

These are not identical measurements of one fundamental scalar. Deriving a
covariant definition of \(C\), and showing that both proxies estimate it, is a
required theory task.

## What is and is not new

The empirical RAR, MOND-like low-acceleration behavior, density-dependent
gravitational permittivity, and relativistic modified-gravity lensing all have
substantial prior art. The project does not claim them. The exact squared
coherence interpolation was not copied from the audited formulas, but an
algebraically uncommon interpolation is not by itself a new physical theory.

The defensible contribution is the test architecture and current empirical
finding: one fixed three-parameter density law, gated by an outcome-blind local
coherence proxy, remains close to galaxy RAR while partially closing the
galaxy-to-cluster acceleration gap.

Primary literature anchors:

- McGaugh, Lelli, and Schombert, [The Radial Acceleration Relation in
  Rotationally Supported Galaxies](https://arxiv.org/abs/1609.05917).
- Tian et al., [The Radial Acceleration Relation in CLASH Galaxy
  Clusters](https://arxiv.org/abs/2001.08340).
- Umetsu et al., [CLASH joint strong/weak-lensing mass
  reconstruction](https://arxiv.org/abs/1507.04385).
- Cesare et al., [DiskMass galaxies in Refracted
  Gravity](https://arxiv.org/abs/2003.07377).

## Independent-nuisance falsification result

The first listed SPARC stage is now complete. Each candidate and comparator
fits its own nuisance parameters on the inner 70% of each galaxy and predicts
the outer 30%. The primary candidate obtains 10.586 km/s outer RMSE versus
10.348 for fixed RAR and 10.385 for simple MOND; all seven density geometries
span 10.368--10.999 km/s. It passes every frozen competitiveness gate but does
not beat RAR.

The added term is inactive, and hence exactly RAR, in 101 of 131 galaxies. In
the 30 active galaxies it improves 12 and worsens 18, with active-subset RMSE
11.559 km/s versus 10.949 for independently refit RAR. See
[`SPARC_INDEPENDENT_NUISANCE_REFIT_RESULTS.md`](SPARC_INDEPENDENT_NUISANCE_REFIT_RESULTS.md).

## Remaining falsification stage

Do not tune another exponent on these data. Freeze the boxed law and pursue:

1. measured radial coherence for disk, S0, elliptical, BCG, group, and cluster
   systems, replacing the mixed \(\lambda_R\)/force-fraction proxies;
2. raw shear, magnification, and/or strong-lens image likelihoods with published
   covariance, not NFW-deprojected accelerations;
3. same-system dynamics plus lensing so one metric must predict both;
4. a covariant action whose weak-field limit produces the coherence gate and
   whose two metric potentials predict photon deflection.

The nuisance-refit stage did not retire the formula, but it narrowed the claim
to a cluster-side RAR extension. Failure on raw lensing retires this exact
squared-gate formula. Passing raw lensing would justify a serious action-level
theory program; it would still not establish that the physical source is void
or negative gravity.

## Reproduction

```powershell
python scripts/run_phenomenology_formula_sweep.py `
  --protocol configs/rar_sharp_coherence_rg_protocol.json `
  --output results/rar_sharp_coherence_rg_sweep

python scripts/run_sparc_coherence_transfer.py `
  --protocol configs/sparc_sharp_coherence_transfer_protocol.json `
  --candidate-report results/rar_sharp_coherence_rg_sweep/report.json `
  --output results/sparc_sharp_coherence_transfer

python scripts/run_sparc_independent_nuisance_refit.py
python scripts/analyze_sparc_independent_nuisance_refit.py
```

Machine-readable results are in
`results/rar_sharp_coherence_rg_sweep/report.json` and
`results/sparc_sharp_coherence_transfer/report.json`. The stricter follow-up is
in `results/sparc_independent_nuisance_refit/`.
