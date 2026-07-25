# SPARC Paired Statistical Validation — Frozen Design

Frozen: 2026-07-18, before computing any new bootstrap, permutation, or nuisance-grid result.

## Question

Does the submitted Σ-Gravity galaxy prescription perform differently from the comparison MOND prescription on SPARC once galaxies—not individual radial samples—are treated as the independent units, and is any conclusion robust to reasonable global nuisance choices?

This is a diagnostic of the submitted phenomenology. It is not an independent test of coherence because the submitted coherence term is evaluated from the model-predicted velocity.

## Locked central models

The central case reproduces `scripts/run_regression_extended.py` without fitting any galaxy:

- disk mass-to-light ratio: 0.5;
- bulge mass-to-light ratio: 0.7;
- Σ amplitude: \(A_0=\exp[1/(2\pi)]\);
- Σ acceleration: \(g^\dagger=cH_0/(4\sqrt{\pi})\), with the constants used by the regression script;
- coherence dispersion: 20 km s\(^{-1}\);
- MOND acceleration: \(a_0=1.2\times10^{-10}\) m s\(^{-2}\);
- MOND interpolation: \(\nu=[1-\exp(-\sqrt{g_{\rm bar}/a_0})]^{-1}\).

For Σ-Gravity, the submitted fixed-point calculation is

\[
C=\frac{V_{\rm pred}^2}{V_{\rm pred}^2+\sigma^2},\qquad
V_{\rm pred}=V_{\rm bar}\sqrt{1+A_0 C h(g_{\rm bar})}.
\]

An acceleration-only ablation sets \(C=1\) everywhere while leaving \(A_0\), \(g^\dagger\), the baryons, and every sample cut unchanged. This ablation tests whether the submitted velocity-derived coherence factor improves prediction; it does not validate a coherence mechanism.

## Sample and exclusions

- Load every local `data/Rotmod_LTG/*_rotmod.dat` file with at least five valid positive-radius, positive-observed-velocity, positive-baryonic-velocity points.
- Reconstruct \(V_{\rm bar}^2={\rm sign}(V_{\rm gas})V_{\rm gas}^2+\Upsilon_dV_{\rm disk}^2+\Upsilon_bV_{\rm bulge}^2\).
- Match the submitted disk-only analysis by excluding points for which the scaled bulge contribution is at least 30% of \(V_{\rm bar}^2\).
- Exclude a galaxy if fewer than three disk points remain.
- Do not exclude SPARC quality-3 galaxies in the primary result because the submitted regression does not exclude them. Report quality-stratified diagnostics using the official Table 1 quality flag.
- No parameter is fitted per galaxy or on a subset of galaxies.

The primary unit of analysis is a galaxy. Radial points are never resampled as if independent galaxies.

## Locked statistics

For each galaxy calculate unweighted velocity RMS for Σ-Gravity, acceleration-only Σ, and MOND. The unweighted RMS is primary because it exactly matches the submitted analysis. Error-weighted chi-square per point is secondary.

Primary paired contrast:

\[
\Delta_i={\rm RMS}_{\Sigma,i}-{\rm RMS}_{{\rm MOND},i}.
\]

Report:

1. galaxy-weighted mean and median \(\Delta\);
2. Σ win fraction, counting strict \({\rm RMS}_\Sigma<{\rm RMS}_{\rm MOND}\);
3. pooled point-level RMSE as a descriptive, non-independent metric;
4. 95% percentile intervals from 20,000 galaxy-cluster bootstrap resamples, seed 20260718;
5. a two-sided exact binomial test of the win fraction against 0.5, with ties omitted;
6. a two-sided paired sign-flip test of mean \(\Delta\), 50,000 random flips, seed 20260718.

The same grouped bootstrap will compare submitted Σ against the acceleration-only ablation.

## Interpretation gates

- Call Σ superior to MOND only if the 95% interval for mean \(\Delta\) is entirely below zero **and** the 95% interval for the win fraction is entirely above 0.5.
- Call MOND superior only if the corresponding intervals are entirely above zero and below 0.5.
- Otherwise report comparable performance; a raw 47/53 or similar split is not a win.
- Say the submitted coherence factor improves the acceleration-only response only if the 95% interval for \({\rm RMS}_{\Sigma}-{\rm RMS}_{C=1}\) is entirely below zero. Even a pass is only an internal ablation because \(C\) is outcome-derived.

## Frozen nuisance diagnostics

Evaluate the Cartesian grid below without re-fitting any parameter (81 cases):

- \((\Upsilon_d,\Upsilon_b)\): (0.3, 0.5), (0.5, 0.7), (0.7, 0.9);
- global distance multiplier: 0.9, 1.0, 1.1;
- global inclination offset: -5°, 0°, +5°, with corrected observed velocities and errors scaled by \(\sin i/\sin(i+\Delta i)\), and adjusted inclination clipped to [10°, 90°];
- Σ amplitude multiplier: 0.9, 1.0, 1.1.

For a distance multiplier \(s\), use \(R\rightarrow sR\) and each baryonic velocity component \(V_j\rightarrow\sqrt{s}V_j\). This is a controlled catalog-distance diagnostic, not a substitute for per-galaxy distance posteriors.

Separately evaluate the submitted coherence calculation at \(\sigma=10,20,30,50\) km s\(^{-1}\). Report whether the sign of the Σ–MOND mean contrast is stable; do not select the best value.

Use the actual SPARC photometric scale length from Table 1 and `sparc_true_rdisk.csv` only to prove that the submitted galaxy prediction is invariant to \(R_d\), \(L_0\), and \(n\). They do not enter the submitted formula.

## Outputs

- `results/per_galaxy_primary.csv`
- `results/bootstrap_summary.json`
- `results/nuisance_grid.csv`
- `results/coherence_sigma_sensitivity.csv`
- `results/quality_strata.csv`
- `results/decision.json`
- `results/sparc_paired_diagnostics.png`

All splits/resamples are reproducible from the fixed seed. Any analysis added after viewing these outputs must be labeled post hoc.
