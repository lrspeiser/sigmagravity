# Universal candidate lensing comparison

## Result

The unchanged squared-coherence candidate is substantially closer than
fixed-\(a_0\) simple MOND to the 20-cluster CLASH lensing-inferred radial field.
Its equal-cluster RMSE is 0.1387 dex, compared with 0.5184 dex for simple MOND.
That is a 73.2% reduction, and the candidate has lower RMSE in all 20 clusters.
A 20,000-draw paired bootstrap over complete clusters gives a candidate-minus-
MOND RMSE interval of \([-0.421,-0.336]\) dex.

The candidate is not as close as a cluster-retuned RAR: 0.1387 versus 0.1083
dex, a 28.1% excess. That retuned comparison changes the acceleration scale
from the galaxy value \(1.2\times10^{-10}\) to
\(2.02\times10^{-9}\ {\rm m\,s^{-2}}\), a factor of 16.83, so it fails this
project's unchanged-setting requirement. The candidate beats it in 7 of 20
clusters; the complete-cluster bootstrap strongly favors the retuned RAR on
this target.

The dark-matter comparison is necessarily asymmetric. The public CLASH
\(g_{\rm obs}\) values were constructed by deprojecting individually fit NFW
lensing profiles, so NFW has zero residual by construction. The candidate lies
0.1387 dex from that NFW-derived field, has a point scatter factor of 1.360,
and an error-normalized RMS of 1.365. Those numbers measure closeness to the
reported field; they are not an independent raw-lensing contest against dark
matter.

The primary 0.1387-dex score is five-fold held-out prediction: one shared
three-parameter law is learned in each training fold and applied unchanged to
all held-out systems in that fold. It uses zero per-cluster parameters, but the
five folds do not use one numerically identical triplet. The exact full-bridge
triplet transferred to SPARC is also scored below as an in-sample descriptive
check; only a future untouched cluster sample can test that one locked triplet
without this calibration caveat.

## Frozen photon rule

The tested formula is nonrelativistic, so a photon rule had to be declared.
The comparison uses a zero-slip, same-effective-potential diagnostic closure:

\[
\Phi=\Psi=\Phi_{\rm eff},\qquad
{d\Phi_{\rm eff}\over dr}=g_{\rm model}(r),
\]

and, for a spherical lens with impact parameter \(b\),

\[
\widehat\alpha(b)={2b\over c^2}
\int_{-\infty}^{+\infty}{g_{\rm model}(\sqrt{b^2+z^2})
\over\sqrt{b^2+z^2}}\,dz.
\]

This is applied to both the candidate and simple MOND with no lensing-only
multiplier or gravitational-slip parameter. It is consistent with the
same-potential lensing convention demonstrated by TeVeS, but it is an added
hypothesis for the candidate rather than a result derived from its own
covariant action. See [Bekenstein's TeVeS paper](https://arxiv.org/abs/astro-ph/0403694)
and the associated [TeVeS lensing derivation](https://arxiv.org/abs/astro-ph/0507332).

Because the present table already supplies a spherical, lensing-deprojected
radial acceleration, the numerical test compares \(g_{\rm model}\) directly
with that field. It does not re-fit image positions or shear catalogs.

## Numerical comparison

| Model | Equal-cluster RMSE (dex) | Point RMSE (dex) | RMSE factor | Error-normalized RMS | Mean predicted/observed | Within factor 1.5 | Object-specific gravity parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Universal candidate** | **0.1387** | **0.1335** | **1.360** | **1.365** | **0.999** | **81.9%** | **0** |
| Fixed simple MOND, galaxy \(a_0\) | 0.5184 | 0.5129 | 3.258 | 6.390 | 0.318 | 1.4% | 0 |
| Cluster-retuned RAR | **0.1083** | **0.1086** | **1.284** | **1.164** | 1.035 | **90.3%** | 0, but a non-universal scale |
| Per-cluster NFW construction | 0 | 0 | 1 | 0 | 1 | 100% | at least 2 per cluster |

The exact locked full-bridge candidate triplet is included in the machine
report under `candidate_locked_full_sample_descriptive`. It is not substituted
for the primary held-out score because the triplet was calibrated using the
same bridge sample. Its equal-cluster RMSE is 0.1356 dex, its point RMSE is
0.1315 dex, and its error-normalized RMS is 1.348--very close to the held-out
result, but not independent evidence.

For the candidate, 54.2% of points are within a factor 1.25, 81.9% within 1.5,
and 98.6% within 2. Its mean log residual is only \(-0.00042\) dex, equivalent
to a post-hoc correction of 1.00097; there is no meaningful normalization
deficit.

Fixed simple MOND has a mean residual of \(-0.49775\) dex. It predicts only
31.8% of the required field on average and would need a forbidden 3.146-fold
lensing multiplier. After subtracting that mean bias, however, its point
scatter is 0.1238 dex, slightly below the candidate's 0.1335 dex. Thus the
candidate's gain over fixed MOND is mainly the correct cluster amplitude, not
a better post-calibration radial shape. That distinction is important and is
why the multiplier is reported only as a diagnostic, never included in the
primary score.

The CLASH comparison is based on the 20-cluster strong-plus-weak-lensing and
X-ray analysis of [Tian et al.](https://arxiv.org/abs/2001.08340), which itself
found that the cluster acceleration scale differs from the galaxy scale.
Independent CLASH lensing work reconstructed cluster mass profiles from joint
strong- and weak-lensing constraints and fitted mass and concentration for
individual clusters; see [Umetsu et al.](https://arxiv.org/abs/1507.04385).
The two-parameter NFW profile originates with
[Navarro, Frenk, and White](https://arxiv.org/abs/astro-ph/9611107).

## Combined galaxy and cluster score

The independent SPARC outer prediction remains the galaxy-side control:

| Test | Candidate | Fixed simple MOND | Tested per-galaxy NFW |
|---|---:|---:|---:|
| SPARC outer RMSE | 10.586 km/s | **10.385 km/s** | 17.804 km/s |
| CLASH derived-field equal-cluster RMSE | **0.1387 dex** | 0.5184 dex | zero by construction |
| Same gravity settings across galaxies and clusters | **yes** | yes, but inadequate clusters | no per-object halo fit |
| Object-specific gravity parameters | **0** | **0** | 262 for 131 galaxies; at least 40 for 20 clusters |

The candidate is 1.94% worse than simple MOND on SPARC but 73.2% better on the
derived cluster target. This is the strongest evidence currently available
that the density/coherence addition fills the fixed-MOND cluster amplitude gap
without sacrificing galaxy performance or introducing per-object gravity
parameters.

The parameter count supports a universality advantage over the declared NFW
controls: zero object-specific candidate gravity parameters versus at least
302 NFW mass/concentration parameters across the tested 131 galaxies and 20
clusters. It does not imply that all dark-matter models have 302 freely
independent physical parameters; cosmological population priors can correlate
halo properties, and the CLASH NFW target is not an independent prediction.

## What is established, and what is not

Established on the current data product:

- galaxy-level parity with fixed simple MOND;
- a large and cluster-bootstrap-stable advantage over fixed simple MOND on all
  20 NFW-deprojected CLASH radial profiles;
- approximate agreement with the derived lensing field at 0.139 dex and 1.365
  times its quoted diagonal error scale;
- no need for an object-specific candidate setting or post-hoc lens multiplier;
- one globally estimated three-parameter law generalizes across held-out
  clusters, although the five validation folds use five fitted triplets.

Not established:

- an independent fit to raw shear, magnification, or multiple-image positions;
- superiority to GR plus dark matter on a common raw likelihood;
- a derived relativistic completion of the candidate;
- a test of non-spherical lensing structure, line-of-sight mass, substructure,
  or the mass-sheet degeneracy;
- a result against every relativistic MOND theory rather than the declared
  fixed simple-MOND same-potential comparator;
- validation of the exact full-bridge triplet on a new, untouched cluster
  sample.

The best raw pilot remains RX J2129: 21 spectroscopic images from seven source
families and a conventional all-image lens RMS of 0.383 arcsec are available.
It cannot yet yield an honest candidate score because its numeric hot-gas
radial profile and other baryonic likelihood pieces are incomplete, its
seven-image holdout has already been inspected, and the candidate lacks field
equations. A candidate image-plane number obtained now would mix those missing
components into the gravity law.

Therefore the strict project verdict remains: **promising and much closer than
fixed MOND on the complete derived target, but not yet a demonstrated win over
MOND or dark matter on raw lensing data.**

## Reproduction

```powershell
python scripts/compare_clash_lensing_models.py
python -m pytest tests/test_lensing_comparison.py tests/test_phenomenology.py
```

Machine-readable outputs:

- `results/clash_lensing_universal_comparison/report.json`
- `results/clash_lensing_universal_comparison/point_comparison.csv`
- `results/clash_lensing_universal_comparison/per_cluster_metrics.csv`
- `results/clash_lensing_universal_comparison/lensing_comparison.png`

The frozen rules are in
`configs/clash_lensing_universal_comparison_protocol.json`.
