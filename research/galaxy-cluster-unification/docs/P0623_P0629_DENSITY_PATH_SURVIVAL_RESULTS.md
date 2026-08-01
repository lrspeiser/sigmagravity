# P0623-P0629: density, path survival, and the dwarf/giant problem

## Executive result

This investigation did not produce a promoted theory, but it did isolate a
repeatable galaxy effect and a sharp cross-domain conflict.

- Baryonic density/path variables are informative. The best local outward
  column law improved four of four development folds by 8.7%, and physical
  pair-crowding laws improved the galaxy score by about 8%.
- Baryonic potential depth was stronger: 14.1% development-CV improvement and
  four of four fold wins. It substantially repaired the dwarf/giant residual
  split.
- Potential-only modulation failed the cluster transfer decisively. Its
  20-system equal-cluster error rose from 0.1942 to 0.3290 dex because a cluster
  is deep even when its baryons are spatially porous.
- A surface-density rescue preserved the galaxy gain and cluster response, but
  no frozen construction simultaneously improved both dwarf and giant mean
  bias, the derived cluster score, Solar proxies, and raw cluster lensing.
- The best cross-domain compromise generated in the opened-data atlas uses one
  global half-strength density response and one global route phase. It improves
  five-fold galaxy OOF RMSE by 7.51%, derived-cluster RMSE by 0.245%, and raw
  equal-system RMS by 0.123%, with all 18 raw roots. It remains 9.28% worse than
  fixed RAR on galaxies and 2.014 times the limited compact-halo raw-lensing
  error.

The defensible conclusion is not “density survival explains dark matter.” It is
that diffuse baryonic structure predicts a useful portion of the remaining
galaxy residual, while the same scalar rule cannot yet reproduce spatially
resolved cluster lensing.

## Question and physical picture

The proposed picture was that individual field/path contributions persist more
completely in diffuse systems and are more strongly redirected, cancelled, or
screened in crowded systems. A dwarf would therefore retain more of the new
channel than a dense giant. A cluster could be deep in total potential while
remaining porous because its baryons are divided among many separated member
galaxies.

This was treated as new phenomenology, not as standard QED. Standard path
integrals do not imply persistent classical light or gravity vectors waiting to
recombine. The calculations below test a mathematical consequence of that
picture without claiming a quantum derivation.

## Frozen test design

P0623 constructed 44 baryon-only features from the SPARC disk, gas, and bulge
profiles:

- mean surface and volume density;
- baryonic potential, acceleration, mass, and size controls;
- normalized pair proximity, pair surface, and pair count at eight physical
  scales from 0.1 to 300 kpc;
- relative pair scales from 0.1 to 3 times `R80`;
- local surface, volume, outward-column, and enclosed-density features.

Each feature was tested with inverse Hill laws, free-floor Hill laws,
log-linear laws, nonmonotonic diagnostics, and wrong-sign controls. In total,
485 formulas received 1,940 grouped training/validation fits. Formula fitting
used equal galaxy weight. No observed rotation velocity entered a feature and
no galaxy or cluster received an individual gravity parameter.

Galaxy folds 0-3 were used for the broad development screen; fold 4 was opened
only after the P0623 family selection. All SPARC and cluster systems remain
project-spent, so even this chronology is not external validation.

## What each stage learned

| Stage | Variation | Main outcome |
|---|---|---|
| P0623 | 44 features x 11 response shapes plus constant | Potential-depth Hill law won: 10.832 km/s development CV, 14.13% better than constant, 4/4 folds |
| P0623 | physical pair/path alternatives | Outward column gained 8.72%; best physical pair laws gained about 8.1% |
| P0623 | chronological fold 4 | Potential law gained only 0.31%; a wrong-sign pair control gained 2.47% but had failed the development consistency gate |
| P0624 | frozen transfer to 20 derived clusters | Potential law failed: 0.3290 versus 0.1942 dex; pair surface gained 1.28%; mean surface gained 0.44% |
| P0624 | Solar and five raw clusters | Unbounded pair law saturated at `q=6`, failed Earth/Mercury, and recovered only 12/18 raw roots |
| P0625 | bounded pair laws and OR combinations | Bounded pair extrapolation improved Solar behavior but still lost raw roots; `max(q_phi,q_surface)` was the best survivor |
| P0626 | 30/100 kpc compact gates plus +90-degree route | Compact gates kept about 10% galaxy gain but reintroduced a missing raw root; direct OR recovered 18/18 |
| P0627 | four global OR strengths x nine global phases | Two opened-data pairs cleared the declared rules; beta 0.5 and phase -67.5 degrees gave the lower raw RMS |
| P0628 | full five-fold OOF synthesis | 11.711 km/s versus constant 12.661 and fixed RAR 10.716; bootstrap galaxy gain 4.90%-10.44% |
| P0629 | compact-potential / extended-porosity hierarchy | Best row gained 9.68%, kept 18 roots, and improved raw RMS 0.074%, but giant mean bias remained slightly worse; 0/12 rows passed all rules |

## The atomic scalar laws

The unchanged P0554 parent can be written as

\[
g_{\rm dyn}(r)=g_b(r)\,[1+q\,A_{0554}(r)],
\]

where `A_0554` is the frozen unit-amplitude path/residence response. It already
contains the acceleration screen, radial residence coordinate, potential
multiplier, and path-ratio multiplier. This investigation changed only the
universal amplitude `q` as a function of baryonic field variables.

For the development fit used in the cluster transfers,

\[
z_\Phi={\log_{10}\Phi_b+7.167482\over0.931835},
\]

\[
q_\Phi={1.787606\over 1+\exp(z_\Phi-1.074325)}.
\]

Here `Phi_b` is the dimensionless baryonic potential depth `|Phi|/c^2`. This
law raises the response in shallow dwarfs and can lower it in deep giants.

The mean-surface branch is

\[
z_\Sigma={\log_{10}(M_b/\pi R_{80}^2)-7.859944\over1.245872},
\]

\[
q_\Sigma=1.100594+(6-1.100594)
 {1\over1+\exp[2(z_\Sigma+1.853764)]}.
\]

This branch remains high when a structure is diffuse in projection, including
some deep clusters.

The P0627 hypothesis-generating compromise is

\[
q_{\rm sel}=q_0+{1\over2}
\left(\max[q_\Phi,q_\Sigma]-q_0\right),
\qquad q_0=1.221417.
\]

Its scalar lens is supplemented only in asymmetric clusters:

\[
\boldsymbol\alpha_{\rm test}
=\boldsymbol\alpha_{\rm scalar}
+{Q^2\over1+\Delta_{80}}
\,\mathcal R_{-67.5^\circ}
 [\delta\boldsymbol\alpha_{\rm route}],
\]

with the previously frozen routed fraction `Delta80/(1+Delta80)`, width
`0.23 R80 sqrt(1+Q^2)`, and return length `0.36 R80`. `Q` is computed from the
observed baryonic member layout. These are global settings, not per-cluster
fits.

This is a phenomenological composite, not an elegant final field equation. It
contains multiple inherited constants and a nonsmooth `max` operator. Its
value is that every term is explicit and falsifiable.

## Dwarf versus giant result

Under five-fold galaxy OOF prediction, the P0627 compromise changed the mass
regimes as follows:

| Regime | Constant RMSE | Selected RMSE | Constant mean residual | Selected mean residual |
|---|---:|---:|---:|---:|
| Dwarf, `M_b < 1e9 Msun` | 9.795 | 8.115 | -6.31 | -3.74 km/s |
| Intermediate | 12.023 | 11.030 | -4.63 | -1.46 km/s |
| Giant, `M_b > 1e10 Msun` | 13.910 | 13.129 | +1.42 | +1.81 km/s |

Thus the selected formula improves RMSE in every mass bin and strongly reduces
the dwarf underprediction, but it slightly worsens the giant mean bias. The
failure is traceable to the OR operator: rescuing a deep-but-diffuse cluster
also prevents enough downward amplitude correction in some dense galaxies.

P0629 tried a continuous size hierarchy,

\[
q=W_L q_\Phi+(1-W_L)q_{\rm porous},\qquad
W_L={1\over1+(R_{80}/L)^2},
\]

at `L=10,30,100 kpc`. It reduced giant bias much more than the simple OR law,
but none of the 12 scale/beta/phase combinations improved both endpoint biases
and all cluster gates. The best raw-compatible row had dwarf residual -1.90
and giant residual +1.44 km/s versus the constant's -6.31 and +1.42.

## Galaxy comparator context

The five-fold OOF equal-galaxy scores are:

| Model | Equal-galaxy RMSE | Parameter context |
|---|---:|---|
| Selected density-route scalar | 11.711 km/s | universal formula parameters trained on other galaxies; zero per-galaxy gravity parameters |
| Constant P0554 amplitude | 12.661 km/s | one universal amplitude refit in each training fold |
| Fixed RAR | 10.716 km/s | same inherited nuisance solution |
| Simple MOND inner refit | 10.708 km/s | nuisance refit on inner radii |
| Weak-prior NFW inner refit | 14.498 km/s | two halo parameters per galaxy plus nuisances, trained only on inner radii |

The candidate is 9.28% worse than fixed RAR and 9.36% worse than this simple
MOND score. It is better than the particular weak-prior, inner-only NFW
extrapolation, but that does not mean it beats dark matter: a conventional halo
fit to the full rotation curve has much more object-specific freedom than this
deliberately difficult comparator.

The paired galaxy bootstrap puts the selected improvement over the constant at
7.52%, with a 95% interval of 4.90%-10.44%. This measures repeatability under
resampling of the same spent galaxies, not out-of-survey generalization.

## Cluster and Solar result

The selected compromise scores 0.193724 dex on the 20 CLASH-derived profiles,
versus 0.194199 for the constant parent, a 0.245% improvement. These targets are
derived through conventional NFW lens models and cannot establish superiority
to dark matter.

On the five raw fixed-geometry systems, beta 0.5 and phase -67.5 degrees recover
all 18 held-out roots. Equal-system RMS is 20.120 arcsec versus 20.145 for the
original P0618 scalar, a 0.123% aggregate improvement. The changes versus the
same-beta scalar are not uniform:

| System | Scalar | Routed | Change |
|---|---:|---:|---:|
| MACS0329 | 19.714 | 19.750 | -0.18% |
| MACS0429 | 13.222 | 13.191 | +0.23% |
| MACS1115 | 27.613 | 27.098 | +1.87% |
| MACS1931 | 27.044 | 26.902 | +0.52% |
| RXJ2129 | 1.513 | 1.435 | +5.17% |

The 20.120-arcsec aggregate is 2.014 times the limited historical compact-halo
comparator of 9.989 arcsec. That comparator is scope-limited, but the gap is too
large to describe the current formula as competitive with dark-matter lensing.

The selected scalar gives a Mercury proxy near -1.924 mas/century, inside the
declared absolute 3.1 margin. Cassini and Earth analytic proxies also pass. No
full ephemeris likelihood was run.

## What was falsified

1. **Potential depth alone is not the cross-domain variable.** It works well in
   galaxies but suppresses deep clusters and worsens derived-cluster error by
   69%.
2. **Unbounded pair persistence is unsafe.** The best log-linear pair law
   extrapolates to the `q=6` cap around the Sun, fails Earth/Mercury, and changes
   raw caustic topology destructively.
3. **Bounding the pair law is not sufficient.** Solar safety returns for some
   bounded laws, but raw roots and RMS remain poor.
4. **A simple diffuse-system boost cannot fix both endpoint biases.** It helps
   dwarfs, but cluster-safe rescue keeps giants slightly overpredicted.
5. **One universal angular phase is only a small correction.** The best
   opened-data phase improves four of five systems relative to its own scalar,
   but the absolute lensing gap remains large.

## What remains promising

The strongest robust clue is not the exact composite formula. It is the
conditional observation that potential depth explains galaxy residuals while
surface density prevents the same variable from misclassifying deep porous
clusters. Low-surface, gas-rich, and late-type galaxies receive the largest OOF
gains; the modest-bulge bin is the one regime that worsens.

A next theory should derive this hierarchy from a local field equation rather
than adding another empirical gate. It must permit the effective scalar channel
to decrease below the constant parent in dense giants, while a distinct
conservative tensor response handles cluster displacement.

## Required next tests

1. Freeze one candidate before obtaining new raw cluster systems. Do not select
   beta or phase again.
2. Use raw multiple-image positions and publish root completeness, not only a
   radial acceleration summary.
3. Replace luminosity-weighted cluster members with stellar mass, gas, and
   diffuse intracluster-light maps.
4. Obtain an external galaxy sample with the same baryonic profile inputs and
   evaluate the dwarf/giant bias without refitting formula shape.
5. Derive a covariant action or conservation equation that produces the scalar
   and angular limits. Until then this is an interpolation architecture, not a
   replacement for GR, MOND, or dark matter.

## Reproducibility

The investigation is reproduced by:

```powershell
python scripts/run_p0623_density_path_survival.py
python scripts/run_p0624_deep_porous_cross_domain.py
python scripts/run_p0625_bounded_porosity_survival.py
python scripts/run_p0626_compact_scalar_angular_route.py
python scripts/run_p0627_or_strength_phase_atlas.py
python scripts/run_p0628_selected_density_route_synthesis.py
python scripts/run_p0629_hierarchical_density_survival.py
```

Every stage has a pre-score protocol in `configs/` and machine-readable results
under its matching `results/p062*` directory.
