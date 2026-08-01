# Universal theory scorecard

## Decision rule

The project now uses two operational comparisons:

1. **Beat fixed-setting MOND:** match MOND on galaxy rotation curves and predict
   raw cluster lensing better with the same unchanged gravity setting.
2. **Beat per-object dark-matter fitting:** match or improve predictive accuracy
   while replacing independently adjusted halo parameters with one universal
   gravity setting across galaxies and clusters.

These are deliberately narrower than claiming that every MOND theory or all
of \(\Lambda\)CDM has been disproved. They define what this project needs to
demonstrate empirically.

## Parameter accounting

The comparison separates three kinds of quantities:

- **Universal gravity settings:** constants and functional choices unchanged
  across every object.
- **Object-specific gravity parameters:** halo or force-law quantities adjusted
  independently for each galaxy or cluster.
- **Observational nuisances and measured inputs:** distance, inclination,
  stellar mass-to-light, gas distribution, baryonic density, and source
  position. These must be treated identically across competing gravity laws
  and are not counted as new-gravity parameters.

The present candidate uses five declared universal settings: the inherited RAR
acceleration scale, three bridge-fitted RG constants, and the fixed squared
coherence exponent. Only three were fitted in the bridge sweep. It has zero
object-specific gravity parameters. The SPARC test gives it the same four
observational nuisances per galaxy as MOND.

The NFW radial-extrapolation control has the same four observational nuisances
plus two object-specific halo parameters per galaxy, \(V_{200}\) and
concentration. Across 131 galaxies that is 262 object-specific gravity
parameters versus zero for the candidate.

## Current numerical scorecard

| Test | Candidate | Fixed simple MOND | Per-galaxy NFW control |
|---|---:|---:|---:|
| SPARC outer RMSE | 10.586 km/s | **10.385 km/s** | 17.804 km/s |
| Object-specific gravity parameters per galaxy | **0** | **0** | 2 |
| BCG RMSE on current bridge | **0.0911 dex** | 0.2853 dex | not independent |
| Cluster RMSE on current derived target | **0.1387 dex** | 0.5184 dex | zero by construction |
| CLASH derived-field equal-cluster RMSE | **0.1387 dex** | 0.5184 dex | zero by construction |
| Error-normalized CLASH derived-field RMS | **1.365** | 6.390 | zero by construction |
| RX J2129 raw held-out image positions | **1.064 arcsec, 7/7 roots** | undefined, 3/7 roots | 2.536 arcsec, 7/7 roots (compact one-halo control) |
| RX J2129 all-image descriptive fit | 0.618 arcsec | undefined, 18/22 roots | 0.624 arcsec compact control; 0.29 arcsec published 71-halo reference |

On galaxies, the candidate is only 1.94% worse than simple MOND and 40.5%
better than the particular inner-fit NFW outer prediction. It therefore passes
the galaxy-parity requirement.

On the current cluster acceleration target, the candidate RMSE is 73.2% lower
than fixed-\(a_0\) simple MOND. That is evidence that the density/coherence term
addresses the cluster gap that fixed-setting MOND leaves. It is not yet the
decisive lensing result because the target was obtained from an NFW-deprojected
lensing reconstruction.

The full residual-distribution comparison strengthens and qualifies that
statement. The candidate beats fixed simple MOND in all 20 clusters, with a
paired complete-cluster bootstrap RMSE advantage of 0.336--0.421 dex at 95%.
Its mean normalization is essentially exact and 81.9% of radial points are
within a factor 1.5. Fixed MOND would need a forbidden 3.146-fold multiplier;
after applying such a post-hoc multiplier only as a diagnostic, its residual
shape is slightly tighter than the candidate. See
[`LENSING_UNIVERSAL_COMPARISON_RESULTS.md`](LENSING_UNIVERSAL_COMPARISON_RESULTS.md).

The 0.1387-dex primary number uses fold-specific globally fitted triplets: each
triplet is shared by every training and held-out system in its fold, and no
cluster receives a private fit. The one exact full-bridge triplet used by the
SPARC transfer is reported separately as an in-sample descriptive CLASH score.
It requires an untouched cluster sample for a strict one-numerical-setting
validation.

The first raw-position pilot is now complete on 22 RX J2129 images in seven
spectroscopic families. The unchanged candidate recovers every held-out image
at 1.064-arcsec RMS, while fixed simple MOND recovers only three of seven. The
candidate also outpredicts the deliberately compact one-halo control on this
split, but it misses the preregistered 0.5-arcsec adequacy gate. Its result is
sensitive to the reconstructed baryonic normalization, and the published
conventional 71-halo all-image fit reaches 0.29 arcsec. See
[`RXJ2129_RAW_LENSING_RESULTS.md`](RXJ2129_RAW_LENSING_RESULTS.md).

## Present verdict

### Against MOND

- Galaxy parity: **passed**.
- Same universal candidate setting across the tested galaxy/BCG/cluster
  domains: **passed**.
- Improvement over fixed-\(a_0\) MOND on the complete 20-cluster derived target:
  **passed, with a model-dependence warning**.
- Raw cluster-lensing prediction under a frozen photon law: **relative pilot
  pass, absolute adequacy fail** (1.064 arcsec versus a 0.5-arcsec gate).

The candidate has reached the threshold for a serious MOND challenger and is
far better than the declared fixed-MOND lensing closure on this cluster. It has
not yet beaten MOND under the adopted definition because “works for cluster
lensing” requires an adequate raw fit, not merely a relative win over a failed
comparator. The single-cluster pilot also lacks a native baryonic likelihood and
a covariant photon law.

### Against per-object dark-matter fitting

- Zero candidate gravity parameters per galaxy: **passed**.
- Better outer-radius prediction than the tested two-parameter NFW control:
  **passed**.
- One unchanged setting predicting raw cluster lensing: **demonstrated at pilot
  level, but inadequate at the absolute gate**.
- Better held-out prediction than the compact one-halo control: **passed on RX
  J2129** (1.064 versus 2.536 arcsec).

The candidate does not yet beat dark matter: the compact halo control is less
flexible than standard cluster analyses, the published RX J2129 all-image model
uses 71 halos and reaches 0.29 arcsec, and no multi-cluster held-out comparison
has been completed. Any future passing claim must remain "better than this
declared per-object halo comparison," not "dark matter is ruled out," because
dark-matter population models can impose cosmological priors and use halo
properties as physical initial conditions.

## Decisive next test

The pilot has identified the next bottleneck: angular baryonic structure and
baryonic uncertainty must be separated from an incorrect radial law. The same
photometry-tied BCG/ICL, gas, and member-galaxy baryon map and covariance must be
sent through three newly frozen forward models:

1. candidate with its current universal settings and no lensing-only parameter;
2. fixed-setting relativistic MOND comparator;
3. GR plus the declared per-object dark-matter halo comparator.

All three receive the same exact image-plane likelihood, source-redshift data,
astrometric covariance, angular baryonic perturbers, and ordinary observational
nuisances. The RX J2129 held-out images are now spent and cannot be used to tune
the force law. Advancement requires passing an absolute accuracy gate on new
spectroscopic cluster families without changing the five universal settings or
introducing an object-specific gravity parameter.

Machine-readable criteria are stored in
`configs/universal_theory_comparison_criteria.json`.
