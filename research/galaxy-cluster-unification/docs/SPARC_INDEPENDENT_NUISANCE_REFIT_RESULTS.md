# Independent SPARC nuisance-refit result

## Outcome

The fixed squared-coherence candidate survives the preregistered galaxy
competitiveness gates. It is 1.94% worse than simple MOND in outer RMSE, which
passes the project's galaxy-parity criterion. Its purpose is not necessarily
to improve on MOND within galaxies, but to retain that accuracy while also
predicting cluster lensing with the same universal setting.

The test removes an important weakness of the first SPARC transfer. Each model
now infers its own mass-to-light, distance, and inclination nuisance parameters
from only the inner 70% of every rotation curve. NFW additionally infers its
own \(V_{200}\) and concentration. All scores use the untouched outer 30%.
The candidate's three bridge parameters remain fixed:

\[
\epsilon_0=0.129505,\qquad
\log_{10}\rho_c=-23.785246,\qquad Q=0.427158.
\]

No global parameter was fit to SPARC for RAR, simple MOND, or the candidate.

## Frozen comparison

The sample contains 131 SPARC galaxies, 2,066 inner calibration points, and 968
outer prediction points. All optimizations use the same four starts and the
same nuisance priors. The saved protocol contains the numerical gates and was
written before scoring.

| Inner-calibrated model | Outer RMSE (km/s) | Equal-galaxy RMSE (km/s) | \(\chi^2/N\) |
|---|---:|---:|---:|
| **Fixed RAR** | **10.348** | 10.716 | **4.780** |
| Simple MOND | 10.385 | **10.708** | 4.836 |
| RAR + squared coherence-gated RG, primary geometry | 10.586 | 11.043 | 5.172 |
| NFW inner-fit radial-extrapolation control | 17.804 | 14.498 | 15.973 |

The primary candidate is 2.30% worse in RMSE and 8.19% worse in
\(\chi^2/N\) than fixed RAR. In a 20,000-draw paired galaxy bootstrap, the
candidate-minus-RAR equal-galaxy RMSE is \(+0.327\) km/s with a 95% interval
of \([-0.293,+1.044]\) km/s. The probability that the candidate is better
than RAR in those draws is 0.164. The data therefore do not distinguish the
aggregate candidate score from RAR at 95%, but they favor RAR in the point
estimate.

The five galaxy-subset RMSE ratios, candidate divided by RAR, are 1.006,
1.005, 1.063, 0.862, and 1.101. All fits are finite; the primary candidate
optimizer succeeds for 100% of galaxies, and 3.05% of galaxy fits touch at
least one declared nuisance bound.

The NFW number is only a controlled inner-to-outer extrapolation result. It
uses one spherical halo with two weakly regularized parameters per galaxy. It
is not a test of all dark-matter halo profiles, baryonic feedback, population
priors, or a full \(\Lambda\)CDM analysis. It must not be described as the
candidate beating dark matter generally.

Under the project's universality criterion, however, the parameter comparison
is meaningful: the candidate uses zero object-specific gravity parameters,
whereas this NFW control uses two per galaxy. Passing raw cluster lensing with
the unchanged candidate settings would therefore beat this declared
per-object halo-fitting protocol. See
[`UNIVERSAL_THEORY_SCORECARD.md`](UNIVERSAL_THEORY_SCORECARD.md).

## Density-geometry sensitivity

The candidate was independently refit under seven frozen 3-D density
geometries.

| Candidate geometry | Outer RMSE (km/s) | \(\chi^2/N\) |
|---|---:|---:|
| Primary | 10.586 | 5.172 |
| Thin stellar disk | **10.368** | **4.985** |
| Thick stellar disk | 10.763 | 5.300 |
| Compact gas disk | 10.543 | 5.187 |
| Extended gas disk | **10.999** | **5.758** |
| Thin gas layer | 10.585 | 5.295 |
| Thick gas layer | 10.875 | 5.543 |

The RMSE range is 10.368--10.999 km/s, or 0.19%--6.29% above fixed RAR.
The extended-gas case is the least favorable. All seven stay within the frozen
15% density-sensitivity ceiling, so the aggregate survival is not caused by
one precise thickness choice.

## Where the new term really acts

The aggregate score conceals the most informative result:

- In 101 of 131 galaxies the local coherence gate never activates on an outer
  point, so the candidate is exactly fixed RAR.
- The RG addition activates in 30 galaxies and 351 of 968 outer points.
- Within those 30 galaxies, the candidate improves 12 and worsens 18 relative
  to independently refit RAR.
- Active-subset RMSE is 11.559 km/s for the candidate and 10.949 km/s for
  independently refit RAR.
- At active points, the candidate's added velocity relative to RAR at the same
  nuisance values has a median of 1.420 km/s, a 95th percentile of 21.344
  km/s, and a maximum of 37.952 km/s.

The largest candidate improvement is UGC02916, whose outer RMSE drops from
26.123 to 15.233 km/s. The largest regression is UGC06614, whose outer RMSE
rises from 24.522 to 37.252 km/s. The complete non-cherry-picked ranking is
saved in `per_galaxy_outer_comparison.csv`.

This changes the interpretation. The equation has demonstrated that it can
protect most disk-dominated galaxies while retaining a cluster-side response.
It has not demonstrated that the added response predicts bulged galaxies
better than RAR. The 30 active galaxies, especially the 12/18 split and the
large outliers, are now the decisive galaxy falsification set.

![SPARC activation diagnostic](../results/sparc_independent_nuisance_refit/activation_diagnostics.png)

## Lensing boundary

No independent lensing score is reported. The existing CLASH acceleration
target is based on spherical NFW-deprojected lensing masses; comparing an NFW
profile to it is circular. The repository now has raw or likelihood-level
image catalogs for 19 of 20 CLASH systems and normalized position likelihoods
for 11, but it has zero metric-neutral Weyl posteriors. The same-system audit
also has zero systems with both complete baryonic forward inputs and a
theory-neutral joint lensing covariance.

More fundamentally, the tested equation specifies a nonrelativistic radial
acceleration, not the two metric potentials that deflect photons. Turning it
into an image-plane lens prediction requires a declared relativistic closure
or action. Assuming zero gravitational slip and identifying the candidate
acceleration with the Weyl acceleration would be a useful diagnostic closure,
but it would be an added hypothesis rather than a consequence of the formula.

The raw-lensing stage should therefore remain blocked until:

1. the photon/Weyl law is frozen;
2. at least one same-system baryonic mass model is complete;
3. image positions, source redshifts, and their measurement covariance can be
   forward modeled without borrowing residuals from a fitted GR/NFW model.

RX J2129 remains the shortest local route to this package; its measurement
pipeline is substantially assembled, but the complete gas likelihood and
metric-neutral gravity response are not yet available.

## Decision

Keep the squared-coherence equation as a **universal MOND/RAR challenger whose
distinctive target is cluster lensing**, not as a formula expected to improve
every galaxy. Do not tune \(\epsilon_0\), \(\rho_c\),
\(Q\), or another gate exponent on SPARC. The next equation-level work should
target a physically common coherence variable and the active-galaxy outliers.
The next empirical advancement gate remains raw same-system lensing under a
frozen relativistic closure.

## Reproduction

```powershell
python scripts/run_sparc_independent_nuisance_refit.py
python scripts/analyze_sparc_independent_nuisance_refit.py
```

Primary machine-readable outputs:

- `results/sparc_independent_nuisance_refit/report.json`
- `results/sparc_independent_nuisance_refit/activation_diagnostics.json`
- `results/sparc_independent_nuisance_refit/per_galaxy_outer_comparison.csv`
- `results/sparc_independent_nuisance_refit/point_predictions.csv`
