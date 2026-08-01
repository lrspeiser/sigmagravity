# Which galaxies does the Solar-screened law help?

## Bottom line

The locked Solar-screened isothermal law beats independently inner-refit fixed
RAR in **41 of 131 SPARC galaxies** and loses in 90. The wins are real under
nearby scoring definitions, but they do not rescue the universal formula: its
pooled outer RMSE remains `18.602 km/s`, versus `10.348 km/s` for fixed RAR and
`10.385 km/s` for simple MOND.

The wins are not best explained by the available void/environment data, bulge
class, Hubble type, gas class, or surface-brightness class. They are best
identified by the way the two laws fit the **inner** rotation curve:

- the winning galaxies have smoother and less-steep inner observed curves;
- they are more often massive and less often viewed edge-on;
- RAR's inner fit assigns them a higher stellar disk mass-to-light ratio;
- most importantly, switching from RAR to the screened law does **not** require
  a large upward change in disk mass-to-light ratio;
- RAR then tends to overpredict their untouched outer velocities, while the
  screened model lands much closer.

This is a post-result forensic analysis. It identifies a reproducible pattern
inside this already-inspected sample; it is not an independent validation set.

## Question and tested equation

The primary question is whether the following locked, cluster-selected law has
lower error than fixed RAR on the outer 30% of each SPARC rotation curve after
each model independently calibrates ordinary nuisance quantities on the inner
70%:

\[
g(r)=g_{\rm bar}(r)+
\lambda\frac{G M_{\rm bar}}{r_*r}
\frac{a_0}{a_0+g_{\rm bar}(r)},
\]

with `lambda = 10.5`, `a0 = 1.2e-10 m/s^2`, and `r* = 200 kpc`. No gravity
parameter is fitted per galaxy. Disk and bulge mass-to-light ratios, distance,
and inclination are fitted separately for each law using only the inner curve.

The primary continuous score is

\[
S=\frac{{\rm MSE}_{\rm RAR}-{\rm MSE}_{\rm screened}}
        {\langle V_{\rm obs}^2\rangle_{\rm outer}},
\]

so positive values favor the screened law.

## Scope of the statistical search

The code inventories **396 local attributes**:

| Tier | Attributes | What is included |
|---|---:|---|
| Core pre-outcome | 167 | SPARC photometry, gas/disk/bulge masses and geometry, morphology, measurement errors, references, distance method, sky position, two void summaries, component-only radial geometry, and inner observed-curve summaries |
| Extended environment | 120 | All locally available multiscale Cosmicflows void-cage, shell, Yukawa, power-law, anisotropy, and tidal descriptors |
| Mechanistic inner-fit | 88 | Inner-fit nuisances and residuals for screened, RAR, MOND, and NFW; screened-minus-RAR fit shifts; source mass and tail fractions evaluated without outer observed velocities |
| Outer descriptive leakage | 21 | Quantities using observed outer velocities; retained only to explain outcomes after the fact |

After eligibility and missing-data checks, the run performs:

- 303 numeric univariate tests and 8 categorical tests;
- every eligible one-variable midpoint split with at least 15 galaxies on each
  side: **53,522 rules**;
- predeclared two-variable, two-direction conjunctions at the 25th, 50th, and
  75th percentiles with at least 12 galaxies on each side: **156,967 rules**;
- Benjamini-Hochberg false-discovery-rate correction within the declared test
  families;
- repeated five-fold out-of-sample classification and continuous-score
  regression for nine feature-family ablations, using both regularized linear
  and nonlinear random-forest models;
- a fixed-label permutation test, 5,000 bootstrap samples for the leading
  one-variable rules, alternate outcome definitions, fit-boundary exclusions,
  quality-1-only scoring, and leave-one-galaxy influence checks.

This is exhaustive over the available measurements and the declared threshold
families. It is not literally exhaustive over all possible mathematical
functions, which would be infinite. The thousands of significant threshold
rows are highly correlated neighboring cuts, not thousands of independent
discoveries.

## The galaxies that improve

The 41 galaxies, ordered by normalized improvement, are:

`UGC07577`, `UGC00731`, `UGC05764`, `F568-V1`, `UGC04325`, `UGC12632`,
`UGC06446`, `NGC2683`, `UGC02916`, `UGC07125`, `UGC12506`, `UGC07524`,
`UGC05829`, `NGC3877`, `UGC05716`, `UGC06983`, `UGC12732`, `UGC05721`,
`NGC0289`, `NGC3953`, `NGC4100`, `UGC05918`, `NGC0801`, `IC4202`,
`UGC06930`, `NGC5033`, `NGC4559`, `UGC02259`, `NGC5055`, `NGC3992`,
`NGC2903`, `NGC2998`, `UGC09133`, `NGC6946`, `NGC5985`, `NGC5907`,
`NGC6674`, `UGC02487`, `NGC5005`, `UGC02953`, and `UGC05253`.

The complete roster includes the two RMSE values, every outcome definition,
and all 396 attributes in
`results/solar_screened_improver_forensics/complete_improver_roster.csv`.

The largest win is `UGC07577`: outer RMSE falls from `11.826` to
`3.095 km/s`. The final ranked win, `UGC05253`, is marginal: `10.096` to
`9.903 km/s`. The result is therefore not one homogeneous galaxy type.

## What is physically different?

### Strong pre-outcome differences

| Attribute | Improvers | Worseners | Effect/test |
|---|---:|---:|---|
| Inner observed fractional scatter | mean `0.171`, median `0.184` | mean `0.274`, median `0.288` | Hedges `g=-0.783`; Mann-Whitney global `q=0.0021`; one-feature AUC `0.719` |
| Inner observed log slope | mean `0.261`, median `0.258` | mean `0.450`, median `0.517` | `g=-0.705`; `q=0.0055`; AUC `0.700` |
| Catalog inclination | mean `57.2 deg`, median `55 deg` | mean `67.3 deg`, median `66 deg` | `g=-0.655`; `q=0.0098`; AUC `0.688` |

In plain language, the winning inner curves rise less sharply and fluctuate
less around a smooth trend. Lower inclination also matters. This may be partly
physical and partly observational: edge-on deprojection, dust, disk thickness,
warps, and noncircular motion are harder to model with the available thin-disk
templates.

### Mass and spatial scale

Improvers have a larger typical baryonic mass: median `3.96e10 Msun`, versus
`6.02e9 Msun` for worseners. The continuous normalized skill rises with log
mass (Spearman `rho=0.406`, global `q=2.12e-5`), but the direct two-group
continuous-mass test is weaker after correction (`q=0.072`). The predeclared
nonlinear mass classes do distinguish the groups (`q=0.0085`, corrected
Cramer `V=0.291`):

| Baryonic-mass class | Wins / total | Win rate |
|---|---:|---:|
| Dwarf, `<1e9 Msun` | 4 / 19 | 21.1% |
| Intermediate, `1e9` to `<1e11 Msun` | 23 / 90 | 25.6% |
| Giant, `>=1e11 Msun` | 14 / 22 | 63.6% |

Improvers also have larger median disk scale, inner endpoint, outer start
radius, HI mass, and characteristic inner speed. Those variables correlate
strongly with mass and curve extent; their direct win/loss group tests do not
remain globally significant after all 303 numeric comparisons. They should be
read as one correlated “larger, faster, better-resolved galaxy” family, not as
separate discoveries.

### Inclination

The predeclared inclination classes also distinguish winners (`q=0.0085`,
corrected Cramer `V=0.281`):

| Inclination class | Wins / total | Win rate |
|---|---:|---:|
| Moderate, `30 <= i < 50 deg` | 14 / 25 | 56.0% |
| Intermediate, `50 <= i < 70 deg` | 19 / 57 | 33.3% |
| Edge-on, `i >= 70 deg` | 8 / 49 | 16.3% |

This is a warning as well as a clue. A successful physical law should not need
to know the observer's viewing angle. An inclination dependence can expose
missing three-dimensional disk geometry or a measurement/deprojection
systematic rather than new gravity.

### Characteristics that do not distinguish the wins

| Family | Win rates | Global categorical q | Conclusion |
|---|---|---:|---|
| Stellar structure | disk 28.8%; mixed disk-bulge 50.0%; bulge 30.8% | `0.316` | No detected bulge-class effect |
| Hubble family | early 30.8%; Sbc-Scd 38.6%; late 26.2% | `0.400` | No detected Hubble-type effect |
| Gas fraction | poor 41.5%; mixed 25.0%; rich 28.0% | `0.316` | No reliable class effect |
| Surface brightness | high 32.8%; intermediate 21.6%; low 40.7% | `0.316` | No reliable class effect |

The continuous stellar bulge fraction is nearly identical: mean `0.0733` in
improvers and `0.0709` in worseners; both medians are zero (`q=0.646`). Thus the
new analysis strengthens the prior conclusion that a large bulge is not the
reason the equation succeeds or fails.

## The strongest distinction is the fitting mechanism

The dominant feature is

\[
\Delta\Upsilon_{\rm disk}=
\Upsilon_{\rm disk,screened}-\Upsilon_{\rm disk,RAR}.
\]

| Inner-fit quantity | Improvers | Worseners | Separation |
|---|---:|---:|---:|
| `screened minus RAR` disk M/L | mean `-0.012`, median `-0.047` | mean `+0.439`, median `+0.242` | AUC `0.802`; `g=-0.719`; global `q=1.02e-5` |
| RAR disk M/L | mean `0.646`, median `0.593` | mean `0.525`, median `0.456` | AUC `0.733`; global `q=0.00137` |
| RAR inner mean bias | `-1.28 km/s` | `+0.92 km/s` | AUC `0.726`; global `q=0.00149` |

The clean interpretation is:

1. In the winners, RAR's inner fit already prefers a relatively substantial
   stellar disk.
2. The screened equation can fit the same inner data without raising the disk
   M/L above the RAR value.
3. On the untouched outer data, RAR then overpredicts by `+10.22 km/s` on
   average, while the screened model's average residual is only `-1.25 km/s`.
4. In the 90 failures, RAR is already nearly unbiased (`-1.14 km/s`), whereas
   the screened model undershoots by `-17.73 km/s`.

The tail itself lowers same-nuisance error in all 41 primary winners and in 54
of the 90 failures. Across all galaxies, it helps 95/131 when the RAR
comparison is forced to use the screened law's nuisance values. But after RAR
is allowed its own inner calibration, only 41 remain wins. This means the
primary result is a **force-law plus nuisance-calibration interaction**, not a
standalone demonstration that the added tail has the correct radial physics.

The strongest data-mined one-variable rule is
`screened-minus-RAR disk M/L <= -0.0336`: 23/30 galaxies inside the rule win
(76.7%), compared with 18/101 outside (17.8%). The odds ratio is `15.15`, the
all-rule FDR value is `0.000127`, and the bootstrapped risk-difference interval
is `[0.412, 0.748]`. This cutoff is exploratory and must be frozen before use
on new galaxies.

An even sharper exploratory conjunction—MOND inner-fit disk M/L above `0.4835`
and the screened-minus-RAR shift at or below `-0.0279`—contains 20 wins among
24 galaxies (83.3%), versus 21/107 outside. Its odds ratio is `20.48` and
pair-family `q=0.000101`. It is also data-mined, correlated with neighboring
rules, and not an independent result.

## Can the differences predict a win out of sample?

Repeated five-fold out-of-fold results provide the most useful guard against
the threshold search overfitting:

| Feature family | Best model | ROC AUC | Average precision | Balanced accuracy |
|---|---|---:|---:|---:|
| Environment/position only (146) | random forest | `0.508` | `0.296` | `0.442` |
| Intrinsic catalog + inner curve (107) | random forest | `0.709` | `0.498` | `0.596` |
| Core pre-outcome (155) | random forest | `0.728` | `0.492` | `0.605` |
| Mechanistic inner-fit only (82) | random forest | **`0.805`** | **`0.677`** | `0.738` |
| Intrinsic + mechanistic (189) | random forest | `0.795` | `0.650` | `0.715` |
| Core + mechanistic (237) | random forest | **`0.809`** | `0.666` | **`0.739`** |
| All pre-outcome (352) | random forest | `0.789` | `0.630` | `0.720` |
| All plus outer descriptive leakage (372) | random forest | `0.781` | `0.620` | `0.726` |

The best repeated-run mean AUC is `0.801 +/- 0.020` for core plus mechanism;
the eight repeats range from `0.764` to `0.830`. A fixed core-only regularized
logistic permutation test gives mean AUC `0.615`, one-sided `p=0.0319` against
500 shuffled-label samples. These are credible exploratory patterns, but the
same 131 galaxies influenced feature construction and the research question.

Continuous improvement magnitude is harder to predict. The best nonlinear
model has out-of-fold Spearman correlation `0.670` but only `R2=0.111`; one
large positive outlier (`UGC07577`) and several large failures strongly affect
magnitude-based summaries. Classification of the win/loss pattern is more
stable than prediction of the exact improvement.

## Environment and void tests

No available environment variable survives the corrected direct improver vs
worsener comparison:

- grouped-CF4 void score: means `0.0086` versus `0.0446`, global `q=0.594`;
- Local-Voids wall score: means `0.0248` versus `0.0366`, global `q=0.410`;
- best of the 120 extended variables by raw group difference, grouped-CF4
  shell void-cell count: raw `p=0.0284`, global `q=0.148`.

Some multiscale variables correlate with the *continuous* skill after FDR, but
the environment-only cross-validation is at chance (AUC `0.508` for the forest
and `0.493` for logistic regression). Adding all extended environment features
to the core degrades random-forest AUC from `0.728` to `0.645`.

Therefore this dataset does not support the claim that measured surrounding
void strength determines which galaxies benefit. The environment catalogs are
coarse and incomplete at individual-galaxy scales, so this is evidence against
the currently measured environmental proxies—not proof that no possible
environment field can matter.

## Observational and outcome robustness

- The same 41 galaxies beat simple MOND, because fixed RAR and simple MOND are
  extremely close for this sample.
- Chi-square weighting gives 40 wins; catalog-space velocities give 39.
- Forty wins exceed 10% MSE improvement and 38 exceed 20%, so the count is not
  mostly numerical ties.
- Quality-1 galaxies give 28/91 wins (30.8%), almost the full-sample 31.3%.
- Excluding screened fits at a nuisance boundary gives 41/121 wins; excluding
  any boundary in either fit gives 39/117.
- Distance-method class is a near-signal but does not survive correction
  (`q=0.070`). Distance, luminosity and inclination errors, reference count,
  sky coordinates, and individual reference-code indicators do not show a
  corrected binary group difference.
- Removing the most influential positive galaxy, `UGC07577`, makes the already
  negative full-sample mean skill more negative; it cannot reverse the failed
  universal verdict.

The outer rotation-shape split is useful only descriptively because it uses
the held-out observed velocities: declining curves win 11/22 (50.0%), flat
curves 23/62 (37.1%), and rising curves 7/47 (14.9%), categorical `q=0.0145`.
It confirms that the screened model especially underpredicts rising outskirts,
but it cannot prospectively select a galaxy without already observing the
answer.

## What the local data cannot distinguish

The analysis leaves nothing out that is present in the local project data, but
the following scientifically important properties are absent:

- bar presence and bar strength;
- spiral-arm multiplicity and pitch angle;
- resolved warps, lopsidedness, and noncircular-flow maps;
- vertical disk thickness and vertical stellar kinematics;
- molecular-gas and circumgalactic-gas profiles;
- stellar age, color, metallicity, IMF, and radial mass-to-light gradients;
- complete group membership, nearest massive neighbor, and external tidal
  field below Cosmicflows resolution;
- an independently measured dark-halo mass or halo shape;
- full point-to-point rotation-curve covariance;
- weak or strong lensing measurements for the same galaxies.

Inclination and inner-curve smoothness make disk thickness, warps, and
noncircular motion particularly important next data. A true three-dimensional
gravity effect must be separated from those observational systematics.

## Scientific conclusion and next test

The 41 wins teach us where this formula's mass normalization happens to work,
not that the formula is the universal law. The existing tail scales, when open,
as `v_tail^2 proportional to Mbar`; it naturally favors a high-mass range and
cannot reproduce the square-root baryonic mass scaling across dwarfs and
giants with one fixed `lambda`.

The most defensible next step is not to assign different gravity parameters to
the discovered subgroups. It is to freeze the inner-fit selection signature
before looking at new outer data, then test it on independent rotation curves
with better 3D and noncircular-motion information. If the rule transfers, it
would identify a real regime in which the screened equation is useful. If it
does not, the apparent regime was sample-specific fitting behavior.

## Reproduction and full outputs

```powershell
python scripts/analyze_solar_screened_improvers.py
```

The protocol is
`configs/solar_screened_improver_forensics_protocol.json`. The output directory
contains the full feature manifest, all univariate tests, all single and paired
rules, repeated cross-validation scores and predictions, permutation null,
bootstrap intervals, robustness table, influence table, per-galaxy outcomes,
complete winner roster, plot, and machine-readable JSON report:

`results/solar_screened_improver_forensics/`
