# Gravity-arc tomography: first cluster test

## Result in one sentence

The data can be represented as conserved baryon-sourced influence traveling
roughly 70--80 kpc before reappearing in the locations conventionally assigned
to dark matter, but the three-cluster pilot does **not** yet identify one
predictive gravity-arc law: the primary analysis fails its held-out gate, while
a stricter galaxy-member selection reveals one intriguing universal
center-return rule that now needs an untouched-cluster test.

## The proposed new-field picture

Let baryonic matter be the only source of gravitational flux, written as

$$
\nabla_\mu J^\mu=4\pi G\rho_b.
$$

This is a bookkeeping law: baryons create the field, so an apparent dark-mass
peak is not permitted to create new flux. New physics enters through a
nonlocal return kernel,

$$
\kappa_{\rm pred}(\mathbf y)=
(1-f)B_w(\mathbf y)+
f\int B(\mathbf x)K_\theta(\mathbf y\mid\mathbf x,E_b)\,d^2x,
$$

with

$$
\int K_\theta(\mathbf y\mid\mathbf x,E_b)\,d^2y=1.
$$

Here $B$ is the observed baryonic-light map, $f$ is the fraction whose
influence is rerouted, and $K_\theta$ says where that influence returns. The
normalization is important: this first test moves the influence associated with
each source but does not create unlimited extra gravity. $E_b$ allows a path to
depend only on the baryonic environment, never on the target dark-matter map.

This is a spatial-shape model. It does not yet predict absolute lensing
strength. A later closure would need something like

$$
\kappa_{\rm physical}=A_bB+A_{\rm arc}qR[B],
$$

where stellar mass, gas mass, and a physically derived path multiplicity $q$
would determine the amplitude.

## How the backtracking works

The inverse calculation treats every galaxy-light pixel as a possible origin
and every positive lensing-convergence pixel as a possible destination. A
balanced optimal-transport calculation assigns origins to destinations while:

1. preserving the total normalized source and destination weights;
2. preferring shorter projected routes; and
3. never inventing a source at the apparent dark-mass location.

The result is the shortest regularized explanation, not the unique historical
path of gravity. It provides path-length and direction statistics that can be
used to design forward laws.

The forward test is harder and more important. Six frozen families predict a
lensing shape using galaxy positions alone: local smoothing, a smooth central
halo control, isotropic return rings, return toward the baryonic center, return
along the combined baryonic field, and return toward the strongest neighboring
galaxy. Parameters were selected using two clusters and scored on the untouched
third cluster.

## Data and frozen test

The pilot uses three RELICS clusters: A2537, MACS J0417, and MACS J0949. The
inputs contain 7,883 catalog rows, 2,275 usable F160W galaxy tracers, 360 strict
photometric-redshift members, and 100 lens-map realizations per cluster. Within
the common 300 kpc scoring aperture, 1,423 sources contribute.

The code scored 1,571 formula/parameter settings, or 4,713 cluster scores for
each of the primary and strict-member source definitions. The protocol and
acquisition records were frozen before comparing galaxy locations with lensing
map structure.

## What the inverse paths say

At the preregistered 50 kpc regularization scale:

| Cluster | Mean path | Median path | 90th percentile | Share within 100 kpc | Share beyond 150 kpc |
|---|---:|---:|---:|---:|---:|
| A2537 | 74.1 kpc | 65.1 kpc | 141.8 kpc | 74.5% | 7.4% |
| MACS J0417 | 80.8 kpc | 76.7 kpc | 138.5 kpc | 67.8% | 6.1% |
| MACS J0949 | 71.9 kpc | 65.6 kpc | 130.1 kpc | 76.3% | 5.1% |

The path scale changes when the optimal-transport regularization changes, so
70--80 kpc is not a measured physical constant. The robust statement is that
the minimum-cost attribution is mostly nonlocal on tens to roughly one hundred
kiloparsecs, with only a small long-distance tail.

The central lensing peaks are still mostly traceable to the brightest central
galaxy: about 72%, 75%, and 81% of their assigned origin weights. Some secondary
peaks contain mixtures of a nearby galaxy and a much more distant source; that
is the kind of feature on which a genuine routing theory must improve over a
smooth halo.

## Held-out prediction results

Lower Jensen--Shannon (JS) divergence is better. Pearson correlation is shown
as a secondary shape score.

### Primary soft-membership analysis

| Held-out cluster | Training-selected winner | JS | Pearson | Improvement over local light | Improvement over central-halo control |
|---|---|---:|---:|---:|---:|
| A2537 | central halo, 100 kpc | 0.0348 | 0.901 | 12.2% | 0.0% |
| MACS J0417 | center return | 0.0821 | 0.802 | 6.9% | -23.0% |
| MACS J0949 | central halo, 100 kpc | 0.0444 | 0.879 | 13.2% | 0.0% |

The frozen success gate fails. A routing law was not the universal winner, and
the center-return rule that training selected for MACS J0417 was substantially
worse than a simple smooth central halo there.

### Strict photo-z member sensitivity

This alternative source definition produced the most interesting clue. Every
leave-one-cluster-out fold independently selected the exact same setting:

$$
\ell(r)=\operatorname{clip}\left[
250\left(\frac{r}{100\ {\rm kpc}}\right)^{-1/2},
50,750\right]\ {\rm kpc},
$$

with 50% of the source weight returned toward the baryonic luminosity center
and deposited in a 50 kpc-wide endpoint. The selected setting improved on
local-light smoothing in all three held-out clusters by 31.5%, 37.3%, and
23.7%.

| Held-out cluster | Arc JS | Local JS | Central-halo JS | Arc vs local | Arc vs central |
|---|---:|---:|---:|---:|---:|
| A2537 | 0.0345 | 0.0503 | 0.0458 | 31.5% better | 24.8% better |
| MACS J0417 | 0.0745 | 0.1188 | 0.0726 | 37.3% better | 2.5% worse |
| MACS J0949 | 0.0523 | 0.0685 | 0.0579 | 23.7% better | 9.8% better |

This still does not pass the gate because it loses narrowly to the central-halo
control in MACS J0417. More importantly, the return scale, negative exponent,
and endpoint width lie at boundaries of the searched grid. The formula can
therefore be acting as an elaborate way to construct a smooth central halo.
Extending and refitting those boundaries on these same three clusters would be
exploration, not confirmation.

## Controls and what they mean

- Randomizing galaxy angles while preserving their radii shows that real
  angular structure is useful in A2537, moderately useful in MACS J0949, and
  not useful in MACS J0417. A universal directional law is therefore not yet
  supported.
- Across 100 lens-map realizations per held-out cluster, the score uncertainty
  is much smaller than the differences above. MACS J0417's failure is not a
  sampling accident within that model ensemble.
- If the broad positive convergence sheet is left unsubtracted, only local or
  smooth central controls win. Background handling is therefore decisive.
- The inverse transport reproduces its required source and target marginals to
  better than $10^{-8}$, so numerical leakage is not creating the effect.

## The largest scientific limitation

The lensing targets are Lenstool reconstructions, not raw light-deflection
observations. Standard lens models assign cluster-member subhalos using galaxy
light, so part of a galaxy-to-convergence correlation can be built into the
target. This makes the current experiment a valid test of spatial
representations, but not independent evidence that gravity physically followed
the inferred arcs. F160W light is also an incomplete baryonic map: it omits the
hot intracluster gas and needs a stellar mass-to-light model.

## Relation to existing ideas

The broad claim that baryons create a nonlocal effective response resembling
dark matter is already present in Mashhoon's nonlocal gravity. Environmental
redirection of gravitational field lines is also central to Refracted Gravity.
What is distinctive here is the falsifiable construction: conserve a
baryon-sourced budget, infer minimum paths with optimal transport, then require
one target-blind directional kernel to predict held-out clusters. That is a new
experimental framework in this project, not yet a demonstrated new fundamental
theory. Relevant starting points are [Mashhoon's nonlocal-gravity
formulation](https://arxiv.org/abs/1101.3752), [observational tests of nonlocal
gravity](https://arxiv.org/abs/1401.4819), and [the original Refracted Gravity
proposal](https://arxiv.org/abs/1603.04943).

The public inputs and their official product definitions are described by the
[RELICS archive at MAST](https://archive.stsci.edu/hlsp/relics).

## The next decisive experiment

**Update:** this experiment has now been completed on ten untouched clusters
with both Lenstool and GLAFIC controls. C0351 failed its confirmation gates, but
a one-at-a-time widening from 50 to 60 kpc improved 8/10 systems in both model
families. See
[`GRAVITY_ARC_FRESH_SAMPLE_RESULTS.md`](GRAVITY_ARC_FRESH_SAMPLE_RESULTS.md).

The strict-member rule and candidate C0351 are now spent: their parameters must
be locked exactly as written. The next run should add at least ten previously
unused clusters and compare that fixed rule against fixed local-light and
central-halo controls. It should use multiple independent reconstruction
families for each cluster, then graduate to raw multiple-image positions and
weak-shear catalogs. A proper baryonic source map must combine stellar mass and
X-ray or SZ gas.

Success would mean the locked routing rule improves both spatial metrics on the
large majority of new clusters and does so across reconstruction methods.
Failure would mean the apparent universal rule was a three-cluster or
source-selection artifact. Only after that test should the project add an
absolute-strength law and ask whether the same field can reproduce galaxy
rotation curves while respecting Solar-system bounds.

## Reproduction

```powershell
python scripts/download_gravity_arc_tomography_catalogs.py
python scripts/audit_gravity_arc_tomography_inputs.py
python scripts/run_gravity_arc_tomography.py
pytest tests/test_gravity_arc_tomography.py tests/test_gravity_arc_tomography_results.py
```

Machine-readable results are in
`results/gravity_arc_tomography/report.json`; the complete score grids and
uncertainty files are in the same directory.
