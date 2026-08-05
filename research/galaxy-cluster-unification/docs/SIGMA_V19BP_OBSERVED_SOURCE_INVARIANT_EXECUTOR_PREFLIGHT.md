# Sigma V19BP observed source-invariant executor preflight

## Decision

The missing terminal integration between the gas posterior and the source-only
decision is now implemented and frozen. V19BP consumes only passing, hash-bound
V19X4 gas-state products and V19BM stellar-morphology controls. It streams those
products through the commissioned V19BO map mathematics and applies the V19BN
decision engine without opening lensing, halo, galaxy-rotation or holdout data.

The preflight passes. Observed execution remains correctly blocked until V19X4
and V19BM have both produced their terminal reports.

## What will be tested

Each of the two development clusters is evaluated under all three frozen
temperature--density normalization correlations, $\rho=-0.9,0,+0.9$. Each
branch must separately survive all six preregistered spatial variants:

| Smoothing FWHM | Aperture radii |
|---:|---:|
| 50 kpc | 250, 350 and 500 kpc |
| 100 kpc | 250, 350 and 500 kpc |

The 50-kpc/350-kpc combination is the primary variant, chosen before observed
source scores exist. The other five are mandatory stability tests, not optional
alternatives. This creates 36 cluster/correlation/spatial conditions per
candidate, or 72 candidate evaluations across I4 and I5. No cluster,
correlation branch or spatial variant is averaged away.

## Frozen source decision

I4 is a projected thermodynamic-gradient tensor. It is tested in two logically
separate parts:

1. its axial direction must be detected, narrow, leave-one-region-out stable
   and stable across every spatial variant; and
2. its amplitude must be detected, nonredundant with the density controls and
   stable under the same perturbations.

I5 is scalar baroclinicity. It may satisfy the amplitude requirement if the I4
amplitude fails, but it can never substitute for the required I4 direction.
Action derivation is authorized only when I4 direction passes in both clusters
and all three gas-correlation branches, and either I4 amplitude or I5 scalar
also passes everywhere.

Every candidate is conditioned on five fixed nuisance predictors: four gas
density/morphology summaries and the within-cluster stellar-light percentile.
At least 32 regions require three-sigma joint gradient support. The candidate
must retain at least 20% PRESS-unexplained variance in 90% of posterior draws,
remain within 10% in activation and 10 degrees in I4 direction, and preserve
those limits through at least 90% of the frozen perturbations.

## Data flow and retained evidence

```text
V19X4 regional gas draws + common physical label grids
                           +
V19BM within-draw stellar morphology ranks
                           |
                           v
V19BO bounded gas-map feature stream
                           |
                           v
V19BN support, posterior, PRESS, omission and variant gates
                           |
                           v
one source-only result per cluster and correlation branch
```

The executor keeps full draw-level activation, I4 axis and PRESS novelty
outputs. It also keeps the support masks and q16/median/q84 regional summaries
of candidate components and all controls. Every terminal input and output is
checked by name, byte size and SHA-256 hash.

## Claim boundary

A pass would show that at least one robust, direction-carrying baryonic source
state exists in these two clusters. It would not show that gravity couples to
that state, that the state predicts lensing, or that it generalizes to the
cluster population. A failure is also informative: it forbids rescuing these
source candidates with lensing targets or a tuned action and instead requires
direct gas velocities or an independently preregistered merger sample.

Solar-System data do not enter this stage. Local/PPN behavior remains a later
hard exclusion gate after a candidate has first demonstrated one-metric
performance across varied galaxies and clusters.

## Reproduction

```powershell
python scripts/run_sigma_v19bp_observed_source_invariant_scoring.py --preflight-only
python -m pytest tests/test_sigma_v19bp_observed_source_invariant_scoring.py -q
```

The frozen evidence is
`results/sigma_v19bp_observed_source_invariant_scoring/preflight_report.json`.
