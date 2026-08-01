# P0554 subcritical route transfer results

## Outcome

The MACS1931-selected route amplitude eta = 0.30 transfers without changing a
single family root count in the other four raw clusters. It also preserves all
observed-seed held-out root recoveries. Its positional effect, however, is too
small and inconsistent to pass the frozen strong-transfer threshold.

Across 15 source families for which both formulas assign every published image,
the other-four-cluster equal-family RMS changes from 6.939 to 6.910 arcseconds:
a 0.412% improvement. Only two of four clusters improve. The frozen strong bar
required at least 1% aggregate improvement and improvement in at least three
clusters, in addition to no topology cost. The formula therefore fails strong
transfer but passes the predeclared weak, topology-safe transfer definition.

## Frozen comparison

The two formulas were

$$
\boldsymbol{\alpha}_0=\boldsymbol{\alpha}_{\rm radial},
$$

and

$$
\boldsymbol{\alpha}_{0.30}=\boldsymbol{\alpha}_{\rm radial}
+0.30\,s_{\rm route}\,\delta\boldsymbol{\alpha}_{\rm A0279}.
$$

Eta was not refitted. Each formula received the same eight optimizer-start
seeds in each cluster and could refit only the same six ordinary lens-geometry
quantities. The test then ran a global root search for both formulas over all 27
source families and 77 published images.

## Position transfer by cluster

MACS1931 is shown for context but excluded from the primary transfer score
because it selected eta = 0.30.

| Cluster | Common complete families | eta=0 RMS | eta=0.30 RMS | Improvement | Primary transfer? |
|---|---:|---:|---:|---:|:---:|
| MACS0329 | 6 | 8.846 | 8.764 | +0.936% | yes |
| MACS0429 | 1 | 4.692 | 4.705 | -0.285% | yes |
| MACS1115 | 2 | 20.385 | 20.382 | +0.014% | yes |
| RXJ2129 | 6 | 0.923 | 0.934 | -1.088% | yes |
| MACS1931 | 4 | 21.662 | 21.560 | +0.468% | no; selection system |

This is not a consistent improvement across objects. The aggregate positive
value is driven mainly by MACS0329; RXJ2129, already the best-fit system in
absolute terms, becomes slightly worse.

## Image multiplicity and held-out roots

The topology result is cleaner:

| Measure across all five clusters | eta=0 | eta=0.30 |
|---|---:|---:|
| Missing-multiplicity families | 8 | 8 |
| Exact-multiplicity families | 12 | 12 |
| Demagnified-only-surplus families | 1 | 1 |
| Potentially observable-surplus families | 6 | 6 |
| Potentially observable-surplus roots | 7 | 7 |
| Observed-seed held-out roots recovered | 17 | 17 |

No one of the 27 family root counts changes. This independently confirms that
eta = 0.30 lies in a topologically quiet regime, not merely on the safe side of
MACS1931 family 2.

## Interpretation for the gravity-flow idea

There is a reproducible, low-amplitude angular response that can move predicted
images without creating new ones. Its sign is not universal at the present
shape: two clusters improve and two worsen. This argues against a single global
volume knob, but it does not reject baryonic gravity redirection.

The next productive variable is where the route field is placed. A physically
useful version should infer its direction and localization from observed
baryonic structure—member-galaxy positions, gas, intracluster light, and their
tidal geometry—rather than increasing one cluster-wide template everywhere.
That matches the inverse problem: start from lensing-inferred excess-deflection
locations, trace candidate conservative field lines backward, and test whether
they terminate on baryonic sources with one universal transport law.

The angular addition has zero monopole, so eta = 0.30 preserves the SPARC,
CLASH radial-profile, Mercury, and Solar-control scores of its radial parent by
construction. That preservation is not an independent fit improvement.

## Protocol metadata erratum

The frozen protocol's human-readable `systems` list accidentally retained
three stale labels. The executable `raw_contexts` loader evaluated RXJ2129,
MACS0329, MACS0429, MACS1115, and MACS1931, which are the same five raw datasets
used by the parent interaction analysis. Coverage still matched the frozen
counts of five systems, 27 families, and 77 images. The protocol was not edited
after scoring; the mismatch and actual labels are recorded in `report.json`.

## Limits

Four of five geometry fits for each formula touch at least one nuisance bound.
The root visibility threshold is not a calibrated completeness model, the lens
is simplified, and the four transfer clusters are still previously examined
project data rather than a new survey. No formula is promoted.

## Reproduction

```powershell
python scripts/run_p0554_subcritical_route_transfer.py
python scripts/run_p0554_subcritical_route_transfer.py --postprocess-only
python -m pytest tests/test_p0554_subcritical_route_transfer.py -q
```

Machine-readable outputs are in `results/p0554_subcritical_route_transfer/`.
