# Sigma V19CG whole-repository cluster holdout contamination

## Decision

The eight-cluster V19BH shortlist is retired from prospective whole-object
holdout use. It remains useful as development source diversity, but it cannot
support the project's eventual blind cluster claim.

This correction has two independent causes.

First, the original alias audit covered the active research subfolder rather
than every tracked repository file. A whole-repository audit found that six of
the eight shortlisted SGAS clusters had already entered older Sigma cluster
catalogs, derived gravity results or coherence calculations. Even if their raw
image coordinates remained unopened, those systems had already supplied
gravity information to the broader project and were not untouched objects.

Second, a metadata search against the SGAS lensing paper unexpectedly returned
its multiple-image table inline. Raw coordinate rows for ten SGAS clusters
became visible, including three V19BH candidates: SDSS J0851+3331,
SDSS J0952+3434 and SDSS J1002+2031. No coordinate value is copied into the
repository and none is used for a score, parameter choice or replacement
selection. The affected systems are nevertheless permanently spent as raw
coordinate holdouts.

## Corrected ledger

| State | Systems |
|---|---:|
| Original V19BH shortlist | 8 |
| Previously used in older Sigma gravity analyses | 6 |
| Raw-coordinate-exposed shortlist systems | 3 |
| Unique systems disqualified by either route | 7 |
| Remaining unspent source reserve | 1 |
| Admitted prospective holdouts | 0 |

The only original candidate not disqualified is SDSS J1226+2149. It remains a
source-incomplete reserve: its imaging differs from the common near-infrared
path and its projected pre-merger pairing requires a separate baryonic
deprojection. It is not admitted.

All six systems that passed V19BT's direct HST-plus-Chandra imaging preflight
are among the six systems already used in earlier whole-repository Sigma work.
Their source products remain valuable for development, but none can be a clean
whole-object holdout.

## What this supersedes

- V19BH remains a valid metadata and acquisition-boundary exercise, but its
  eight systems cannot supply the final untouched sample.
- V19BT remains a valid source-imaging preflight for development data, but its
  six direct systems cannot supply the final untouched sample.
- V19CF's galaxy-readiness conclusion is unchanged.
- V19CF's statement that the cluster source universe is prospectively ready is
  superseded. A new cluster universe is required.

## New acquisition rule

The next cluster candidate search must begin with a whole-repository identity
and alias audit. At least eight systems must be absent from every earlier
Sigma gravity fit, derived cluster score, mechanism diagnostic and raw target
opening. Source files and raw lensing targets must be separate acquisition
containers: source products can be inspected, while raw coordinates remain a
hash-only sealed payload until the action and universal constants are frozen.

No replacement is selected here. That prevents an accidental exposure from
turning into an outcome-informed substitution.

## Reproduction

```powershell
python scripts/audit_sigma_v19cg_whole_repo_cluster_holdout_contamination.py
python -m pytest tests/test_sigma_v19cg_whole_repo_cluster_holdout_contamination.py -q
```

The machine-readable result is
`results/sigma_v19cg_whole_repo_cluster_holdout_contamination/report.json`.
