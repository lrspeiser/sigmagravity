# Sigma v19AA member-counterpart association results

## Result

The frozen position-only protocol completed for all 141 spectroscopic members,
and every integrity gate passed.  It produced 831 unified HSC/NSC candidates
from 779 unique HSC detections and 226 unique NSC detections.  The catalog
unification contains 174 reciprocal cross-survey pairs.

| Cluster | Members | Secure | Ambiguous | No candidate |
|---|---:|---:|---:|---:|
| Abell 2146 | 63 | 13 | 46 | 4 |
| Bullet | 78 | 0 | 77 | 1 |

This is a catalog-association result.  No photometric transformation, stellar
mass, mass current, lensing target, halo map, gravity residual, or gravity
parameter entered the calculation.

## Why the Bullet result is zero

The zero is not evidence against a directional or long-wave gravity model.  It
is a measured information limit in the published member coordinates.

At the Bullet declination, a right ascension rounded to one whole time-second
corresponds to an east-west half-bin of about 4.2 arcsec.  The north-south
half-bin is 0.5 arcsec.  This broad rectangle contains many catalog sources:
the 77 nonempty Bullet rows have a median of 9 unified candidates and a maximum
of 53.

Under the frozen `Q = 0.80` prior, the largest observed minimum-prior posterior
is 0.8454.  It therefore cannot cross the preregistered 0.90 secure threshold,
even when a candidate has a large likelihood advantage over its neighbors.
The protocol correctly retained all 77 as ambiguous instead of lowering the
threshold after seeing the answer.

## Gate anatomy

The counts below show how many members satisfy each gate separately; they are
not sequential counts.

| Gate | Abell 2146 | Bullet |
|---|---:|---:|
| At least one candidate | 59 | 77 |
| Posterior at least 0.90 at every frozen prior | 14 | 0 |
| Top/second likelihood ratio at least 10 | 59 | 53 |
| Global assignment agrees with local top candidate | 17 | 53 |
| Dual-survey or repeated-detection support | 58 | 64 |
| Not rejected as a high-PM point source | 59 | 77 |
| All secure gates | 13 | 0 |

The Abell 2146 result also warns against treating a nonempty cone as an
identity.  Its published coordinates are much more precise, but only 13 rows
simultaneously beat the null hypothesis and the global one-to-one competition.

## Consequence for the field-theory program

The long-wavelength proposal says that an additional gravitational response is
nearly constant over a star system but can change over kiloparsec or larger
separations.  Its directional versions need the actual baryonic locations,
motions, or stresses.  V19AA establishes that Abell 2146 has a small secure
anchor set, while the Bullet layout must be represented as a probability
distribution rather than a single catalog map.

The next defensible step is a separately frozen photometric/SED association
likelihood.  It may use the published Bullet B/R/I values and the HSC/NSC
multi-band measurements, calibrated on unambiguous associations, but it may
not use lensing residuals or select identities because they improve a gravity
fit.  Any mass-current map must then marginalize the remaining counterpart and
transverse-velocity uncertainty.

## Reproducibility

- Frozen protocol: `configs/sigma_v19aa_member_counterpart_association.json`
- Runner: `scripts/run_sigma_v19aa_member_counterpart_association.py`
- Unified candidates: `data/derived/sigma_v19aa_member_counterpart_association/unified_candidates.csv`
- Candidate posteriors: `data/derived/sigma_v19aa_member_counterpart_association/candidate_posteriors.csv`
- Member decisions: `data/derived/sigma_v19aa_member_counterpart_association/member_associations.csv`
- Machine-readable report: `results/sigma_v19aa_member_counterpart_association/report.json`

The result tests independently verify hashes, exact state counts, posterior
normalization at all four priors, the one-to-one assignment, every secure gate,
and the measurement-only claim boundary.
