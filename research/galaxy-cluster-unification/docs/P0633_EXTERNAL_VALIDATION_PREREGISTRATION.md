# P0633: untouched external validation and rejection gates

P0633 freezes the next evaluation before downloading or scoring any selected
target product. Its purpose is to prevent another nominal holdout from becoming
a development set through repeated inspection.

## Locked samples

The galaxy test contains 13 LITTLE THINGS dwarfs with public resolved HI cubes,
moment maps, and optical/infrared imaging. They are the Iorio et al. 3D-kinematic
sample after excluding the three SPARC overlaps (`DDO154`, `DDO168`, and
`NGC2366`) and conservatively excluding `WLM` because its token already occurred
in baseline metadata. The remaining 13 target names and compact aliases have
zero occurrences in the preserved Git tree.

The cluster test contains four RELICS systems: Abell S295, MACS J0025.4-1222,
MACS J0159.8-0849, and PLCK G287.0+32.9. Each has public HST products and two
archived lens-model methods, but no canonical name or archive slug occurred in
the preserved tree. The published lens models are comparators only and may not
be used to construct the baryonic input.

The official data sources are the [NRAO LITTLE THINGS data
release](https://science.nrao.edu/science/surveys/littlethings/data), the
[Iorio et al. resolved 3D kinematic analysis](https://doi.org/10.1093/mnras/stw3285),
and the [MAST RELICS archive](https://archive.stsci.edu/hlsp/relics) (DOI
`10.17909/T9SP45`).

## What stays sealed

The baryonic inputs may be opened: HI moment-0 maps, calibrated stellar light,
instrument metadata, cluster-member light, and X-ray products needed for a gas
model. Dynamics and lensing targets remain sealed until the field solvers, the
candidate equation, every universal parameter, and a prediction manifest are
committed and hashed.

`DDO154` is explicitly project-spent and is the only LITTLE THINGS commissioning
object. It can expose every ingestion and projection bug without spending a
P0633 target.

## Gates fixed before fitting

All Newtonian Poisson, AQUAL, and QUMOND analytic and convergence tests must pass
before external scoring. The candidate then must:

- use zero per-object gravity parameters;
- remain within 5% of the best frozen full-field MOND comparator on both the
  equal-galaxy circular-speed and resolved velocity-field metrics;
- keep every predeclared morphology bin within 25% of that comparator;
- converge every held-out strong-lens image root and reproduce every held-out
  family topology;
- achieve raw image and critical-curve errors no more than 1.25 times the
  per-cluster compact-halo comparator while closing at least 75% of the
  baryon-only-to-halo image-RMS gap;
- satisfy the existing 2.3e-5 PPN-gamma and 3.1 mas/century Mercury bounds, with
  metric quantities derived rather than assumed.

The gates are conjunctive. A good galaxy average cannot rescue failed cluster
topology, and one unusually favorable cluster cannot rescue the cluster sample.

## Frozen evidence

Run:

```powershell
$env:PYTHONPATH='src'
python scripts/freeze_p0633_external_validation.py
python -m pytest tests/test_p0633_external_validation_preregistration.py -q
```

The generated ledger records the protocol hash, the baseline commit, every
historical alias check, and the fact that none of the four target directories
existed at freeze time. Once any target observable is unsealed, P0633 becomes
project-spent and later formula changes are exploratory.
