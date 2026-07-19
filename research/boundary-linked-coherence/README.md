# Boundary-Linked Coherence Evaluation

**Status:** research plan; no result in this folder is evidence for the mechanism yet.

This folder defines a falsifiable evaluation of a speculative physical interpretation of
Σ-Gravity: long-baseline correlations from interior baryonic sources may remain open in
low-acceleration, weakly screened directions and couple collectively to outer baryonic matter.
The resulting response could become disk-aligned and fall approximately as `1/r`.

The proposal is called **boundary-linked coherence (BLC)** here. “Link” means a nonlocal
correlation kernel, not a literal string and not a claim that photons gain mass or gravity.

## Why this is a separate research track

The canonical repository model is

\[
\Sigma_0 = 1 + A(L)\,\mathcal C\,h(g_N),
\]

with a QUMOND-like field equation, a kinematic coherence scalar, and a disk approximation
`W(r) = r/(ξ+r)`. It already contains two related exploratory implementations:

- `derivations/test_nonlocal_coherence_kernel.py` integrates a one-dimensional survival
  probability through a rough radial density profile.
- Supplementary Information §13b averages the canonical response with an exponential radial
  kernel and finds nearly the same aggregate results as the baseline.

Neither implementation tests the specific BLC claim. In particular, neither constructs a
symmetric source-to-boundary openness measure, separates mass-normalized from luminosity-driven
links, predicts directional effects, or establishes a conserved causal dynamics. This folder
keeps those questions isolated from the published baseline.

## The question this track must answer

Can one kernel, derived only from source-side observables, do all of the following?

1. Reduce to ordinary gravity in compact, high-acceleration, or incoherent systems.
2. Reproduce the canonical Σ-Gravity response without per-galaxy tuning.
3. Produce a `1/r` outer acceleration without unbounded `N²` growth.
4. Obey reciprocity, momentum conservation, weak-equivalence-principle behavior, and causal
   propagation.
5. Make at least one correct held-out prediction that the canonical radial window, MOND, and a
   conventional halo fit do not make.

If it cannot, BLC should be rejected as a physical mechanism even if a flexible version can fit
rotation curves.

## Competing hypotheses

| ID | Hypothesis | Link source | What would distinguish it |
|---|---|---|---|
| `H0` | No boundary links | Canonical `Σ₀` only | Openness, luminosity, and external orientation add no held-out information |
| `HM` | Mass-normalized BLC | Baryonic stress-energy | Boundary topology or environment matters after controlling for `g_N`, geometry, and kinematics; present luminosity does not |
| `HL` | Literal luminosity BLC | Emissivity or radiative energy | Present luminosity and stellar-population tracers matter at fixed baryonic structure |

`HM` is the cleaner primary hypothesis. Independent stellar emission is not expected to possess
a galaxy-wide shared optical phase, and a luminosity-weighted force risks assigning different
gravity to otherwise similar old and young populations. `HL` is retained as a useful falsifiable
alternative rather than being ruled out by wording alone.

## Research rules

- The canonical model and its locked parameters remain unchanged.
- No observed rotation velocity may enter a feature used to predict that same rotation velocity.
- All new tunable quantities are global or hierarchy-level parameters; no per-galaxy BLC knobs.
- Calibration and validation are separated by galaxy, not by radial point.
- A model-complexity penalty and uncertainty propagation are reported with every fit metric.
- Null, mass-normalized, and luminosity variants run through the same data filters and splits.
- Failed predictions remain in `RESULTS.md` when execution begins; they are not silently removed.

## Folder map

- [`MODEL_SPEC.md`](MODEL_SPEC.md) — minimal operational model, variants, and consistency
  requirements.
- [`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) — phased work plan, data products, statistics, and
  go/no-go gates.
- [`EXPERIMENT_MATRIX.csv`](EXPERIMENT_MATRIX.csv) — machine-readable experiment and decision
  registry.
- [`REFERENCES.md`](REFERENCES.md) — primary sources and their role in the evaluation.

## Recommended implementation layout

The first code change should add the following without editing the baseline regression formulas:

```text
research/boundary-linked-coherence/
├── blc/
│   ├── kernel.py          # openness, coherence, range, and normalization factors
│   ├── model.py           # H0/HM/HL predictions through one interface
│   ├── synthetic.py       # analytic disks, rings, gaps, companions, and null injections
│   └── metrics.py         # likelihood, information criteria, and held-out scores
├── configs/
│   ├── preregistration.yaml
│   └── splits.json
├── scripts/
│   ├── run_theory_checks.py
│   ├── run_synthetic_suite.py
│   ├── run_sparc_ablation.py
│   └── run_environment_test.py
├── outputs/               # generated, with provenance manifests
└── RESULTS.md             # append-only decision record
```

The implementation should import canonical constants and data loaders from the existing
regression code where practical, but it should write all generated artifacts under this folder.

## Definition of success

BLC can advance as an **interpretation** if it derives or replaces `W(r)` with no more than two
new global parameters, stays within 5% of canonical held-out SPARC error, and passes every theory
gate. It can advance as a **predictive extension** only if a preregistered boundary or orientation
observable improves held-out likelihood after complexity penalties and survives the negative
controls in the evaluation plan.

Anything weaker remains a descriptive analogy, not a mechanism.
