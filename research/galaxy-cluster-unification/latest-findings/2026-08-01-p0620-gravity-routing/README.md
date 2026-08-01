# P0620 gravity-routing research snapshot

Frozen: 2026-08-01

This folder collects the latest galaxy/cluster unification findings in one
reviewable package. It includes the scientific summaries, prior-art audit,
frozen protocols, analysis programs, key implementation modules, validation
tests, derived tables, JSON reports, and generated figures used in the P0612
through P0620 synthesis.

## Current diagnostic

The most promising tested construction is

\[
\boldsymbol\alpha_{\rm test}(\mathbf x)
=\boldsymbol\alpha_{0554}(r)
+{Q^2\over1+\Delta_{80}}
\,\mathcal R_{90}
\!\left[\delta\boldsymbol\alpha_{\rm route}(\mathbf x)\right].
\]

Here `P0554` supplies a potential- and profile-dependent radial enhancement,
`Q` is the projected baryonic quadrupole, and `Delta80` is the P0554 excess at
the radius containing 80 percent of the projected baryonic proxy weight. The
route template has width `0.23 R80 sqrt(1 + Q^2)`, return length `0.36 R80`, and
a shared 90-degree phase before annular monopole removal. The final angular
field is potential-derived and numerically curl-free. No gravity parameter is
fit separately to each object.

This is a phenomenological diagnostic, not a promoted field theory.

## Headline results

| Domain | Current result | Interpretation |
|---|---:|---|
| SPARC outer rotation | 12.592 km/s RMSE vs 10.348 fixed RAR | 21.7% worse; the P0554 radial parent carries the galaxy result. |
| Solar proxies | Mercury -1.730 mas/century inside the 3.1 margin | Passes the tested proxies, mainly through screening and symmetry nulls. |
| Five-cluster fixed geometry | +1.685% mean; 18/18 roots; 3/5 improve | Angular phase matters, but the sign is not universal. |
| RX J2129 phase diagnostic | +8.241% | Largest single response to the shared phase. |
| A383 frozen full refit | +0.174%; 9.081 arcsec RMS | Directional gain transferred, but absolute accuracy is inadequate. |
| Raw validation | 19.076 vs 9.989 arcsec compact halo | The current model has 1.91 times the compact-halo error. |

The angular layer has zero annular monopole. It can redistribute convergence,
move caustics, and change exact image roots, but it cannot provide missing total
radial convergence. That explains why some angular lens configurations improve
while absolute cluster accuracy remains poor.

## Scientific conclusion

The exact P0620 construction was not found in the targeted literature, but its
ingredients overlap substantially with QUMOND phantom density, refracted
gravity, EMOND potential dependence, relativistic MOND lensing, gravitational
polarization, and conventional lens shear or multipoles. The defensible name is
therefore:

> baryon-sourced conservative anisotropic effective-density ansatz

The next decisive test is a matched comparison against ordinary external shear
and a generic zero-monopole quadrupole, using a baryon-only direction predictor
frozen before raw image positions are scored.

## Package contents

- `docs/`: P0554 background, P0612-P0620 findings, and the P0621 prior-art and
  first-principles explanation.
- `protocols/`: frozen JSON protocols for the radial parent and P0612-P0619.
- `analysis/`: the exact stage runners and P0620 synthesis builder.
- `implementation/`: key radial-invariant, baryonic-metric, route-template, and
  conservative lens-field modules.
- `tests/`: result-contract and field-construction tests for the packaged stages.
- `results/current/`: complete P0612-P0620 CSV, JSON, Markdown, and PNG outputs.
- `results/synthesis-inputs/`: the compact prior-stage result artifacts consumed
  by the parameter-impact synthesis, plus the principal P0554/SPARC baselines.

Canonical actively developed files remain in the parent project's `docs/`,
`configs/`, `scripts/`, `src/`, `tests/`, and `results/` directories. The copies
here are an archival snapshot for review and citation.

## Reproduction

From `research/galaxy-cluster-unification`, the final sequence is:

```powershell
python scripts/run_p0612_cross_stage_parameter_impact.py
python scripts/run_p0613_bounded_endpoint_cross_domain.py
python scripts/run_p0614_composite_formula_audit.py
python scripts/run_p0615_self_coupled_quadrupole_route.py
python scripts/run_p0616_frozen_self_coupled_transfer.py
python scripts/run_p0617_self_coupled_support_phase_atlas.py
python scripts/run_p0618_universal_route_phase.py
python scripts/run_p0619_frozen_tangential_transfer.py
python scripts/build_p0620_parameter_impact_synthesis.py
python -m pytest tests/test_p0612_cross_stage_parameter_impact_results.py tests/test_p0613_bounded_endpoint_cross_domain_results.py tests/test_p0614_composite_formula_audit_results.py tests/test_p0615_self_coupled_quadrupole_route_results.py tests/test_p0616_frozen_self_coupled_transfer_results.py tests/test_p0617_self_coupled_support_phase_atlas_results.py tests/test_p0618_universal_route_phase_results.py tests/test_p0619_frozen_tangential_transfer_results.py tests/test_p0620_parameter_impact_synthesis_results.py -q
```

The parent repository stores the scientific source data, including large HST,
Chandra, XMM, Gemini, MUSE, and model-chain products, through Git LFS. Their
provenance records, acquisition protocols, hashes, and download/reduction
programs remain alongside them. Machine-local virtual environments, compiler
toolchains, and scratch products are intentionally excluded.
