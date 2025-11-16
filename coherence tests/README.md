# Coherence Tests

Speculative playground for mapping observational Î£-Gravity coherence into
microphysical root causes without touching the production paper or data.
The folder structure mirrors the five candidate mechanisms so we can run
apples-to-apples tests later.

> Data status: **no real SPARC/GAIA data or fits are stored here yet** â€”
> everything is currently analytic/toy math (see `coherence_models.py`).

## Layout

- `coherence_models.py` â€“ shared Burr-XII window plus five multiplier
  closures. Swap these into any pipeline by calling
  `apply_coherence_model(name, ...)`.
- `path_interference/` â€“ holds experiments for quantum path interference.
- `metric_resonance/` â€“ metric fluctuation resonance scaffolding.
- `entanglement/` â€“ matter-geometry entanglement studies.
- `vacuum_condensation/` â€“ stochastic vacuum condensation sweeps.
- `graviton_pairing/` â€“ non-local graviton pairing / correlation checks.

When we run numerical tests, drop notebooks, configs, and result summaries
inside the associated subfolder and clearly call out whether the inputs are
real observational data or synthetic toy sets. Every run should end with a
short summary (success metrics, fit residuals, caveats) so that we can trace
observations back to first-principle causes.
