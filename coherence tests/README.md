Coherence Tests Sandbox
=======================

Purpose
-------
- Provide a speculative workspace for mapping observed Î£-Gravity behaviour back to five hypothesised microphysical root causes without touching any production data, figures, or paper sources in the repo root.
- Keep all experiment scaffolding in one place so each mechanism can grow independently.

Current Status
--------------
- No real observational data is loaded here yet; everything is placeholder-ready for future wiring.
- Summary-of-results guidelines: whenever a test script is run in this tree, record whether it used real or synthetic inputs and capture the resulting kernel fit / diagnostic metrics in a short Markdown or JSON log inside the corresponding subfolder.

Layout
------
- coherence_models.py &mdash; shared Burr-XII window helpers plus five drop-in multipliers ( = 1+K) that match the Î£-Gravity pipeline signature.
- path_interference/ &mdash; coherence loss from high-Ïƒ path decoherence.
- metric_resonance/ &mdash; Î»-space resonance between spacetime fluctuations and baryonic structure.
- entanglement/ &mdash; temperature-suppressed matter-geometry entanglement strength.
- acuum_condensation/ &mdash; phase-transition-like vacuum order parameter below a dispersion threshold.
- graviton_pairing/ &mdash; non-local graviton pairing and coherence-length scaling.

Next Steps
----------
- Fit each mechanismâ€™s parameters against existing Î£-Gravity kernels (using SPARC or Gaia inputs stored outside this folder) and stash the fit reports locally.
- Document observational-to-first-principles reasoning chains per mechanism within their subfolders as analyses progress.
