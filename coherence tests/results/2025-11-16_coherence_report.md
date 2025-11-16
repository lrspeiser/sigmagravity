# Coherence Model Sweep â€” 2025-11-16

**Data provenance:** real SPARC rotation curves from `data/Rotmod_LTG` with galaxy metadata from `data/sparc/sparc_combined.csv`. No synthetic data or paper inputs were touched.

Baseline (pure GR) global RMS = **43.98 km/s** (median |dv| 29.89 km/s).

| Model | Global RMS (km/s) | Median |dv| (km/s) | 95th |dv| (km/s) | Reduced Ï‡Â² |
| --- | --- | --- | --- | --- | --- |
| Path interference | 40.03 | 27.78 | 77.45 | 123.37 |
| Metric resonance | **38.54** | **25.31** | 76.63 | **111.10** |
| Entanglement | 40.03 | 27.77 | 77.45 | 123.37 |
| Vacuum condensation | 40.03 | 27.77 | 77.45 | 123.37 |
| Graviton pairing | 39.84 | 26.09 | 78.61 | 117.07 |

**Parameter notes**
- Path/entanglement/vacuum fits push the amplitude and coherence scale to extremely large values, reproducing the empirical Î£-Gravity shape but offering little microphysical discrimination. The optimizer effectively wants an almost-constant boost.
- Metric resonance converges with finite amplitude (~1.47) but drives â„“â‚€ to the lower bound, highlighting that the log-normal Î» window carries most of the weight.
- Graviton pairing stabilizes with Î³â‚“áµ¢ â‰ˆ 30 (saturating at the upper bound), meaning the model mimics a fixed coherence length rather than a Ïƒ-sensitive effect.

**Recommendation:** metric resonance provides the best RMS/Ï‡Â² trade-off while keeping parameters in a quasi-physical range. Next steps should scan tighter priors on â„“â‚€ and allow galaxy-dependent Î»â‚˜â‚€ (e.g., tying it to `R_disk`) to confirm whether the resonance width alone can explain the Î£-Gravity kernel.
