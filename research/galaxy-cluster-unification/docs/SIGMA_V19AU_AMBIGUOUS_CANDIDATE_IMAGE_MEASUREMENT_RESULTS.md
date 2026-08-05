# Sigma V19AU ambiguous-candidate image-measurement results

## Decision

V19AU **failed closed** because only 59.74% of individual
candidate–exposure measurements had positive flux, below the frozen 80%
source-sufficiency gate. The method, candidates, exposures and gate were not
changed after seeing the result.

The failure is not a processing or coverage failure. All 40,812 planned rows
were measured and retained, and none raised a processing exception. It is a
failure of the assumption that most individual exposures of every catalog
candidate should yield positive four-arcsecond flux.

## Frozen result

- Unique candidates: **568/568**
- Detector groups: **123/123**
- Planned and retained measurements: **40,812/40,812**
- Positive-flux measurements: **24,382**
- Signed non-positive measurements: **16,430**
- Processing failures: **0**
- Overall positive fraction: **59.742%** (required at least 80%)
- Candidates with at least one positive exposure in every `griz`: **461/568
  (81.16%)**
- Candidates complete in `grizY`: **454/568 (79.93%)**
- Bessel/positional association scores computed: **none**

| Filter | Planned | Positive | Non-positive | Positive fraction |
|---|---:|---:|---:|---:|
| `g` | 7,553 | 4,384 | 3,169 | 58.04% |
| `r` | 13,757 | 8,378 | 5,379 | 60.90% |
| `i` | 8,133 | 4,911 | 3,222 | 60.38% |
| `z` | 6,175 | 3,696 | 2,479 | 59.85% |
| `Y` | 5,194 | 3,013 | 2,181 | 58.01% |

## What the failure means

The roughly 60% rate is similar in every filter, which argues against one bad
band or zeropoint. The candidate list intentionally includes every HSC/NSC
source in broad, quantization-aware spectroscopic cones. Many are faint or not
the true optical counterpart, so a single shallow exposure can fluctuate below
zero after local background subtraction.

The result also shows why discarding those rows would be wrong. A large
majority of candidates—461—still have at least one positive exposure in all
four color bands. Every one of the 57 members has at least one such candidate,
and 528/640 member–candidate hypotheses are `griz` complete by that loose
criterion. The missing information is distributed signal-to-noise, not WCS
coverage or a broken image algorithm.

## Next defensible route

V19AU does not authorize a positive-magnitude-only candidate likelihood. The
next materially different test should combine **signed fluxes** across epochs:

1. transform every flux and uncertainty to a common per-filter reference
   zeropoint;
2. compute a robust inverse-variance multi-epoch flux, retaining negative
   measurements;
3. report signal-to-noise and upper limits instead of requiring every exposure
   to be positive;
4. verify that the same stacking transform preserves the already-measured
   development/validation color behavior; and
5. only then freeze a joint positional/color likelihood with explicit null and
   ambiguous states.

This is not a retrospective lowering of the 80% gate. It is a different
observable: a multi-epoch flux likelihood rather than a count of positive
single-exposure magnitudes.

## Reproducibility

- Frozen protocol: `configs/sigma_v19au_ambiguous_candidate_image_measurement.json`
- Plan builder: `scripts/build_sigma_v19au_ambiguous_candidate_image_plan.py`
- Measurement runner: `scripts/run_sigma_v19au_ambiguous_candidate_image_measurement.py`
- Machine report: `results/sigma_v19au_ambiguous_candidate_image_measurement/report.json`
- Measurements and aggregates: `data/derived/sigma_v19au_ambiguous_candidate_image_measurement/`
