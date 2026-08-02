# P0735 raw multiple-image lensing adapter

## Outcome

P0735 passes its frozen numerical and engineering gates. A researcher can now
submit one typed three-dimensional photon-acceleration field and compare its
predicted strong-lensing image positions directly with observed image
families. This is materially different from comparing against a reconstructed
convergence map: the adapter profiles the unknown source position, searches
the full declared image plane for lens-equation roots, assigns distinct roots
to distinct observed images, and scores the raw east/north positions with the
declared measurement errors.

The adapter adds no gravity parameter. It records exactly two observational
nuisance parameters per source family: the profiled east and north source
coordinates.

## Mapping and score

For each source family, the existing P0734 projection is reused without a
second interpretation of the photon field:

`alpha_perp = -(2 distanceRatio / c^2) integral(a_photon,perp dl)`

The observed image coordinates `theta_i` are ray-shot to the source plane and
their mean defines the profiled source `beta`. The worker then solves

`theta - alpha(theta) - beta = 0`

from global sign-change and residual-minimum seeds. A minimum-cost one-to-one
assignment pairs predicted roots with observed images. A complete family is
scored in two coordinates per image. Two degrees of freedom per family are
subtracted for the profiled source position.

If fewer roots are found than observed images, the family and target become
`incomplete_topology`. Aggregate RMS, chi-square, and likelihood are null. A
matched-subset RMS may appear only as a named diagnostic. Extra roots are
retained and counted; deciding that an extra image would be detectable needs a
separate selection model and is not smuggled into this adapter.

## Acceptance results

- Analytic two-image fixture: 2/2 observed images matched.
- Profiled source error: `5.55e-17 arcsec`.
- Image-plane RMS: `0 arcsec`.
- Maximum numerical root closure: `5.55e-17 arcsec`.
- Missing-multiplicity fixture: `incomplete_topology`, with null aggregate RMS
  and chi-square.
- Axis-permuted storage and two independent source-distance ratios pass.
- Integrated field jobs and decoupled observation jobs emit byte-identical
  scores, predictions, family tables, and root archives.
- The batch reporter retains `image_position_arcsec` separately from velocity,
  deflection-map, and reduced-shear channels.
- Targeted Python acceptance: 20 tests passed.
- Hosted contract and batch acceptance: 70 tests passed.

The rasterized singular-isothermal fixture also finds one extremely
demagnified interpolation root at the unresolved singular center. It is
disclosed as an excess root rather than silently deleted. The two observable
roots are recovered exactly. Real finite-resolution maps need an explicit
detectability/resolution policy before excess roots can be classified.

## Real-catalog audit

The secure P0713/P0714 subset contains 65 images in 18 source families:

- AS295: 18 images, 6 families.
- PLCKG287: 47 images, 12 families.

All image coordinates were converted with the established east/north sky
convention and survived a JSON target round trip with `0 arcsec` numerical
change. However, the parsed catalog has no published per-image positional
uncertainty column. P0735 therefore does not invent errors and does not call
that catalog score-ready. The next data task is to source or preregister
defensible astrometric uncertainties from the original measurement products.

## Artifacts

Each raw-image observation evaluation can publish:

- `observation_multiple_image_predictions.csv`, retaining every observed image
  including unmatched ones;
- `observation_multiple_image_families.csv`, retaining source positions,
  multiplicity, topology state, scores, nuisance counts, and gravity-parameter
  counts;
- `observation_multiple_image_roots.npz`, retaining all roots, closure errors,
  magnifications, and optional critical-curve points;
- `observation_scores.json`, with the separate `image_position_arcsec` score
  channel.

The full frozen evidence is in
`results/p0735_raw_multiple_image_adapter/report.json`.

## Scientific boundary

P0735 validates the observation adapter, not a candidate gravity theory. It
does not infer cosmological distances, fit lens mass, choose exclusions, or
claim that a baryon-only field explains a real cluster. Time delays, flux
ratios, microlensing, line-of-sight structure, source morphology, and an
independently measured critical-curve likelihood remain outside this stage.

