# Inverse baryon-to-response workbench milestone

Date: 2026-08-02

Release: hosted simulator v0.18

Status: local reference worker complete; public durable execution not connected

## Outcome

The simulator now has a formula-neutral way to ask a bounded discovery
question:

> Is there one compact spatial response pattern which, when applied to the
> measured baryonic maps of several development systems, resembles their
> separately inferred halo-like response maps?

The worker fits one stationary 2D or 3D kernel and one amplitude jointly across
all submitted systems. It does not fit a different force parameter to each
galaxy or cluster. It propagates the supplied target uncertainty, compares the
fit with a radius-preserving angle-shuffle null, and reports whether the
kernel is identifiable. Every artifact is deterministic and content hashed.

This is a hypothesis generator. The input target is a product of a lens or
mass model, not a direct picture of dark matter and not a raw observation. A
kernel recovered from that target cannot validate itself.

## Exact problem solved

For each development system `s`, the worker evaluates

```text
predicted_s(y) = A sum_x source_s(x) K(y - x) dV
```

where:

- `source_s` is a non-negative baryonic map;
- `K` is one shared compact stationary kernel;
- `A` is one shared fitted response amplitude;
- `dV` is the physical cell area or volume from the submitted grid spacing;
- the kernel origin is its center sample;
- convolution is linear and same-sized with zero padding outside the submitted
  domain; and
- no automatic normalization is hidden in the numerical operation.

The solve minimizes uncertainty-weighted residuals with declared ridge and
smoothness penalties. The public v1 can constrain the kernel to be non-negative
or permit a signed response. It supports Cartesian 2D and Cartesian 3D arrays.

The normalized kernel is a shape diagnostic. For a non-negative kernel its
physical integral is one; for a signed or exactly compensated kernel its
physical L1 integral is one. The separately reported positive amplitude carries
the total response strength. This separation prevents a trivial
kernel-amplitude rescaling degeneracy from being presented as two physical
measurements and still permits a zero-net signed response.

## Input roles enforced by the API

Each NPZ array has both an operational role and a scientific role:

| Array | Operational role | Required scientific role | Meaning |
|---|---|---|---|
| Baryonic source | `source` | `baryonic_input` | Stellar, gas, or other declared baryonic field |
| Response target | `auxiliary` | `model_derived_discovery_target` | Halo-like or effective-response product inferred by another model |
| Target uncertainty | `uncertainty` | `nuisance_or_calibration` | Per-cell uncertainty used in the weighted fit and ensemble |
| Optional mask | `mask` | `nuisance_or_calibration` | Cells allowed to influence fitting and scoring |

An array labeled `raw_observation` is rejected as an inverse target. Source,
target, and uncertainty units must agree, system identifiers must be unique,
all named arrays must share the declared grid, and the source must be
non-negative with positive total content.

## What a completed job emits

- `scientific_result.json`: aggregate and per-system fit metrics, amplitude and
  interval, identifiability, null results, sensitivity, parameter accounting,
  data-role audit, and claim boundaries.
- `kernels.npz` and `kernel.csv`: raw and normalized kernels plus 2.5%, median,
  and 97.5% ensemble values at every cell.
- `system_predictions.npz` and `per_system.csv`: predicted response maps and
  per-system residual scores.
- `null_controls.csv`: every radius-preserving angle-shuffle score.
- `regularization_sensitivity.csv`: response changes under preregistered
  regularization multipliers.
- `report.html`: deterministic human-readable metrics and kernel visualization.
- `llm_briefing.md`: deterministic facts an optional external LLM may explain;
  it may not refit, exclude, score, or decide the result.
- `reproduction.txt`, `resource_log.json`, `artifact_index.json`, and
  `manifest.json`: the exact command, resources, file hashes, worker-source
  hash, and scientific-result hash.

## Acceptance evidence

The known-answer HTTP test uploads two synthetic baryonic maps whose targets
were made with an asymmetric 5 by 5 kernel and amplitude 1.6. Through the same
upload, queue, worker, polling, and artifact-download path used by researchers,
it obtains:

| Gate | Result |
|---|---:|
| Recovered-kernel cosine similarity | `0.99999999999937` |
| Recovered amplitude | `1.5999999988039162` |
| Aggregate R-squared | `0.9999999999999991` |
| Radial-angle null p-value with 19 permutations | `0.05` |
| Signal-against-null verdict | `true` |
| Per-system fitted gravity parameters | `0` |
| Downloaded artifacts with valid hashes | `14 / 14` |

Additional automated gates establish that:

- a constant target does not claim a signal merely because a permutation
  p-value is small; a minimum explanatory-effect gate is also required;
- empirical 95% amplitude and central-kernel intervals cover the injected
  values in the declared fraction of ten noisy trials;
- a rank-one constant source reports the remaining 24 kernel directions as
  non-identifiable instead of manufacturing a unique pattern;
- a Cartesian 3D impulse exactly recovers a 3 by 3 by 3 kernel;
- radial-angle shuffling preserves shell values while changing angular
  placement;
- repeat jobs have identical scientific hashes; and
- the gateway and worker agree on the exact worker source hash.

Passing these gates shows that the code solves the declared inverse problem.
It does not show that any real cluster has this response law.

## Parameter accounting

The synthetic example fits 25 kernel cells and one universal amplitude. The
report therefore calls them discovery coefficients. It does not advertise the
result as a one-parameter competitor to a dark-matter halo or MOND. A compact
physical theory would need to replace those cells with a small analytic closure
whose constants, units, symmetries, conservation behavior, and weak-field
limit are declared before the holdout is opened.

## What still needs to be built for useful Sigma Gravity discovery

### 1. Register complete, uncertainty-aware real inputs

- Cluster member and brightest-cluster-galaxy stellar maps.
- Intracluster light.
- X-ray gas density and temperature plus SZ constraints.
- Foreground/background structure and line-of-sight geometry.
- At least two independent lens-model posterior ensembles for each development
  cluster.
- Raw strong-lensing image families, redshifts, positions, covariances, and
  selection information held separately from derived targets.
- Galaxy stellar, H I, molecular-gas, velocity-field, distance, inclination,
  beam, mask, and mass-to-light posterior products.
- Provenance, licenses, WCS, coordinate transforms, checksums, and declared
  missing-baryon uncertainty for every product.

### 2. Replace a single best 2D map with posterior ensembles

The current job perturbs a supplied target by its per-cell uncertainty. A real
campaign must repeat the inverse across baryonic deprojections and full lens-
model posterior draws. It must separate statistical noise, source-model
choices, mass-sheet/source-position degeneracies, line-of-sight structure,
and baryonic incompleteness. Stability must be checked across independent lens
model families rather than one map plus Gaussian pixel noise.

### 3. Complete the null and falsification library

- Target-system permutation.
- Fourier phase scrambling.
- Central-halo and local-light baselines.
- Radius-preserving angle shuffle for both sources and targets.
- Missing-gas and missing-intracluster-light perturbations.
- WCS, centroid, redshift, and mass-to-light perturbations.
- Lens-model-family and reconstruction-code swaps.
- Negative controls with no injected nonlocal response.
- Conservative-redirection ledgers that require compensating deficits rather
  than allowing unexplained amplification.

### 4. Search interpretable response families, not only free cells

The free kernel can reveal scale, anisotropy, offsets, or multiple lobes worth
testing. The next layer should compare small preregistered families such as:

- isotropic kernels with one length scale;
- density- or tidal-gated length scales;
- anisotropic kernels aligned with the baryonic Hessian or tidal tensor;
- normalized conservative transport plus a divergence-free return field;
- two-potential matter/photon closures; and
- signed or compensated kernels whose integral is fixed by a conservation law.

Each family needs dimensional consistency, boundary behavior, an explicit
Solar-System limit, no object-class switch, and a parameter-count disclosure.
Complexity penalties and held-out likelihood—not visual resemblance—should
choose among them.

### 5. Freeze candidates and remove every derived halo target

For each surviving compact law:

1. choose the form, constants, solver, grids, exclusions, and thresholds using
   development systems only;
2. cryptographically freeze the manifest;
3. provide only permitted baryonic and environmental inputs for validation and
   holdout systems;
4. solve the fields and ray-trace them to raw image positions, shear,
   magnification, and, where available, time delays;
5. predict galaxy rotation curves and resolved velocity fields with the same
   physical constants; and
6. run Solar-System, conservation, stability, and resolution gates before
   revealing the holdout scores.

This is the step that can distinguish an alternative gravity law from a
flexible reconstruction of a halo map.

### 6. Compare fairly with trusted baselines

The deterministic joint report must show baryons-only Newtonian/GR, fixed
MOND/RAR, and published dark-matter lens/dynamics baselines. It must place raw
accuracy beside global, hierarchical, nuisance, and per-object parameter
counts. A universal model may remain interesting while fitting less closely
than a flexible halo, but the numerical and complexity gaps must both be
visible. Failure topology, excluded points, and unconverged systems must never
be silently dropped.

### 7. Connect production infrastructure

Vercel should remain the browser and short-request gateway. Useful public
execution still requires content-addressed object storage, a durable queue and
database, isolated network-disabled scientific containers, authentication,
project isolation, quotas, cancellation, retry policy, caching, signed
manifests, license enforcement, monitoring, and reproducible pinned runtimes.
Uploaded advanced code must run only in that isolated worker tier.

## Decision rule

Do not continue indefinitely by adding free coefficients. If complete baryons,
independent lens models, and the full null suite erase the learned response,
record that negative result. If a compact frozen law cannot predict unseen raw
cluster lensing and galaxy dynamics after at most three materially different
closures, reassess the premise instead of rescuing it with per-object gravity
parameters or the holdout system's halo map.
