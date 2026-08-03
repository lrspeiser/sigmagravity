# Full simulator completion audit

Date: 2026-08-02

This audit compares the current repository with the full research-platform
specification. “Built” means exercised by deterministic tests. “Partial” means
there is a real implementation with an important scientific or operational
gap. A public schema alone is not counted as hosted execution.

## Requirement status

| Required capability | Status | Current evidence | Missing before it is genuinely useful |
|---|---|---|---|
| General confirmed model manifest | Built | Typed fields, sources, equations, units, parameters, geometry, boundaries, data, photon/matter targets, observables, solver, exact canonical confirmation | Broader boundary types and time-dependent solver classes |
| Safe formula operators | Partial | Grad/div/curl/Laplacian, products, gates, coupled scalar elliptic fields, LOS/path operations, Cartesian convolution, multiple potentials | General tensor algebra, cylindrical nonlocal kernels, broader nonlinear constitutive forms |
| Isolated advanced-code plug-ins | Missing | Contract direction only | Signed upload, single-use network-disabled containers, read-only data, quotas, pinned runtime, malware/policy controls; never execute plug-ins in Vercel |
| Generic Cartesian 2D/3D worker | Partial | Finite-volume scalar elliptic, nonlinear Picard/Anderson/Newton–Krylov, coupled fields, diagnostics, nonlocal Cartesian fixture | FFT path integration, vector/tensor solves, mixed/Neumann/periodic library, refinement, production scaling |
| Axisymmetric `(r,z)` worker and observables | Built for scalar elliptic fields, massive tracers, photon maps and raw image positions | Regular-axis finite volume, variable coefficients, second-order Bessel acceptance, direct circular-speed curves, inclined resolved velocity maps, inclination-aware deflection/shear maps, finite-support raw roots and immutable async job path | Cylindrical convolution, weak-lensing catalogs, time delays and non-axisymmetric structure |
| Coordinate-safe lensing/ray tracing | Partial | Typed Cartesian and axisymmetric photon acceleration, deflection/shear maps, raw Cartesian and cylindrical multiple-image roots and scores | Validated physical cosmological normalization, weak-lensing catalogs, time delays and magnification-selection likelihoods |
| Resolved observational catalog | Partial | SPARC radial catalog, eight frozen resolved-twin systems, four-cluster evidence registry, selected local maps | Licensed homogeneous light/gas/cube/PSF/WCS/noise/mask packages across a morphology-diverse sample and untouched clusters |
| Uncertainty-aware 2D-to-3D reconstruction | Partial | Seeded prior ensembles, exact projection checks, first gravity-independent surface likelihood and weight diagnostics | Full covariance, PSF/beam/dust, bulge depth, scale-height and warp likelihoods, adaptive posterior sampler, adequate effective sample size |
| Gravity-independent inverse baryon extractor | Partial | Content-addressed baryonic parameter extraction and strict separation from gravity/velocity/lensing targets | Validated inference from raw multiband images and cubes with posterior calibration and withheld-data checks |
| Full forward galaxy generator | Partial | Seeded 2D/3D density ensembles, projection, structural perturbations, observation metadata | Physical equilibrium, intrinsic velocity/dispersions, spectral cubes, radiative/beam pipeline, bars/arms/clumps/thickness calibrated to observations |
| Round-trip validation | Partial | Baryonic map comparisons, frozen development/validation/holdout evidence, formula-transport diagnostics | Large morphology-diverse validation set, uncertainty-calibrated acceptance, Fourier/pixel/cube statistics, withheld modalities |
| Asynchronous batch API | Built locally / contract on Vercel | Upload, jobs/events/artifacts, batches, composed observation jobs, cancellation/recovery tests | Durable production queue, workers, database, object storage, auth, project isolation and no 25-system production limit |
| Parameter policies and accounting | Built | Published fixed, universal, train/validation/holdout, hierarchical/per-object disclosure; gravity and nuisance counts separated | Comparator-wide effective-complexity and posterior-volume reporting |
| Deterministic reporting | Partial | Hashed JSON/CSV/NPZ/HTML artifacts, residual histories, reproduction metadata, LLM briefing paths | Uniform PDF bundle, sensitivity plots for every job class, signed manifests from production storage |
| Optional LLM explanation | Correctly non-authoritative | Deterministic engine owns scores, exclusions, manifests and pass/fail | Optional explanation service only after reports are complete; no LLM key is needed for scientific execution |
| Production computation infrastructure | Missing | Vercel control plane publishes honest `production_worker_not_connected` errors | Queue/scheduler, isolated Python workers, Postgres, S3/R2, auth, quotas, cancellation, retries, cache, licensing enforcement, audit logs |
| Formula-independence acceptance suite | Partial | Newtonian, AQUAL-like, QUMOND, Refracted Gravity, nonlocal, two-potential, nonlinear and axisymmetric fixtures | A real isolated plug-in fixture plus broader vector/tensor and external-researcher manifests |

## The bounded path to a useful Sigma Gravity test

These are the remaining research deliverables, in order. Work outside them is
polish, not the critical path.

1. **Write one exact law.** Express Sigma Gravity as a confirmed manifest with
   a small set of universal constants, units, boundaries, photon/matter
   coupling, and an explicit Newtonian/GR Solar-System limit.
2. **Validate cylindrical raw lensing on registered observations.** Direct
   circular-speed, resolved velocity-field, photon-map and raw multiple-image
   adapters now pass independent analytic and asynchronous acceptance. Register
   complete real image catalogs, propagate baryonic/domain uncertainty, add
   weak-lensing likelihoods, and retain resolution sensitivity.
3. **Register morphology-diverse baryons.** Assemble licensed gas, stellar,
   bulge, distance, inclination, PSF/beam, mask and uncertainty packages that
   were not created using the candidate gravity law.
4. **Calibrate baryonic posteriors.** Replace collapsed two-draw importance
   weights with enough draws or adaptive sampling to achieve preregistered
   effective-sample and posterior-predictive gates.
5. **Freeze constants once.** Fit only on a named development set, then lock
   the manifest hash and constants before validation, final galaxy holdout, and
   cluster testing.
6. **Score raw observations.** Galaxies must use withheld velocity fields or
   curves. Clusters must use image positions, shear, magnification, or time
   delays—not the target dark-matter map used to invent the law.
7. **Run fair comparators.** Report baryons-only GR/Newtonian, fixed MOND/RAR,
   and published dark-matter models with their gravity, halo and nuisance
   parameter counts displayed together.
8. **Require cross-domain gates.** A candidate remains exploratory until the
   same settings pass galaxies, clusters, Solar System, resolution, boundary,
   conservation and held-out prediction gates.

## The bounded path from dark-matter clouds to an alternative law

Dark-matter maps are not direct observations, so they should be used only in
the discovery stage:

1. Register several teams’ halo/posterior maps and keep them separate from raw
   lensing data and baryonic inputs.
2. Infer response patterns from baryons to the model-derived excess using the
   shared-kernel inverse workbench and all declared null families.
3. Test whether recovered features survive changes in lens-model method,
   baryonic realization, central-light treatment, angular shuffling, phase
   scrambling, target permutation and synthetic missing baryons.
4. Measure identifiability. A 25-cell kernel is 25 discovery coefficients, not
   a one-parameter theory; compatible null-space alternatives must be shown.
5. Compress any repeatable pattern into a small analytic, conservation-aware
   forward law based only on baryonic density, gradients, curvature, topology
   or declared environmental fields.
6. Freeze that law and discard every target halo map.
7. Predict untouched raw galaxy and cluster observations from baryons alone.
   Failure here rejects the proposed interpretation even if the discovery map
   looked halo-like.

## Practical product boundary

The repository is already useful for developing and falsifying well-specified
stationary field equations locally. It is not yet a self-service hosted
research platform. The next product milestone should therefore be a narrow
end-to-end production slice—authenticated upload, one isolated worker, durable
artifacts, and one confirmed axisymmetric Poisson job—before adding more UI or
formula variations.
