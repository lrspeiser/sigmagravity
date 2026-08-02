# Public simulator: 2D/3D field and galaxy-generator roadmap

Date: 2026-08-02

## Product outcome

The intended product is not a Refracted Gravity endpoint or a SigmaGravity
endpoint. A researcher supplies a typed mathematical model, declares its
parameters and required data, selects real or synthetic systems, and receives
observation-space predictions and comparator scores. The service must disclose
exactly which parameters were universal, fitted on a training sample, or fitted
per object.

“Paste a formula” initially means paste or build a safe, dimension-aware JSON
equation tree. Natural-language or LaTeX conversion may help construct that
tree, but the researcher must confirm the canonical tree before execution.
Unrestricted submitted Python is a separate later sandbox tier, not part of the
trusted solver.

## What is complete now

| Capability | Current evidence | Honest limit |
|---|---|---|
| Radial hosted benchmark | 175 SPARC systems, 3,391 points, Newtonian and fixed-MOND comparators | radial mass-model components only |
| Seeded radial generator | deterministic mass, gas fraction, bulge fraction, scale, noise, and seed | not a morphology-matched 2D galaxy |
| Observation-matched replica layer | P0631 recreates radial photometry and supplied velocity observables for 131 systems | axisymmetric reconstruction; dynamics are inputs in replica mode |
| Real resolved map path | P0639 provides registered gas, stellar, and total-baryon maps for 13 LITTLE THINGS galaxies; matching velocity fields exist separately | gas-rich dwarf sample; the maps inherit preprocessing and stellar mass-to-light assumptions |
| Typed field-model contract | `sigma-field-model/1` validates units, ranks, operators, boundaries, data keys, solver family, and parameter policy | execution is not connected to the hosted gateway |
| Cross-theory conformance fixtures | Newtonian, AQUAL, QUMOND, Refracted Gravity, and two-potential manifests pass one validator | validation does not prove a numerical or physical result |
| Neutral local field worker | one expression-driven engine solves scalar divergence-form equations on Cartesian 2D and 3D grids | no hosted queue, tensors, nonlocal operators, axisymmetric coordinates, or arbitrary-code sandbox yet |
| Numerical acceptance tests | analytic 2D, analytic 3D, variable-coefficient, and exact Refracted Gravity-tree executions pass | convergence order and production-scale resource classes are not frozen yet |
| Content-addressed field job | verified array bundle, deterministic job/scientific hashes, residual history, output hashes, resource log, artifact index, and CLI pass end to end | durable cloud upload, database, scheduler, and hosted worker are not connected |
| Worker container definition | pinned Python/NumPy/SciPy/JCS environment, non-root user, read-only/network-disabled run instructions | Docker/Podman is unavailable on this machine, so image build and runtime isolation still need CI or a container host |
| Asynchronous reference API | immutable NPZ upload, queue, polling, events, cancellation, restart recovery, and rehashed artifact downloads pass through real HTTP and the Python worker | local single-user filesystem queue; no durable cloud store, auth, or deployed container scheduler |
| Gravity-independent galaxy extraction | P0720 converts each real gas and stellar map into content-hashed radial/Fourier/local-feature parameters without velocity targets or gravity parameters | operates on registered maps, not raw images or uncertainty posteriors |
| Parameter-controlled 2D generation | P0720 deterministically replays known systems and changes mass, scale, angular structure, clumps, rotation, and offsets | finite basis leaves visible low-level artifacts; not yet a survey observation simulator |
| Explicit 3D prior ensembles | 78 varied thickness/flaring realizations project back to their 2D maps at numerical precision | demonstrates non-uniqueness; does not recover true depth |
| Known-map commissioning round trip | 13/13 total-baryon maps: median normalized error 0.168, worst 0.257, median correlation 0.986 | thresholds saw the same data during prototyping and are not blind validation |
| Asynchronous galaxy-job API | immutable map upload, extraction, 2D/3D generation, polling, events, cancellation, verified downloads, and parameter-controlled replay pass through real HTTP | local single-user worker; public Vercel route advertises the schema but cannot execute it yet |
| Asynchronous multi-system batch API | one frozen manifest and policy run across generated/uploaded 2D or 3D bundles, with polling, cancellation, child jobs, restart recovery, and deterministic rehashed reports | numerical execution only; rotation and lensing observation adapters are not connected |

The generic worker dispatches from equation structure, not a theory name. It
supports `laplacian(phi)=source` and
`divergence(coefficient*gradient(phi))=source`, including coefficients and
sources computed from the safe expression tree. This is enough for the first
Poisson, density-dependent Refracted Gravity, AQUAL-like, and coupled
QUMOND-like execution path, subject to the boundary and convergence limits
below.

## Everything still required

| Workstream | Remaining work | Concrete acceptance outcome |
|---|---|---|
| 1. Formula authoring | Add a browser equation builder, JSON editor, LaTeX preview, schema help, and an optional LaTeX-to-tree draft converter. Require explicit confirmation of the canonical tree. | A new user can reproduce all five example hashes without editing application code; ambiguous LaTeX never queues automatically. |
| 2. Contract coverage | Publish a formal JSON Schema; add coordinate frames, field shapes, uncertainty fields, priors, parameter bounds, fit/freeze stages, and observable scoring definitions. | Invalid dimensions, missing data, missing boundaries, and hidden per-object parameters fail before job creation; schema and implementation conformance tests agree. |
| 3. Generic numerical engine | Add controlled isolated-boundary approximations, axisymmetric cylindrical grids, tensor coefficients, coupled nonlinear convergence controls, nonlocal kernels, line-of-sight operators, adaptive resource estimates, and checkpoint/restart. | Manufactured solutions demonstrate the expected convergence order in every supported solver family; divergence and nonconvergence return diagnostics rather than a plausible-looking field. |
| 4. Safe advanced tier | Define a container plug-in ABI for models outside the safe language; run each in a network-blocked, single-use sandbox with read-only inputs and hard CPU, memory, wall-time, and output limits. | A hostile fixture cannot read credentials, reach the network, or affect another run; a valid plug-in reproduces a safe-language fixture within tolerance. |
| 5. Data ingestion | Immutable local NPZ upload with unit, hash, provenance, and license gates is implemented. Add resumable object-storage uploads for FITS, HDF5, CSV, tables, and catalogs plus frames, masks, PSFs/beams, uncertainty maps, distance, and inclination. | Uploading the same bytes and metadata returns the same data ID; missing units, frame, license, or checksum blocks scientific execution. |
| 6. Versioned real-galaxy maps | Thirteen registered dwarf maps are now packaged internally. Add raw/processed public releases, licenses, masks, PSFs/beams, uncertainties, and at least seven spiral/bulge/LSB systems. | At least 20 morphologically varied galaxies pass registration, mass conservation, and observation-forward reconstruction gates before any theory score is opened. |
| 7. 2D-to-3D reconstruction | P0720 now preserves vertical-prior ambiguity and exact projection. Add bulge deprojection, inclination/distance/warp uncertainty, PSF-aware inference, and posterior rather than prior-only draws. | Synthetic recovery tests cover known thickness/bulge/inclination; real results report the spread across allowed reconstructions. |
| 8. Inverse parameter extraction | Deterministic mass, centroid, radial, Fourier, and local-feature extraction now works on registered maps. Add raw-image light/gas separation, uncertainty distributions, bulge/thickness inference, calibration checks, and missing-structure diagnostics. | On synthetic images with hidden truth, calibrated intervals contain the generating values at their advertised rate; holdout residuals reveal model misspecification. |
| 9. Forward galaxy generator | Component maps can now be replayed and controlled in mass, scale, angular structure, clumps, rotation, and offsets. Add intrinsic bulges/warps, projected sky images, spectral/velocity cubes, beams/PSFs, noise, masks, and survey selection. | Same seed is bitwise reproducible; changing one declared parameter produces the expected controlled change; mass/light are conserved within frozen tolerances. |
| 10. Known-galaxy round trip | The first 13 registered-map photometric round trips pass commissioning gates. Add raw-image, uncertainty, kinematic, covariance, train/validation, and untouched whole-galaxy holdouts. | Predeclared photometric, kinematic, morphology, and conservation gates pass on train, validation, and untouched whole-galaxy holdouts. A radial match alone is insufficient. |
| 11. Theory-to-observable adapters | Convert solved potentials/fields into circular speeds, line-of-sight velocity fields, weak shear, convergence, deflection, critical curves, and raw multiple-image roots. Keep photon and massive-tracer mappings explicit. | Newtonian/MOND published fixtures reproduce within declared numerical and data-processing tolerance; photons are never silently scored with a massive-particle rule. |
| 12. Fair scoring | Add fixed train/development/holdout splits, no-target-access execution, nuisance-policy declarations, universal/per-object parameter counts, likelihoods with covariance, and same-input comparators. | A batch report separates galaxies, clusters, topology, and Solar-System tests and shows performance versus Newtonian, fixed MOND/RAR, and a declared halo baseline without a single blended score. |
| 13. Cluster data | Version member light, intracluster gas, geometry, source redshifts, weak-shear catalogs, and raw strong-lens image positions with licensing and sealed/open states. | Multiple clusters can be run with one frozen gravity parameter set; raw image/topology holdouts are scored, not only reconstructed dark-matter maps. |
| 14. Asynchronous API | Local upload, queue, lifecycle events, cancellation, restart recovery, caching identity, artifact indexes, stable errors, and batches up to 1,000 systems are implemented. Add durable adapters, model registration, retry policy, and production resource classes. | Identical model/data/solver/seed/worker hashes return the cached immutable run; a browser never holds a request open for a long solve. |
| 15. Storage and reproducibility | Add durable database and object storage for model, dataset, code, solver, container, seed, grid, boundary, logs, predictions, plots, and manifests. Sign citation-ready manifests. | A clean worker can reproduce selected run hashes/tolerances from only the manifest and permitted inputs; failed runs retain useful artifacts. |
| 16. Hosting and operations | Containerize the Python worker, deploy CPU resource classes, connect it to the Vercel control plane, and add auth, quotas, monitoring, cost controls, backups, abuse handling, and uptime/support policy. | Public users can submit a bounded batch, follow status, download artifacts, and cannot exhaust shared resources or access sealed data. |
| 17. Reports and SDKs | Deterministic batch JSON, CSV, HTML, failure table, parameter accounting, hashes, and LLM briefing are implemented. Add observation predictions, CSV/Parquet and FITS exports, plots, residual maps, sensitivity analyses, a methods report, and Python/HTTP SDKs. | Another researcher can reproduce and independently interpret a run without copying output into an LLM. |
| 18. Optional LLM explanation | Generate a compact, redacted briefing artifact and allow an opt-in LLM to summarize it. The model cannot alter scores, manifests, or scientific state. | Deterministic results remain complete without an LLM key; generated prose is labeled and traceable to the briefing hash. |
| 19. External replication | Ask scientists unaffiliated with SigmaGravity to encode Refracted Gravity and at least two other models, upload a new system, run a batch, and report friction. | External users reproduce expected fixtures without repository assistance; every failure is retained and classified as model, data, numerical, or product error. |

## Delivery order and rethink gates

1. Finish the field contract, formal schema, and conformance fixtures. **Initial stationary-field contract complete.**
2. Harden the generic local worker through convergence, nonlinear, boundary,
   coupled-field, and resource tests.
3. Package the worker and connect an asynchronous job path with immutable
   artifacts. **Local reference path complete; durable hosted adapters remain.**
4. Publish the first resolved multi-galaxy data release and observation-forward
   adapters.
5. Implement inverse extraction and a generator round trip on synthetic truth,
   then on known-galaxy holdouts.
6. Add raw cluster lensing and common comparator reports.
7. Run external replication pilots before calling the API production-ready.

Pause and rethink a layer when its numerical known-answer tests fail, when
inferred parameters are not identifiable under realistic noise, when generated
images pass radial checks but fail 2D morphology, or when results depend on
undisclosed per-object gravity parameters. More formula freedom must not be
used to hide a broken reconstruction, solver, or observation adapter.

## Current next milestone

The local field API, real-map extraction/generation round trip, asynchronous
galaxy-job contract, and chained multi-system batch now pass. Galaxy jobs emit
SI surface- and volume-density bundles that any compatible confirmed manifest
can consume. One batch freezes its manifest and parameter policy and produces
content-addressed per-system and aggregate reports. The next product milestone
is an observation-scoring layer that converts fields into rotation and lensing
predictions with explicit uncertainty and comparator policies.
The next scientific milestone is synthetic hidden-truth recovery with noise,
PSF, inclination, bulge, thickness, and calibrated uncertainties, followed by
an untouched morphologically varied whole-galaxy holdout. Production hosting
still requires durable object storage, job metadata, isolated workers, auth,
quotas, and monitoring.
