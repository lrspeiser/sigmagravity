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
| Real resolved map path | P0635 ingests DDO154 H I and optical images and lifts them to a 3D density field | one spent commissioning galaxy, not a public multi-galaxy release |
| Typed field-model contract | `sigma-field-model/1` validates units, ranks, operators, boundaries, data keys, solver family, and parameter policy | execution is not connected to the hosted gateway |
| Cross-theory conformance fixtures | Newtonian, AQUAL, QUMOND, Refracted Gravity, and two-potential manifests pass one validator | validation does not prove a numerical or physical result |
| Neutral local field worker | one expression-driven engine solves scalar divergence-form equations on Cartesian 2D and 3D grids | no hosted queue, tensors, nonlocal operators, axisymmetric coordinates, or arbitrary-code sandbox yet |
| Numerical acceptance tests | analytic 2D, analytic 3D, variable-coefficient, and exact Refracted Gravity-tree executions pass | convergence order and production-scale resource classes are not frozen yet |
| Content-addressed field job | verified array bundle, deterministic job/scientific hashes, residual history, output hashes, resource log, artifact index, and CLI pass end to end | durable upload, queue, storage, and hosted worker are not connected |
| Worker container definition | pinned Python/NumPy/SciPy/JCS environment, non-root user, read-only/network-disabled run instructions | Docker/Podman is unavailable on this machine, so image build and runtime isolation still need CI or a container host |

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
| 5. Data ingestion | Add resumable uploads for FITS, HDF5, NumPy, CSV, tables, and catalogs; unit/frame metadata; checksums; license/provenance; masks; PSFs/beams; uncertainty maps; distance and inclination uncertainties. | Uploading the same bytes and metadata returns the same data ID; missing units, frame, license, or checksum blocks scientific execution. |
| 6. Versioned real-galaxy maps | Build an open catalog sharing resolved stellar-light, H I/gas, distance, inclination, PSF/beam, masks, and velocity maps for the same systems. Preserve raw and processed versions. | At least 20 morphologically varied galaxies pass registration, mass conservation, and observation-forward reconstruction gates before any theory score is opened. |
| 7. 2D-to-3D reconstruction | Support transparent vertical priors, bulge deprojection, gas thickness, inclination, distance, warp/bar flags, and Monte Carlo nuisance draws. Never call a non-unique reconstruction “the galaxy’s true 3D density.” | Synthetic recovery tests cover known thickness/bulge/inclination; real results report the spread across allowed reconstructions. |
| 8. Inverse parameter extraction | Infer light/gas components, scale lengths, bulge fraction, thickness, inclination, asymmetry, Fourier modes, clumps, and uncertainty distributions from observations. Separate measured, externally supplied, and inferred quantities. | On synthetic images with hidden truth, calibrated intervals contain the generating values at their advertised rate; holdout residuals reveal model misspecification. |
| 9. Forward galaxy generator | Generate intrinsic stellar/gas density, projected images, spectral/velocity maps, beams/PSFs, noise, masks, and survey selection from a seed and physical/observational parameters. Provide both parametric and nonparametric modes. | Same seed is bitwise reproducible; changing one declared parameter produces the expected controlled change; mass/light are conserved within frozen tolerances. |
| 10. Known-galaxy round trip | Feed inferred posterior parameters into the generator and compare generated products with the source observations, including radial profiles, pixels, Fourier modes, gas layout, velocity fields, and covariances. | Predeclared photometric, kinematic, morphology, and conservation gates pass on train, validation, and untouched whole-galaxy holdouts. A radial match alone is insufficient. |
| 11. Theory-to-observable adapters | Convert solved potentials/fields into circular speeds, line-of-sight velocity fields, weak shear, convergence, deflection, critical curves, and raw multiple-image roots. Keep photon and massive-tracer mappings explicit. | Newtonian/MOND published fixtures reproduce within declared numerical and data-processing tolerance; photons are never silently scored with a massive-particle rule. |
| 12. Fair scoring | Add fixed train/development/holdout splits, no-target-access execution, nuisance-policy declarations, universal/per-object parameter counts, likelihoods with covariance, and same-input comparators. | A batch report separates galaxies, clusters, topology, and Solar-System tests and shows performance versus Newtonian, fixed MOND/RAR, and a declared halo baseline without a single blended score. |
| 13. Cluster data | Version member light, intracluster gas, geometry, source redshifts, weak-shear catalogs, and raw strong-lens image positions with licensing and sealed/open states. | Multiple clusters can be run with one frozen gravity parameter set; raw image/topology holdouts are scored, not only reconstructed dark-matter maps. |
| 14. Asynchronous API | Add model registration, data registration, job queue, lifecycle events, cancellation, retries, caching, batch runs, artifact indexes, and stable error states. | Identical model/data/solver/seed hashes return the cached immutable run; a browser never holds a request open for a long solve. |
| 15. Storage and reproducibility | Add durable database and object storage for model, dataset, code, solver, container, seed, grid, boundary, logs, predictions, plots, and manifests. Sign citation-ready manifests. | A clean worker can reproduce selected run hashes/tolerances from only the manifest and permitted inputs; failed runs retain useful artifacts. |
| 16. Hosting and operations | Containerize the Python worker, deploy CPU resource classes, connect it to the Vercel control plane, and add auth, quotas, monitoring, cost controls, backups, abuse handling, and uptime/support policy. | Public users can submit a bounded batch, follow status, download artifacts, and cannot exhaust shared resources or access sealed data. |
| 17. Reports and SDKs | Generate JSON, CSV/Parquet, FITS/NumPy fields, plots, residual maps, convergence histories, and a deterministic methods report. Add Python and HTTP examples. | Another researcher can reproduce and independently interpret a run without copying output into an LLM. |
| 18. Optional LLM explanation | Generate a compact, redacted briefing artifact and allow an opt-in LLM to summarize it. The model cannot alter scores, manifests, or scientific state. | Deterministic results remain complete without an LLM key; generated prose is labeled and traceable to the briefing hash. |
| 19. External replication | Ask scientists unaffiliated with SigmaGravity to encode Refracted Gravity and at least two other models, upload a new system, run a batch, and report friction. | External users reproduce expected fixtures without repository assistance; every failure is retained and classified as model, data, numerical, or product error. |

## Delivery order and rethink gates

1. Finish the field contract, formal schema, and conformance fixtures. **Initial stationary-field contract complete.**
2. Harden the generic local worker through convergence, nonlinear, boundary,
   coupled-field, and resource tests.
3. Package the worker and connect an asynchronous job path with immutable
   artifacts.
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

The container-ready local field-job milestone now passes: one canonical model
manifest plus one content-hashed array bundle enters the generic worker and a
known-answer field, residual history, resource log, output hashes, and immutable
result manifest come out. The immediate next milestone is durable array upload
and an asynchronous queue that sends this unchanged job format to the worker.
