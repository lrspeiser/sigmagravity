# Sigma Gravity Research Simulator — hosted preview

Production: <https://sigma-gravity-research-simulator-five.vercel.app>

- Vercel team: `Horizon3` (`horizon3`)
- Vercel project: `sigma-gravity-research-simulator`

This package is the deployable Vercel control plane for the galaxy/cluster
research repository. The radial version is narrow but functional:

- all 175 published SPARC radial mass models are packaged with provenance;
- researchers can retrieve real systems or generate a seeded synthetic radial
  system;
- formulas use a bounded, dimension-aware JSON AST rather than `eval`;
- universal and per-object parameter counts are explicit;
- interactive runs compare the submitted equation with Newtonian baryons and
  a fixed simple-MOND law; and
- every result includes a canonical formula hash and run-manifest hash.

The 0.6 preview also accepts a theory-neutral field-model manifest. It declares
scalar, vector, or tensor fields; 2D or 3D coordinates; equations; units;
boundary conditions; observables for matter and/or photons; solver settings;
and universal versus per-object parameters. The validator checks dimensions,
operator types, complexity, data requirements, and parameter disclosure, then
returns content hashes and the exact worker capabilities the model needs.

The same contract represents the included Newtonian Poisson, AQUAL, QUMOND,
Refracted Gravity, and two-potential examples under `examples/models`. The
local Python research package now contains a theory-neutral Cartesian 2D/3D
divergence-form worker prototype. It passes manufactured known-answer tests
and executes the Refracted Gravity and QUMOND examples without theory-name
branches. QUMOND explicitly declares the zero-vector limit of its singular
isotropic flux with the general `multiply_zero_vector_limit` operator; this
prevents an implicit numerical convention from being hidden in the worker.
That worker is not connected to this Vercel gateway yet.

The hosted preview does **not** claim to execute the repository's full 2D/3D
Poisson, AQUAL, QUMOND, or raw cluster-lensing workflows. Those require an
asynchronous scientific worker and immutable artifact storage. Unsupported
tests return `worker_not_connected` instead of a proxy result.

## Local verification

From this directory:

```powershell
npm run build:catalog
npm test
npm run build
npm run dev
```

The dependency-free development server listens on `http://127.0.0.1:4173` by
default and exercises the same route handlers used by Vercel. Use
`npm run dev:vercel` when the directory has been linked to an authenticated
Vercel project.

The catalog build reads `../data/raw/sparc/table1.dat` and all 175 files under
`../data/raw/sparc/rotmod`. The generated `data/sparc-v1.json` is committed so
Vercel does not need access to the large scientific repository during deploy.
`npm run build` copies the HTML, JavaScript, and CSS into the ignored `dist`
directory expected by the Vercel project; serverless functions remain under
`api`.

## Vercel deployment

Deploy this directory as its own Vercel project. Do not deploy the repository
root, because it contains large scientific datasets and environments that are
not part of the web control plane.

```powershell
npx vercel --prod
```

Do not store a Vercel token in this directory. Use an authenticated CLI session
or pass the token through a process environment variable. `.vercel`, `.env*`,
and dependency directories are ignored.

## API examples

```text
GET  /api/v1/health
GET  /api/v1/datasets
GET  /api/v1/systems?q=DDO
GET  /api/v1/systems/DDO154
POST /api/v1/synthetic-galaxies
POST /api/v1/formulas/validate
POST /api/v1/models/validate
POST /api/v1/field-jobs/prepare
POST /api/v1/data-uploads
PUT  /api/v1/data-uploads/{id}/content
POST /api/v1/field-jobs
GET  /api/v1/field-jobs/{id}
GET  /api/v1/field-jobs/{id}/events
GET  /api/v1/field-jobs/{id}/artifacts
POST /api/v1/galaxy-jobs
GET  /api/v1/galaxy-jobs/{id}
GET  /api/v1/galaxy-jobs/{id}/events
GET  /api/v1/galaxy-jobs/{id}/artifacts
POST /api/v1/batches
GET  /api/v1/batches/{id}
GET  /api/v1/batches/{id}/events
GET  /api/v1/batches/{id}/artifacts
POST /api/v1/runs
GET  /api/v1/openapi.json
```

The initial formula AST supports constants, declared parameters, `g_bar`,
`radius`, `surface_density`, and the operators `add`, `sub`, `mul`, `div`,
`pow`, `min`, `max`, `sqrt`, `abs`, `exp`, and `log`. Formula trees are limited
to 128 nodes and depth 24.

`POST /api/v1/models/validate` is the field-equation path. A request uses the
`sigma-field-model/1` schema demonstrated by `examples/models/*.json`. It does
not execute pasted JavaScript or Python. A successful validation currently
returns `executionReadiness.state=worker_not_connected`; this is deliberate
until queue, data storage, and the Python worker are deployed.

`POST /api/v1/field-jobs/prepare` binds a valid model to a content-hashed array
manifest, grid spacing, boundaries, requested observables, seed, and parameter
policy. It rejects missing or unit-incompatible arrays and returns a stable
preflight hash plus a resource estimate. It cannot verify or execute array
bytes until uploads and the worker queue are connected, so it reports both
blockers explicitly.

The Node development server now supplies a real asynchronous reference backend
for the upload and field-job endpoints. `POST /api/v1/data-uploads` registers a
completed `sigma-array-bundle/1` manifest plus the expected NPZ byte hash and
size. `PUT` sends those exact bytes. `POST /api/v1/field-jobs` then queues the
confirmed field model, upload ID, boundaries, observables, solver settings, and
seed. Status, ordered lifecycle events, cancellation, and artifact downloads
are separate short requests. Artifact bytes are rehashed before download.

Run the complete HTTP known-answer test while `npm run dev` is active:

```text
npm run smoke:field-jobs
```

The local store defaults to `../tmp/hosted-field-job-service` and can be moved
with `SIMULATOR_LOCAL_STORE`. It is a single-user reference adapter with one
worker, bounded upload size, bounded estimated memory, bounded job count, and a
runtime timeout. It accepts only safe manifests and NPZ data; it never executes
uploaded researcher code.

The local worker CLI is `../scripts/run_generic_field_job.py`. It packages NPZ
arrays into a verified `sigma-array-bundle/1` directory and runs a
`sigma-field-job-request/1`. Every run writes separate deterministic job and
scientific-result hashes, output-array hashes, residual history, resource log,
artifact index, and reproduction manifest. See `../worker` for the pinned,
non-root container definition.

The 0.6 local reference API also supports formula-independent resolved-galaxy
jobs:

```text
POST /api/v1/galaxy-jobs
GET  /api/v1/galaxy-jobs/{id}
GET  /api/v1/galaxy-jobs/{id}/events
GET  /api/v1/galaxy-jobs/{id}/artifacts
POST /api/v1/galaxy-jobs/{id}/cancel
```

`extract_roundtrip` consumes an immutable Cartesian 2D NPZ bundle containing
`gas_surface_density` and `stellar_surface_density` in `M_sun/kpc^2`. It emits
a content-hashed baryonic parameter package, regenerated 2D density bundle,
an explicitly prior-based 3D density bundle, and round-trip metrics. `generate`
accepts one of those parameter packages plus controlled changes to mass,
radial scale, Fourier/asymmetry strength, local features, rotation, and
component offsets. The resulting surface or volume bundle can then be supplied
to any compatible `/api/v1/field-jobs` model; the galaxy job never selects a
gravity theory itself.

Generation may also declare `outputGrid.cellsPerAxis` and an `extentScale`
from 1 to 4. The latter expands the physical box about its original center
without changing the extracted baryonic parameter package. Pairing a 1.5x
extent with 1.5x as many intervals keeps the transverse cell spacing fixed and
provides a direct zero-boundary-proximity diagnostic.

The 3D density is not labeled as a uniquely recovered galaxy. Its scale height,
vertical profile, and flaring are declared priors, and multiple different 3D
realizations can project to the same 2D map. The public Vercel route currently
returns `production_worker_not_connected`; real execution is available through
the local reference server until durable workers and object storage are
deployed.

The same local server now chains generated surface- or volume-density bundles
into one multi-system field batch without registering a formula-specific route:

```text
POST /api/v1/batches
GET  /api/v1/batches/{id}
GET  /api/v1/batches/{id}/events
GET  /api/v1/batches/{id}/artifacts
POST /api/v1/batches/{id}/cancel
```

A batch freezes one `sigma-field-model/1` manifest and one declared parameter
policy across as many as 1,000 source bundles. The executable reference modes
are `published_fixed` and `universal_fixed`. Fitted, hierarchical, and
per-object policies are represented explicitly but rejected until their
no-target-access fitting workflows exist; they are never silently approximated.
The deterministic report includes per-system convergence, equation residuals,
parameter counts, failures, content hashes, and an optional-LLM briefing. It
reports `observationScoresAvailable=false` when a batch has no compatible
targets; it never substitutes numerical convergence for an observational fit.

The first observation adapter is now implemented for massive-tracer circular
speeds. A system can attach a content-hashed `sigma-observation-target/1` curve
with uncertainties or a full covariance matrix. After the field converges, the
adapter azimuthally samples any declared `massive_tracers` vector acceleration,
computes `v_c(R)=sqrt(R g_R)`, and writes predicted points, residuals, RMSE,
chi-square, reduced chi-square, and Gaussian log likelihood. It rejects photon
observables for this mapping. The local DDO101 acceptance retains all ten
published points and reproduces the earlier frozen Newtonian curve within
`0.496 km/s` RMS on a deliberately coarse grid. Photon lensing and resolved
velocity-field adapters remain future work.

The full P0723 commissioning run then used the same generic HTTP path for all
13 registered galaxies and four published-fixed manifests. All 52 child solves
converged and all 161 circular-speed points per manifest were scored with zero
per-object gravity parameters. Equal-galaxy RMSE was `23.154 km/s` for
Newtonian Poisson, `13.131 km/s` for AQUAL, `12.486 km/s` for QUMOND, and
`14.439 km/s` for the Refracted Gravity fixture. This spent-sample result
validates the formula-neutral execution and reporting path, not any gravity
theory; all aggregate reduced chi-square values remain much larger than one.
See `../docs/P0723_FORMULA_NEUTRAL_RESOLVED_COMPARATOR_RESULTS.md`.

P0724 exercised the same route under six frozen geometry/reconstruction
scenarios on four sentinel galaxies. It retained 94 converged jobs and two
fine-grid AQUAL nonconvergences. Expanded boundaries and two vertical-prior
draws passed their stability gates; the coarse AQUAL aggregate fit was
resolution-sensitive. Partial rows are now excluded from model ranks and
plotted as missing. See
`../docs/P0724_GRID_BOX_VERTICAL_SENSITIVITY_RESULTS.md`.

P0725 added generic nonlinear initialization plus explicit requested/effective
iteration-limit metadata. A unit-coefficient linearized warm start with fixed
damping `0.20` converged both failed fine-grid AQUAL inputs under one universal
setting. It is not yet the production default because no second successful
method was available for the frozen solution-agreement gate. See
`../docs/P0725_AQUAL_SOLVER_ROBUSTNESS_RESULTS.md`.

P0726 added generic Anderson and Newton--Krylov residual methods. The frozen
Newton--GMRES path independently reproduced the DDO53 Picard field and speeds
to better than `5e-9` normalized RMS, but did not converge DDO101 from the
linearized field. The API therefore retains diagnostics and no universal
production solver is selected. See
`../docs/P0726_INDEPENDENT_NONLINEAR_CROSSCHECK_RESULTS.md`.

Run the chained 3D HTTP acceptance test while `npm run dev` is active:

```text
npm run smoke:batches
```

## Production worker connection

The local queue proves the API contract and the unchanged Python worker process,
but it is not the production deployment. Production still needs direct object-
storage uploads, a durable queue/database, container scheduling, authentication,
project isolation, retries, quotas, and monitoring. Vercel remains the UI and
short-request gateway; it must not execute Python or hold an HTTP connection
open during field or lensing solves.
