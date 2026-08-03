# Sigma Gravity Research Simulator — hosted preview

Production: <https://sigma-gravity-research-simulator-five.vercel.app>

- Vercel team: `Horizon3` (`horizon3`)
- Vercel project: `sigma-gravity-research-simulator`

This package is the deployable Vercel control plane for the galaxy/cluster
research repository. The radial version is narrow but functional:

- all 175 published SPARC radial mass models are packaged with provenance;
- researchers can retrieve real systems or generate a seeded synthetic radial
  system;
- any real catalog system can be regenerated as a compressed baryonic radial
  twin while its observed speed and uncertainty columns are withheld;
- formulas use a bounded, dimension-aware JSON AST rather than `eval`;
- universal and per-object parameter counts are explicit;
- interactive runs compare the submitted equation with Newtonian baryons and
  a fixed simple-MOND law; and
- every result includes a canonical formula hash and run-manifest hash.

Version 0.11 adds `POST /api/v1/twin-runs`. It applies the same submitted
formula to both the published baryonic channels and a six-control-point radial
twin, then reveals the measured rotation curve for scoring. The response keeps
three errors separate: baryonic source reconstruction, formula-versus-observed
speed, and the prediction change caused by transporting the formula from the
published source to the generated twin. The parameter package lists all
withheld columns, contains no gravity parameters, and is invariant when the
observed velocities are perturbed.

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
POST /api/v1/observation-evaluation-jobs
GET  /api/v1/observation-evaluation-jobs/{id}
GET  /api/v1/observation-evaluation-jobs/{id}/events
GET  /api/v1/observation-evaluation-jobs/{id}/artifacts
POST /api/v1/galaxy-jobs
GET  /api/v1/galaxy-jobs/{id}
GET  /api/v1/galaxy-jobs/{id}/events
GET  /api/v1/galaxy-jobs/{id}/artifacts
POST /api/v1/batches
GET  /api/v1/batches/{id}
GET  /api/v1/batches/{id}/events
GET  /api/v1/batches/{id}/artifacts
POST /api/v1/runs
POST /api/v1/twin-runs
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

A completed field can now be scored against new data without solving it again:

```text
POST /api/v1/observation-evaluation-jobs
GET  /api/v1/observation-evaluation-jobs/{id}
GET  /api/v1/observation-evaluation-jobs/{id}/events
GET  /api/v1/observation-evaluation-jobs/{id}/artifacts
POST /api/v1/observation-evaluation-jobs/{id}/cancel
```

The submission names a successful `fieldJobId`, a separately uploaded
observation bundle, and one or more full `sigma-observation-target/1` objects.
The service verifies the source field manifest, model, job, scientific result,
and observable archive before queuing. Field identity and observation identity
remain separate: changing a mask, beam, uncertainty, or dataset creates a new
observation job and zero field-solver calls. P0732 byte-matches both 2D curve
and 3D resolved-map artifacts against integrated evaluation. Run the real HTTP
acceptance while `npm run dev` is active with:

```text
npm run smoke:observation-jobs
```

See `../docs/P0732_DECOUPLED_OBSERVATION_EVALUATION_MILESTONE.md`.

The 0.8 local reference API also supports formula-independent resolved-galaxy
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

P0733 makes each scored system a two-stage composition. `dataUploadId` supplies
the field-equation arrays, while optional `observationDataUploadId` supplies
measured velocity, mask, uncertainty, intensity, or beam arrays. The batch
creates or reuses the field job first, then creates or reuses at most one
`observation-evaluation-jobs` child after the field succeeds. Omitting
`observationDataUploadId` reuses the field upload for compatibility. Reports
retain both child IDs, both scientific and manifest hashes, both lifecycle
states, and the observation child's artifact hashes. Changing only observation
data therefore changes the batch and observation identities but preserves the
field identity. See
`../docs/P0733_COMPOSED_BATCH_OBSERVATION_JOBS_MILESTONE.md`.

Two massive-tracer adapters and two separately typed photon adapters are now
implemented. A system can
attach a content-hashed `sigma-observation-target/1` circular-speed curve or
resolved line-of-sight velocity map. After the field converges, the curve
adapter azimuthally samples any declared `massive_tracers` vector acceleration
and computes `v_c(R)=sqrt(R g_R)`. The map adapter samples the same declared
field at content-hashed disk-plane coordinates, projects circular speed through
an explicit inclination and handedness, and can apply an intensity-weighted
beam convolution before scoring observed pixels. Both write predictions,
residuals, RMSE, chi-square, reduced chi-square, and Gaussian log likelihood.
Neither adapter changes the field equation or accepts a photon observable.

`photon_lensing_map` instead requires a Cartesian 3D vector observable in
`m/s^2` whose manifest target is `photons` or `both`. It declares the north,
east, and line-of-sight storage axes, an explicit `D_ls/D_s` distance ratio,
and an explicit lens angular-diameter distance. The worker computes

`alpha_perp = -(2 D_ls / (c^2 D_s)) integral(a_photon,perp dl)`

and publishes named east/north deflection maps plus convergence, both shear
components, reduced shear, rotation, Jacobian determinant/eigenvalues, and
absolute magnification. Optional paired deflection and reduced-shear maps are
scored in separate `arcsec` and dimensionless channels; neither can be blended
with the velocity score. The deterministic NPZ map archive is content-hashed
and byte-identical between integrated and separately cached observation jobs.
See `examples/observation-targets/photon-lensing-map.json` and
`../docs/P0734_TYPED_PHOTON_LENSING_ADAPTER.md`.

`multiple_image_systems` uses the same explicit 3D photon projection but tests
raw observed image positions rather than a reconstructed convergence map. For
each source family it profiles two source coordinates, globally searches the
declared image plane for every closed lens-equation root, and uses a minimum-
cost one-to-one assignment. Those source coordinates are counted as
observational nuisance parameters and add no gravity parameter. If the field
predicts fewer roots than observed images, the target is marked
`incomplete_topology`; aggregate RMS, chi-square, and likelihood remain null
instead of scoring only the matched subset. Extra roots remain disclosed
because classifying their detectability requires an additional selection
model. See `examples/observation-targets/multiple-image-systems.json` and
`../docs/P0735_RAW_MULTIPLE_IMAGE_ADAPTER.md`.

The map target contract is illustrated by
`examples/observation-targets/line-of-sight-velocity-field.json`. Its named
coordinate, observation, uncertainty, intensity, mask, and beam arrays must be
included in the same immutable NPZ upload and declared in the array bundle with
their units and hashes. One unchanged target form works with either a 2D or 3D
Cartesian model whose requested observable is a massive-tracer acceleration in
`m/s^2`. `emissionMaskArrayKey` controls which intrinsic predicted pixels enter
beam convolution; `scoreMaskArrayKey` controls which convolved pixels are
compared with data. The legacy `maskArrayKey` remains a score-mask alias. An
intensity array may retain physical flux units because its overall scale
cancels in the normalized moment and weighted score. The public default
`nonPositiveInwardPolicy=exclude` rejects pixels without inward circular
support; `zero_speed` is available only when a frozen protocol explicitly
requires that convention.

The local DDO101 curve acceptance retains all ten published points and
reproduces the earlier frozen Newtonian curve within `0.496 km/s` RMS on a
deliberately coarse grid. Resolved-map manufactured solutions pass for both 2D
and 3D fields, including beam convolution and batch aggregation. P0731 also
passes real LITTLE THINGS parity for 13 galaxies and four fixed model
manifests: 52/52 evaluations have exact pixel support, valid hashes, zero
per-galaxy gravity parameters, and at most `1.96e-10 m/s` prediction parity RMS
against an independent frozen implementation. This is spent-sample adapter
commissioning, not theory validation. Pressure support/non-circular motions,
spectral cubes, time delays, flux-ratio selection, and independently observed
critical-curve catalogs remain future acceptance work. See
`../docs/P0731_REAL_VELOCITY_FIELD_ADAPTER_PARITY_RESULTS.md`.

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

P0727 added a formula-neutral bounded Picard warm-up before the same generic
Newton--GMRES residual solve. The frozen 40-step warm-up converged both
difficult fields and agreed with the independent Picard references to better
than `5.4e-8` normalized RMS across potential, acceleration, and predicted
speed. Warm-up steps count against the declared total iteration limit, and the
validator exposes all controls. See
`../docs/P0727_HYBRID_NONLINEAR_CROSSCHECK_RESULTS.md`.

P0728 applied the selected 40-step hybrid universally to four fine-grid
sentinels. Three converged and agreed with independent Picard references;
NGC1569 missed the relative-update gate, so deterministic reporting excludes
the partial AQUAL row from ranks and leaves sensitivity incomplete. See
`../docs/P0728_COMPLETE_FINE_GRID_AQUAL_RESULTS.md`.

P0729 used the 80-step variant that had already qualified in P0727. It
converged all four fine-grid fields, agreed with independent references, scored
all 55 points, and restored a fair complete model comparison. The inherited
resolution gates pass, with NGC1569 remaining close to the sensitivity limit.
See `../docs/P0729_QUALIFIED_80STEP_FINE_GRID_AQUAL_RESULTS.md`.

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
