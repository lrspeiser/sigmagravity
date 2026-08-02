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

The 0.2 preview also accepts a theory-neutral field-model manifest. It declares
scalar, vector, or tensor fields; 2D or 3D coordinates; equations; units;
boundary conditions; observables for matter and/or photons; solver settings;
and universal versus per-object parameters. The validator checks dimensions,
operator types, complexity, data requirements, and parameter disclosure, then
returns content hashes and the exact worker capabilities the model needs.

The same contract represents the included Newtonian Poisson, AQUAL, QUMOND,
Refracted Gravity, and two-potential examples under `examples/models`. The
local Python research package now contains a theory-neutral Cartesian 2D/3D
divergence-form worker prototype. It passes manufactured known-answer tests
and executes the Refracted Gravity example without a theory-name branch. That
worker is not connected to this Vercel gateway yet.

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

## Worker connection

The next deployment milestone is to containerize and connect the existing
generic Python worker. The
gateway will submit immutable jobs containing dataset, formula, solver, seed,
and code hashes. The worker will upload JSON/CSV/PNG artifacts to object
storage and report state through a durable database. Vercel remains the UI and
short-request gateway; it does not hold an HTTP connection open during field
or lensing solves.
