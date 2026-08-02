# Hosted simulator deployment status

Date: 2026-08-02

## Outcome

The radial research service is live in the Horizon3 Vercel team:

- production URL:
  <https://sigma-gravity-research-simulator-five.vercel.app>
- team and scope: `Horizon3` / `horizon3`
- project: `sigma-gravity-research-simulator`
- production deployment inspected at:
  <https://vercel.com/horizon3/sigma-gravity-research-simulator/EYbBFCaLGLvBUH7jfY7WcLRL9QAP>

The service passes its local production build, ten automated tests, and a live
HTTP smoke suite. The deployment credential was supplied only to the CLI
process and was not stored in a file, repository setting, or generated
artifact.

## Deployable artifact

The isolated deployment root is [`../hosted-simulator`](../hosted-simulator/README.md).
It contains only the public web/API application and a 502 KB generated catalog,
not the repository's large raw datasets or Python environments.

Implemented public capabilities:

1. List a versioned SPARC release and all 175 real galaxies.
2. Retrieve a named galaxy and its published radial mass-model data.
3. Create a deterministic seeded radial synthetic galaxy.
4. Validate and hash a bounded, dimension-aware formula AST.
5. Reject hidden per-object gravity parameters.
6. Run the formula against up to 25 systems and return full curves, fixed-MOND
   and Newtonian comparators, scores, assumptions, caveats, and an immutable
   manifest hash.
7. Download the result from the browser.
8. Return an explicit `worker_not_connected` state for field and lensing tests.

## Verification evidence

```powershell
cd research/galaxy-cluster-unification/hosted-simulator
npm.cmd test
npm.cmd run build
```

The current result is ten passing tests and a build check confirming 175
galaxies. The catalog generator separately confirms 3,391 radial points and
the release hash
`a5df1cb7c7a52da415a167d145a831fe0e0625243b46dd38047ca43ba0299681`.

## Redeployment

Use either an authenticated local Vercel CLI session or a Vercel personal
access token with permission to deploy inside `horizon3`. Never commit the
token or place it in `.env` tracked by Git.

```powershell
cd research/galaxy-cluster-unification/hosted-simulator
npx.cmd vercel --yes
npx.cmd vercel --prod --yes --scope horizon3 --project sigma-gravity-research-simulator
```

Then verify:

```text
GET  https://<deployment>/api/v1/health
GET  https://<deployment>/api/v1/datasets
GET  https://<deployment>/api/v1/systems/DDO154
POST https://<deployment>/api/v1/formulas/validate
POST https://<deployment>/api/v1/runs
```

The deployment is not accepted merely because the homepage loads. The hosted
DDO154 fixed-MOND response must match the local run ID, manifest hash, scores,
and point arrays. The accepted production smoke values are:

- formula SHA-256:
  `7461db9401d4396e4e7ad7f675007bc28adeace523a174b0211c73c2a5a27ce2`
- run ID: `run_5b5a7ce1fb73c49abc643832`
- manifest SHA-256:
  `5b5a7ce1fb73c49abc643832a54cafe34fc5afe3aeaae9d15a8ed0efd11cc9d9`
- fixed-MOND DDO154 RMSE: `4.451772996259156 km/s`
- Newtonian-baryon DDO154 RMSE: `23.71217692693497 km/s`

The first attempted project was accidentally created in the personal
`lrspeisers-projects` scope and contains only a failed build. It is not the
production target and should be removed separately if account cleanup is
desired.

## Worker milestone after Vercel

The next deployable component is a containerized Python worker plus durable
job and artifact storage. It must invoke the repository's existing Poisson,
AQUAL, QUMOND, coordinate-safe root, and raw-lensing engines. Vercel will
enqueue those jobs and return immediately; it will not run the long scientific
solve inside the web request.
