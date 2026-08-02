# Hosted simulator deployment status

Date: 2026-08-02

## Outcome

The Vercel-ready radial research service is implemented and passes its local
production build and ten automated tests. It has not been published because
Vercel rejected the supplied credential as unauthorized for both identity and
project listing. Its `vck_` prefix identifies it as a Vercel API key, not the
`vcp_` personal access token accepted by the deployment CLI. No Vercel project
was created, no deployment URL exists, and the credential was not stored in a
file, repository setting, or generated artifact.

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

## Authentication handoff

Use either an already authenticated local Vercel CLI session or a valid Vercel
personal access token with permission to create and deploy a project in the
intended account/team. Never commit the token or place it in `.env` tracked by
Git.

Once authentication succeeds:

```powershell
cd research/galaxy-cluster-unification/hosted-simulator
npx.cmd vercel --yes
npx.cmd vercel --prod --yes
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
and point arrays.

## Worker milestone after Vercel

The next deployable component is a containerized Python worker plus durable
job and artifact storage. It must invoke the repository's existing Poisson,
AQUAL, QUMOND, coordinate-safe root, and raw-lensing engines. Vercel will
enqueue those jobs and return immediately; it will not run the long scientific
solve inside the web request.
