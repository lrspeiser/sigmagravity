# Hosted simulator deployment status

Date: 2026-08-02

## Outcome

The radial research service is live in the Horizon3 Vercel team:

- production URL:
  <https://sigma-gravity-research-simulator-five.vercel.app>
- team and scope: `Horizon3` / `horizon3`
- project: `sigma-gravity-research-simulator`
- production deployment inspected at:
  <https://vercel.com/horizon3/sigma-gravity-research-simulator/69F46H1818YVZhukGzJYDrSFBzL9>
- immutable deployment URL:
  <https://sigma-gravity-research-simulator-7r9j4n26t-horizon3.vercel.app>
- deployment ID: `dpl_69F46H1818YVZhukGzJYDrSFBzL9`
- public contract version: `0.20.0-preview`

The service passes its local production build, 85 automated hosted tests, all
1,568 Python scientific tests, and the live HTTP smoke suite. No deployment
credential is stored in a file, repository setting, or generated artifact.

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
8. Publish the formula-neutral 2D/3D model, field-job, galaxy-job, batch, and
   decoupled observation-evaluation contracts, including separate field and
   observation upload identities for composed batches.
9. Return an explicit `worker_not_connected` state for production field,
   observation-evaluation, and lensing tests instead of substituting a radial
   proxy.
10. Publish the typed `photon_lensing_map` target, example, and schema for
    coordinate-safe 3D photon projection into deflection, convergence, shear,
    rotation, Jacobian, and magnification maps. The local reference worker
    executes it; velocity, deflection, and reduced-shear scores remain separate.
11. Publish the `multiple_image_systems` target, example, and schema for raw
    image-plane source profiling, global root search, one-to-one assignment,
    explicit topology failure, and a separate image-position score channel.
    The local reference worker executes it and adds no gravity parameter.
12. Regenerate a selected real galaxy's radial baryonic source profile while
    withholding its measured rotation speeds and uncertainties, evaluate the
    same submitted formula on the generated twin and measured baryons, and
    display both predictions against the held-out observations with residuals.
13. Require a separate researcher acknowledgement bound to the exact validated
    computational model hash before any 2D/3D field or batch preflight. The
    Python worker repeats the check before solving, so a changed equation cannot
    reuse an earlier confirmation.
14. Publish a researcher guide that distinguishes live production, local-only,
    and unbuilt capabilities; gives concrete input/output examples; explains
    what each result can and cannot establish; and gives a bounded roadmap for
    Sigma Gravity and inverse halo-response research.
15. Publish and validate a generic nonlocal baryon-to-response manifest. The
    local worker executes its convolution with a centered kernel, linear
    zero-padded boundaries, physical cell-volume integration, and no hidden
    normalization; public heavy execution remains disconnected.
16. Publish the inverse baryon-to-response discovery contract and researcher
    explanation. The local worker fits one shared compact 2D/3D kernel and one
    amplitude across systems, propagates target uncertainty, evaluates a
    radial-angle null, exposes rank/nullity and compatible-kernel diagnostics,
    supports non-negative or signed zero-net responses, counts every discovery
    coefficient, and emits 14 verified deterministic artifacts. Production
    returns an explicit 503 until durable isolated workers are connected.
17. Publish and locally execute a non-degenerate two-potential photon/matter
    fixture through the generic multi-field worker. The exact confirmed model
    returns separately typed matter and photon accelerations, discloses its
    sequential Gauss-Seidel update scheme, uses zero per-object gravity
    parameters, and has a separate manufactured bidirectional-coupling test.
18. Publish a backward-compatible multi-null inverse-response contract. The
    local 2D/3D worker now supports source radial-angle shuffle, source
    Fourier-phase scramble, target-system permutation, target radial-angle
    shuffle, and total-preserving synthetic missing-baryon dropout. It reports
    each family and passes globally only when the observed pairing beats every
    declared family.

## Verification evidence

```powershell
cd research/galaxy-cluster-unification/hosted-simulator
npm.cmd test
npm.cmd run build
```

The current result is 85 passing hosted tests, 1,568 passing Python scientific
tests, and a build check confirming 175
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
GET  https://<deployment>/api/v1/openapi.json
GET  https://<deployment>/schemas/observation-evaluation-job-submit-v1.schema.json
GET  https://<deployment>/schemas/inverse-response-job-submit-v1.schema.json
POST https://<deployment>/api/v1/formulas/validate
POST https://<deployment>/api/v1/models/validate
POST https://<deployment>/api/v1/models/confirm
POST https://<deployment>/api/v1/runs
POST https://<deployment>/api/v1/twin-runs
POST https://<deployment>/api/v1/observation-evaluation-jobs
POST https://<deployment>/api/v1/inverse-response-jobs
GET  https://<deployment>/schemas/model-confirmation-request-v1.schema.json
```

The deployment is not accepted merely because the homepage loads. The hosted
DDO154 fixed-MOND response must match the local run ID, manifest hash, scores,
and point arrays. The accepted production smoke values are:

- formula SHA-256:
  `7461db9401d4396e4e7ad7f675007bc28adeace523a174b0211c73c2a5a27ce2`
- run ID: `run_4a8ce420ff4fb9387e8b45a2`
- manifest SHA-256:
  `4a8ce420ff4fb9387e8b45a21b7cb5ad0ee0e59f3e741bf2ace0453a994a3955`
- fixed-MOND DDO154 RMSE: `4.451772996259156 km/s`
- Newtonian-baryon DDO154 RMSE: `23.71217692693497 km/s`

The v0.9 live checks additionally require the homepage and OpenAPI document to
advertise `/api/v1/observation-evaluation-jobs`, the observation submission
schema to resolve with HTTP 200, and the production submission route to return
HTTP 503 with `production_worker_not_connected` until durable storage and an
isolated scientific worker are connected. The local reference backend does
execute this contract and passed byte-parity, content-identity, cancellation,
restart-recovery, and artifact-rehash gates without rerunning the source field
solve. The batch schema must also expose `observationDataUploadId`, and the
local P0733 acceptance must preserve a field job while changed observational
uncertainty creates a new evaluation job.

The v0.10 photon checks additionally require the health document to report
`localTypedPhotonLensingMaps=available_in_dev_server`, the observation-target
schema to include `photon_lensing_map`, the published photon target example to
name all three axes, and production heavy execution to remain honestly marked
`production_worker_not_connected`. P0734 passed its frozen analytic normalization,
point-mass, affine-invariant, channel-separation, deterministic-map, decoupled
parity, and composed-batch gates before deployment.

The v0.10 raw-image checks additionally require the homepage to show local raw
image roots as available, health to report
`localRawMultipleImageLensing=available_in_dev_server`, the observation-target
schema to include `multiple_image_systems`, and the published example to retain
families, positions, uncertainties, axes, distance ratios, and root controls.
P0735 passed source profiling, root closure, missing-topology non-scoring, axis
permutation, multiple-distance-ratio, byte-parity, batch-channel, and real
catalog round-trip gates before deployment. The AS295/PLCKG287 catalog is not
called score-ready because it lacks published per-image positional errors.

The v0.11 held-out-twin checks additionally require health to report
`heldoutObservedGalaxyTwins=available`, OpenAPI to advertise
`/api/v1/twin-runs`, and the browser to plot the observed curve, the submitted
formula on the generated twin, the same formula on measured baryons, fixed
MOND on the twin, Newtonian baryons on the twin, and an uncertainty-aware
residual panel. The accepted DDO154 smoke result has twin run ID
`twinrun_eaed32ca1924b4a31050bb30`, source-gravity normalized RMSE
`0.000008625358785734849`, and submitted-formula twin RMSE
`4.459265029781337 km/s`. Across all 175 systems, the P0737 audit confirmed
that twin packages are invariant to mutations of all held-out velocity data,
uses zero gravity parameters, and obtained a median fixed-MOND transport RMSE
of `0.7518162551 km/s`. Its frozen worst-transport gate remains honestly failed:
NGC2903 reaches `6.9564143643 km/s` against a `5 km/s` limit, so this radial
twin is a useful diagnostic rather than a completed 2D/3D galaxy generator.

The v0.14 resolved checks require eight systems, 146,532 scored H I velocity
pixels, the P0752 evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`,
and separate protocol and formula verdicts. A protocol pass must not be
rendered as a theory pass.

The v0.15 confirmation checks require a draft model to report
`awaiting_researcher_confirmation`, the published schema to resolve, and the
homepage to expose a separate **Confirm exact hash** action. The live
Newtonian-Poisson confirmation receipt is
`bc59e0053cf7a37f5dc0ef2d98be7b835f9927e69554337cfddaf7258364d795`.
Changing a solver control in the browser invalidated the prior confirmation,
enabled the separate confirmation action, and produced a different bound hash
without browser errors. The real local HTTP field, observation, and composed
batch smoke suites also passed with gateway/worker source-hash agreement.

The v0.16 documentation checks require `/guide.html` to resolve on the public
alias, health and OpenAPI to report `0.16.0-preview`, and the guide to retain its
capability matrix, concrete radial/field/worker-boundary examples, scientific
limitations, and seven-step discovery roadmap. Desktop and 390-pixel mobile
browser checks found no settled page-level horizontal overflow or console
errors. The live HTTP smoke reproduced all radial scores, twin metrics, and the
resolved-evidence hash above; `POST /api/v1/field-jobs` still returns HTTP 503
with `production_worker_not_connected` until the durable worker is connected.

The v0.17 nonlocal checks require the public health document to report
`localNonlocalConvolution=available_in_dev_server`, the workbench and guide to
link `/examples/models/nonlocal-response.json`, and model validation to report
the `convolution` capability for the exact confirmed model hash
`3a2d7dd1b296d5c7a8e9b99b58f739547896c8dee7e4ba692174440286eaaeca`.
Local manufactured tests require physical-volume impulse normalization and no
opposite-edge periodic wraparound. The published example solves in 3D through
the generic worker, while the production gateway continues to report
`worker_not_connected` for heavy field execution.

The v0.18 inverse-discovery checks require health to report
`localInverseHaloResponseDiscovery=available_in_dev_server`, OpenAPI to expose
`/api/v1/inverse-response-jobs`, the public inverse submission schema to return
HTTP 200, and the guide to show a concrete request, shortened result, artifact
list, parameter count, and forward-test boundary. Production submission must
return HTTP 503 with `production_worker_not_connected` and classification
`hypothesis_generator_not_forward_theory_test`. The local real-HTTP
known-answer test recovered the injected asymmetric kernel with cosine
`0.99999999999937`, amplitude `1.5999999988039162`, and aggregate R-squared
`0.9999999999999991`; all 14 downloaded artifact hashes and the independent
gateway/worker source hashes agreed. The worker used zero per-system gravity
parameters, recorded its target as `model_derived_discovery_target`, and
recorded that no held-out raw observation entered the inverse.

The v0.19 coupled-field checks require health to report
`localCoupledTwoPotentialPhotonMatter=available_in_dev_server`, the guide to
show both the numerical acceptance and its scientific limitations, and the
published two-potential model to validate at exact hash
`bcc7c218ec4d11ee77c85837530daa342e98748c3eb04e460b35f93a7e17accc`.
The local real-HTTP job solved two 3D potentials, verified all eight downloaded
artifact hashes, used zero per-object gravity parameters, recovered
`Phi=1.5 Psi` with relative error `4.16e-16`, and recovered photon acceleration
equal to `1.25` times matter acceleration with maximum relative error
`4.19e-16`. A separate known-answer system makes each field depend on the
other, so this acceptance cannot pass by solving two independent equations.
The public alias passes the complete v0.19 HTTP smoke and model validation,
while heavy execution remains explicitly `worker_not_connected`.

The v0.20 inverse-control checks require health to report
`localInverseResponseMultiNullSuite=available_in_dev_server`, the published
schema to expose the `all_declared_families` suite, and the guide to explain
all five controls and their limits. The local real-HTTP known-answer run used
95 null fits, recovered the injected kernel with cosine
`0.99999999999937`, recovered amplitude `1.5999999988039162`, and obtained
maximum family Monte Carlo p-value `0.05` with every family passing. All 14
downloaded artifact hashes and the gateway/worker source hashes agreed; no
held-out raw observation or per-system gravity parameter entered the inverse.
The production alias passed the complete radial HTTP smoke at v0.20. Desktop
and 390-pixel browser checks found no page-level horizontal overflow or console
errors. Public heavy execution remains explicitly
`production_worker_not_connected`.

The first attempted project was accidentally created in the personal
`lrspeisers-projects` scope and contains only a failed build. It is not the
production target and should be removed separately if account cleanup is
desired.

## Worker milestone after Vercel

The next deployable component is durable job and artifact storage plus a
containerized Python worker running the already-tested local field, galaxy,
batch, and decoupled observation-evaluation services. It must invoke the
repository's existing Poisson, AQUAL, QUMOND, nonlocal convolution,
coordinate-safe root, and
raw-lensing engines. Vercel will enqueue those jobs and return immediately; it
will not run the long scientific solve inside the web request.
