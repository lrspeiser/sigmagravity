# Hosted simulator deployment status

Date: 2026-08-03

## Outcome

The research simulator and public API gateway are live in the Horizon3 Vercel
team:

- production URL:
  <https://sigma-gravity-research-simulator-five.vercel.app>
- team and scope: `Horizon3` / `horizon3`
- project: `sigma-gravity-research-simulator`
- production deployment inspected at:
  <https://vercel.com/horizon3/sigma-gravity-research-simulator/FL6AriDMSpit9bxZCd33VSgpYxuZ>
- immutable deployment URL:
  <https://sigma-gravity-research-simulator-hfdl9zo71-horizon3.vercel.app>
- deployment ID: `dpl_FL6AriDMSpit9bxZCd33VSgpYxuZ`
- deployed implementation commit: `4e6a2d4028b12e83f730aff00b421f1ca7eca55c`
- public contract version: `0.32.0-preview`

The service passes its local production build, 147 automated hosted tests, the
live deterministic HTTP smoke suite, a real private-queue canary, and the
GitHub Linux container acceptance for the field and galaxy workers. No
deployment credential is stored in a tracked file or generated artifact.

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
19. Publish a deterministic four-cluster RELICS evidence registry that keeps
    registered projected baryonic maps, model-derived GLAFIC/Zitrin lens maps,
    and raw multiple-image observations in separate scientific roles. It
    exposes readiness, uncertainty limits, hashes, and blockers without
    claiming that the spent P0633 sample is a blind holdout or that any current
    raw target is likelihood-ready.
20. Publish the bounded local baryonic uncertainty-ensemble contract and guide.
    The gravity-independent worker retains every seeded 2D surface draw and
    every corresponding 3D vertical draw, emits percentile maps and draw
    tables, verifies exact projection, and rejects more than 256 MiB of raw
    ensemble arrays. The public gateway advertises and validates the contract;
    production execution remains disconnected.
21. Materialize selected 2D surface-density and 3D volume-density ensemble
    draws into verified SI field bundles, fan one unchanged confirmed model
    across them, and report per-realization results plus within-system
    p16/p50/p84 prediction-score spread. The selection, parent artifact hash,
    exact realization indices, units, and source-array content hashes are bound
    into every child identity. These are prior-prediction spreads, not
    likelihood-weighted posterior credible intervals.
22. Execute stationary scalar elliptic manifests on a verified axisymmetric
    cylindrical `(r,z)` grid in the local worker. The immutable job binds
    `axisOrder=["r","z"]` and `origin=[0,z0]`; the symmetry axis uses a regular
    zero-radial-flux limit. Bessel manufactured solutions show second-order
    convergence. Public production advertises this local capability but still
    does not execute the heavy worker.
23. Convert a solved axisymmetric radial acceleration into circular-speed
    curves or inclined resolved line-of-sight velocity maps. The local worker
    samples the declared `(r,z)` midplane directly, retains uncertainty, masks,
    beam convolution, covariance and nuisance accounting, and rejects
    Cartesian axis conventions. The exact known-answer acceptance introduces
    zero per-object gravity parameters. This is an observation adapter, not a
    claim that circular equilibrium is complete or that the formula fits a
    real galaxy.
24. Publish and locally execute projection of one solved axisymmetric
    photon-acceleration field directly into
    inclination-aware deflection, convergence, shear, rotation, Jacobian, and
    magnification maps. Each ray is clipped to the finite solved cylinder and
    reconstructs its local Cartesian vector without a 3D proxy. Photon and
    massive-tracer targets can be scored in one immutable job with separate
    channels and zero per-object gravity parameters. Time-delay, weak-lensing,
    and non-axisymmetric cluster likelihoods remain future work.
25. Convert the same axisymmetric photon field into one archived
    distance-ratio-one deflection map, scale it for each declared source
    family, profile two source coordinates per family, find global
    lens-equation roots, assign raw observed images one-to-one, and score the
    image positions. The finite-support gate covers the full root square and
    Jacobian stencil; unsupported topology cannot be converted into a finite
    fit score. This path remains local-only and is an analytic software
    acceptance, not a real cluster result.
26. Provide an authenticated, separately deployable field-worker HTTP boundary
    and a server-side Vercel connector for the exact upload and field-job
    lifecycle routes. The non-root container runs the existing 2D, 3D,
    axisymmetric, massive-tracer, photon-map, and raw-image fixtures, rehashes
    every published artifact, rejects unindexed output and quota violations,
    and recovers or invalidates jobs safely after restart. Vercel has no worker
    token or worker URL configured yet, so production continues to fail closed
    with an explicit 503 rather than implying durable heavy execution.
27. Expose the gravity-independent resolved-galaxy extraction and generation
    lifecycle through the same authenticated worker and allow-listed Vercel
    connector. The extraction job accepts registered gas and stellar maps,
    emits content-hashed 2D/3D baryonic ensembles and a compact parameter
    package, and cannot inspect velocity, lensing, dark-matter, or gravity-model
    targets. A second job generates declared structural variants from that
    package. The Linux container acceptance downloads and rehashes every
    artifact and reports zero gravity parameters; public Vercel still returns
    an explicit 503 because no persistent external worker is connected.

## Verification evidence

```powershell
cd research/galaxy-cluster-unification/hosted-simulator
npm.cmd test
npm.cmd run build
```

The current result is 121 passing hosted tests, 1,609 passing Python scientific
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
npx.cmd vercel --prod --yes --scope horizon3
```

Then verify:

```text
GET  https://<deployment>/api/v1/health
GET  https://<deployment>/api/v1/datasets
GET  https://<deployment>/api/v1/systems/DDO154
GET  https://<deployment>/api/v1/cluster-evidence?system=AS295
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
- manifest SHA-256:
  `d541618c4fd3b2a1dca0e963514841741086c912634dd13ade4c434194409e8f`
- run ID: `run_d541618c4fd3b2a1dca0e963`
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

The v0.21 cluster-evidence checks require health to report
`resolvedClusterEvidenceRegistry=available`, OpenAPI to advertise
`/api/v1/cluster-evidence`, and the registry to reproduce SHA-256
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
The public response must contain exactly four registered baryonic field inputs,
four systems with two model-derived lens-map methods each, two systems that
passed the old raw-catalog readiness gate, zero score-ready raw forward targets,
and zero prospective holdouts. The stable public alias passed the complete HTTP
smoke with twin run `twinrun_9b5466d74274760f88f497ca`. Desktop and 390-pixel
mobile browser checks found no page-level horizontal overflow or console
warnings/errors; the wide capability table scrolls inside its own container on
mobile. Public heavy execution remains explicitly
`production_worker_not_connected`.

The v0.22 baryonic-ensemble checks require health to report
`localBaryonicUncertaintyEnsembles=available_in_dev_server`, the public galaxy
schema to expose 1 to 16 surface realizations and all bounded prior fields, and
the guide to label the result
`observation_conditioned_prior_not_posterior`. The local DDO101 HTTP smoke
retained a `3 x 65 x 65` surface ensemble and a
`3 x 2 x 65 x 65 x 25` volume ensemble, rehashed all 21 extraction artifacts,
and projected every 3D draw to its matching 2D map with maximum relative error
`3.64e-16`. The stable public alias passed the complete v0.22 HTTP smoke with
run `run_6ae2980fde9ea35e58d1f17c`, twin run
`twinrun_54a7b74eefc70c0383650a73`, and unchanged cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
Desktop and 390-pixel mobile browser checks found no page-level overflow or
console warnings/errors; the capability table remains internally scrollable
on mobile. This verifies the public contract and local artifact path, not an
observation-derived baryonic posterior.

The v0.23 ensemble-propagation checks require health to report
`localBaryonicEnsemblePropagation=available_in_dev_server`, the public batch
schema to accept `surface_density_ensemble` and `volume_density_ensemble`, and
the guide to show example 09 with exact inputs, shortened outputs, and a clear
prior-versus-posterior boundary. The real local DDO101 HTTP batch used one
confirmed Newtonian-Poisson model on four parent systems and five solver
children, including two selected 3D realizations. All five solves converged,
the maximum equation residual was `2.6226280458858716e-10`, all 16 artifacts
rehash correctly, and the run introduced zero per-object or
observation-derived gravity parameters. The three scored parents contain 30
published rotation-curve points; aggregate RMSE is
`40.37055168881182 km/s`, weighted RMSE is `43.07208501177664 km/s`, and
reduced chi-square is `201.38128805090903`. Those poor Newtonian scores are a
useful falsification result, not a platform failure. The public alias passed
the complete v0.23 smoke with run `run_0197e79c5c42ddbeb7ace796`, manifest
`0197e79c5c42ddbeb7ace796747b3e533e3e0832e1ebc30eaafa6a845d6e8699`,
twin run `twinrun_6a8f3a18fd0e124e9b345472`, and unchanged cluster registry
hash `875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
Production `POST /api/v1/batches` still returns HTTP 503 with
`production_worker_not_connected`; the Vercel deployment publishes and
validates the contract but does not claim to run the local scientific worker.

The v0.24 baryonic-conditioning checks require health to report
`localBaryonicImageConditioning=available_in_dev_server`, the public galaxy-job
schema to expose the fixed `diagonal_gaussian_surface_density` likelihood, and
the guide to show Example 10 with its request, artifacts, data boundary, and
collapsed-weight interpretation. The real local DDO101 HTTP batch used
gravity-independent gas and stellar surface-density uncertainties, selected
two 3D draws, and produced weights `[1.0, 0.0]`, ESS `1.0`, ten weighted
prediction rows, and the explicit status `degenerate_importance_weights` with
`credibleIntervalReady=false`. All five field children converged, maximum
equation residual was `2.6226280458858716e-10`, all 17 artifacts rehashed, and
no per-object or observation-derived gravity parameter was introduced. The
aggregate and conditioned-weight rotation RMSE values were respectively
`40.17397086769832 km/s` and `42.88650710545306 km/s`; the poor Newtonian fit
is retained rather than repaired with the measured velocity target.

Deployment `dpl_DkNa41DMSz1xvf8ShdioxfRp4HnF` is production-ready at the stable
alias and serves `0.24.0-preview`. The complete live HTTP smoke reproduces run
`run_55745def401569f80dd362a6`, manifest
`55745def401569f80dd362a6aade0cb212d326e4793a7fa26a20c107f2858539`, twin run
`twinrun_208ecd0b698c0ec8107e05c1`, and unchanged cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
The fresh guide has ten examples, no page-level horizontal overflow at the
normal viewport, and no site console warning or error in a fresh tab. Public
`POST /api/v1/batches` continues to return HTTP 503 with
`production_worker_not_connected`, so the page does not imply that the Vercel
function executes the local 3D worker.

The v0.25 axisymmetric checks require health to report
`localAxisymmetricCylindricalFields=available_in_dev_server`, both array-bundle
schemas to require `axisOrder=["r","z"]` and `origin=[0,z0]`, and the guide to
show Example 11 with concrete input, output, verified error scaling, and explicit
limits. The local manufactured Bessel field has relative errors
`1.2597932529074785e-3`, `3.1487840159771037e-4`, and
`7.871635258224039e-5` at 25, 49, and 97 cells, respectively. The full
immutable-job test also verifies the regular axis, spatially varying
coefficients, worker version `1.2.0-preview`, artifacts, origin and axis order.

Deployment `dpl_7biUPj86kfyryd1TVqTjVJAfAe3Q` is production-ready at the stable
alias and serves `0.25.0-preview`. The complete live HTTP smoke reproduces run
`run_c98323738095e3ab59484d0c`, manifest
`c98323738095e3ab59484d0c398a89509c457979ed2ce608f52ce08843c11956`, twin run
`twinrun_67c06c459e0f89232b3d931f`, and unchanged cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
The radial submitted, fixed-MOND, and Newtonian RMSE values remain unchanged,
showing that the new geometry path did not alter the hosted radial benchmark.
Axisymmetric field execution remains local-only; Vercel heavy routes continue
to return `production_worker_not_connected`.

The v0.26 axisymmetric-observation checks require health to report
`localAxisymmetricGalaxyObservations=available_in_dev_server`, preflight and
worker validation to bind `axisOrder=["r","z"]`, `origin=[0,z0]`, and
`centerM=[0,z_midplane]`, and the guide to show Example 12 with concrete
inputs, outputs, coordinate rules, and scientific limits. Cartesian
`planeAxes` and azimuthal sampling are rejected for this path. The real local
asynchronous HTTP acceptance solved Cartesian 2D, Cartesian 3D, and
axisymmetric jobs, downloaded and rehashed every artifact, and used zero
per-object gravity parameters. Its axisymmetric solid-body known answer had
field relative L2 error `3.4364145737847694e-15` and circular-speed RMSE
`4.220673123283083e-15 m/s` using
`axisymmetric_midplane_direct`. This establishes software normalization and
coordinate consistency, not a fit to a real galaxy or a completed dynamical
model.

Deployment `dpl_2DEdgRigbtRLY8B5XdWbpmo7hyRg` is production-ready at the
stable alias and serves `0.26.0-preview`. The stable alias passed the complete
live HTTP smoke with run `run_f0748afc66904deb7bb60545`, manifest
`f0748afc66904deb7bb6054565a9d02be08a34f44fab68e21b17e462096fee42`, twin run
`twinrun_32ff131ec6beb66181526ead`, resolved evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`, and
cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
The immutable hostname is protected by Vercel's deployment challenge and was
verified through authenticated `vercel curl`; the public stable alias remains
directly accessible. Public heavy submissions still return HTTP 503 with
`production_worker_not_connected`.

The v0.27 axisymmetric-photon checks require health to report
`localAxisymmetricPhotonLensing=available_in_dev_server`; preflight and worker
validation bind the cylindrical field to `axisOrder=["r","z"]`, a zero radial
origin, an explicit sky shape, inclination, line-of-sight sample count, and the
same solved-field origin. The guide's Example 13 gives a complete target and
known-answer result. Independent gates recover the exact face-on affine map,
the edge-on finite-cylinder chord, and the point-mass
`4GM/(c^2 R)` normalization with improving line-of-sight resolution. The
Cartesian photon path is unchanged.

The real local asynchronous HTTP release run solved Cartesian 2D, Cartesian
3D, and axisymmetric jobs and rehashed every downloaded artifact. One
axisymmetric job used the same confirmed field for massive tracers and photons:
field relative L2 error was `3.4364145737847694e-15`, circular-speed RMSE was
`4.220673123283083e-15 m/s`, photon-deflection RMSE was
`5.490987717737826e-26 arcsec`, 11 artifacts rehashed successfully, and the
per-object gravity-parameter count was zero. These are analytic software and
normalization fixtures, not fits to released galaxy or cluster observations.

Deployment `dpl_HMGHfumGfJwju8dYw31PBST5HJxK` is production-ready at the
stable alias and serves `0.27.0-preview`. The live HTTP smoke reproduces run
`run_dfd14b92341185b2c1a2a275`, manifest
`dfd14b92341185b2c1a2a2757b6c9d9b012755f73325d841139b0cbca586105f`, twin run
`twinrun_6f40fa73cbb05da72369687f`, resolved evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`, and
cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
The public guide follows its canonical redirect and contains the new input,
output, and limitations. The immutable hostname passed authenticated
`vercel curl`. Public heavy submissions deliberately remain HTTP 503 until a
durable isolated worker is connected.

The v0.28 axisymmetric raw-image checks require health to report
`localAxisymmetricRawMultipleImageLensing=available_in_dev_server`; preflight
and worker validation bind `axisOrder=["r","z"]`, the zero radial origin,
inclination, sky shape, line-of-sight sampling, root bounds, source-family
distance ratios, and the exact solved-field origin. The worker archives one
distance-ratio-one projection rather than integrating the same rays again for
every source family. It refuses a root domain whose bilinear/Jacobian support
touches a truncated ray, and keeps topology failure separate from fit score.

The real local asynchronous HTTP release run used a smooth cored-isothermal
axisymmetric known answer. Field relative L2 error was
`0.0014789730022103532`; the two recovered outer images scored
`0.001692053225097455 arcsec` RMS; all 13 artifacts rehashed successfully; and
the per-object gravity-parameter count was zero. This validates coordinate
composition, distance-ratio scaling, root recovery, artifact identity, and
nuisance accounting. It is not a fit to released cluster observations.

Deployment `dpl_4oCqdTRhzf6Tv9Z48CQR7kfS24A5` is production-ready at the
stable alias and serves `0.28.0-preview`. The live HTTP smoke reproduces run
`run_96ee4fa523cac6cf9790ff18`, manifest
`96ee4fa523cac6cf9790ff18a5581747cde7422125114fb25c1b1ba9ae3f6b3b`, twin
run `twinrun_c80a954c24843d444f12d119`, resolved evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`, and
cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
The public guide contains the complete input/output example, limitations, and
nine-step Sigma Gravity/inverse-halo roadmap. The immutable hostname passed
authenticated `vercel curl`; public heavy submissions return HTTP 503 with
`production_storage_not_connected` until durable storage and isolated workers
are connected.

The v0.29 authenticated-worker checks add a real deployment boundary without
claiming that Vercel runs the scientific solver. A separate server accepts only
the bounded upload and field-job lifecycle, requires a constant-time-checked
bearer token, applies body and response quotas, and exposes no galaxy, inverse,
batch, arbitrary-code, filesystem-path, or secret surface. The gateway requires
HTTPS outside localhost, refuses redirects and path traversal, never returns the
worker credential, and keeps every unconfigured production route explicitly
unavailable. Artifact publication verifies the scientific manifest, artifact
index, job identity, file names, count, aggregate size, individual size, and
every SHA-256 before a result can become visible. Restart recovery repeats
those checks and converts mutated completed output into an infrastructure
failure rather than serving it as science.

The real local HTTP acceptance ran the worker in a process separate from the
gateway and completed Cartesian 2D, Cartesian 3D, axisymmetric galaxy/photon,
and raw multiple-image jobs. The manufactured relative field errors were
`0.0014291183165795044` in 2D, `0.003218964440079798` in 3D, and
`3.4364145737847694e-15` for the solid-body axisymmetric fixture. Its
circular-speed RMSE was `4.220673123283083e-15 m/s`, photon-deflection RMSE was
`5.490987717737826e-26 arcsec`, raw-lensing field relative error was
`0.0014789730022103532`, and raw image-position RMS was
`0.001692053225097455 arcsec`. All 13 raw-lensing artifacts and the other job
artifacts rehashed correctly; every fixture used zero per-object gravity
parameters.

GitHub Actions independently built the actual non-root Docker image and ran the
same four-case scientific HTTP acceptance successfully at
<https://github.com/lrspeiser/sigmagravity/actions/runs/30794032742>. The run
used implementation commit `63a260c0d510152750e895ed5591214131ea727b` and did
not publish an image or receive a production worker secret. This closes the
container-build gap but does not prove persistence on a future container host.

Deployment `dpl_H6E8D6C3rgytzNV6YqBK2AC2puS1` is production-ready at the
stable alias and serves `0.29.0-preview`; its immutable URL is
<https://sigma-gravity-research-simulator-uxlcl8hmt-horizon3.vercel.app>. The
live health document advertises the authenticated connector as available only
when an external worker is configured. Collection, detail, content, event,
artifact-download, and cancellation routes were all exercised on the stable
alias and return `production_storage_not_connected` or
`production_worker_not_connected` with HTTP 503. The ordinary public smoke
reproduces run `run_d541618c4fd3b2a1dca0e963`, manifest
`d541618c4fd3b2a1dca0e963514841741086c912634dd13ade4c434194409e8f`, twin run
`twinrun_eb785c4006fe30dd70a2db9b`, twin source error
`0.000008625358785734849`, twin submitted-formula RMSE
`4.459265029781337 km/s`, resolved evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`, and
cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
No external worker, persistent volume, database, or object store is connected
to production yet; heavy science remains local/CI verified, not hosted.

The v0.30 authenticated-galaxy-worker release extends the allow-listed worker
surface to `data-uploads` plus the galaxy-job collection, status, events,
artifacts, artifact download, and cancellation routes. It does not expose a
generic filesystem route, arbitrary code, inverse-response jobs, observation
jobs, or batches. The gravity-independent extractor cannot inspect a velocity
target, lensing target, dark-matter map, or submitted gravity model.

GitHub Actions built the updated non-root Linux image and completed both the
four-case field suite and a two-job galaxy extraction/generation suite at
<https://github.com/lrspeiser/sigmagravity/actions/runs/30795427239>. The
synthetic 65-by-65 baryonic-map fixture achieved total-map normalized L2 error
`0.05478050421226611` and pixel correlation `0.9972455569454114`. It retained
a `3 x 65 x 65` surface ensemble and `3 x 2 x 65 x 65 x 25` volume ensemble,
projected 3D density back to 2D with maximum relative error
`2.466495895856214e-16`, reproduced a requested gas-mass scale of `1.25` as
`1.2500000000000002`, downloaded and rehashed 23 extraction plus 22 generation
artifacts, used zero gravity parameters, and did not use velocity targets.
The image pins Astropy `7.1.1`, the missing runtime dependency identified by
the first failed container attempt, and CI now prints bounded subprocess stderr
on future failures.

Deployment `dpl_H31k9KURECH5j3mDyHhZwS7HSWig` is production-ready at the
stable alias and serves `0.30.0-preview`; its immutable URL is
<https://sigma-gravity-research-simulator-jflwrga1z-horizon3.vercel.app>.
The stable alias passed the complete deterministic HTTP smoke with run
`run_d541618c4fd3b2a1dca0e963`, manifest
`d541618c4fd3b2a1dca0e963514841741086c912634dd13ade4c434194409e8f`, twin run
`twinrun_eb785c4006fe30dd70a2db9b`, resolved evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`, and
cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
Valid public field and galaxy heavy routes return HTTP 503 while malformed job
identifiers return 404; this confirms fail-closed routing rather than hosted
scientific execution.

The first attempted project was accidentally created in the personal
`lrspeisers-projects` scope and contains only a failed build. It is not the
production target and should be removed separately if account cleanup is
desired.

## v0.31 durable private object-storage release

The v0.31 release connects the first durable production layer without claiming
that heavy scientific jobs are hosted. The Horizon3 project has a private
Vercel Blob store, and the provider adapter creates bounded immutable objects
at SHA-256-derived paths, disables overwrite/random suffixes, treats repeated
writes as idempotent, and downloads and rehashes the complete object after
write and on every read. Unit acceptance covers partial credentials, malformed
namespaces, traversal, quotas, reference mutation, stored-byte mutation, and
idempotency.

The real store canary ran in two separate processes. Both returned the same
private 160-byte object at
`sigma/v1/objects/deployment-canary/sha256/3b3aa125f6eb843feec8c90f35b70223a00375108a63150d5ad7b1761127017b.json`
and verified the same SHA-256 after download. Credentials remain in ignored
local/Vercel environment state and were absent from the tracked diff.

The hosted suite now has 126 passing Node tests, the static build verifies 175
SPARC galaxies, eight resolved-galaxy evidence packages, and four cluster
packages, and the isolated local HTTP smoke passed. GitHub Actions independently
built the non-root Linux image and passed the gateway plus real field-and-galaxy
container acceptance at
<https://github.com/lrspeiser/sigmagravity/actions/runs/30797087474> for
implementation commit `7dfffd9f33bcabb50fa0b225f070b51326725968`.

Production deployment `dpl_GiCv8TD7vFwJr8emRz6ENgFDKR5W` is ready at the
stable alias and serves `0.31.0-preview`; its immutable URL is
<https://sigma-gravity-research-simulator-apcli8huh-horizon3.vercel.app>.
The live deterministic HTTP smoke reproduces run
`run_d541618c4fd3b2a1dca0e963`, manifest
`d541618c4fd3b2a1dca0e963514841741086c912634dd13ade4c434194409e8f`, twin run
`twinrun_eb785c4006fe30dd70a2db9b`, resolved evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`, and
cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.

The public health and storage-readiness documents report the private object
store as configured while reporting the queue, job metadata database, and
stateless scientific container as not connected and production execution as
not ready. A real public field-job submission remains HTTP 503. The canonical
guide redirect was followed and verified to contain the v0.31 capability,
limitations, Sigma Gravity path, and inverse-halo discovery path.

## v0.32 durable control-plane and queue release

The v0.32 release adds a formula-neutral, transactional production job-control
implementation. Its nine-table PostgreSQL schema keeps projects, confirmed
models, uploads, jobs, ordered events, attempts and leases, artifacts, and a
transactional queue outbox separate. The job repository rejects idempotency-key
reuse with different scientific inputs, handles duplicate queue delivery,
recovers expired leases, gives cancellation precedence over late results, and
prevents a stale worker from publishing. Result finalization downloads and
rehashes the manifest and every artifact before making a job terminal.

The 147-test hosted suite executes the real SQL migration twice in embedded
PostgreSQL and covers transient publication failure, deterministic retry,
duplicate delivery, lease recovery, cancellation races, retryable worker
failure, stale-worker rejection, and manifest/index/path/hash/byte-count
agreement. GitHub Actions independently built the non-root Linux worker and
passed its real field-and-galaxy container acceptance at
<https://github.com/lrspeiser/sigmagravity/actions/runs/30798650630> for commit
`4e6a2d4028b12e83f730aff00b421f1ca7eca55c`.

Production deployment `dpl_FL6AriDMSpit9bxZCd33VSgpYxuZ` is ready at the
stable alias and serves `0.32.0-preview`; its immutable URL is
<https://sigma-gravity-research-simulator-hfdl9zo71-horizon3.vercel.app>. A
real public canary was published to `sigma-control-plane-canary-v1`, invoked a
private queue-trigger consumer, and wrote a private content-addressed
acknowledgement. A second independent smoke returned the same deployment hash
`c086d4e4d4868fc1d27e939ad7ae8c638daff1737faa7e8248968bd692f82aa4` and
acknowledgement hash
`ab7746edc88e31a6dd358ea2dbe340bc3f8290ba66beaa05eb077fe84c308c2e`.
`GET /api/v1/storage-readiness` consequently reports
`durable_storage_and_queue_connected` and `verified_consumed`.

The deterministic public smoke still reproduces run
`run_d541618c4fd3b2a1dca0e963`, manifest
`d541618c4fd3b2a1dca0e963514841741086c912634dd13ade4c434194409e8f`, twin run
`twinrun_eb785c4006fe30dd70a2db9b`, resolved evidence hash
`8fed5429efecb7a0b5055a15928b8edf48e5713454ba18b42c9503305778d1b7`, and
cluster registry hash
`875b04d5ee32465545262a30ab2cee300eb2c34407f1bcccf6f4012128ad6a79`.
The canonical guide follows its redirect and includes v0.32 inputs, outputs,
limits, Sigma Gravity, inverse-response discovery, and the remaining database
approval. A real field-job POST still returns HTTP 503 with
`production_worker_not_connected`.

Production research jobs are not enabled yet. An authorized Horizon3 user must
first accept Neon's Marketplace terms at
<https://vercel.com/horizon3/~/integrations/accept-terms/neon?source=cli>, after
which the free `iad1` database can be provisioned and migrated. A stateless
scientific container must then be deployed and connected through the server-only
worker URL/token pair. The system deliberately fails closed until both are
verified.

## Worker milestone after Vercel

The container and authenticated gateway connector pass a real Docker
acceptance, and the durable private object store, queue delivery, PostgreSQL
schema/outbox, and stateless handoff/finalization contracts now exist. The next
deployable milestone is to provision and migrate Neon, run that image on a
container host without relying on its local filesystem for durable state,
configure the server-only worker URL/token pair, and complete a production job
through queue, database lease, worker, verified artifact finalization, and
restart recovery.

That remains a bounded infrastructure milestone, not final scientific-platform
readiness. The v0.33 release below builds and tests researcher authentication,
project isolation, transactional cancellation authorization, quotas, and audit
logs. Production still needs their hosted database activation plus cache and
license policy, signed result manifests, monitoring, backups, cost controls,
and an image publication/promotion policy. Only after those controls are
verified should the same boundary admit inverse-response, observation,
composed-batch, and signed advanced plug-in workloads.

## v0.33 project-scoped research API release

The v0.33 release publishes a formula-neutral durable lifecycle for confirmed
models, immutable uploads, jobs, events, artifacts, and cancellation. Project
bearer tokens are stored only as SHA-256 hashes. The second PostgreSQL migration
adds credentials, immutable audit events, and project quotas; state changes and
their audit entries commit in the same transaction. The public API deliberately
separates model/data registration from worker execution and keeps submission
disabled until the recurring dispatcher and stateless worker are verified.

The 152-test hosted suite passes, including repeatable eleven-table migrations,
credential isolation, exact model-receipt verification, immutable byte rehashes,
idempotency conflicts, active-job/upload/attempt quotas, transactional rollback,
cancellation, events, and project-scoped artifact download. GitHub Actions run
<https://github.com/lrspeiser/sigmagravity/actions/runs/30801012875> passed the
real non-root Linux worker and field/galaxy container acceptance for commit
`a8c4ef2b72d1ce2f72c541b1033b22e95c74c475`.

Production deployment `dpl_B2jrYAiTfZxGdyWzYXnB4dtQCG5f` serves
`0.33.0-preview` at the stable alias. Its immutable URL is
<https://sigma-gravity-research-simulator-n8vt853cb-horizon3.vercel.app>.
The live guide contains the production job input, current fail-closed output,
expected connected output, capability boundaries, and scientific limitations.
The OpenAPI document publishes project bearer authentication and the generic
job route.

Two independent live queue smokes verified deployment identity hash
`b55efdabcc436d6aee6981688284f075166bb3efa82af69f80763e4204cd28d6`
and private acknowledgement hash
`9b7e503b64a9328217a2da2eaf47a515f2f8c036d600a7cff26f009c1f408d48`.
Storage readiness reports `durable_storage_and_queue_connected`; PostgreSQL,
the recurring outbox scheduler, and the stateless scientific container remain
`not_configured`, so `productionExecution` remains `not_ready` and a job POST
returns HTTP 503 `production_control_plane_not_connected`.

## v0.34 signed advanced plug-in sandbox release

The v0.34 release adds a cryptographically identified advanced-code package
and a second, deliberately separate execution boundary. Public Vercel preflight
validates a domain-separated Ed25519 signature but neither trusts the publisher
nor reads package bytes nor executes code. The host launcher requires an active
operator trust record, rehashes the entire package, rejects undeclared files and
links, and requires a digest-pinned production image. Advanced production jobs
remain fail-closed until a package registry and dedicated sandbox host exist.

The 157-test hosted suite passes. GitHub Actions run
<https://github.com/lrspeiser/sigmagravity/actions/runs/30802568501> built the
new Python 3.13.7 sandbox and passed its real Linux isolation acceptance for
commit `7e3bdd8e65a7b8bdbdede9a78cbc4bbe1838184c`. The external fixed-MOND
fixture matched the safe AST on two runs and reported non-root identity, zero
capabilities, no new privileges, blocked network and writes, no credentials or
Docker socket, fresh temporary state, and pre-container rejection of a changed
signed source byte. Its package hash was
`642021f60f929c7e6fda5a2c9ee593a7a4ada42923b0c53b8b668519cde3a301`.

Production deployment `dpl_ACjmcy4E9dQQTh6JoJTHkkcbKvDi` is ready at
<https://sigma-gravity-research-simulator-6dvfme76k-horizon3.vercel.app> and
aliased to the stable site. Health and OpenAPI report `0.34.0-preview`; the live
guide publishes the input/output distinction and threat boundary. A fresh-key
HTTP smoke verified that signature validity returns without publisher trust,
package-byte verification, or Vercel execution.

Two queue smokes verified deployment identity hash
`ba7c8d5d3229af944925fc288ff33b949a40844494d9c375b2409eb5a63966f4`
and private acknowledgement hash
`85a5b7b031412c1712ed176767d3d0d538c686fd5ff38b33992a19f8b2de269e`.
The database, outbox scheduler, safe scientific worker, trusted plug-in
registry, and sandbox host remain unconnected; no hosted heavy or uploaded-code
execution is claimed.
