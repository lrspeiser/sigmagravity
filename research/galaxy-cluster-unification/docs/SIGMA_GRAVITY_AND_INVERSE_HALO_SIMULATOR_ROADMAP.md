# Roadmap: a simulator for Sigma Gravity and inverse halo-response discovery

Status: active implementation plan, updated 2026-08-02 for the hosted v0.18
inverse-response workbench milestone.

## Outcome

The current simulator is a credible research preview, not yet a public theory
discovery service. It can run bounded radial formulas, expose frozen resolved
evidence, validate and confirm exact 2D/3D field manifests, and execute a broad
reference workflow locally. The next useful product is not another cosmetic
refinement. It is a public, asynchronous system that can:

1. represent a Sigma Gravity candidate without theory-specific application
   code;
2. run that frozen model on registered galaxy and cluster data;
3. use lens-model halo maps to discover candidate response structure while
   clearly labeling the inverse as non-unique;
4. remove the halo target and predict raw held-out observations from baryons;
5. compare the prediction with baryons-only GR, fixed MOND/RAR, and disclosed
   dark-matter baselines; and
6. return one deterministic report with uncertainty, parameter counts,
   provenance, numerical diagnostics, and pass/fail gates.

The defining boundary is simple: **an inverse reconstruction is a hypothesis
generator; a frozen forward prediction on unseen raw data is the test.**

## What the existing work already tells us

- The live radial route evaluates a safe algebraic acceleration formula on all
  175 packaged SPARC systems. One universal parameter and one parameter per
  galaxy are counted separately.
- The local worker already supports formula-neutral stationary elliptic field
  solves, content-addressed arrays, generated galaxy jobs, decoupled
  observation evaluation, typed photon maps, raw multiple-image roots, and
  composed batches.
- Eight resolved THINGS systems and 146,532 velocity pixels demonstrate that
  generator fidelity, formula transport, and formula accuracy can be measured
  independently. That protocol passed; fixed simple MOND was competitive but
  incomplete, and Newtonian baryons failed the final holdout criterion.
- In ten RELICS clusters, the baryon-to-lensing-excess inverse maps were stable
  between Lenstool and GLAFIC (median path-density correlation 0.969), but real
  angular layouts rarely beat radius-preserving nulls. Much of the apparent
  routing structure can therefore be radial morphology.
- The P0568 baryon-only tensor forward model improved the locked cluster-map
  score by 8.01% over local light, missing its frozen 10% gate. Its amplitude
  transferred badly to SPARC, remaining 5.5--6.3 times worse than fixed RAR.
- The largest cluster-map driver was effective baryonic extent, not tensor
  orientation. A 75--125 kpc scale may be compensating for omitted X-ray gas,
  intracluster light, or BCG structure rather than revealing a propagation
  length.

These results argue against spending more time on member-galaxy-light-only
formula variations. The next experiment needs complete measured baryons and
raw lensing targets.

## The data distinction the platform must enforce

### Observations

These are the data a forward theory should ultimately predict:

- strong-lensing image positions, family assignments, redshifts, parities,
  time delays, and uncertainties;
- weak-lensing ellipticities or calibrated shear catalogs;
- magnification constraints where available;
- galaxy H I or stellar velocity fields, rotation curves, and dispersions;
- Solar-System and laboratory constraints used by the declared weak/strong
  field limit.

### Baryonic inputs

These may be used to construct a prediction if their uncertainty is propagated:

- stellar-mass or multi-band light maps;
- H I and molecular-gas maps;
- cluster X-ray gas and SZ pressure/density products;
- BCG and intracluster-light maps;
- distance, inclination, WCS, PSF/beam, masks, selection functions, and
  mass-to-light priors.

### Model-derived discovery targets

Dark-matter halo centers, convergence components, or posterior mass maps are
not direct pictures of dark matter. They are inferences made under a lens
model, priors, geometry, and baryonic assumptions. They are useful as
development targets for asking, "what response would be required here?" They
must not be supplied to a held-out forward prediction that claims to explain
them.

Every stored product should therefore declare one of:

- `raw_observation`;
- `baryonic_input`;
- `nuisance_or_calibration`;
- `model_derived_discovery_target`; or
- `withheld_score_target`.

The job planner must reject data leakage across those roles.

## Build 1: connect durable production computation

Priority: **P0; this is the shortest path to making the existing local system
useful to another researcher.**

Build:

- a queue and scheduler outside Vercel request execution;
- network-disabled, single-use Python worker containers;
- Postgres metadata for projects, models, jobs, events, and permissions;
- S3/R2-compatible immutable array and report storage;
- content hashing at upload, worker input, artifact creation, and download;
- authentication, project isolation, quotas, cancellation, retry, timeout,
  caching, and audit logs;
- signed result manifests and exact worker image/version identities; and
- license and redistribution enforcement at dataset and artifact level.

Acceptance evidence:

- the public field-job route returns `202`, survives a gateway restart, and
  produces the same artifact hashes as the local reference run;
- cancellation, timeout, quota, corrupt-upload, worker-crash, and artifact
  rehash tests pass;
- changing only observation uncertainty reuses the field solve while creating
  a new observation-evaluation job; and
- a malicious uploaded plug-in has no network, host, secret, or cross-project
  access.

## Build 2: create a resolved galaxy-and-cluster registry

Priority: **P0 scientific dependency.**

Each galaxy package needs:

- stellar surface-brightness or stellar-mass maps;
- H I and, where available, molecular gas;
- a 2D velocity field or spectral cube;
- distance, inclination, position angle, WCS, PSF/beam, channel width, masks,
  uncertainties, and covariance;
- bulge, disk, bar, arm, warp, clump, and scale-height metadata or priors; and
- citation, license, transformation history, and content hashes.

Each cluster package needs:

- member-galaxy stellar-mass maps, not just positions or counts;
- BCG and intracluster light;
- registered X-ray gas density and temperature plus SZ constraints;
- strong-lensing multiple-image catalogs with per-image uncertainty;
- weak-lensing shear catalogs and magnification where available;
- competing lens-model posterior samples labeled as derived discovery targets;
- redshift and distance-ratio products; and
- consistent coordinates, masks, footprints, selection functions, and
  provenance.

Acceptance evidence:

- every scored pixel or image constraint traces to a public source and hash;
- coordinate round trips remain within a declared fraction of a pixel;
- integrated baryonic mass agrees across registered representations within
  propagated uncertainty;
- at least two independent lens reconstructions can be compared without
  overwriting their method identity; and
- withheld raw observations cannot be fetched by an inference/generator job.

The first cluster release should prefer a small number of systems with complete
baryons and defensible raw lensing errors over a large member-light-only sample.

## Build 3: finish uncertainty-aware resolved galaxy twins

Priority: **P1.**

The inverse parameter extractor must infer only baryonic structure and
observation conditions. It must not see rotation speed targets or any gravity
parameter. Its output should be a posterior ensemble over:

- stellar and gas scale heights;
- inclination, warp, and lopsided geometry;
- bulge shape and disk/bar/arm components;
- gas flaring, turbulence, clumps, and missing/obscured baryons;
- mass-to-light nuisance parameters; and
- PSF, beam, sampling, mask, and noise conditions.

The forward generator must turn a parameter draw and seed into intrinsic 3D
baryonic density and a simulated observation. Gravity is then a separate,
replaceable stage.

Acceptance evidence:

- real observations -> inferred ensemble -> regenerated observations recover
  withheld light, gas, and morphology within measurement uncertainty;
- the result reports radial profiles, 2D residuals, Fourier modes,
  concentration, asymmetry, clumpiness, thickness, and mass conservation;
- the parameter package is byte-invariant when velocity targets are perturbed;
- every gravity model receives exactly the same source ensemble; and
- poor deprojection identifiability appears as a broad prediction interval,
  not one confident fake galaxy.

This build answers the user's practical visual question: a fake galaxy with
the same inferred configuration should show both the observed speed field and
the selected formula's predicted field, with uncertainty and residual maps.

## Build 4: express Sigma Gravity as a complete model, not a label

Priority: **P1.**

The manifest must state:

- dynamical fields and whether each is scalar, vector, tensor, or nonlocal;
- the action or stationary weak-field equations actually being tested;
- baryonic sources and any permitted environmental variables;
- matter and photon couplings;
- parameters, dimensions, priors/bounds, and universal/per-object policy;
- geometry, boundary conditions, screening, and Solar-System limit;
- requested observables and numerical tolerances; and
- the domain where the approximation is claimed to apply.

The safe field language now executes one fixed-semantics nonlocal convolution
fixture in the local 2D/3D worker. It still needs robust support for flexible
nonlocal kernels, coupled potentials, path projections, smoothly gated
constitutive laws, and vector/tensor contractions. A signed plug-in route is
needed only when a model cannot be represented declaratively.

Acceptance evidence:

- Newtonian Poisson, algebraic MOND/RAR, AQUAL, QUMOND, Refracted Gravity, one
  nonlocal convolution, one two-potential photon/matter theory, and Sigma
  Gravity execute without theory-name branches in the application;
- manufactured known-answer and resolution-convergence tests pass;
- units, boundaries, singular limits, and parameter roles are machine-audited;
- any equation or solver-control change invalidates confirmation and caches;
  and
- the same physical constants are used for galaxy dynamics, cluster lensing,
  and the Solar-System gate unless the report explicitly labels nuisance
  parameters.

## Build 5: add an inverse halo-response workbench

Priority: **P1 discovery tool, not the final test.**

Implementation status: **local v1 complete; real-data scientific acceptance
incomplete.** The reference service now accepts content-hashed Cartesian 2D or
3D baryonic, model-derived target, uncertainty, and mask arrays; fits one
stationary compact kernel and one amplitude across all submitted systems; and
emits uncertainty ensembles, a radial-angle null, rank/nullity,
regularization-sensitivity, compatible-kernel, deterministic report, hash, and
reproduction artifacts. It rejects raw observations as an inverse target and
counts every kernel cell as a fitted discovery coefficient. Synthetic
injection, null, 3D, degeneracy, determinism, artifact-integrity, and empirical
interval-coverage gates pass. Remaining acceptance work requires complete real
baryons, multiple independent lens-model posterior ensembles, the additional
null families below, and a frozen raw-observation holdout.

For a cluster development set, allow a researcher to supply:

- a baryonic 2D/3D ensemble;
- one or more model-derived effective-mass/convergence posterior ensembles;
- a declared conservation/amplification hypothesis; and
- a candidate response family such as a normalized transport kernel,
  divergence-free return field, anisotropic constitutive tensor, or nonlocal
  Green function.

The inverse engine should return:

- posterior source-to-response couplings or kernel coefficients;
- route-length, direction, density, tidal, gas, and concentration summaries;
- sensitivity to baryonic catalog completeness and lens-model family;
- radius-preserving, angle-shuffled, phase-scrambled, central-halo, local-light,
  and missing-baryon nulls;
- identifiability diagnostics showing which materially different kernels fit;
- a conservation ledger and predicted compensating deficits for conservative
  redirection; and
- candidate compact forward laws, with complexity and parameter counts.

It must never return "the path gravity took." It should return "families of
responses compatible with these assumptions and this derived target."

Acceptance evidence:

- synthetic injected kernels are recovered with calibrated uncertainty;
- null data do not produce confident routes;
- posterior coverage is measured across baryonic and lens-map realizations;
- results remain stable across at least two independent lens-model methods; and
- the workbench can reveal non-identifiability rather than force one answer.

The local v1 satisfies the synthetic recovery, simple-null, calibrated-noise,
and explicit non-identifiability portions. It does not yet satisfy stability
across independent real lens models. Target permutation, phase scrambling,
central-halo, local-light, missing-baryon, conservation, and multi-method
posterior controls remain open rather than being inferred from the one
radial-angle control.

## Build 6: freeze a baryon-only forward law and remove the halo target

Priority: **P0 scientific decision point after Builds 1--5.**

Use development systems to compress inverse patterns into a small law
`K_theta(y | x, E_b)` or another declared field closure. Then:

1. freeze the formula, universal constants, screens, numerical settings,
   exclusions, and score thresholds;
2. hide all target halo maps for validation and holdout clusters;
3. calculate the field from measured baryons and permitted environmental data;
4. ray-trace to raw multiple-image roots and predict shear/magnification;
5. predict galaxy velocity fields with the same physical settings;
6. run Solar-System and numerical-stability gates; and
7. reveal the observations once for the registered holdout report.

Acceptance evidence must keep separate:

- image-plane positional error and missing/extra-image topology failures;
- reduced shear, magnification, time-delay, convergence-shape, and dynamics
  channels;
- source reconstruction and observation-systematic nuisance parameters;
- per-system and aggregate scores;
- comparator parameter counts; and
- numerical failure versus empirical failure.

A cluster result that is only close to a derived halo map is insufficient. A
formula must predict the raw measurements that caused researchers to infer the
halo.

## Build 7: publish a deterministic joint report and external API

Priority: **P1 public usefulness.**

Every batch should emit:

- `manifest.json` and exact confirmed `model.json`;
- input and artifact hashes;
- `per_system.csv`, `aggregate_scores.json`, and `failures.csv`;
- predicted and residual maps/curves with uncertainty bands;
- solver residuals, conservation, convergence, and resolution sensitivity;
- parameter accounting and comparator assumptions;
- train/validation/holdout disclosure;
- HTML and printable PDF reports;
- a reproduction command or notebook; and
- `llm_briefing.md` containing only deterministic facts.

The API should allow researchers to register a model, select or upload systems,
declare a parameter policy, submit an asynchronous batch, monitor events, and
download signed artifacts. An LLM may explain the completed report, but must
not reinterpret equations, fit parameters, choose exclusions, compute scores,
or decide pass/fail.

## Joint scientific scorecard

The platform should report, not conceal, these distinct comparisons:

| Question | Minimum comparator | Required disclosure |
|---|---|---|
| Does the model beat baryons-only gravity? | Newtonian/GR baryons | Same baryonic maps and observation operator |
| Is it competitive with a universal empirical law? | Fixed MOND/RAR | Fixed constants and declared formulation |
| Is it competitive with dark-matter modeling? | Published halo baseline | Halo parameter count, priors, and data reused in fitting |
| Does one setting transfer? | Frozen cross-domain model | Universal, hierarchical, nuisance, and per-object counts |
| Is the apparent routing real? | Radial/angle and missing-baryon nulls | Lens-model and baryonic posterior sensitivity |
| Is the field physically viable? | Solar/PPN, wave speed, stability | Domain and approximation used |

"Beating dark matter" should not be reduced to one residual number. A
universal model may be scientifically interesting while scoring somewhat worse
than a highly flexible halo fit, but the numerical gap, parameter-count gap,
and failed systems must all remain visible.

## Stop rules: avoid building for months by incremental tweaks

Do not move to a new UI refinement or formula family unless it closes one of
the acceptance gaps above. Use these three release decisions:

### Release A: public executor

Stop and reassess if the existing local reference jobs cannot be made durable,
isolated, reproducible, and hash-identical in production. Without this, the
site remains documentation plus a radial demo.

### Release B: identifiable data foundation

Stop member-light-only routing work if complete baryonic maps erase the
preferred 75--125 kpc scale or if independent lens reconstructions and nulls
show no stable nonlocal signal. That is a useful negative result, not a reason
to add parameters.

### Release C: blind joint prediction

After at most three materially different compact forward closures, reassess the
premise if all fail for the same reason on held-out raw lensing or galaxy
dynamics. Do not rescue them with per-object force parameters, test-object halo
targets, class labels, or a lensing-only normalization.

## Recommended implementation order

1. Production queue, workers, metadata, and artifact storage.
2. One small complete-baryon cluster release plus two independent lens-model
   ensembles and raw strong-lensing constraints.
3. A public end-to-end Newtonian and fixed-MOND field batch as infrastructure
   validation.
4. The inverse halo-response workbench with synthetic recovery and null tests.
5. One frozen candidate forward kernel tested on unused raw cluster lensing.
6. Joint galaxy, cluster, Solar-System, parameter-count, and sensitivity report.
7. Broader data catalog and external signed plug-ins only after that end-to-end
   path works.

That sequence answers the practical product question quickly: if step 3 works,
the platform is already useful to other researchers; if steps 4--6 also work,
it becomes useful for discovering and falsifying Sigma Gravity-like field
ideas.
