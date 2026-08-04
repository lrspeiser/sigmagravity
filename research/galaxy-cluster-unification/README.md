# Void Screening Lab

An independent, falsification-first project for testing whether a smoothly
activated outer-galaxy acceleration can predict SPARC rotation curves, and
whether any surviving amplitude is independently associated with cosmic void
environment.

This is deliberately isolated from SigmaGravity's main theory and application
code even though it now lives under `research/galaxy-cluster-unification` in the
same repository. SigmaGravity is used only as a read-only source for a copied
SPARC snapshot; this subproject has its own package, configurations, data,
tests, and results. Every imported SPARC file is recorded with its source path,
source Git commit, size, and SHA-256 hash in
`data/raw/sparc/provenance.json`. See
[`docs/SIGMAGRAVITY_PLACEMENT.md`](docs/SIGMAGRAVITY_PLACEMENT.md) for the
placement and origin record.

The current theory-only result is
[`docs/SIGMA_V11C_BIOT_STRETCH_FALSIFICATION.md`](docs/SIGMA_V11C_BIOT_STRETCH_FALSIFICATION.md).
Exact v11C is retired before data. Replacing Green strain by Biot stretch
repairs v11B's one-dimensional negative quartic, but a finite
orientation-preserving anisotropic stretch makes a mixed-shear stiffness
larger than the tilted time-kinetic budget. The physical material Hessian
crosses zero while `Q=0` material flow remains timelike. V11A, v11B, and v11C
are three distinct post-reset failures at nonlinear kinetic rank, so the
material-memory mechanism is reset rather than patched with v11D.

The preceding theory-only result is
[`docs/SIGMA_V11B_TILTED_RANK_FALSIFICATION.md`](docs/SIGMA_V11B_TILTED_RANK_FALSIFICATION.md).
Exact v11B is retired before data. On a finite tilted slice its strain square
becomes quartic in a physical material velocity; the Legendre Hessian crosses
zero and negative while material flow remains timelike. V11A and v11B are now
two distinct post-reset failures at the nonlinear kinetic-rank gate.
The preceding theory-only selection is
[`docs/SIGMA_V11B_ELASTIC_TRIAD_SELECTION.md`](docs/SIGMA_V11B_ELASTIC_TRIAD_SELECTION.md).
V11B treats the Sigma sector as a stress-free elastic spacetime triad. Three
connection-free scalar material coordinates have zero action and stress in the
unstrained vacuum, while carrying two shear phonons at squared speed `3/11`
and one longitudinal phonon at `3/4`. The metric TT front remains
Einstein-Hilbert, and one new rigidity length retains the five-constant cap.
This advances only to nonlinear tilted-rank and full metric-constraint gates;
no data are authorized.
The preceding theory-only result is
[`docs/SIGMA_V11A_TILTED_RANK_FALSIFICATION.md`](docs/SIGMA_V11A_TILTED_RANK_FALSIFICATION.md).
Exact v11A is retired before data. On a finite tilted-aether background, its
bounded alignment is a concave function of the AeST scalar coordinate
velocity. An allowed finite memory gradient drives the scalar velocity
Hessian through zero and negative; increasing the positive base kinetic term
only moves this finite surface. The next candidate may not put a dynamical
alignment gradient inside another field's kinetic coefficient without an
exact global degeneracy identity.
The preceding theory-only selection is
[`docs/SIGMA_V11A_ANISOTROPIC_SCALAR_MEMORY_SELECTION.md`](docs/SIGMA_V11A_ANISOTROPIC_SCALAR_MEMORY_SELECTION.md).
After the required v10 mechanism reset, v11A replaces the rank-two carrier
with one bounded anisotropic scalar memory. Its fixed-background spatial
stiffness stays between `9/44` and `3/11`, the static Schur margin is at least
`1/44`, and all mixed roots remain positive and no greater than one. Because a
scalar derivative has no metric connection, the aether-rest TT metric symbol
does not inherit v10D's automatic cone widening. This is selection only:
nonlinear rank, tilted cones, weak metric/lensing equations, PPN/Solar limits,
and data remain unresolved.
The preceding theory-only result is
[`docs/SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md`](docs/SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md).
Exact v10D is retired before data. On the exact axisymmetric spin-two sector,
its nonzero carrier background gives
`c_TT^2=1+c_P^2(p_parallel-p_perp)^2`, so every anisotropy widens the physical
metric cone. V10B, v10C, and v10D are three materially distinct closures of
the same aether-tidal carrier family that fail the mathematical
causality/stability gate; the family is reset rather than patched with v10E.
No observational holdout was opened.
The preceding theory-only subgate is
[`docs/SIGMA_V10D_ADM_RANK.md`](docs/SIGMA_V10D_ADM_RANK.md).
In an aether-rest ADM frame the carrier derivative is the perfect square
`W=dot(P)-K.P-P.K`. The triangular velocity map `(dot(h),dot(P))` to
`(dot(h),W)` has determinant one for every `P`, so the combined Legendre
inertia remains one inherited DeWitt negative direction and fifteen positive
directions, with no zero. The generic count is the published six AeST modes
plus six carrier modes. That local rank result remains correct but was not
sufficient to guarantee a causal spatial characteristic.
The preceding source-block result is
[`docs/SIGMA_V10D_ANISOTROPIC_CHARACTERISTICS.md`](docs/SIGMA_V10D_ANISOTROPIC_CHARACTERISTICS.md).
For arbitrary carrier orientation, the completed aether kinetic matrix obeys
`F>=I`, while a wave direction enters through
`R=(I+n n^T)/2` with `I/2<=R<=I`. These bounds keep the full fixed-metric
source-block static Schur complement at least `I/3` and all mixed roots inside
the metric cone, even when `F` and `R` do not commute. Nonzero background
`J` changes only lower-order terms in this block. The dynamical metric, AeST
scalar and full ADM constraints remain the next gate.
The preceding selection is
[`docs/SIGMA_V10D_EXPONENTIAL_KINETIC_SELECTION.md`](docs/SIGMA_V10D_EXPONENTIAL_KINETIC_SELECTION.md).
V10D replaces v10C's unsafe vector kinetic factor with the fixed covariant
matrix `K_B[exp(X)-X]`, `X=(beta/K_B)P`. Since `exp(x)-x` has global minimum
one for every real eigenvalue, the finite-amplitude ghost is removed without a
new constant or state cutoff. Zero-background response and cones are unchanged;
amplitude scans keep the local static and hyperbolic blocks positive and causal.
This is selection only: full tilted ADM constraints, nonzero-`J`
characteristics, PPN/Solar limits and numerics remain mandatory.
The preceding falsification is
[`docs/SIGMA_V10C_NONLINEAR_KINETIC_FALSIFICATION.md`](docs/SIGMA_V10C_NONLINEAR_KINETIC_FALSIFICATION.md).
Exact v10C is retired before data. Enforcing that `P` stays spatial relative to
the moving aether changes the physical vector kinetic matrix to
`K_B I-beta P`. At the finite isotropic amplitude
`P=sqrt(11 K_B/2) I` its Legendre map becomes singular; above it the three
AeST vector directions are ghosts. The quartic potential is finite there and
the spatiality constraint does not bound amplitude. A successor must use a
materially different globally positive coupling, not an object-selected
carrier cutoff.
The preceding variation result is
[`docs/SIGMA_V10C_COVARIANT_VARIATION.md`](docs/SIGMA_V10C_COVARIANT_VARIATION.md).
The exact projected carrier momentum passes a tilted-aether finite-difference
check, the spatiality constraint has rank four and leaves six carrier
components, the source has a first-derivative boundary form, and the all-field
Noether identity closes on-shell conservation. Every Euler equation is at most
second order. This advances v10C only to its nonlinear ADM constraint gate; it
does not establish the physical degree count, arbitrary-background cones, or
Solar viability.
The preceding applicability result is
[`docs/SIGMA_V10C_COVARIANT_PPN_PRECHECK.md`](docs/SIGMA_V10C_COVARIANT_PPN_PRECHECK.md).
The spatial-aether counterterm has the exact unit-aether map
`(c1,c2,c3,c4)=(K_B u,0,-K_B u,K_B(1-u))`, preserving `c13=0` and
`c14=K_B`. A pure Einstein-aether proxy leaves `alpha1=-4 K_B` unchanged
from the original Maxwell base, while its `alpha2` formula is inapplicable at
`c123=0` because it omits the AeST scalar and the new `P` carrier. V10C is
therefore not independently retired, but this is explicitly not a PPN pass:
the complete moving-source AeST-plus-`P` derivation remains a mandatory gate.
The preceding selection is
[`docs/SIGMA_V10C_HYPERBOLIC_AETHER_TIDAL_SELECTION.md`](docs/SIGMA_V10C_HYPERBOLIC_AETHER_TIDAL_SELECTION.md).
V10C gives the surviving aether-tidal trace/STF tensor a hyperbolic time
kinetic term. Requiring both the v10B threefold static capacity and a luminal
upper cone at the frozen AeST scalar speed derives `c_P^2=3/11` and
`beta^2/K_B=2/11`; no sixth constant is added. The worst static determinant is
`1/11`. Longitudinal squared speeds are `9/44` and `1`, transverse speeds are
`0.232009/0.881627`, unmixed carrier modes have `3/11`, and flat TT remains
luminal. This is a selection pass only: full covariant variation, nonlinear
ADM constraints, arbitrary-background cones, PPN, Solar screening, cosmology,
and numerics precede data.
The preceding auxiliary implementation is
[`docs/SIGMA_V10B_AUXILIARY_AETHER_TIDAL_FALSIFICATION.md`](docs/SIGMA_V10B_AUXILIARY_AETHER_TIDAL_FALSIFICATION.md).
V10B moves the trace/STF tensor source from the vanishing-stiffness MOND scalar
to the constant-stiffness aether acceleration and removes the tensor's time
kinetic term. Its worst static eigenvalue is `0.183503`, its Schur complement
is exactly `1/3`, its longitudinal capacity is `3`, and six primary plus six
secondary second-class constraints remove all six carrier states with a
positive reduced Hamiltonian. It is still retired before data: eliminating a
finite-range second-class carrier leaves a nonzero equal-time Yukawa tail in
the physical transverse aether mode. The successor must be a hyperbolic causal
completion of this positive aether-tidal block.
The preceding constant-mixing result is
[`docs/SIGMA_V10A_SPATIAL_POLARIZATION_FALSIFICATION.md`](docs/SIGMA_V10A_SPATIAL_POLARIZATION_FALSIFICATION.md).
V10A is the first post-v9B carrier to retain both trace and shear orientation.
Its necessary flat block is positive and causal, its convex fixed-source state
is unique, it distinguishes equal-acceleration systems through `M/r^3`, and its
normalized response capacity is `4`. The exact constant derivative coupling is
nevertheless retired before data: the simple-mu quasistatic scalar stiffness
vanishes in deep fields, so every nonzero constant mixing produces a negative
high-k gradient eigenvalue. At the selected row, transverse fields require
`x>1.285714` and longitudinal fields require `x>0.511858`; the zero-field
eigenvalue is `-0.270285`. A successor must preserve the tensor geometry in a
manifestly positive or degenerate complete principal form.
The preceding mechanism closure is
[`docs/SIGMA_V9B_LOCAL_FIRST_GRADIENT_CLOSURE.md`](docs/SIGMA_V9B_LOCAL_FIRST_GRADIENT_CLOSURE.md).
It proves that every regular, unique, static local completion `F(Y,Z,U)` reduces
to one universal acceleration relation in spherical symmetry. In the spent
development products, all 72 cluster points lie inside the SPARC outer
acceleration range and match a galaxy point within `0.00145 dex` at the median,
yet require `0.509 dex` (factor `3.23`) more enhancement. The local first-
gradient lane is closed; another angle or saturation function cannot supply the
missing state. The successor must use a uniquely baryon-forced finite-
environment/tidal variable with both monopole and shear response.
The exact v9A interaction failure is retained in
[`docs/SIGMA_V9A_BOUNDED_ALIGNMENT_FALSIFICATION.md`](docs/SIGMA_V9A_BOUNDED_ALIGNMENT_FALSIFICATION.md).
The preceding higher-derivative v8B failure remains in
[`docs/SIGMA_V8B_GLOBAL_RANK_FALSIFICATION.md`](docs/SIGMA_V8B_GLOBAL_RANK_FALSIFICATION.md).

The latest formula-development status is
[`docs/P0715_P0718_LENSING_STRUCTURE_AND_TRANSFER_RESULTS.md`](docs/P0715_P0718_LENSING_STRUCTURE_AND_TRANSFER_RESULTS.md).
The coordinate-safe lens engine now has analytic and archived-map conformance
tests, and the candidate's one-root behavior is numerically robust. At the real
arcs it lacks both convergence and shear. A Solar-screened AQUAL contrast and a
new nonlinear-before-summation member law improve root completeness, reaching
`0.851` on PLCKG287, but fail cross-cluster image-position and topology gates;
no formula advances. The preceding untouched external verdict remains in
[`docs/P0710_P0714_EXTERNAL_VALIDATION_RESULTS.md`](docs/P0710_P0714_EXTERNAL_VALIDATION_RESULTS.md):
the candidate beats the frozen full-field MOND comparators on 13 new dwarf
rotation curves, while the four-cluster test is not validly evaluable because
only two selected targets meet the frozen catalog-readiness rules. The previous review entry is
[`latest-findings/2026-08-02-p0695b-cubic-path-audit`](latest-findings/2026-08-02-p0695b-cubic-path-audit/README.md).

The first action cycle under the renewed Sigma goal is now in
[`docs/SIGMA_V1_NONMETRICITY_ACTION_RESULTS.md`](docs/SIGMA_V1_NONMETRICITY_ACTION_RESULTS.md).
It derives the standard-mu galaxy limit from a one-parameter covariant
nonmetricity action, but proves that the regular isolated branch has zero
metric slip and is exactly AQUAL.  It passes the external dwarf-galaxy gate and
fails the inherited raw cluster topology gate, so it is retired without
fitting.  The next action must contain a baryon-predictable anisotropic-stress
state rather than another scalar interpolation.
The post-v4 action selection is in
[`docs/SIGMA_V5_ORIENTATION_TRANSPORT_POSTULATES_AND_ACTION_SELECTION.md`](docs/SIGMA_V5_ORIENTATION_TRANSPORT_POSTULATES_AND_ACTION_SELECTION.md).
It derives a nonmetricity invariant that acts directly on the photon Weyl
potential and leaves the vacuum linear TT trace null, while also proving that
a plain nonminimal scalar-curvature factor cancels from linear lensing. The
candidate anisotropic trace coupling remains an envelope, not a theory: its
causal no-free-state action, complete variation, constraint count, and health
proof must pass before another map score.
The first local completion is documented in
[`docs/SIGMA_V5A_CAUSAL_POLARIZATION_ACTION_AUDIT.md`](docs/SIGMA_V5A_CAUSAL_POLARIZATION_ACTION_AUDIT.md).
It introduces one massive gravitational polarization mode with a transition-
band source and a bounded orientation-dependent kinetic cone. All current
theory-only algebraic, local-mode, causality, Solar-source, and parameter gates
pass, but the full nonmetricity mode count, cosmological tensor speed, weak
metric equations, and PPN limits remain unresolved; observational fitting is
still prohibited.
Its complete leading static variation is now in
[`docs/SIGMA_V5A_WEAK_FIELD_DERIVATION.md`](docs/SIGMA_V5A_WEAK_FIELD_DERIVATION.md).
The metric, flat-connection, and polarization Euler equations are stated in
compact exact form, and the coupled weak equations for `Psi`, `Phi`, and the
polarization include every transition-source and anisotropic-chain term. All
manufactured variation checks pass; background mode, cosmological `c_T`, and
PPN proofs still precede any data fit.
The subsequent cosmological-domain audit retires the exact v5A base in
[`docs/SIGMA_V5A_COSMOLOGICAL_BRANCH_RESULTS.md`](docs/SIGMA_V5A_COSMOLOGICAL_BRANCH_RESULTS.md):
the inherited Sigma-v2 primitive is not real and differentiable around FLRW.
The causal polarization survives unchanged in the cleaner
[`docs/SIGMA_V5B_STEGR_POLARIZATION_ACTION.md`](docs/SIGMA_V5B_STEGR_POLARIZATION_ACTION.md),
which starts from STEGR/GR. Its homogeneous background is GR, its TT cone is
luminal, and both galaxy and cluster departures must come from the same four-
constant polarization field. Hamiltonian, PPN, cosmological-stability, and
prior-art gates still prohibit fitting.
The nonlinear constraint audit then retires exact v5B in
[`docs/SIGMA_V5B_NONLINEAR_DEGENERACY_RESULTS.md`](docs/SIGMA_V5B_NONLINEAR_DEGENERACY_RESULTS.md).
Although its `sigma=0` FLRW branch and free scalar are healthy, the transition
source and orientation transport each make the lapse/flat-connection kinetic
matrix full rank on generic polarized static backgrounds. The lost STEGR
degeneracy exposes an additional sign-changing kinetic direction. No data were
opened; a successor must prove degeneracy before source phenomenology.
That successor action envelope is selected in
[`docs/SIGMA_V5C_DEGENERACY_FIRST_ACTION_SELECTION.md`](docs/SIGMA_V5C_DEGENERACY_FIRST_ACTION_SELECTION.md).
It is a four-constant, one-metric member of the established luminal Class-Ia
DHOST family. Its tensor speed and constraint degeneracy are algebraic
identities, and one fixed signed-safe scalar-Hessian coefficient can carry
baryon-derived orientation. The class is prior art and the row is not yet a
theory: full equations, cosmological/scalar health, PPN, branch uniqueness, and
prior-art comparison still prohibit observational fitting.
Its first physical feasibility gate is now complete in
[`docs/SIGMA_V5C_EXTERIOR_LAW_RESULTS.md`](docs/SIGMA_V5C_EXTERIOR_LAW_RESULTS.md),
and retires the fixed row before full variation. The screened DHOST exterior is
exactly GR, while the unscreened canonical massive scalar falls at least as
`1/r^2`; neither can sustain flat outer galaxy speeds. This removes the current
local one-scalar lane and requires a causal baryon-forced memory/orientation
carrier that persists through vacuum without a free halo profile.
The latest mechanism result is
[`docs/SIGMA_V4C_BARYON_SEEDED_COHERENCE_TRACE_RESULTS.md`](docs/SIGMA_V4C_BARYON_SEEDED_COHERENCE_TRACE_RESULTS.md).
Its positive baryon-seeded trace is broad, stable, and the strongest v4 source,
reducing the spent joint cluster-map error by `10.23%`. It improves PLCKG287
by `20.66%` but AS295 by only `0.57%`, worsens one shear channel, fails
cross-cluster transfer, and drives its high-field scale to the bound. Together
with v4A and v4B, it activates the scoped stop decision in
[`docs/SIGMA_V4_SCALAR_MEMORY_MECHANISM_FALSIFICATION.md`](docs/SIGMA_V4_SCALAR_MEMORY_MECHANISM_FALSIFICATION.md):
do not tune another one-scale isotropic scalar-memory closure. The next lane
must derive a sourced trace plus orientation-preserving tensor transport from
an action before another map fit.
It completes the only preregistered numerical repair of the straight-ray path
operator without reading observational scores. The prior P0695, P0694, P0693, P0692, P0691, P0690, P0689, P0688, P0687, P0686, P0684, P0683, P0682, and P0677 snapshots remain in
[`latest-findings/2026-08-02-p0695-radial-path-math-audit`](latest-findings/2026-08-02-p0695-radial-path-math-audit/README.md),
[`latest-findings/2026-08-02-p0694-ddo154-routing-continuum`](latest-findings/2026-08-02-p0694-ddo154-routing-continuum/README.md),
[`latest-findings/2026-08-02-p0693-projected-spectral-joint-screen`](latest-findings/2026-08-02-p0693-projected-spectral-joint-screen/README.md),
[`latest-findings/2026-08-02-p0692-routing-continuum`](latest-findings/2026-08-02-p0692-routing-continuum/README.md),
[`latest-findings/2026-08-02-p0691-multipole-gated-routing`](latest-findings/2026-08-02-p0691-multipole-gated-routing/README.md),
[`latest-findings/2026-08-02-p0690-full-routing-screen`](latest-findings/2026-08-02-p0690-full-routing-screen/README.md),
[`latest-findings/2026-08-02-p0689-source-routing-audit`](latest-findings/2026-08-02-p0689-source-routing-audit/README.md),
[`latest-findings/2026-08-02-p0688-monotone-envelope`](latest-findings/2026-08-02-p0688-monotone-envelope/README.md),
[`latest-findings/2026-08-02-p0687-system-path-coordinate`](latest-findings/2026-08-02-p0687-system-path-coordinate/README.md),
[`latest-findings/2026-08-02-p0686-locked-path-topology`](latest-findings/2026-08-02-p0686-locked-path-topology/README.md),
[`latest-findings/2026-08-02-p0684-path-diluted-qumond`](latest-findings/2026-08-02-p0684-path-diluted-qumond/README.md),
[`latest-findings/2026-08-02-p0683-potential-channel-qumond`](latest-findings/2026-08-02-p0683-potential-channel-qumond/README.md),
[`latest-findings/2026-08-02-p0682-spent-deflection-atlas`](latest-findings/2026-08-02-p0682-spent-deflection-atlas/README.md)
and
[`latest-findings/2026-08-02-p0677-absolute-field-audit`](latest-findings/2026-08-02-p0677-absolute-field-audit/README.md),
and the complete P0612-P0621 archival bundle remains in
[`latest-findings/2026-08-01-p0620-gravity-routing`](latest-findings/2026-08-01-p0620-gravity-routing/README.md).

The public simulator/API delivery plan is maintained in
[`docs/PUBLIC_SIMULATOR_API_PLAN.md`](docs/PUBLIC_SIMULATOR_API_PLAN.md). It
defines real and seeded synthetic galaxy/cluster calls, safe formula
submission, asynchronous field-solver jobs, frozen comparators, parameter
accounting, Vercel hosting for the web/gateway layer, and isolated Cloud Run or
Modal workers for scientific computation. The first deployable control-plane
preview now lives in [`hosted-simulator`](hosted-simulator/README.md): it
packages all 175 SPARC galaxies, validates bounded dimension-aware formula
trees, creates deterministic radial synthetic systems, and scores submitted
laws against fixed MOND and Newtonian baryons. The local reference backend now
also executes generic Cartesian 2D/3D field manifests, extracts and generates
resolved baryonic galaxies, and runs one frozen model across asynchronous
multi-system batches with deterministic reports. The public 2D/3D and
raw-lensing routes remain explicitly unavailable on the public deployment
until the new authenticated worker connector is configured against verified
durable storage.
The latest implementation evidence and remaining boundary are recorded in
[`docs/ASYNC_MULTI_SYSTEM_BATCH_API_MILESTONE.md`](docs/ASYNC_MULTI_SYSTEM_BATCH_API_MILESTONE.md).
The first formula-independent observation-space result is documented in
[`docs/MASSIVE_TRACER_OBSERVATION_ADAPTER_MILESTONE.md`](docs/MASSIVE_TRACER_OBSERVATION_ADAPTER_MILESTONE.md):
the local API now converts a solved Cartesian acceleration field into a
circular-speed curve and scores declared uncertainties without exposing the
target to the field equation or baryonic extractor.
The verified Horizon3 production deployment is
<https://sigma-gravity-research-simulator-five.vercel.app>.

The v0.30 milestone exposes the gravity-independent resolved-galaxy extractor
and generator through the same authenticated worker boundary as confirmed
field jobs. A real separated process extracted DDO101 without velocity targets,
retained its 2D/3D ensemble, regenerated a controlled variant, exactly
transferred a 1.25 gas-mass change, and rehashed all 45 artifacts with zero
gravity parameters. No external worker or durable volume has been deployed
yet, so public heavy routes correctly remain HTTP 503. See
[`docs/AUTHENTICATED_GALAXY_WORKER_MILESTONE.md`](docs/AUTHENTICATED_GALAXY_WORKER_MILESTONE.md).

The v0.29 milestone first separated the confirmed-manifest field worker from
the development server, including the non-root container, bounded gateway,
restart recovery, and immutable artifact verification. See
[`docs/AUTHENTICATED_FIELD_WORKER_MILESTONE.md`](docs/AUTHENTICATED_FIELD_WORKER_MILESTONE.md).

The v0.28 local worker now feeds a cylindrical photon field into the raw
strong-lensing pipeline. It archives one distance-ratio-one projection,
profiles and counts two source coordinates per family, finds and assigns global
lens-equation roots, and scores image positions only when predicted topology is
complete. A finite-support gate prevents zero-filled cells outside the solved
cylinder from entering the root or Jacobian region. The real asynchronous HTTP
fixture scored its analytic images at `0.001692 arcsec` RMS, rehashed all 13
artifacts, and added zero per-object gravity parameters. The contract and its
real-cluster limitations are documented in
[`docs/AXISYMMETRIC_RAW_MULTIPLE_IMAGE_LENSING_MILESTONE.md`](docs/AXISYMMETRIC_RAW_MULTIPLE_IMAGE_LENSING_MILESTONE.md).

The v0.27 local worker now turns the same confirmed axisymmetric field into
both galaxy-motion and photon-lensing predictions. It reconstructs and
integrates a cylindrical `(a_r,a_z)` field at a declared inclination, returns
deflection/shear/invariant maps, and preserves separate photon and velocity
score channels with zero added gravity parameters. Independent affine,
chord-length and point-mass fixtures freeze the normalization; the finite
solved-domain and raw-likelihood limits are documented in
[`docs/AXISYMMETRIC_PHOTON_LENSING_MILESTONE.md`](docs/AXISYMMETRIC_PHOTON_LENSING_MILESTONE.md).

## Absolute-field update through P0682

The newest first-principles-style branch starts from divergence-form AQUAL,
uses real registered two-dimensional baryon maps lifted into physical 3D, and
integrates zero-slip photon deflections from the solved potential. Its compound
path activation cleanly separates galaxies from clusters, but the absolute
RX J2129 raw-lensing verdict is negative: the field retains one image root per
family and no critical curves. Reorienting the tensor to suppress transverse
leakage raises the field by 15%; the unique two-transverse-dimension survival
law raises it by 30%. Both miss their frozen advancement gates, and arbitrary
fitted powers are prohibited.

The supported conclusion is that angular routing is not the current
bottleneck. P0682 now repeats the required-field decomposition across six
spent clusters. All five non-boundary systems pass the frozen radial-morphology
gate; their median radial halo/baryon ratios have geometric mean `8.59` and
scatter `0.127 dex`. Only three compact-halo targets meet the reliability rule,
so no baryonic amplitude predictor advances. The next development target is a
baryon-predictable **radial monopole/effective-extent law**, tested against both
spherical and registered 3D baryon maps before any new sealed galaxy or cluster
outcome is opened. See
[`docs/P0682_SPENT_MULTICLUSTER_DEFLECTION_ATLAS_RESULTS.md`](docs/P0682_SPENT_MULTICLUSTER_DEFLECTION_ATLAS_RESULTS.md)
for exact metrics and stop/go thresholds.

P0683 then tested a single QUMOND-style equation whose exponent moves from the
fixed RAR galaxy response toward a deeper-potential cluster response. With one
universal transition setting, the dimension-fixed primary remains within
`1.035x` fixed RAR on 131 spent galaxies and closes `51.2%` of the spent
cluster radial gap. It fails the frozen cluster accuracy gates (`0.285 dex` on
five systems and `0.309 dex` on the reliable three) because potential depth
orders the individual cluster amplitudes in the wrong direction. No formula is
advanced. The next formula generator is the dimensionless baryonic potential
path ratio, not another per-cluster amplitude. See
[`docs/P0683_POTENTIAL_CHANNEL_QUMOND_RESULTS.md`](docs/P0683_POTENTIAL_CHANNEL_QUMOND_RESULTS.md).

P0684 adds the dimensionless baryonic path ratio
`eta=|Phi_b|/(r g_b)`. The fixed primary improves the cluster score to
`0.240 dex` and passes the reliable-three gate without degrading galaxies, but
still fails the all-five and 75%-gap rules. Two predeclared diagnostic rows
clear every numerical spent gate. The cleaner inverse-square-root row becomes
a new formula generator—not a promoted result—and is now locked for a 3D
QUMOND raw-topology test. See
[`docs/P0684_PATH_DILUTED_QUMOND_RESULTS.md`](docs/P0684_PATH_DILUTED_QUMOND_RESULTS.md).

P0685 implements the clean diagnostic generator as a locked registered 3D
QUMOND operator. Its numerical field passes every frozen gate and produces
`9.96 arcsec` strong-lens RMS physical deflection, `3.336x` the scalar-AQUAL
field, with negligible curl and no fitted amplitude. P0686, whose topology
gates were frozen before that field was calculated, rejects it: only `14/15`
training and `6/7` heldout exact roots converge, three source families are
missing images, only five recover both parities, and both shear nuisances hit
their bounds. The local path ratio creates a hollow response: median exponent
rises from `1.28` inside 15 kpc to `2.71` at 150-225 kpc. The next generator
uses one baryon-derived system path coordinate rather than this singular local
coordinate. See
[`docs/P0685_P0686_LOCKED_PATH_QUMOND_RESULTS.md`](docs/P0685_P0686_LOCKED_PATH_QUMOND_RESULTS.md).

P0687 replaces the centrally singular local ratio with one scale-free
baryonic system coordinate, `eta_sys=max|Phi_b|/max(r g_b)`. This guarantees
that the fixed primary exponent decreases outward and leaves galaxy accuracy
at `1.029x` fixed RAR. It does not preserve the full cluster result: the
all-five/reliable-three scores are `0.234/0.201 dex` and it closes only 59.9%
of the fixed-RAR gap. A capped-local diagnostic retains `0.154/0.177 dex` but
also retains the forbidden outward-rising exponent. No form advances. The next
generator is a parameter-free inward monotone envelope of the local exponent.
See [`docs/P0687_SYSTEM_PATH_COORDINATE_QUMOND_RESULTS.md`](docs/P0687_SYSTEM_PATH_COORDINATE_QUMOND_RESULTS.md).

P0688 applies the smallest pointwise nonhollow repair,
`p_env(r)=max_{s>=r} p_local(s)`. It adds no constant, preserves every local
response, guarantees a nonincreasing exponent, and remains `1.030x` fixed RAR
on galaxies. It nevertheless scores `0.216/0.271 dex` on the all-five and
reliable-three cluster sets and closes 63.0% of the gap. It nearly fixes the
two most underpredicted clusters but over-bends MACS1931, RXJ1347, and RXJ2129
by `0.29-0.35 dex`. The scalar local-path family is retired. The next branch
will redistribute the effective source while conserving its far-field
monopole and preserving baryonic multipoles. See
[`docs/P0688_MONOTONE_ENVELOPE_QUMOND_RESULTS.md`](docs/P0688_MONOTONE_ENVELOPE_QUMOND_RESULTS.md).

P0689 changes source placement instead of reshaping another scalar exponent.
It routes the positive extra QUMOND source onto the observed baryonic
morphology and places an equal negative polarization source on the existing
potential-transition shell, conserving the far-field monopole without a new
constant. The no-observation audit passes: routed residual `2.47e-14`,
positive/negative mismatch `1.57e-16`, net added source fraction `9.29e-17`,
and 78.1% of shell weight inside the boundary. No photon, radial, root, or
sealed score was computed. The next step is a separately frozen empirical
screen. See
[`docs/P0689_SOURCE_CONSERVING_BARYONIC_ROUTING_AUDIT.md`](docs/P0689_SOURCE_CONSERVING_BARYONIC_ROUTING_AUDIT.md).

P0690 rejects routing the entire positive generator: cluster radial error
becomes `0.871/1.089 dex`, median physical deflection reaches `23.81 arcsec`,
and only `14/15` training plus `4/7` heldout roots converge. But all seven
families now recover both parities and critical curves, versus five parity-
diverse families for P0686. Source placement is affecting the correct
observable. The next generator uses the normalized baryonic quadrupole,
`q_b=0.11886` on RX J2129, to mix local and routed sources with exact spherical
and line-like limits rather than a fitted routing fraction. See
[`docs/P0690_SOURCE_ROUTING_EMPIRICAL_SCREEN_RESULTS.md`](docs/P0690_SOURCE_ROUTING_EMPIRICAL_SCREEN_RESULTS.md).

P0691 calculates the routing fraction from the baryonic quadrupole rather than
fitting it. RX J2129 gives `q_b=0.118863`; the resulting field passes its
residual, identity, boundary, amplitude, and curl gates, but raw topology does
not. Only `13/15` training and `5/7` heldout roots converge, three families are
missing images, only five recover both parities, and one nuisance reaches its
bound. A global shape scalar is therefore insufficient. The next step is a
frozen, non-promotable continuum atlas to decide whether any linear blend has
the required topology before the whole family is retired. See
[`docs/P0691_MULTIPOLE_GATED_SOURCE_ROUTING_RESULTS.md`](docs/P0691_MULTIPOLE_GATED_SOURCE_ROUTING_RESULTS.md).

P0692 freezes 17 routing fractions and treats every row as spent diagnostic
evidence. One row, `f=0.30`, passes every viability gate: `15/15` training and
`7/7` heldout roots, `0.495/2.692 arcsec` RMS, no missing family, all seven
parity-diverse and critical, two allowed surplus-image families, and no
near-bound nuisance. Its heldout error is `1.0615x` the object-specific compact
halo. The fraction is not selected or advanced. Instead, the result generates
a parameter-free projected spectral-anisotropy hypothesis,
`e_2D=1-lambda_min/lambda_max=0.272023`, which must face a separately frozen
real-2D galaxy and RX J2129 test. See
[`docs/P0692_SPENT_LINEAR_ROUTING_CONTINUUM_RESULTS.md`](docs/P0692_SPENT_LINEAR_ROUTING_CONTINUUM_RESULTS.md).

P0693 replaces the spent `f=0.30` clue with a parameter-free projected
spectral anisotropy, `e_2D=1-lambda_min/lambda_max`. It produces the strongest
source-routing cluster result: RX J2129 calculates `0.272023`, recovers all
training and heldout roots, scores `0.601/2.670 arcsec`, has no missing family,
all parities and critical curves, two allowed surplus-image families, and a
heldout error only `1.053x` the object-specific compact halo. The joint verdict
is nevertheless fail. Real-map DDO154 scores `3.943 km/s`, `1.352x` algebraic
MOND and slightly worse than ordinary 3D QUMOND. The global covariance scalar
does not advance; the next frozen diagnostic asks whether any allowed mixture
of the current endpoints can close the galaxy gap. See
[`docs/P0693_PROJECTED_SPECTRAL_ROUTING_JOINT_RESULTS.md`](docs/P0693_PROJECTED_SPECTRAL_ROUTING_JOINT_RESULTS.md).

P0694 freezes the full allowed DDO154 source-mixture interval. Zero of 13 rows
is competitive with algebraic MOND; the best is `f=0` at `3.943 km/s`,
`1.352x` the ordinary and `2.671x` the weighted algebraic-MOND errors. The
response changes only in the eighth decimal place from zero to full routing,
so a new transform of the global controller cannot repair the galaxy within
these endpoints. The pair is retired. The next generator constructs a
curl-free potential by integrating the algebraic force along baryon-centered
rays, then adds the successful cluster relocation as a zero-boundary potential
difference. See
[`docs/P0694_SPENT_DDO154_ROUTING_CONTINUUM_RESULTS.md`](docs/P0694_SPENT_DDO154_ROUTING_CONTINUUM_RESULTS.md).

P0695 implements a scalar potential obtained by integrating the algebraic
simple-MOND force along rays from the baryonic centroid, then adds the cluster
routing correction as a zero-boundary potential difference. The no-observation
audit passes its radial limit (`1.17%` RMS), 24/48 quadrature convergence,
90-degree rotation covariance, curl, hybrid identity, and boundary gates. It
does not advance because first-order interpolation produces `9.15%`
tangential/radial power and `5.79%` maximum spherical angular scatter. A new
audit may change only interpolation to cubic; no physical term or threshold is
retuned. See
[`docs/P0695_RADIAL_PATH_POTENTIAL_MATH_AUDIT_RESULTS.md`](docs/P0695_RADIAL_PATH_POTENTIAL_MATH_AUDIT_RESULTS.md).

P0695B changes only interpolation from linear to cubic. Radial error improves
to `0.652%`, quadrature convergence to `0.0043%`, and all rotation, curl,
identity, and boundary gates pass. Tangential leakage (`6.61%`) and maximum
angular scatter (`6.01%`) still fail unchanged thresholds, so straight-ray
Cartesian transport is retired before any observational score. The next
generator boosts the coherent spherical monopole, retains measured Newtonian
multipoles, and adds the cluster routing correction as a zero-boundary
potential difference. See
[`docs/P0695B_CUBIC_RADIAL_PATH_AUDIT_RESULTS.md`](docs/P0695B_CUBIC_RADIAL_PATH_AUDIT_RESULTS.md).

## Untouched external validation (P0633)

The next verdict is now preregistered before any selected target product is
downloaded or scored. P0633 locks 13 non-SPARC LITTLE THINGS dwarfs with
resolved HI data and four previously absent RELICS clusters. Dynamics and raw
lensing targets remain sealed until the Poisson, AQUAL, and QUMOND solvers, the
candidate equation, its one universal parameter vector, and all predictions are
committed and hashed. The galaxy, raw-image, topology, critical-curve,
universality, and Solar-System rejection gates are conjunctive.

See
[`docs/P0633_EXTERNAL_VALIDATION_PREREGISTRATION.md`](docs/P0633_EXTERNAL_VALIDATION_PREREGISTRATION.md)
and reproduce the freeze with `python
scripts/freeze_p0633_external_validation.py`.

## Latest density/path-survival investigation

P0623-P0629 rigorously test the proposed dwarf/giant path-interference idea
with 44 baryon-only density, pair-crowding, and local-column features; 485
formula shapes; grouped galaxy cross-validation; derived and raw cluster
lensing; and Solar proxies. Baryonic potential depth gives a repeatable galaxy
gain but fails deep clusters. A potential-plus-surface compromise improves the
five-fold galaxy score by 7.51% over the constant parent and retains all 18 raw
lens roots, but it remains 9.28% worse than fixed RAR and 2.014 times the
limited compact-halo raw-lensing error. No formula passes the final combined
dwarf, giant, cluster, and Solar rules.

See
[`docs/P0623_P0629_DENSITY_PATH_SURVIVAL_RESULTS.md`](docs/P0623_P0629_DENSITY_PATH_SURVIVAL_RESULTS.md)
for the equations, every variation, comparator accounting, failure modes, and
the next external test.

## Latest field-physics candidate

The newest directional stage turns every observed baryonic source into a
positive, normalized Green kernel and derives a curl-free two-dimensional
deflection field. Thirty tensor variants show that tidal orientation is almost
irrelevant; routed fraction, width, and travel scale dominate. A simplified
720-law sweep finds one short, broad inward route in all five whole-cluster
folds. It improves the equal-galaxy RMSE by 2.00%, held Lenstool morphology by
11.5% versus local light and 5.9% versus a central blob, and passes the screened
Solar proxy. It still trails the cluster-only W060 kernel by 19.8%.

The stronger raw calculation rejects the spherical projection of that route.
Route+RAR scores 5.043 arcsec on the seven spent RX J2129 images, while adding
the frozen P0599 amplitude loses three exact roots. A 16-variant sensitivity
sweep finds routed fraction overwhelmingly dominant and selects zero routing
on training data.

The component-direction follow-up uses registered HST light and Chandra gas
morphology without pretending that X-ray brightness is gas mass. Forty signed
raw variants test discrete members, continuous stars, gas, and two blends. The
best positive route is gas-directed but at only `s_theta=0.0025`; its exact
refit improves training RMS by 0.018% and worsens the spent held-out RMS by
0.108%. An opposite-starlight control fits training slightly better and also
worsens held-out data. The supported inference is therefore a near-zero
directional correction, not a detected arc. See
[`docs/P0607_COMPONENT_DIRECTION_RESULTS.md`](docs/P0607_COMPONENT_DIRECTION_RESULTS.md)
for the raw component test and why source-redshift tomography is the next
falsifiable arc observable. That tomography now spans seven source redshifts
from 0.679 to 3.427. Varying the redshift exponent from 0 to 3 changes training
RMS by only 0.000155 arcsec at the allowed route strength, and a corrected
48-fit random-start audit finds the gamma difference smaller than the ordinary
geometry-basin width. The current data therefore identify neither a redshift
law nor a hidden arc height. See
[`docs/P0608_ROUTE_REDSHIFT_TOMOGRAPHY_RESULTS.md`](docs/P0608_ROUTE_REDSHIFT_TOMOGRAPHY_RESULTS.md).
The no-retuning transfer to four other raw clusters also fails: the standard
gas route improves the matched aggregate by 7%, but only MACS0429 improves,
MACS1931 still loses a held-out root, and equal-system RMS remains 19.9
arcseconds. This isolates gas/member misalignment as a conditional observation,
not a universal correction. See
[`docs/P0609_GAS_ROUTE_MULTICLUSTER_RESULTS.md`](docs/P0609_GAS_ROUTE_MULTICLUSTER_RESULTS.md)
for the transfer. A posthoc audit sharpens that clue: MACS0429 is the only
measured system where gas disagrees strongly with both discrete members and
smooth starlight. A dual-misalignment gate activates at 0.938 there and below
0.150 everywhere else, but its Pearson correlation falls from 0.963 to 0.230
when MACS0429 is removed. It is therefore a frozen prediction generator, not
evidence. See
[`docs/P0610_DUAL_MISALIGNMENT_DRIVER_RESULTS.md`](docs/P0610_DUAL_MISALIGNMENT_DRIVER_RESULTS.md)
for the candidate. Its exact chronologically prospective transfer to A383 and
MS2137 fails: the former activates at effectively zero, the latter at 0.595,
but the high-activation route changes the held-out-only RMS by just +0.026% and
still misses one training root. The gate is not promoted or retuned. See
[`docs/P0611_FROZEN_DUAL_MISALIGNMENT_TRANSFER_RESULTS.md`](docs/P0611_FROZEN_DUAL_MISALIGNMENT_TRANSFER_RESULTS.md)
and the revised next-test design in
[`docs/GRAVITY_ARC_NEXT_FALSIFICATION.md`](docs/GRAVITY_ARC_NEXT_FALSIFICATION.md).
See
[`docs/P0603_P0606_DIRECTIONAL_FIELD_RESULTS.md`](docs/P0603_P0606_DIRECTIONAL_FIELD_RESULTS.md)
for the operator, 750 formula variants, cross-domain frontier, raw decomposition,
and parameter rankings. The preceding bounded-potential stage remains in
[`docs/P0599_P0602_POTENTIAL_FIELD_RESULTS.md`](docs/P0599_P0602_POTENTIAL_FIELD_RESULTS.md).

## Gravity-arc tomography

The first direct test of the new field-routing picture is complete. It asks
whether conserved gravitational influence sourced only by baryonic galaxy
light can travel along nonlocal paths and reappear in the locations that
standard lens models assign to dark matter. An inverse optimal-transport map
backtracks those paths, while six target-blind forward laws are selected on two
RELICS clusters and scored on the third.

The primary three-cluster test fails its predictive gate. A strict
photometric-member sensitivity does select the same center-return formula in
all three holdouts and beats local-light smoothing by 24--37%, but it narrowly
loses to a smooth central-halo control in one cluster and sits at the search-grid
boundaries. It is a concrete lead for a locked, fresh-cluster test rather than
evidence against dark matter. See
[`docs/GRAVITY_ARC_TOMOGRAPHY_RESULTS.md`](docs/GRAVITY_ARC_TOMOGRAPHY_RESULTS.md)
for the field equation, backtracking method, full scores, limitations, and next
falsification experiment.

That falsification has now been run on ten untouched clusters with 1,000
Lenstool uncertainty maps and ten GLAFIC controls. The exact C0351 rule fails
both frozen gates. Across twelve locked perturbations, return direction is the
dominant ingredient and the returned fraction is the least influential. A
single favorable change—widening the endpoint from 50 to 60 kpc—improves 8/10
systems under both reconstruction methods and is now a candidate for a new
untouched holdout, not a retroactive rescue. See
[`docs/GRAVITY_ARC_FRESH_SAMPLE_RESULTS.md`](docs/GRAVITY_ARC_FRESH_SAMPLE_RESULTS.md).

The same ten systems now have a full inverse baryon-to-lensing-excess
reconstruction. Balanced optimal transport backtracks each excess pixel to
candidate member-galaxy origins and supplies projected path-density maps under
both Lenstool and GLAFIC. The methods agree well on the representation and give
a typical minimum route near 90 kpc, but the true angular galaxy pattern beats
radius-preserving shuffles in only 6/10 and 5/10 systems. The strongest
post-hoc clue is an extent-gated inward response: $R_{50}/R_{80}$ predicts the
inward transport share across both methods. This generates a new scale-free
kernel for a future untouched test; it does not demonstrate physical gravity
arcs. See
[`docs/GRAVITY_FLOW_INVERSE_RESULTS.md`](docs/GRAVITY_FLOW_INVERSE_RESULTS.md).

That inverse result has now been translated into a universal arc-apogee force
law and swept across 131 SPARC galaxies, ten cluster maps under two lens
reconstructions, and Solar-System proxies. The decisive finding is about where
the new control belongs: baryonic concentration harms galaxy rotation when it
multiplies scalar force, but helps cluster morphology when it only steers the
field between local and arcing paths. The separated candidate uses one global
$q=1.4557$ and passes the specified Solar proxies, but its held-out galaxy RMSE
is 12.966 km/s versus 10.348 km/s for fixed RAR. Its cluster-map candidate beats
local light in median JS divergence but trails the earlier C0351 and W060 arc
controls. See
[`docs/ARC_APOGEE_CROSS_DOMAIN_RESULTS.md`](docs/ARC_APOGEE_CROSS_DOMAIN_RESULTS.md)
for the equations, 1,980 formula variants, morphology breakdown, and claim
limits.

The next absolute-lensing stage shows why normalized morphology was not enough:
the arc-apogee baseline misses the CLASH acceleration by a median factor of
3.68. Across 58 one-at-a-time changes and a 576-law fine grid, baryonic
potential depth is the dominant scalar control; the old apogee, concentration,
and screening adjustments barely move absolute cluster error. A combined
potential/path response reaches 0.163 dex on CLASH without gravitational slip,
while a photon-weighted compromise scores 12.592 km/s on galaxies and 0.199 dex
on CLASH. On raw RX J2129 positions, however, the best shortlisted law reaches
only 1.245 arcsec, worse than the previous 1.064-arcsec project candidate and
above the 0.5-arcsec gate. See
[`docs/ARC_INVARIANT_ABSOLUTE_LENSING_RESULTS.md`](docs/ARC_INVARIANT_ABSOLUTE_LENSING_RESULTS.md)
for the field invariants, parameter-impact ranking, Solar checks, morphology
breakdown, and raw-image falsification.

The follow-up directional interaction test holds those scalar parents fixed and
adds only a zero-radial-average redistribution based on the 66 measured RX
J2129 member galaxies. A 722-law training screen strongly prefers compact,
positive routing shared across lower-mass members, but the gain does not
transfer: P0554 worsens from 1.256 to 1.307 arcsec on seven held-out images,
P0396 loses one of seven nonlinear roots, and the measured layout is not special
relative to radius-preserving angle shuffles. This rules out the direct local
member overlay, not all nonlocal field arcs. See
[`docs/ARC_MEMBER_INTERACTION_RESULTS.md`](docs/ARC_MEMBER_INTERACTION_RESULTS.md).
The general mass-budget-preserving nonlocal field equation and the inverse
source-to-arrival reconstruction are specified in
[`docs/CONSERVATIVE_GRAVITY_ROUTE_FIELD.md`](docs/CONSERVATIVE_GRAVITY_ROUTE_FIELD.md).

## Screened Sigma-field exploration

The first direct Newtonian `+ Sigma` field model has been implemented and run
through galaxy, cluster, finite-void, disk-geometry, and zero-slip lensing
checks.  Its cleanest new result is a calculable finite-void onset at
$R_{\rm critical}=\pi L_\Sigma$; its present bounded coupling gives a useful
universal boost but returns to Keplerian behavior after the transition.  See
[`docs/SIGMA_FIELD_EXPLORATION_RESULTS.md`](docs/SIGMA_FIELD_EXPLORATION_RESULTS.md)
for the equations, all six exploratory outcomes, and reproducible artifacts.

The action-constrained follow-up identifies the Sigma equation with established
symmetron screening and compares its native conformal force with two AQUAL-based
couplings.  A minimal Sigma-refracted AQUAL action removes the first model's
Keplerian far-field return while remaining comparably close to the galaxy and
cluster targets.  Its remaining universal-amplitude and raw-lensing tension is
documented in
[`docs/SIGMA_ACTION_EXPLORATION_RESULTS.md`](docs/SIGMA_ACTION_EXPLORATION_RESULTS.md).

The subsequent complete-action check lets gravity reshape Sigma and counts the
field's own stored energy as gravitating mass.  Feedback is negligible, while a
field heavy enough to help the cluster becomes much too heavy in the galaxy.
The result, explained with ordinary mismatch factors rather than only dex, is in
[`docs/SIGMA_COMPLETE_ACTION_RESULTS.md`](docs/SIGMA_COMPLETE_ACTION_RESULTS.md).

## Current strongest galaxy/cluster bridge

The completed MOND/dark-matter comparator sweep found one exploratory formula
worth advancing: fixed galaxy RAR plus a squared low-coherence Refracted-Gravity
susceptibility. It scores 0.1174 dex in held-out BCG+CLASH prediction and remains
within 2.3% of fixed RAR on the independent-nuisance SPARC outer refit (10.586
versus 10.348 km/s); it is 1.94% worse than simple MOND. On all 20 CLASH
clusters, its equal-cluster radial-field RMSE is 0.1387 dex versus 0.5184 dex
for fixed simple MOND, and a complete-cluster bootstrap confirms the advantage.
The CLASH target is NFW-deprojected, so this is not an independent victory over
dark matter or a raw relativistic lensing result.

See
[`docs/MOND_DARK_MATTER_FORMULA_SWEEP_RESULTS.md`](docs/MOND_DARK_MATTER_FORMULA_SWEEP_RESULTS.md)
for the equation, every failed intermediate formula, prior-art boundary,
limitations, and the next falsification stage.
The complete lensing closeness analysis and its raw-data boundary are in
[`docs/LENSING_UNIVERSAL_COMPARISON_RESULTS.md`](docs/LENSING_UNIVERSAL_COMPARISON_RESULTS.md).

## Raw RX J2129 strong-lensing pilot

The frozen raw-position pipeline has now been built and executed on 22
spectroscopic images in seven RX J2129 source families. The unchanged universal
candidate recovers all seven held-out images at 1.064-arcsec radial RMS, versus
only three recovered roots for fixed simple MOND and 2.536 arcsec for a compact
GR-plus-one-halo control. It does not pass the preregistered 0.5-arcsec absolute
gate. Its separately labeled all-image RMS is 0.618 arcsec; the published
conventional 71-halo RX J2129 model reports 0.29 arcsec on its all-image fit.

This is the first comparison here that scores raw image positions rather than
an NFW-deprojected acceleration target. The baryonic input remains a literature
radial reconstruction, and the zero-slip photon law is a diagnostic closure,
not a covariant theory. RX J2129 also contributed to the earlier derived-field
calibration, so the withheld coordinates are new but the cluster is not an
independent validation object. See
[`docs/RXJ2129_RAW_LENSING_RESULTS.md`](docs/RXJ2129_RAW_LENSING_RESULTS.md) for
the method, exact scores, sensitivities, and claim boundary. Reproduce it with:

```powershell
python scripts/run_rxj2129_raw_theory_lensing.py
```

## What the first milestone tests

The primary phenomenological model is

$$
g_{\rm pred}=g_{\rm bar}+
A_0 e^{\beta \mathcal V},a_t
\left(\frac{g_{\rm bar}}{a_t}\right)^p
S(g_{\rm bar}),
$$

with a gradual unscreening function

$$
S(g)=\left[1+\exp\left(\frac{\log_{10}g-\log_{10}a_t}{w}\right)\right]^{-1}.
$$

The data are allowed to determine the exponent $p$. A value near $p=1/2$
produces a flat added rotation contribution in the point-mass outer limit. The
same nuisance treatment and radial holdout are used for Newtonian, empirical
RAR, and NFW comparators.

SPARC rotation curves are mostly H I and H-alpha gas kinematics, not tracks of
individual outer stars. “Negative gravity” is therefore only an operational
label here for an additional inward effective acceleration. A SPARC fit alone
cannot establish a void origin.

## Current scope

- Parse all 175 SPARC mass-model files and the published metadata table.
- Preserve signed gas contributions when constructing $V_{\rm bar}^2$.
- Apply explicit quality, inclination, and minimum-point cuts.
- Fit common physical parameters plus per-galaxy mass-to-light, distance, and
  inclination nuisance parameters with stated priors.
- Train on the inner 70% of each retained curve and score the untouched outer
  30%.
- Compare Newtonian, RAR, NFW, free-$p$ void, and fixed-$p=1/2$ void models.
- Include a frozen, independent Cosmicflows-4 density score for every SPARC
  galaxy. The code never derives an environment score from rotation-curve
  velocities or residuals.
- Run on CPU or CUDA and save machine-readable summaries, predictions, model
  state, optimization history, and diagnostic plots.

The preregistered decision rules and project phases are in
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md).
The first tested CPU/CUDA engineering baseline is recorded in
[`docs/INITIAL_STATUS.md`](docs/INITIAL_STATUS.md).

## Quick start on this machine

The SPARC snapshot is already imported. To rebuild it from the local source:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/import_sigmagravity_data.ps1
```

Create an isolated environment and install the package:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the RTX 5090, the included setup script creates a Python 3.12 environment
and installs the official PyTorch 2.12.1 CUDA 13.0 wheel. The version and index
are explicit so the environment is reproducible:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_cuda.ps1
.venv\Scripts\Activate.ps1
```

Confirm the environment rather than trusting the presence of `nvidia-smi`:

```powershell
python check_device.py
```

If PyTorch has released a newer stable build, verify the current Windows/Pip/CUDA
command at <https://pytorch.org/get-started/locally/> before changing the pinned
version.

Run a single model:

```powershell
python fit.py --model void --device auto --steps 5000 --output results/void_free_p
python fit.py --model void --fixed-flat-power --device auto --steps 5000 --output results/void_p05
```

Run the initial comparator suite:

```powershell
python compare_models.py --device auto --steps 5000 --output results/comparison
```

For a fast execution check, add `--steps 25`. That verifies the pipeline, not
scientific convergence.

## Independent environment input

Download the official Cosmicflows-4 grids and catalog, verify their byte sizes
and hashes, and rebuild the frozen SPARC cross-match with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_cosmicflows4.ps1
python scripts/build_cf4_environment.py
```

The generated `data/derived/void_scores_cf4.csv` contains one row per galaxy.
Its required model-input columns begin:

```text
galaxy,void_score
CamB,...
DDO154,...
...
```

The primary score is the negative grouped-grid density contrast, so larger
values mean a more underdense reconstructed environment. The ungrouped 64^3
grid and official 128^3 grid are retained as prespecified sensitivity columns.
The construction uses only sky position, SPARC distance, and the external CF4
density field. Run the environment-enabled model with:

```powershell
python fit.py --model void --environment-csv data/derived/void_scores_cf4.csv `
  --device cuda --steps 10000 --output results/void_environment
```

The full coordinate convention, grid hashes, score distributions, cross-grid
correlations, and catalog validation are recorded in
`data/derived/cf4_environment_report.json`.

## Current CF4 theory-test result

The completed 5,000-step radial and five-fold galaxy tests do **not** support
the specific prediction that a stronger CF4 void environment increases the
additional galactic acceleration. The smooth low-acceleration law remains much
better than Newtonian baryons alone, but the environment effect harms strict
held-out-galaxy prediction for both 64^3 reconstructions, and the 128^3 result
has the opposite beta sign. See `docs/CF4_THEORY_TEST.md` for the full results,
bootstrap intervals, limitations, and decision-rule audit.

The prior-art audit and frozen next-model sequence are in
[`docs/PRIOR_ART_AND_NEXT_TESTS.md`](docs/PRIOR_ART_AND_NEXT_TESTS.md). It
separates known ideas from this project's narrower test architecture and defines
potential-screened, environment-shifted, boundary-layer, void-wall, and physical
tidal checks before any of those variants are fit.

The next theory-development registry is in
[`docs/MAXWELL_HEAVISIDE_VOID_EXTENSIONS.md`](docs/MAXWELL_HEAVISIDE_VOID_EXTENSIONS.md).
It treats Maxwell-Heaviside four-potential gravity as a weak-field scaffold,
audits nine void/environment extensions against prior art, and ranks covariant
scalar-vector-tensor, generalized-Aether, and MOG-like completions by whether
they can predict both dynamics and lensing from one physical metric.

Quantitative continue, success, and rethink thresholds for that program are in
[`docs/THEORY_DEVELOPMENT_STAGE_GATES.md`](docs/THEORY_DEVELOPMENT_STAGE_GATES.md)
and `configs/theory_stage_gates.json`. The first active stage reconstructs the
constitutive response required by every SPARC and CLASH point before choosing a
new field interpolation.

The live gate audit, action-derived weak-field results, and the frozen 34-system
SPIDERS--MaNGA host-profile validation are in
[`docs/THEORY_DEVELOPMENT_PROGRESS.md`](docs/THEORY_DEVELOPMENT_PROGRESS.md).
Both simple- and standard-$\mu$ closures improve the U0 bridge, but neither
advances because a parameter reaches a hard bound in multiple folds. The bridge
sample and independent profile-constraint route pass the frozen coverage gate.
Integrating BCG, gas, and satellite profiles gives $\chi^2/N=1.658$, RMS 0.132
dex, and mean residual -0.083 dex, passing the Stage 4 scientific thresholds in
all 5,000 uncertainty realizations without a fitted host normalization. This
supports the baryonic-potential environment variable but does not override the
H7s hard-bound failure. The subsequent local EA-Q0 action also fails before
fitting: the reciprocal source required by variation changes $Q$ by at least a
factor 216 for every frozen BCG, versus the 5% gate. The action, conservation
identity, spherical calculation, and decision are in
[`docs/EAQ0_DERIVATION.md`](docs/EAQ0_DERIVATION.md). EA-Q0 is retired. The
subsequent five-parameter
[`environmental MOG checkpoint`](docs/ENVIRONMENTAL_MOG_CONTROL.md) is also
complete and retired before fitting. Its monotone chameleon response has an
analytic 59.0% minimum-error contradiction within CLASH, and its universal
Yukawa force resembles $1/r$ over only 0.055 dex. The action, equations,
conservation identity, same-metric lensing solution, and numerical envelope are
in [`docs/ENVIRONMENTAL_MOG0_DERIVATION.md`](docs/ENVIRONMENTAL_MOG0_DERIVATION.md).
The research program is now at its preregistered premise-level rethink.
The subsequent formula-level comparison of Sigma Gravity with published
Refracted Gravity is in
[`docs/SIGMA_REFRACTED_GRAVITY_COMPARISON.md`](docs/SIGMA_REFRACTED_GRAVITY_COMPARISON.md).
It shows that the retired NBP0 scalar permittivity contains the ordinary RG
transition as its zero-smoothing limit, rejects a multiplicative Sigma x RG
double boost, and defines CPR0, a coherence-partitioned interpolation with exact
Sigma and RG endpoints. An exploratory 20-cluster proxy screen improves on the
old fixed cluster Sigma amplitude but not on a freely recalibrated cluster-only
Sigma control; CPR0 therefore advances only to local-density and independently
measured-coherence acquisition, not to a gravity claim.
That acquisition and test sequence is now complete. The hash-locked ACCEPT
snapshot supplies measured Chandra densities for 18 CLASH clusters, reducing
the galaxy--cluster density gap to 0.421 dex; the CLASH BCG catalog adds observed
stellar masses and effective radii for all 20 central points. Density-only RG
scores 0.1300 dex on the 52-point outer-cluster test and 0.1776 dex after adding
the central stellar-density points. In the final shared fit, cluster RMSE is
0.1556 dex for RG and 0.1546 dex for CPR0, while measured `Lambda_Re` never
provides the required improvement. No measured-data protocol passes all gates,
so CPR0 is now a rejected baseline. The full audit and the narrower, genuinely
new data requirements are in
[`docs/CPR0_MEASURED_DENSITY_AND_COHERENCE_RESULTS.md`](docs/CPR0_MEASURED_DENSITY_AND_COHERENCE_RESULTS.md).
The next R0--R3 outcomes—including a raw-observable audit and a same-object
two-potential identifiability sample—are in
[`docs/PREMISE_LEVEL_RETHINK.md`](docs/PREMISE_LEVEL_RETHINK.md).
R0 now includes both its 27-row scored-column lineage and a hash-locked
19,030-row scalar provenance ledger covering every value currently scored for
131 SPARC galaxies, all 20 CLASH systems, and 34 frozen BCG hosts. The latter
records the exact input file and SHA-256, value and unit, transformation,
metric/dynamical assumptions, covariance disposition, and alternative-theory
forward-model status for each scalar. This completes provenance, not the
same-system likelihood or identifiability gates.
The requirement ledger in `data/derived/r0_r2_goal_progress.csv` keeps those
stages separate: provenance and the two acquisition-disposition boundaries are
closed, while the 10-system freeze and all three R2 reconstruction/validation
requirements remain incomplete or unauthorized. Its hashed audit is
`results/r0_r2_goal_progress/report.json`.
The residual-blind attainability audit also proves that the current 15-system
candidate universe has a structural ceiling of only three rank-qualified systems:
Abell 1689, MACS J1206 and RX J2129. Even repairing every downstream baryon and
covariance gap would leave the sample at most 3/10, so the next acquisition cycle
must add at least seven new systems with pre-fit radial-rank ceiling >=3 instead of
reprocessing rank-zero candidates. See
`results/r1_ten_system_attainability/report.json`.

The first search outside that 15-system universe has also been audited without
viewing science pixels or gravity residuals. SDSS J0946+1006 (the Jackpot lens)
has three spectroscopic source planes, public HST imaging, and eleven public ESO
MUSE cube products. It nevertheless fails the frozen admission gate: the
published model-valid stellar-dynamics support stops at 1.95 arcsec, so only the
1.4-arcsec ring scale overlaps it; the 2.1- and 2.5-arcsec scales do not. The
public record supplies raw archives and analysis scripts but no downloadable
normalized theory-neutral likelihood or chains. Jackpot is therefore retained
as a rank-one repair candidate, not counted as a fourth structural promotion,
and no science-pixel download is authorized. The source and archive audit is in
`results/r1_j0946_jackpot_feasibility/report.json`.

The second external candidate, ESO 325-G004, passes a narrower pre-pixel gate but
is not promoted. Its public 2.95-arcsec extended ring lies inside the published
central 4-arcsec MUSE modelling support, and the primary source explicitly says
that arc widths and shapes constrain radial magnification beyond a single
Einstein-radius mass. Public metadata resolve one level-2 MUSE science cube, two
sky cubes, 18,882 seconds of F814W imaging, and 4,800 seconds of F475W imaging.
Because one ring radius is still only one mode and thousands of source pixels are
not thousands of independent mass-profile constraints, the structural ceiling
remains 3/10. A frozen next-stage protocol acquires four exact HST DRC mosaics
and a 19.7-MB MUSE SODA cutout, then requires at least three >=3-sigma
nuisance-projected response singular directions stable across every source-grid,
regularization, PSF, mask, finite-difference, synthetic-injection, and held-out-
visit control, plus three overlapping numerical MUSE bins with covariance. A
failure retains E325 only at its measured lower rank; it cannot trigger a
threshold change. See `results/r1_e325_feasibility/report.json` and
`configs/r1_e325_acquisition_jacobian_protocol.json`.

The current replacement-sample gate and local public-data inventory are in
[`docs/R1_REPLACEMENT_SAMPLE_INVENTORY.md`](docs/R1_REPLACEMENT_SAMPLE_INVENTORY.md).
Ten systems pass the published-count screen, but none yet passes strict R1.
All ten now have observable-level image positions and declared image-plane
likelihood errors; 106 positions also have spectroscopic source redshifts. Seven
have reconstructed BCG stellar profiles, and three have RELICS MCMC convergence-
map ensembles. The structural three-radial-mode gate also fails 0/10: A383's
radial-rank upper bound is 2, A2537 and MS2137 each have an upper bound of 1, and
the rest have 0. The active R1A.2 stage is therefore a residual-blind search toward
30 source-screened BCG hosts (or a documented hard public-data shortfall), with at
least two structurally qualified replacements required before strict-readiness
preparation can begin.
Complete baryonic inputs, rerunnable lens nuisance models, and systematic covariance
remain limiting requirements.
The first replacement audit promotes MACS J1206 structurally (11 strict inner
images over six BCG-dynamics annuli) but not yet to R1-ready status.
An official archive audit found no numerical BCG dispersion table, covariance, or
likelihood for MACS J1206 or Abell S1063, but did verify public level-3 ESO MUSE
cubes for both. A frozen independent pPXF reconstruction was run on the level-3
cutout and on six homogeneous level-2 central-pointing cutouts. The level-2 result
removes the catastrophic level-3 outer velocity split, but still fails the outer
formal-uncertainty, opposite-half sigma, and leave-one-product-out sigma gates.
MACS J1206 therefore remains a structural promotion, not an R1-ready numerical
likelihood substitute, and gravity fitting remains unauthorized.
Cycle 1 has now exhausted the Sand six-cluster sample and explicitly excludes
RXJ 1133, Abell 1201, and MACS J0416 on observable-level grounds. Cycle 2 retains
SDSS J0100+1818 as a group-scale bridge/control but does not promote it: only one
of its 18 spectroscopic images overlaps its six-bin, 3-arcsec BGG kinematic
support, so its structural radial-rank ceiling is 1.

RX J2129 is now the second structural promotion. Its three spectroscopic inner
images from three families overlap a frozen four-bin, 0-5 arcsec public-MUSE BCG
profile. A deterministic audit found that the original E-MILES templates were
broader than MUSE in the fitted rest-frame range, invalidating that attempted
resolution sensitivity. The frozen correction uses higher-resolution XSL templates
without changing the center, annuli, mask, or thresholds. Its dispersions are
293.1, 306.4, 322.0, and 343.6 km/s, and every predeclared baseline and opposite-
half check passes. All 100 block bootstraps and all nine non-baseline protocols
complete; the positive-definite 4x4 covariance has total errors of 6.74, 1.91,
2.07, and 3.32 km/s, and the maximum protocol shift is 3.91%.
Cycle 3 exhausts the complete 32-BCG Loubser et al. spatial-kinematics sample.
After five overlaps it adds 27 unique hosts, so the current counts are 45/30
source-screened hosts, 2/2 non-disturbed structural promotions, and 0 strict
R1-ready systems. All 27 new hosts have ingested Gemini kinematics, but none has a
local image-position/source-redshift likelihood in the current normalized ledger.

The active next-stage target tests whether RX J2129 can become the first strict-ready
system by completing numeric baryonic inputs and a rerunnable 21-image lens
likelihood with nuisance uncertainty. The kinematic covariance workstream is now
complete. A residual-blind baryonic audit has also reconstructed the published
four-bin Hernquist BCG baseline and its shared covariance. The original audit
failed the complete component gate because it had only one 14.3-kpc Chandra gas
anchor, no identifiable BCG/ICL split, and no normalized off-center satellite
likelihood. See
`results/r1_rxj2129_baryons/report.json`.

The frozen empirical-PSF prerequisite now passes in both bands using three
predeclared field stars; its encircled-energy, pairwise-profile, and leave-one-out
checks are recorded in `results/r1_rxj2129_hst_psf/report.json`. The Cooke source
also shows that the stellar mass came from MAGPHYS with SDSS Petrosian and WISE
profile-fit photometry, not an HST aperture matched to a BCG/ICL split.

The measured SDSS r-band Petrosian radius is 9.373521 arcsec, fixing Cooke's SDSS
flux aperture at 18.747042 arcsec. The frozen masked HST extraction now passes too:
49 radial bins per band survive the coverage gate, their joint 98x98 covariance is
positive definite, and the light centroid is 0.0464 arcsec from the dynamics
center. The preregistered PSF-convolved model comparison is also complete. Two
Sersic terms reduce held-out chi2 by 69.8% in F125W and 77.1% in F814W, showing
that a one-component total-light shape is inadequate. However, the putative outer
term contains 94.9% of F125W light inside 30 arcsec, outside the frozen 5-80%
physical-component range. The result is therefore total-light structure plus an
explicit BCG/ICL non-identifiability finding, not a component mass decomposition.
See `results/r1_rxj2129_bcg_icl/report.json`.

The satellite stellar term is now numeric. A 43-label candidate-domain model
improves grouped held-out Brier score by 57.1%, reaches AUC 0.918, and preserves
500 bootstrap probability vectors. Two thousand off-center force draws give a
positive 4x4 covariance; profile-size sensitivity peaks at 1.18% of the BCG
acceleration and a worst-case bound for all 199 candidates beyond 30 arcsec peaks
at 0.871%, below its frozen 1% gate. See
`results/r1_rxj2129_satellite_membership/force_report.json`.

The source-traceable lens observables are also assembled: 21 spectroscopic images
in seven families, a 42x42 coordinate covariance with 0.5-arcsec errors, and three
images from three families inside the 5-arcsec dynamics support. Four photometric
off-center galaxy-galaxy images are retained in the ledger but excluded from the
likelihood. No published GR map or model residual was ingested. See
`results/r1_rxj2129_lens_observables/report.json`.

The separately implemented image-plane model is now an informative failed blind
gate rather than a missing workstream. Its smooth model fits all 21 images with
0.3833-arcsec exact radial RMS and a finite 24x24 Laplace covariance, but the
seven-image holdout has 1.4299-arcsec RMS. Adding the 66 candidate members improves
training residuals while worsening heldout RMS to 2.7265 arcsec. The holdout can
reject that added layer, but the frozen protocol omitted a numerical heldout-
adequacy threshold. Consequently the all-image result is retained only as an
engineering fit and no predictive or Weyl-response claim is authorized. See
`results/r1_rxj2129_lens_model/report.json`.

The frozen Chandra reduction is now a closed failed route. ObsID 552 retains
81.809% of its exposure versus the 90% floor; the blank-sky `BKGSCAL` values are
outside 0.5-2.0; and the event lineage does not meet the frozen CALDB requirement.
No RX J2129 gas likelihood is authorized. Abell 1689 is also closed as a dynamics
route: its 200/200 bootstrap succeeds, but the frozen 27-run systematic grid has
signed-bin dispersion shifts up to 36.6% versus the 10% ceiling. It remains
geometry-only and no final covariance was assembled.

The extended-kinematics cycle is now closed without threshold changes. Abell 2261
failed its frozen continuum-center gate; A383 failed its 0.200-Angstrom arc-RMS
gate; and MS2137 failed its pre-pPXF registration/geometry gate. The final A2537
disturbed engineering control passed exact raw acquisition and environment checks
but stopped before science processing because its two CuAr solutions have 0.2248
and 0.2205 Angstrom RMS versus the frozen 0.2000-Angstrom ceiling. The cycle added
zero structural promotions, triggering the planned data-route rethink.

The active route now tries to finish RX J2129 rather than acquire more BCG
spectroscopy. Its public ObsID 0093030201 tree is local and checksum-provenanced.
XMM X1-X2 now pass with MOS2 and pn: calibration, flare-cleaned exposure, the
immutable 87-source mask, sector-level FWC/corner scales, and the frozen local
650-900 kpc background transfer all meet the predeclared gates. MOS1 is excluded
because CCD5 fails its sector scale, even though its pooled diagnostic passes.
X3 now passes for all six immutable MOS2+pn annuli: 76,279.260 conservative net
counts in total and a minimum annular S/N of 84.091. The active targets are now
coverage-complete direct responses, a 6x6 PSF cross-region ARF matrix for each
detector, central-source response vectors, and the already frozen HST H1-H3
21-image, 42x42 centroid-covariance measurement. The default X3 responses are not
admissible for fitting because the outer region exceeded the default detector-map
coverage; X3 was only a count-information and product-existence gate. See
[`docs/R1_NEXT_STAGE_2026-07-26.md`](docs/R1_NEXT_STAGE_2026-07-26.md).
The quarantined X4 interface now passes for MOS2 and pn. A subsequent engineering
gate found that SAS realized the nominal 650 map request with 81-detector-unit
pixels, just above the frozen 80-unit ceiling. That baseline is rejected without
changing the science geometry or thresholds; the active convergence comparison is
the predeclared promoted 920 request versus its 1302-pixel sqrt(2) refinement.
That gate now passes: MOS2 changes by 1.803%, 1.841%, and 2.034% for the
integrated, median-shape, and p95-shape metrics; pn changes by 0.464%, 0.463%,
and 0.523%. Full X4 response production and its immutable manifest audit are now
active. The independent HST H1 calibration also passes after preserving one
invalid engineering run and correcting only its drizzled-WCS SIP handling and
declared Moffat-FWHM parameterization. The corrected run has 178 mutual matches,
0.024001-arcsec leave-one-out RMS, 607/612 successful F814W PSF fits, 203/203
successful F125W fits, and all 500-draw registration and PSF bootstraps. Its
segmentation, draw arrays, source ledgers, PSF field, and diagnostic are
hash-locked. The exact H2 centroid implementation now also passes a no-pixel
synthetic recovery and static freeze audit and is running on all 21 immutable
images. Its first attempt stopped before writing a centroid because Astropy
rejected an equivalent but non-identical coordinate frame; that event is logged,
and only the explicit frame conversion was corrected and re-hashed. H3 remains
locked until at least 18 images, including 5.2, 6.3, and 8.2, pass unchanged
two-band and bootstrap gates.
No gas temperature/density likelihood, full Jacobian, Weyl response, or gravity-law
fit is authorized until X4 response calibration and the independent measurement
gates pass.
The ordered acquisition milestones, numeric advancement thresholds, and rethink
clock are in [`docs/R1_EXECUTION_TARGETS.md`](docs/R1_EXECUTION_TARGETS.md) and
`configs/r1_execution_targets.json`.

Reproduce the independent-data inventory and frozen score with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_bcg_environment_catalogs.ps1
python scripts/download_host_profile_catalog.py
python scripts/inventory_host_profile_coverage.py
python scripts/validate_measured_host_profiles.py
python scripts/check_eaq0_derivation.py
python scripts/check_environmental_mog0.py
```

Those staged tests are now complete. The self-potential screen is competitive
with, but does not outperform, fixed RAR; CF4 threshold shifting, a boundary
layer, and independently cataloged void-wall depth all fail strict held-out-
galaxy prediction. Ordinary CF4 tides are about five orders of magnitude too
small at the median. See [`docs/NEXT_MODEL_RESULTS.md`](docs/NEXT_MODEL_RESULTS.md).

## Joint galaxy-dynamics and cluster-lensing result

The project now tests a stricter target: one baryon-linked acceleration field,
with zero gravitational slip and no lensing-only multiplier, must predict SPARC
circular speeds and CLASH lensing accelerations. The strongest tested bridge is
an explicitly **EMOND-like prior-art control** in which the RAR acceleration
scale rises with baryonic potential depth. In five-fold whole-system validation
it reduces the equal-domain score from 25.98 to 7.16 while making the SPARC
score 3.94% worse. It therefore clears the frozen relative advancement gate,
but its cluster chi-square remains too large for it to be called a completed
theory.

On 50 MaNGA BCG dynamical points that were not used in fitting, the frozen law
improves chi-square per point from 9.96 to 7.15, but a separately labeled
cluster-scale RAR reaches 2.19. The bridge is therefore partial. A post-hoc
inverse check finds that the missing transition corresponds to a plausible host
baryonic potential scale, but that quantity must be independently measured
before it can be used predictively. See
[`docs/UNIFIED_GALAXY_CLUSTER_RESULTS.md`](docs/UNIFIED_GALAXY_CLUSTER_RESULTS.md).

Reproduce the joint and external tests with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_clash_rar.ps1
python scripts/cross_validate_unified.py
powershell -ExecutionPolicy Bypass -File scripts/download_manga_bcg.ps1
python scripts/build_manga_bcg_table.py
python scripts/test_external_bcg.py
python scripts/diagnose_bcg_host_potential.py
```

Reproduce the strict validation and report with:

```powershell
python scripts/cross_validate_cf4.py --device cuda --steps 5000 `
  --folds 5 --bootstrap-draws 100000 --output results/cf4_galaxy_cv_5000
python scripts/summarize_cf4_test.py
```

## Axisymmetric disk-versus-bulge result

The scalar nonlocal-permittivity branch has now been tested with an explicit 2D
axisymmetric solver rather than a slab proxy.  A 1,023-case sweep confirms that
disk and bulge geometry changes the field, but the predicted outer radial ordering
is not stable: only 28.9% of 128 matched parameter environments favor a disk over
a bulge at 4, 6, and 8 disk scales simultaneously.  The frozen SPARC morphology
test also fails.  Adding bulge fraction and bulge scale worsens held-out RMSE for
all nine tested stellar mass assumptions.  This closes scalar retuning and points
the next geometry-aware branch toward a genuinely directional constitutive law,
with vertical dynamics and lensing ellipticity as mandatory controls.  See
[`docs/NBP0_MORPHOLOGY_RESULTS.md`](docs/NBP0_MORPHOLOGY_RESULTS.md).

## Member-tidal metric result

A curl-free anisotropic-Poisson test now uses the observed CLASH member layout
to modify only the fixed-RAR extra potential with one universal tensor coupling.
The frozen grid selects `t=0`. Its two-cluster held-out RMS is 18.432 arcsec,
unchanged from scalar metric slip and above the 9.989-arcsec compact-halo
comparator. A post-result `t=-0.6` check improves this only to 18.015 arcsec;
stronger positive settings lose an exact image root. The member-only tensor is
therefore retired. A second frozen test retained the member field's circular
tidal stress instead of subtracting it; it also selected `t=0` and obtained
18.433 arcsec. See
[`docs/MEMBER_TIDAL_METRIC_RESULTS.md`](docs/MEMBER_TIDAL_METRIC_RESULTS.md) and
[`docs/MEMBER_FULL_TIDAL_METRIC_RESULTS.md`](docs/MEMBER_FULL_TIDAL_METRIC_RESULTS.md).

Reproduce with:

```powershell
python -m pytest tests/test_tidal_metric.py -q
python scripts/run_member_tidal_metric.py
python scripts/run_member_tidal_nonzero_diagnostic.py
python scripts/run_member_full_tidal_metric.py
```

## Complete formula scorecard

The consolidated audit now contains 124 scientifically distinct formula/test
rows. It reports every available original error and a descriptive normalized
proximity percentage while keeping observational galaxy dynamics, GR/NFW-
derived lensing accelerations, and raw image-coordinate lensing in separate
columns. See [`docs/FORMULA_SCORECARD.md`](docs/FORMULA_SCORECARD.md), with
machine-readable outputs in `results/formula_scorecard/`.

Rebuild and verify it with:

```powershell
python scripts/build_formula_scorecard.py
python -m pytest tests/test_formula_scorecard.py -q
```

## One-parameter multi-cluster raw-lensing result

A frozen eight-family search selected one shared baryon-normalized isothermal
tail, `g = g_bar + 9 g_bar(200 kpc) (200 kpc/r)`. The value was chosen on
MACS0329+0429 and locked before a MACS1115+1931 replay holdout. It obtains 9.423
arcsec equal-cluster RMS, improving baryons-only GR (25.199), fixed MOND
(25.636), and fixed RAR (25.673), and narrowly beating a deliberately compact
halo aggregate (9.989). It still fails the frozen 2-arcsec accuracy gate, has a
worse pooled chi-square than the compact halo, and performs poorly on the
post-lock RXJ2129 stress replay. This is a useful one-parameter phenomenology,
not a replacement for dark matter. See
[`docs/ONE_PARAMETER_MULTICLUSTER_LENS_RESULTS.md`](docs/ONE_PARAMETER_MULTICLUSTER_LENS_RESULTS.md).

Reproduce with:

```powershell
python scripts/run_one_parameter_multicluster_lens.py
python scripts/run_one_parameter_multicluster_lens_stress.py
python -m pytest tests/test_one_parameter_multicluster_lens_results.py -q
```

## Solar-screened one-parameter result

The one-parameter isothermal tail now has a fixed high-acceleration screen,
`a0/(a0+g_bar)`. A frozen development replay selects the single universal value
`lambda=10.5`. The locked law predicts only `-2.36e-7` milliarcseconds/century
of supplementary Mercury precession, inside the unchanged `+/-3.1` margin by a
factor of 13.1 million. Its MACS1115+1931 replay holdout improves to 5.261 arcsec
and pooled chi-square 942.9, compared with 9.989 arcsec and 1615.4 for the
limited compact-halo comparator. It still fails the earlier 2-arcsec absolute
target and scores 6.17 times the compact-halo error on the RXJ2129 stress replay.
This is a surviving one-parameter phenomenology, not a raw ephemeris test or a
covariant theory. See
[`docs/SOLAR_SCREENED_ISOTHERMAL_RESULTS.md`](docs/SOLAR_SCREENED_ISOTHERMAL_RESULTS.md).

The locked value has now also been tested on 131 SPARC galaxies with the same
inner-70% nuisance calibration and untouched outer-30% prediction split used by
the trusted galaxy controls. It fails that transfer: outer RMSE is 18.602 km/s
versus 10.348 for fixed RAR and 10.385 for simple MOND. Disk-dominated galaxies
score 20.324 km/s, dwarfs 16.520 km/s, and rising outer curves 21.164 km/s. The
open-screen tail has `v_tail^4 proportional to Mbar^2`, not the observed
approximately linear baryonic Tully-Fisher scaling, so one constant lambda
cannot cover dwarfs through giants. See
[`docs/SOLAR_SCREENED_GALAXY_MORPHOLOGY_RESULTS.md`](docs/SOLAR_SCREENED_GALAXY_MORPHOLOGY_RESULTS.md).

Reproduce with:

```powershell
python -m pytest tests/test_one_parameter_lens.py tests/test_solar_system_tail.py -q
python scripts/run_solar_screened_isothermal.py
python -m pytest tests/test_solar_screened_isothermal_results.py -q
python scripts/run_solar_screened_galaxy_morphology.py
python -m pytest tests/test_solar_screened_galaxy_morphology_results.py -q
```

## Adaptive baryonic gravity-route result

A conservative route model now treats apparent halo structure as a possible
arrival pattern for the P0554 gravity excess generated by observed baryons.
The A0279 member-extent kernel transferred across derived maps, but failed at
unit strength on raw RX J2129 images. Its post-hoc `s_theta=f_route^2.5` bridge
was then locked and replayed on four other raw clusters. It recovered a missing
MACS1931 image root, yet improved the three mutually complete clusters by only
0.276%, retained 19.160-arcsecond absolute RMS, and was 1.886 times the limited
compact-halo validation error. The formula fails as a cluster-lensing solution;
the topology response motivates inverse launch-to-arrival reconstruction with
complete galaxy, ICL, and gas maps. See
[`docs/ADAPTIVE_ROUTE_RESULTS.md`](docs/ADAPTIVE_ROUTE_RESULTS.md) and
[`docs/CONSERVATIVE_GRAVITY_ROUTE_FIELD.md`](docs/CONSERVATIVE_GRAVITY_ROUTE_FIELD.md).

Reproduce the final replay and tests with:

```powershell
python scripts/run_adaptive_route_multicluster_raw.py
python -m pytest tests/test_adaptive_route_multicluster_raw.py tests/test_adaptive_route_results.py -q
```

## P0554 local parameter-impact result

A frozen 23-formula scan now isolates which small P0554 changes matter in each
domain. Radial accumulation shape is the largest galaxy lever, baryonic
transition-radius scaling is the largest radial cluster lever, weak
concentration/extent response is the largest RX J2129 topology lever, and
high-acceleration screen sharpness dominates the Solar boundary. A second
12-formula exact-refit stage found that `extent_leak=0.05` plus
`screen_scale=0.8` recovers all 18 raw held-out roots and improves the four
parent-complete systems by 3.23%, but still scores 16.383 arcsec, worsens SPARC,
and remains worse than the limited compact-halo validation comparator. This is
a mechanistic clue, not a promoted formula. See
[`docs/P0554_LOCAL_SENSITIVITY_RESULTS.md`](docs/P0554_LOCAL_SENSITIVITY_RESULTS.md).

Reproduce with:

```powershell
python scripts/run_p0554_local_cross_domain_sensitivity.py
python scripts/run_p0554_compensated_interactions.py
python -m pytest tests/test_p0554_local_cross_domain_sensitivity.py tests/test_p0554_compensated_interactions_results.py -q
```

The follow-up multi-scale experiment replaces the single low/high interval
with 89 central perturbations. It finds smooth galaxy and CLASH derivatives but
raw image-root bifurcations under several 1--2% changes. The mass-radius
exponent is the strongest stable CLASH lever but prefers opposite signs in RX
J2129 and the other four raw clusters; only screen location has a common
material fixed-geometry direction, and that RX direction does not survive an
ordinary geometry refit. See
[`docs/P0554_MULTISCALE_ELASTICITY_RESULTS.md`](docs/P0554_MULTISCALE_ELASTICITY_RESULTS.md).

```powershell
python scripts/run_p0554_multiscale_elasticity.py
python -m pytest tests/test_p0554_multiscale_elasticity.py -q
```

A separate 73-formula structural grid changes the algebra rather than the old
coefficients. The dynamics addition law is the largest galaxy and Solar lever;
the lensing addition law is the largest CLASH lever. Both fail universality:
the former pulls galaxies and clusters in opposite directions, while the
latter prefers opposite signs in RX J2129 and the other four raw clusters.
Only screen softness has a partially common fixed-geometry direction. See
[`docs/P0554_STRUCTURAL_MICROVARIATIONS_RESULTS.md`](docs/P0554_STRUCTURAL_MICROVARIATIONS_RESULTS.md).

```powershell
python scripts/run_p0554_structural_microvariations.py
python -m pytest tests/test_arc_invariants.py tests/test_p0554_structural_microvariations.py -q
```

The structural shortlist has now been replayed with ordinary lens geometry
refit. All seven apparent root recoveries disappear: every formula solves the
same 17/18 images. Dynamics softness 0.98 gives the largest matched continuous
gain (2.50%) but worsens SPARC. Lensing softness 0.98 improves CLASH, RX J2129,
and the other three complete clusters without changing galaxies or Mercury,
but leaves MACS1931 incomplete and slightly worsens the only directly matched
historical-validation system. See
[`docs/P0554_STRUCTURAL_EXACT_REFIT_RESULTS.md`](docs/P0554_STRUCTURAL_EXACT_REFIT_RESULTS.md).

```powershell
python scripts/run_p0554_structural_exact_refit.py
python -m pytest tests/test_p0554_structural_exact_refit.py -q
```

The two useful but incomplete effects have now been combined directly. A
softer photon-addition law improves continuous residuals but leaves one
observed-seed MACS1931 solution unresolved; the conservative A0279 angular
route recovers that solution but slightly worsens the four parent-complete
systems. Their combined parent recovers all 18/18 observed-seed roots and preserves the P0554 galaxy and Solar
scores, yet improves matched raw-lensing RMS by only 0.310% and remains 1.816
times the limited compact-halo validation error. Across seven small coordinates,
photon softness is the dominant continuous lever; route fraction, strength,
length, and source weighting mainly toggle caustic topology, while width and
extent have only small smooth effects. See
[`docs/P0554_ROUTE_SOFTNESS_INTERACTION_RESULTS.md`](docs/P0554_ROUTE_SOFTNESS_INTERACTION_RESULTS.md).

A frozen global root and caustic audit corrects the tempting “missing image”
interpretation. All 18 formulas have the three roots required by the observed
MACS1931 family, but the 11 observed-seed successes have five global roots and
the seven failures have three. The extra pair always lies near image 2c. A
branch-based source-to-caustic margin separates the two regimes perfectly in
this spent family, whereas the Jacobian at the observed coordinate is weak.
The extra companion is now an observational prediction/liability, not a model
promotion. See
[`docs/P0554_CAUSTIC_MARGIN_RESULTS.md`](docs/P0554_CAUSTIC_MARGIN_RESULTS.md).

That companion prediction has now been checked directly in the local CLASH
F160W mosaic. The five-root formulas predict opposite-parity companions with
0.917--1.251 times the anchor brightness and 1.782--10.859-arcsecond separation.
No published family-2 companion exists. After registering each model anchor to
observed image 2c, ten variants point to clean blank sky and one is contaminated
but unconfirmed. The route-induced extra pair is therefore disfavored, removing
the apparent topology advantage of the earlier 18/18 score. See
[`docs/P0554_MACS1931_COMPANION_AUDIT_RESULTS.md`](docs/P0554_MACS1931_COMPANION_AUDIT_RESULTS.md).

The multiplicity audit now spans all 27 source families in the five raw
clusters. No formula has exact published multiplicity everywhere. Route-only
lowers equal-family position RMS from 10.531 to 9.228 arcseconds, but raises
potentially observable surplus roots from eight to twelve. Its root-count
changes are confined to MACS1931 families 2 and 3, where it adds one pair to
each; the other 25 families are unchanged. This surplus direction survives
relative-magnification thresholds from 0.10 to 1.00. The next target is a
subcritical route strength that retains the positional gain without crossing
either MACS1931 caustic. See
[`docs/P0554_MULTIFAMILY_MULTIPLICITY_RESULTS.md`](docs/P0554_MULTIFAMILY_MULTIPLICITY_RESULTS.md).

The frozen route-amplitude continuation has now tested that target. The best
topology-preserving setting is eta=0.30, but it improves MACS1931 assigned
positions by only 0.468%. Family 3 creates an extra pair at eta=0.60; family 2
does so at full strength. The apparently large 24.855% full-strength gain is
therefore mostly reached after unwanted caustic crossings and raises the
potentially observable surplus-root count from two to six. Amplitude-only
tuning is not promoted. See
[`docs/P0554_SUBCRITICAL_ROUTE_SCAN_RESULTS.md`](docs/P0554_SUBCRITICAL_ROUTE_SCAN_RESULTS.md).

The selected eta=0.30 setting has also been transferred unchanged to the other
four raw clusters with paired exact geometry refits. It changes none of the 27
family root counts and preserves all held-out root recoveries. Its primary
equal-family RMS improves by only 0.412%, however, with two clusters improving
and two worsening. It passes a weak topology-safe transfer definition but fails
the frozen strong-transfer gates; the next useful lever is route localization,
not a larger global amplitude. See
[`docs/P0554_SUBCRITICAL_ROUTE_TRANSFER_RESULTS.md`](docs/P0554_SUBCRITICAL_ROUTE_TRANSFER_RESULTS.md).

Route localization has now been tested independently of route amplitude. A
200-kpc-softened direction computed from neighboring baryonic member galaxies
beats the original global-centroid direction after eight-start exact geometry
refits. Across 15 common complete transfer families it improves RMS by 0.745%
versus no route and 0.335% versus the global route, improves all four exact
held-out cluster scores, and changes none of 27 family root counts. It misses
the frozen 1% gate and is not promoted, but establishes local baryonic direction
as a more impactful lever than additional global strength. See
[`docs/P0554_ROUTE_LOCALIZATION_RESULTS.md`](docs/P0554_ROUTE_LOCALIZATION_RESULTS.md).

A frozen one-at-a-time follow-up varies the local route's smoothing scale,
distance falloff, and baryonic light weighting. The finite coherence scale is
the largest lever: 200 kpc is the bracketed aggregate optimum, although the
individual clusters prefer 100--250 kpc. A $1/R$ neighbor influence is 0.103%
worse than the inverse-square parent; changing the falloff exponent is the
weakest lever. Squaring the light weights gains 0.337% in aggregate but helps
only two of four primary clusters. No variant passes the predeclared exact-
follow-up rule. See
[`docs/P0554_LOCAL_NEIGHBOR_PARAMETER_RESULTS.md`](docs/P0554_LOCAL_NEIGHBOR_PARAMETER_RESULTS.md).

Registered continuous baryon proxies have now been tested as route directions.
The acquisition supplies F160W coverage at all 84 cataloged image coordinates
and 307.3 ks of Chandra exposure across the five clusters. Continuous stars,
masked/unmasked X-ray morphology, and their blends all remain better than no
route, but none beats the discrete member-only direction. This disfavors the
simple idea that gravity merely follows the smooth gradient of all visible
matter; the next inverse test must reconstruct paths from the assumed excess-
gravity arrival field back to individual baryonic launch points. See
[`docs/P0554_ALL_BARYON_ROUTE_RESULTS.md`](docs/P0554_ALL_BARYON_ROUTE_RESULTS.md).

```powershell
python scripts/run_p0554_route_softness_interaction.py
python -m pytest tests/test_p0554_route_softness_interaction.py -q
python scripts/run_p0554_caustic_margin.py --postprocess-only
python -m pytest tests/test_p0554_caustic_margin.py -q
python scripts/run_p0554_macs1931_companion_audit.py
python scripts/run_p0554_macs1931_relative_companion.py
python -m pytest tests/test_p0554_macs1931_companion_audit.py -q
python scripts/run_p0554_multifamily_multiplicity.py --postprocess-only
python -m pytest tests/test_p0554_multifamily_multiplicity.py -q
python scripts/run_p0554_subcritical_route_scan.py --postprocess-only
python -m pytest tests/test_p0554_subcritical_route_scan.py -q
python scripts/run_p0554_subcritical_route_transfer.py --postprocess-only
python -m pytest tests/test_p0554_subcritical_route_transfer.py -q
python scripts/run_p0554_route_localization_screen.py --postprocess-only
python scripts/run_p0554_local_neighbor_exact.py --postprocess-only
python scripts/run_p0554_local_neighbor_parameter_screen.py --postprocess-only
powershell -ExecutionPolicy Bypass -File scripts/download_p0554_all_baryon_route.ps1
python scripts/audit_p0554_all_baryon_route_inputs.py
python scripts/run_p0554_all_baryon_route_screen.py --postprocess-only
python -m pytest tests/test_route_template.py tests/test_baryon_morphology.py tests/test_p0554_route_localization_screen.py tests/test_p0554_local_neighbor_exact.py tests/test_p0554_local_neighbor_parameter_screen.py tests/test_p0554_all_baryon_route_inputs.py tests/test_p0554_all_baryon_route_screen.py -q
```

## Reproducibility

The squared-coherence survivor has now completed its independent SPARC
nuisance refit. Its primary outer RMSE is 10.586 km/s versus 10.348 for fixed
RAR and 10.385 for simple MOND; seven density geometries span
10.368--10.999 km/s. It passes the frozen galaxy-parity gates. The theory-level
target is now explicit: preserve MOND-level galaxy accuracy while predicting
raw cluster lensing with the same universal settings, and compare that zero-
per-object-gravity-parameter law against halo fits with object-specific mass
parameters. See
[`docs/UNIVERSAL_THEORY_SCORECARD.md`](docs/UNIVERSAL_THEORY_SCORECARD.md).
The new term is inactive in 101/131 galaxies; among the 30 active
galaxies it improves 12 and worsens 18. See
[`docs/SPARC_INDEPENDENT_NUISANCE_REFIT_RESULTS.md`](docs/SPARC_INDEPENDENT_NUISANCE_REFIT_RESULTS.md).
The complete 20-cluster derived-lensing comparison is reproducible with
`python scripts/compare_clash_lensing_models.py`; it records the frozen photon
closure, residual distributions, and complete-cluster bootstrap in
[`docs/LENSING_UNIVERSAL_COMPARISON_RESULTS.md`](docs/LENSING_UNIVERSAL_COMPARISON_RESULTS.md).

All CLI defaults are stored in [`configs/baseline.json`](configs/baseline.json).
Each result summary records the seed, data hash, device, cuts, parameter count,
and train/holdout metrics. The source data are never edited in place.

The current cross-stage sensitivity synthesis is reproducible with
`python scripts/run_p0612_cross_stage_parameter_impact.py`. It normalizes each
parameter response only within its own observable, keeps harmful/topology-losing
responses distinct from improvement, and identifies width, path length,
endpoint residence, saturation, and universal strength as the bounded next
coordinates. See
[`docs/P0612_CROSS_STAGE_PARAMETER_IMPACT_RESULTS.md`](docs/P0612_CROSS_STAGE_PARAMETER_IMPACT_RESULTS.md).

The first atlas-chosen factorial is reproducible with
`python scripts/run_p0613_bounded_endpoint_cross_domain.py`. It holds endpoint
residence fixed and crosses only baryonic-size width, universal routed strength,
and smooth saturation on exact raw lens roots, 131 SPARC galaxies, and the
Solar point-source null. See
[`docs/P0613_BOUNDED_ENDPOINT_CROSS_DOMAIN_RESULTS.md`](docs/P0613_BOUNDED_ENDPOINT_CROSS_DOMAIN_RESULTS.md).

The corresponding same-equation accounting audit is reproducible with
`python scripts/run_p0614_composite_formula_audit.py`. It shows explicitly that
P0554 carries the galaxy result, the endpoint layer mainly changes cluster
topology, and the composite fails its compact-halo and RXJ2129 route-transfer
comparisons despite passing Solar proxies. See
[`docs/P0614_COMPOSITE_FORMULA_AUDIT_RESULTS.md`](docs/P0614_COMPOSITE_FORMULA_AUDIT_RESULTS.md).

The parameter-reducing follow-up is reproducible with
`python scripts/run_p0615_self_coupled_quadrupole_route.py`. It derives route
fraction from the P0554 scalar excess and angular amplitude from a baryonic
quadrupole invariant, fitting no new strength on any object. See
[`docs/P0615_SELF_COUPLED_QUADRUPOLE_ROUTE_RESULTS.md`](docs/P0615_SELF_COUPLED_QUADRUPOLE_ROUTE_RESULTS.md).

Its chronologically frozen A383/MS2137 transfer is reproducible with
`python scripts/run_p0616_frozen_self_coupled_transfer.py`. It derives both
cluster amplitudes from baryonic data before paired full geometry refits and
fits no gravity parameter per system. See
[`docs/P0616_FROZEN_SELF_COUPLED_TRANSFER_RESULTS.md`](docs/P0616_FROZEN_SELF_COUPLED_TRANSFER_RESULTS.md).

The follow-up support and phase atlas is reproducible with
`python scripts/run_p0617_self_coupled_support_phase_atlas.py`. It freezes the
P0615 amplitude and varies only baryon-derived route width, return length, and
whether endpoints may cross the baryonic center. See
[`docs/P0617_SELF_COUPLED_SUPPORT_PHASE_ATLAS_RESULTS.md`](docs/P0617_SELF_COUPLED_SUPPORT_PHASE_ATLAS_RESULTS.md).

The universal angular-phase diagnostic is reproducible with
`python scripts/run_p0618_universal_route_phase.py`. It tests radial, oblique,
and tangential-like rotations shared by every cluster, while explicitly
forbidding a per-cluster phase choice. See
[`docs/P0618_UNIVERSAL_ROUTE_PHASE_RESULTS.md`](docs/P0618_UNIVERSAL_ROUTE_PHASE_RESULTS.md).

The frozen +90-degree full-refit transfer is reproducible with
`python scripts/run_p0619_frozen_tangential_transfer.py`. It evaluates the
shared tangential-like rule on A383 and MS2137 with no per-system gravity
settings. See
[`docs/P0619_FROZEN_TANGENTIAL_TRANSFER_RESULTS.md`](docs/P0619_FROZEN_TANGENTIAL_TRANSFER_RESULTS.md).

The complete P0612-P0619 parameter-impact synthesis is reproducible with
`python scripts/build_p0620_parameter_impact_synthesis.py`. It separates the
most recurrent coordinate (width), the largest new lens response (phase), and
the most explosive but destructive coordinate (route strength), then records
the current galaxy, cluster, and Solar scorecard. See
[`docs/P0620_PARAMETER_IMPACT_SYNTHESIS.md`](docs/P0620_PARAMETER_IMPACT_SYNTHESIS.md).

The corresponding equation-by-equation prior-art audit and plain-language
first-principles explanation are in
[`docs/P0621_PRIOR_ART_AND_FIRST_PRINCIPLES.md`](docs/P0621_PRIOR_ART_AND_FIRST_PRINCIPLES.md).
It distinguishes the exact P0620 test construction from QUMOND, refracted
gravity, EMOND, relativistic MOND, gravitational polarization, and standard
lens shear, and states the claim boundary required before treating the ansatz
as a physical theory.

The broad SigmaGravity validation pattern has now been adapted into a stricter
regime-diagnostic suite with `python
scripts/run_p0622_comprehensive_regime_diagnostics.py`. It reproduces the 131-
galaxy outer holdout, splits results by ten physical/morphological dimensions,
tests 32 two-way interaction bins with bootstrap intervals, audits five-cluster
phase influence with leave-one-system-out scores, and keeps raw roots, derived
lens products, Solar proxies, inherited nulls, and synthetic invariants in
separate evidence classes. The key new findings are a dwarf-to-giant velocity-
bias reversal and an RXJ2129-dominated cluster mean: the shared +90-degree route
falls from +1.685% mean improvement to +0.046% when RXJ2129 is omitted. See
[`docs/P0622_COMPREHENSIVE_REGIME_DIAGNOSTICS.md`](docs/P0622_COMPREHENSIVE_REGIME_DIAGNOSTICS.md).

```powershell
python scripts/run_p0622_validation_suite.py
```

## Observation-matched simulator correction (P0630-P0631)

P0630 is a gravity-law forward laboratory, not by itself proof that its
parametric particle scenes look like real galaxies. P0631 adds the missing
observation-matched layer using official SPARC radial photometry and bulge/disk
decompositions. It produces projected light maps, line-of-sight velocity maps,
and deterministic luminosity-tracer scenes for the same 131-galaxy sample.

The full replica run passes its frozen reconstruction gates, including all 23
whole-galaxy holdouts: median angular-profile error is 0.0000308 dex, median
finite-camera rotation loss is 0.221 km/s, and median total-light error is
0.367%. Observed speed is deliberately an input only in replica mode. The blind
renderer requires an explicit theory-predicted curve and has no observed-speed
fallback.

This validates the simulator's radial observation layer; it does not validate
the gravity formula. The current P0630 transport law remains worse than fixed
RAR on held-back galaxy speeds and worse than object-specific compact halos on
raw cluster image positions. See
[`docs/P0630_SYNTHETIC_UNIVERSE_RESULTS.md`](docs/P0630_SYNTHETIC_UNIVERSE_RESULTS.md)
and
[`docs/P0631_OBSERVATION_MATCHED_REPLICA_RESULTS.md`](docs/P0631_OBSERVATION_MATCHED_REPLICA_RESULTS.md).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0631_sparc_replica_data.ps1
$env:PYTHONPATH='src'
python scripts/run_p0631_observation_matched_replicas.py
python -m pytest tests/test_galaxy_replica.py tests/test_p0631_observation_matched_replicas.py -q
```

## Published MOND/RAR simulator calibration (P0632)

The simulator now reproduces the Li et al. (2018) individual-SPARC RAR/MOND
analysis from the authors' source table: all 175 fits, the 153-galaxy and 2,694-
point scatter sample, 0.057161 dex versus 0.057 dex published after nuisance
refits, and a 0.999976 correlation with the published per-galaxy reduced
chi-square values. With catalog distance/inclination and fixed stellar
mass-to-light ratios, it independently recovers 0.132766 dex versus the
published approximately 0.13 dex.

On the chronologically frozen 23 whole-galaxy holdouts, the fixed published
RAR/MOND equation scores 23.326 km/s, simple-μ MOND 23.800 km/s, standard-μ
MOND 22.715 km/s, and Newtonian baryons 52.180 km/s. The publication
replication and blind holdout are kept separate because the former optimizes
three or four nuisance quantities for each galaxy. See
[`docs/P0632_PUBLISHED_MOND_REPLICATION_RESULTS.md`](docs/P0632_PUBLISHED_MOND_REPLICATION_RESULTS.md).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0632_li2018_benchmark.ps1
$env:PYTHONPATH='src'
python scripts/run_p0632_published_mond_replication.py
python -m pytest tests/test_mond_benchmark.py tests/test_p0632_published_mond_replication.py -q
```

## Real Cartesian field solvers (P0634)

P0634 implements isolated three-dimensional Newtonian Poisson, two-step
QUMOND, and nonlinear finite-volume AQUAL solvers on one shared baryonic grid.
They pass the twelve numerical gates frozen in P0633: manufactured-solution
grid convergence, Plummer forces, equation residuals, spherical MOND limits,
and the high-acceleration return to Newtonian gravity. This validates the
solver machinery, not an observational theory or a relativistic photon law.
See [`docs/P0634_REAL_FIELD_SOLVER_VALIDATION.md`](docs/P0634_REAL_FIELD_SOLVER_VALIDATION.md).

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0634_field_solver_validation.py
python -m pytest tests/test_field_solvers.py tests/test_p0634_field_solver_validation.py -q
```

## First real 2D baryonic map (P0635)

The validated field equations now run on the official LITTLE THINGS DDO154 H I
moment-0 map and optical stellar morphology. The gridded map retains 99.14% of
the raw gas mass and is 94.44% gas by baryonic mass. On the deliberately spent
DDO154 rotation curve, Newtonian/QUMOND/AQUAL score 25.03/3.94/3.60 km/s RMSE,
while algebraic simple MOND scores 2.92 km/s. Thickness and axisymmetry
ablations show that the remaining full-field difference is not a registration
artifact. See
[`docs/P0635_REAL_2D_DDO154_COMMISSIONING.md`](docs/P0635_REAL_2D_DDO154_COMMISSIONING.md).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0635_ddo154_maps.ps1
$env:PYTHONPATH='src'
python scripts/run_p0635_ddo154_map_commissioning.py
python scripts/run_p0635_map_geometry_sensitivity.py
python -m pytest tests/test_galaxy_maps.py tests/test_p0635_ddo154_map_commissioning.py -q
```

The eventual public researcher service—real/synthetic object catalog, safe
formula submissions, reproducible batch scoring, and a Vercel front end backed
by asynchronous field-solver workers—is specified in
[`docs/PUBLIC_SIMULATOR_API_PLAN.md`](docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Sealed validation baryons acquired (P0636)

All 13 P0633 LITTLE THINGS galaxies now have content-addressed H I moment-0,
B-band, V-band, and UBV-calibration inputs: 52 permitted products totaling
300,811,128 bytes. Every radio beam and FITS image passes ingestion checks.
Kinematic cubes, moment-1/2 maps, circular velocities, and target scores remain
sealed. See
[`docs/P0636_LITTLE_THINGS_BARYON_ACQUISITION.md`](docs/P0636_LITTLE_THINGS_BARYON_ACQUISITION.md).

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0636_little_things_baryons.ps1
$env:PYTHONPATH='src'
python scripts/audit_p0636_little_things_baryons.py
python -m pytest tests/test_p0636_little_things_baryon_acquisition.py -q
```

## Formula-neutral 2D/3D simulator commissioning (P0720-P0730)

The simulator now has a gravity-independent inverse/forward galaxy path and a
shared asynchronous field API. Registered gas and stellar maps are converted
into content-hashed structural parameters, replayed as 2D maps and explicitly
prior-based 3D density ensembles, and passed to confirmed typed field
manifests. The model and the galaxy generator remain separate.

The v0.22 ensemble milestone now retains every requested 2D and 3D realization
instead of saving only the first vertical draw. Bounded seeded priors can vary
mass, radial and angular structure, local features, distance, thin-map
inclination deprojection, warp, thickness, and a disclosed co-spatial unseen
baryon fraction. Jobs emit percentile maps, draw tables, hashes, and exact
projection diagnostics while continuing to use zero gravity parameters and no
velocity target. These are explicit prior ensembles, not likelihood-derived
posteriors. See
[`docs/BARYONIC_UNCERTAINTY_ENSEMBLE_MILESTONE.md`](docs/BARYONIC_UNCERTAINTY_ENSEMBLE_MILESTONE.md).

Version 0.26 connects axisymmetric fields to circular-speed curves and
resolved line-of-sight velocity maps without a Cartesian proxy. The adapter
uses `v_c^2=r(-a_r)` at a declared midplane, retains inclination, handedness,
masks, beam, uncertainty and nuisance accounting, and rejects coordinate
controls that have no cylindrical meaning. The local asynchronous known-answer
case rehashes ten artifacts with zero per-object gravity parameters and
`4.22e-15 m/s` rotation RMSE. See
[`docs/AXISYMMETRIC_GALAXY_OBSERVATION_ADAPTER_MILESTONE.md`](docs/AXISYMMETRIC_GALAXY_OBSERVATION_ADAPTER_MILESTONE.md).

Version 0.25 adds a formula-independent axisymmetric cylindrical `(r,z)`
finite-volume worker for stationary scalar elliptic laws. The symmetry axis is
handled by its regular zero-radial-flux limit, manufactured Bessel fields show
second-order convergence, variable coefficients execute through the same
expression path, and the immutable job identity binds axis order and origin.
This makes disk-versus-bulge field tests materially more faithful, but does not
yet provide cylindrical nonlocal kernels or axisymmetric rotation/lensing
observation adapters. See
[`docs/AXISYMMETRIC_FIELD_WORKER_MILESTONE.md`](docs/AXISYMMETRIC_FIELD_WORKER_MILESTONE.md)
and [`docs/FULL_PLATFORM_COMPLETION_AUDIT.md`](docs/FULL_PLATFORM_COMPLETION_AUDIT.md).

Version 0.24 adds the first gravity-independent baryonic image likelihood.
Gas and stellar surface-density uncertainties can weight the generated surface
draws before any gravity model or velocity/lensing target is used. Those
immutable weights propagate into 2D/3D batches, weighted score summaries, and
per-radius prediction bands. The real two-draw DDO101 commissioning run
collapsed to ESS 1.0 and is explicitly not a credible interval; this exposes
the need for real covariance/PSF products and adaptive sampling. See
[`docs/BARYONIC_IMAGE_CONDITIONING_MILESTONE.md`](docs/BARYONIC_IMAGE_CONDITIONING_MILESTONE.md).

P0723 ran Newtonian Poisson, AQUAL, QUMOND, and Refracted Gravity through that
same path on all 13 registered galaxies. All 52 solves converged, all 161
circular-speed points per model were scored, downloaded artifacts rehashed
successfully, and no per-galaxy gravity parameter was used. QUMOND had the
lowest equal-galaxy RMSE (`12.486 km/s`), followed by AQUAL (`13.131 km/s`),
Refracted Gravity (`14.439 km/s`), and Newtonian baryons (`23.154 km/s`). The
aggregate reduced chi-square remains poor for every manifest, so this is a
formula-neutral engineering result rather than evidence that any tested model
is sufficient. See
[`docs/P0723_FORMULA_NEUTRAL_RESOLVED_COMPARATOR_RESULTS.md`](docs/P0723_FORMULA_NEUTRAL_RESOLVED_COMPARATOR_RESULTS.md)
and
[`docs/PUBLIC_SIMULATOR_2D_3D_GENERATOR_ROADMAP.md`](docs/PUBLIC_SIMULATOR_2D_3D_GENERATOR_ROADMAP.md).

P0724 then ran 96 frozen grid, box, and vertical-prior sensitivity jobs on four
spent sentinel galaxies. Ninety-four converged. Expanded boundaries and two
independent vertical draws passed; the coarse AQUAL fit changed by 77.5%, and
fine-grid AQUAL retained nonconvergence diagnostics for DDO53 and DDO101.
Incomplete rows are excluded from ranks. This makes nonlinear solver
hardening--without changing the physics--the next gate. See
[`docs/P0724_GRID_BOX_VERTICAL_SENSITIVITY_RESULTS.md`](docs/P0724_GRID_BOX_VERTICAL_SENSITIVITY_RESULTS.md).

P0725 then froze six universal solver variants on the two failed fine-grid
inputs. A generic unit-coefficient warm start with damping `0.20` converged
both at the strict `1e-8` residual target, while every other real-system
variant failed. The stage remains a failed selection gate because a second
independent successful method is required before choosing a production
default. See
[`docs/P0725_AQUAL_SOLVER_ROBUSTNESS_RESULTS.md`](docs/P0725_AQUAL_SOLVER_ROBUSTNESS_RESULTS.md).

P0726 added a direct Newton--Krylov residual solver. Newton--GMRES independently
reproduced the converged DDO53 Picard potential, acceleration, and circular
speeds to better than `5e-9` normalized RMS, but no direct-from-linearized
variant converged DDO101. The universal cross-method gate therefore remains
open. See
[`docs/P0726_INDEPENDENT_NONLINEAR_CROSSCHECK_RESULTS.md`](docs/P0726_INDEPENDENT_NONLINEAR_CROSSCHECK_RESULTS.md).

P0727 closed that numerical gate with a frozen hybrid: 40 universal Picard
warm-up steps at damping `0.20`, followed by Newton--GMRES/Armijo. It converged
both difficult fine-grid fields and reproduced the independent P0725 Picard
potential, acceleration, and circular-speed results to better than `5.4e-8`
normalized RMS. The 20-step hybrid failed DDO101 and the 80-step hybrid also
passed, so 40 is the smallest preregistered success. This validates the
discrete roots, not AQUAL as physical theory. See
[`docs/P0727_HYBRID_NONLINEAR_CROSSCHECK_RESULTS.md`](docs/P0727_HYBRID_NONLINEAR_CROSSCHECK_RESULTS.md).

P0728 applied that selected solver to all four fine-grid sentinels. DDO53,
DDO101, and DDO50 converged and independently agreed with prior Picard fields,
but NGC1569 missed the `1e-8` relative-update tolerance at `1.76e-8` despite a
`2.40e-10` equation residual. The AQUAL row remains incomplete and is excluded
from ranking and stability classification. See
[`docs/P0728_COMPLETE_FINE_GRID_AQUAL_RESULTS.md`](docs/P0728_COMPLETE_FINE_GRID_AQUAL_RESULTS.md).

P0729 then applied the already-preregistered P0727 80-step hybrid universally.
All four fine-grid fields converged and independently matched Picard
references. AQUAL's complete equal-galaxy RMSE is `21.636 km/s`, ranking behind
QUMOND (`16.502`) and the Refracted Gravity fixture (`18.982`) but ahead of
Newtonian baryons (`24.420`). The reconstructed fine-grid stability gates pass,
although NGC1569's `24.09%` prediction change is close to the scenario limit.
See
[`docs/P0729_QUALIFIED_80STEP_FINE_GRID_AQUAL_RESULTS.md`](docs/P0729_QUALIFIED_80STEP_FINE_GRID_AQUAL_RESULTS.md).

P0730 adds a formula-neutral resolved line-of-sight velocity-field target to
the same local field and batch APIs. Any compatible 2D or 3D manifest can now
project its declared massive-tracer acceleration through explicit disk
coordinates, inclination, handedness, masks, uncertainties, intensity weights,
and a beam kernel. Manufactured known-answer, worker-artifact, preflight, and
batch-aggregation tests pass.
See
[`docs/P0730_RESOLVED_VELOCITY_FIELD_API_MILESTONE.md`](docs/P0730_RESOLVED_VELOCITY_FIELD_API_MILESTONE.md).

P0731 then commissioned that same theory-neutral target on real LITTLE THINGS
moment maps. Four unchanged P0723 field manifests were evaluated on all 13
galaxies, giving 52 resolved-map scores with zero per-galaxy gravity
parameters. The production and independent frozen-P0712 implementations agree
to `1.96e-10 m/s` maximum prediction RMS, `1.79e-8 m/s` maximum absolute pixel
difference, and `5.82e-11 m/s` maximum score difference. All immutable field
and observation hashes validate. QUMOND has the lowest spent-sample aggregate
RMSE (`17.764 km/s`), but it wins only two individual galaxies; Newtonian wins
five, AQUAL three, and the Refracted Gravity fixture three. This validates the
adapter, not a gravity theory: the sample is spent and dwarf-only, and circular
equilibrium omits pressure support, warps, and non-circular motion. See
[`docs/P0731_REAL_VELOCITY_FIELD_ADAPTER_PARITY_RESULTS.md`](docs/P0731_REAL_VELOCITY_FIELD_ADAPTER_PARITY_RESULTS.md).

P0732 separates observation evaluation from the expensive field solve. A new
content-addressed asynchronous job consumes a successful immutable 2D/3D field
artifact plus a separate observation upload and full target declaration. Its
2D circular-curve and 3D velocity-map score and prediction artifacts are byte
identical to integrated field-job evaluation, while the real HTTP acceptance
records zero field-solver calls and zero added gravity parameters. Duplicate
submissions reuse one job identity; changed data or targets create a new
evaluation without changing the source field. Cancellation, restart recovery,
and downloaded-artifact rehashing pass. See
[`docs/P0732_DECOUPLED_OBSERVATION_EVALUATION_MILESTONE.md`](docs/P0732_DECOUPLED_OBSERVATION_EVALUATION_MILESTONE.md).

P0733 composes that independent evaluation boundary into the multi-system
batch service. Each system now creates or reuses an observation-independent
field child first and, only after a successful field solve, creates or reuses
one separately hashed observation child. Field and measured-observation arrays
may use different upload IDs. The frozen real-HTTP run proves that changing an
uncertainty preserves the field job while changing the observation job; field
rejection, observation rejection, cancellation, restart recovery, artifact
hashes, and zero added gravity parameters also pass. See
[`docs/P0733_COMPOSED_BATCH_OBSERVATION_JOBS_MILESTONE.md`](docs/P0733_COMPOSED_BATCH_OBSERVATION_JOBS_MILESTONE.md).

```powershell
python scripts/run_p0723_formula_neutral_api_comparators.py `
  --base-url http://127.0.0.1:4173

python scripts/run_p0724_grid_box_vertical_sensitivity.py `
  --base-url http://127.0.0.1:4189

python scripts/run_p0725_aqual_solver_robustness.py

python scripts/run_p0726_independent_nonlinear_crosscheck.py

python scripts/run_p0727_hybrid_nonlinear_crosscheck.py

python scripts/run_p0728_complete_fine_grid_aqual.py

python scripts/run_p0729_qualified_80step_fine_grid_aqual.py

python scripts/run_p0731_real_velocity_field_adapter_parity.py

python scripts/run_p0732_decoupled_observation_evaluation.py

python scripts/run_p0733_composed_batch_observation_jobs.py
```
