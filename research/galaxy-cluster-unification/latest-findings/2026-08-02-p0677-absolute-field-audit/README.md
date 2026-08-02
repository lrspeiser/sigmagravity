# P0677 absolute-field research snapshot

Frozen: 2026-08-02

## Bottom line

The project now has a real three-dimensional divergence-form field solver,
registered baryonic maps, zero-slip photon integration, exact raw-lens root
tests, and strict protocol-before-score controls. That infrastructure is a
substantial result. The current physical candidate is **not** a successful
replacement for dark matter or MOND.

The most promising empirical formula remains the locked
RAR-plus-coherence/refracted-gravity diagnostic, which reached `1.064 arcsec`
on seven spent-heldout RX J2129 images. It is useful evidence that a universal
low-acceleration/environment response can fit both domains, but it inherits an
RAR-like galaxy sector and does not supply one elegant field equation.

The most promising first-principles clue is narrower: a parameter-free
compound path statistic separates the registered domains by more than four
orders of magnitude. Turning that statistic into the tested constitutive
tensors does not create the missing cluster lens topology.

## Latest equation chain

The elementary, baryon-derived route probability is

\[
\epsilon=\sqrt{1-e^{-(D^2+Q^2)}}
{a_0\over a_0+|g_N|}C_\perp,
\qquad N=\left({\ell\over L_c}\right)^2,
\]

and repeated coherent opportunities give

\[
\sigma=1-(1-\epsilon)^N.
\]

The first tensor field tested

\[
\nabla\!\cdot\left[\mu(I-\sigma\hat h\hat h)\nabla\Phi\right]
=4\pi G\rho_b.
\]

It suppresses mobility along `h`. The opposite, waveguide-like orientation was

\[
\nabla\!\cdot\left[
\mu\{(1-\sigma)I+\sigma\hat h\hat h\}\nabla\Phi
\right]=4\pi G\rho_b,
\]

with one open route eigenmode and two suppressed transverse modes. The only
stronger version allowed without fitting a power used `(1-sigma)^2`, because
there are exactly two transverse spatial dimensions.

## Frozen outcomes

| Stage | Frozen outcome | Meaning |
|---|---:|---|
| P0673 compound activation | pass; galaxy/cluster medians `5.33e-5 / 0.556` | Geometry can activate strongly in clusters while remaining nearly null in galaxies. |
| P0674 compound field | pass; `5.78%` vector change, `0.951x` scalar RMS | The nonlinear equation converges, but the original orientation weakens the field. |
| P0675 raw topology | fail; `17.83 arcsec` heldout, 7/7 missing multiplicity | Large activation still gives one root per family, one parity, and no critical curves. |
| P0676 transverse confinement | fail advancement; `1.151x` scalar RMS | The direction is better, but the one-pass response misses the frozen 20% gates. |
| P0677 dual transverse survival | fail advancement; `1.303x` scalar RMS | Even the dimensionally fixed square is far below the frozen `2x/100%` requirement. |
| Compact-halo comparator | `2.536 arcsec` spent-heldout | Still much better than the absolute baryonic field, despite being only a limited halo model. |
| Published multi-halo reference | `0.29 arcsec` | Context only; not a same-parameter-count comparison. |

The P0675 compound/compact-halo heldout error ratio is `7.03`. The current
absolute law is therefore not close to a conventional cluster solution.

## What has survived

1. **One universal activation is possible.** P0673 adds no post-P0659 constant
   and no per-object setting, yet distinguishes 13 registered galaxies from
   four registered clusters under all frozen mass sensitivities.
2. **The equations are numerically real.** Scalar and tensor AQUAL converge on
   the same physical mass cube and boundary; deflections are integrated from
   the gradient field and remain curl-free.
3. **Solar screening is not the hard part.** Earlier acceleration-gated
   operators are negligible under Solar accelerations and satisfy the tested
   Cassini/Mercury proxies.
4. **Angular geometry alone is insufficient.** P0568, P0672, P0675, P0676,
   and P0677 independently point to radial strength/effective baryon extent as
   the dominant unresolved coordinate.
5. **Inverse maps contain a real candidate clue, not a prediction.** P0567
   finds `95.8%` local positive-tensor feasibility and typical required
   anisotropy of `1.56:1`, while its baryon-only forward compression misses its
   transfer gate. Effective extent (`75-125 kpc`) dominates orientation.

## What is retired

- `mu(I-sigma h h)` as a solution to cluster topology;
- transverse-confinement powers beyond the fixed two-dimensional survival law;
- another member-only angular orientation parameter;
- any claim that a coefficient-level galaxy/cluster separation is evidence of
  successful lensing; and
- opening P0633/P0640 sealed outcomes before a spent field passes topology and
  resolution robustness.

## Next scientific stages and concrete outcomes

1. **Spent inverse monopole decomposition.** On RX J2129 and existing inverse
   cluster maps, subtract the absolute baryonic scalar field from the compact
   comparator and measure the required radial boost, effective extent,
   quadrupole, curl, and alignment with stars, gas, acceleration, potential,
   and tidal path statistics. Outcome: a table of predictable versus
   target-derived components; no new candidate score.
2. **Baryon-only law generation.** Admit only formulas whose amplitude and
   spatial scale are computed from measured stellar+gas distributions and
   global dimensionless invariants. Outcome: one formula frozen before another
   raw score, with no cluster amplitude and no arbitrary confinement power.
3. **Spent topology gate.** Require every RX J2129 family to have acceptable
   multiplicity, both parities, critical curves, heldout RMS under `3 arcsec`,
   and no nuisance parameter at a bound. Failure returns to stage 1; thresholds
   do not move.
4. **Resolution and baryon uncertainty.** A topology survivor must reproduce
   its sign and root count on at least three grids and fixed stellar/gas mass
   scenarios. Failure retires the discretization-dependent law.
5. **Open the frozen external sample once.** Run P0633 galaxy kinematics and
   P0640 multi-cluster raw lensing with one global parameter vector. Require
   MOND-competitive galaxy errors, compact-halo-competitive cluster topology,
   and the already frozen Solar limits. No per-object gravity settings.
6. **Prior-art/covariant audit.** Only a predictive survivor earns a proposed
   action, stress-energy conservation analysis, stability/causality study, and
   a new comparison with refracted gravity, BIMOND/TeVeS-like theories,
   gravitational polarization, and nonlocal gravity.

## Researcher simulator and public API

The simulator should become the reference implementation, not a separate demo.
The delivery path is:

1. serialize real/synthetic `GalaxyMap`, `ClusterMap`, `Formula`, `RunSpec`, and
   `RunResult` objects with units, provenance, licenses, hashes, and sealed/open
   state;
2. expose deterministic local calls through FastAPI and generate OpenAPI plus a
   Python SDK;
3. accept formulas through a dimension-aware, allowlisted expression language,
   never `eval` or arbitrary Python in the main service;
4. provide catalog calls for named real galaxies/clusters and seeded synthetic
   creation calls;
5. enqueue galaxy, cluster, Solar, topology, and batch evaluations as immutable
   asynchronous jobs with comparator and parameter-accounting outputs;
6. deploy the web application and thin authenticated gateway on Vercel;
7. run CPU-heavy Poisson/AQUAL/QUMOND and root searches in versioned Cloud Run
   Jobs or Modal workers, storing artifacts in content-addressed object storage;
8. reproduce selected local pass **and failure** fixture hashes in staging;
9. add a network-disabled arbitrary-code sandbox only after threat modeling;
   and
10. invite external researchers to submit formulas, call specific real or
    synthetic objects, download predictions, and cite permanent run manifests.

The detailed endpoints, request example, security model, deployment split, and
launch acceptance criteria are in
[`docs/PUBLIC_SIMULATOR_API_PLAN.md`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md).

## Canonical evidence

- [`P0673 results`](../../docs/P0673_COMPOUND_PATH_ACTIVATION_RESULTS.md)
- [`P0674 results`](../../docs/P0674_SPENT_RXJ2129_COMPOUND_3D_FIELD_RESULTS.md)
- [`P0675 results`](../../docs/P0675_SPENT_RXJ2129_COMPOUND_RAW_TOPOLOGY_RESULTS.md)
- [`P0676 results`](../../docs/P0676_SPENT_RXJ2129_TRANSVERSE_CONFINEMENT_FIELD_RESULTS.md)
- [`P0677 results`](../../docs/P0677_SPENT_RXJ2129_DUAL_TRANSVERSE_SURVIVAL_FIELD_RESULTS.md)
- [`P0567 inverse tensor result`](../../docs/P0567_BARYON_FLUX_TENSOR_BACKTRACK_RESULTS.md)
- [`P0568 forward tensor result`](../../docs/P0568_BARYON_ONLY_TENSOR_FORWARD_RESULTS.md)
- [`public simulator/API plan`](../../docs/PUBLIC_SIMULATOR_API_PLAN.md)
