# P0715-P0718 lensing structure and transfer results

## Bottom line

The numerical lensing engine is now trustworthy enough to distinguish a
physics failure from an axis or root-search failure. The present formulas do
not pass that test.

- P0715 makes the sky contract explicit: arrays are `[north, east]` and vector
  components are `(alpha_east, alpha_north)`. It reproduces every archived
  P0714 source, root count, and finite RMS to numerical precision in legacy
  mode. Its production root search returns the same root counts at 81, 161,
  and 241 grid points for all 36 real family/model cases.
- P0716 shows why the candidate produces one root. At the 65 observed arcs it
  has median deficits of `0.324` in convergence and `0.145` in shear, giving a
  `0.467` minimum-Jacobian-eigenvalue gap. Zero candidate arcs are locally
  near-critical versus `52.3%` in the compact-halo map. Candidate and halo
  shear magnitudes are uncorrelated (`r=-0.067`), with a `17.7 degree` median
  shear-axis error.
- P0717 tests the one-parameter screened metric contrast
  `W=Phi_N+q(Phi_AQUAL-Phi_N)` and related controls. AQUAL's Hessian fit gives
  `q=10.56` on AS295 and `10.08` on PLCKG287, a promising `4.6%` parameter
  disagreement, while automatically returning to the Newtonian Solar limit.
  Raw roots and image positions do not transfer: no form passes the completeness,
  topology, finite-RMS, and compact-halo-relative gates.
- P0718 tests the distinct nonlinear ordering hypothesis
  `W=Phi_N+q[(Phi_AQUAL,total-Phi_N,total)+sum_i(Phi_AQUAL,i-Phi_N,i)]` on 125
  and 138 photometrically selected member galaxies. Its fitted values differ
  by only `6.1%`, and PLCKG287 root completeness rises to `0.851`. AS295 reaches
  only `0.556`, while median image RMS remains `13.47x` and `20.82x` the
  compact-halo comparator in the two transfer directions. It is rejected.

The most promising surviving clue is therefore not a finished formula: the
order in which nonlinear component fields are combined strongly affects image
multiplicity, but the current construction integrates to the wrong global
source mapping.

## What P0715 fixed

The earlier frozen cluster maps were stored as image rows north and image
columns east, but P0708 called the two components x and y. P0715 preserves the
frozen output exactly and introduces a separate coordinate-safe interface.
It measures, in one basis:

- ray-shooted source position;
- the full lens Jacobian;
- convergence, both shear components, rotation, determinant, and eigenvalues;
- global roots and absolute magnification;
- critical-curve cells and point-cloud distances; and
- minimum-cost observed-image/root assignment.

Analytic checks cover an asymmetric affine lens, an SIS two-image solution,
the Einstein critical ring, source profiling, root-density stability, and the
north/east photon integration wrapper. The production root search uses the
union of 81, 161, and 241 grids because the archived compact-halo maps contain
narrow complementary basins at those phases. This is an explicit numerical
floor, not a hidden score adjustment.

## What the local field is missing

For a nearly curl-free lens, the smaller Jacobian eigenvalue is approximately

`lambda_min = 1 - kappa - |gamma|`.

Multiple images appear when this eigenvalue crosses zero. P0716 finds a median
candidate value of `0.434` versus `0.049` for the compact-halo map. The
difference is almost exactly the sum of missing convergence and missing shear.

The split is not the same in both clusters. AS295 already has roughly the
needed median convergence and is mainly missing shear. PLCKG287 lacks both.
That is why another universal radial multiplier is structurally incapable of
repairing both systems. It can make the bending stronger but cannot put the
correctly oriented folds and caustics around the arcs.

## Screened contrast test

P0717 retains the two-potential metric interpretation:

- massive tracers respond to `Psi`, which stays on the frozen RAR/coherent
  galaxy law; and
- photons respond to the Weyl potential `W=(Psi+Phi)/2`.

The tested correction is

`W = Phi_base + q (Phi_nonlinear - Phi_N)`.

The contrast tends to zero as `a0/g` in high acceleration. The Solar-limb PPN
slip proxies are about `1e-11`, Mercury proxies about `1e-7`, and the matter
potential is unchanged. This is why the form is more defensible than simply
multiplying gravity everywhere by six or ten.

The AQUAL and QUMOND values transfer reasonably, but local-Hessian agreement
does not imply a correct global potential. On PLCKG287, the QUMOND source-fit
row recovers `74.5%` of observed images, close to the archived glafic map's
`76.6%`, yet its median finite image RMS is about `19.4 arcsec` rather than
`2.5 arcsec`. In the reverse transfer, completeness falls to `44.4%`.

## Resolution and nonlinear summation order

The P0708 nonlinear maps contain only 65 cells across 900 kpc (`14.06 kpc` per
cell). P0718 therefore adds an exact zero-padded FFT thin-lens solver and
reconstructs the central baryon maps on 257 cells (`3.52 kpc` per cell).
This changes median arc deflections by `0.674 arcsec` on AS295 and `1.004
arcsec` on PLCKG287, but it does not repair cross-cluster source consistency.
Resolution is relevant, not decisive.

P0718 then applies the simple-mu nonlinear excess to every measured stellar
member before summing vectors. The point-member kernel has the correct limits:
it vanishes in the high-acceleration interior and approaches the constant
deep-MOND deflection proportional to `sqrt(M)` outside `sqrt(GM/a0)`.

| Train to test | Root completeness | Finite-RMS families | Observable topology | Median RMS / halo |
|---|---:|---:|---:|---:|
| AS295 to PLCKG287 | 0.851 | 0.833 | 0.167 | 13.473 |
| PLCKG287 to AS295 | 0.556 | 0.333 | 0.333 | 20.817 |

Half/double softening and the frozen low/high stellar mass-to-light bounds do
not change the conclusion. The formula is useful as a generator of
multiplicity, but it is not a universal lens solution.

## Current scientific status

What is supported:

1. The frozen two-potential candidate remains competitive with or better than
   the frozen MOND fields on the new dwarf rotation curves.
2. Its cluster one-root result is not caused by the discovered axis bug,
   insufficient root starts, or the first map resolution.
3. Cluster topology is controlled by a two-dimensional Hessian pattern, not
   total bending amplitude alone.
4. Nonlinear-before-summation component fields are worth retaining as a clue
   because they create much more multiplicity with transferable amplitude.

What is not supported:

1. No current formula beats or matches compact-halo raw image positions across
   these clusters with one universal setting.
2. No current formula is a covariant relativistic field theory.
3. P0715-P0718 are post-unseal diagnostics on a spent two-cluster subset, not
   new validation.
4. The PLCKG287 2019 glafic map is not a clean comparator for all 2024 image
   families; its own absolute errors must remain visible.

## Next research stages and stop conditions

### 1. Freeze a genuinely new raw-lensing sample

Select at least four clusters whose public products already contain three
secure families, one spectroscopic family, eight images, a baryon map, and a
comparator built for the same catalog version. Prefer catalogs with observed
arc orientation or parity. Freeze formula, coordinates, root floors, nuisance
policy, and thresholds before downloading target positions.

Stop or redesign if fewer than four targets pass readiness again. Do not
silently replace a target after any prediction is opened.

### 2. Derive a forward geometric operator, not another amplitude

The next equation must predict the missing convergence and shear orientation
from baryons before seeing lensing. Candidate ingredients are a conservative
path-integrated tensor or a component-overlap operator with a well-defined
continuum limit. It must conserve the far-field monopole, remain curl-free,
and reduce to GR/Newtonian behavior in the Solar limit.

Stop the current route/amplitude family if a new operator requires a
cluster-specific orientation, shear, smoothing length, or amplitude. P0716-
P0718 already show that those additions can imitate local structure without
transferring image positions.

### 3. Supply the missing relativistic theory

An algebraic Weyl potential is not enough. A defensible model needs an action
or closed field equations that generate `Psi` and `Phi`, a conserved matter
stress tensor, stable perturbations, causal propagation, a PPN limit, and
consistency with gravitational-wave speed and cosmological slip constraints.

### 4. Build the public researcher service

The API contract already exists in
[`../api/openapi.yaml`](../api/openapi.yaml), and the deployment/security plan
is [`PUBLIC_SIMULATOR_API_PLAN.md`](PUBLIC_SIMULATOR_API_PLAN.md). The practical
implementation sequence is:

1. Package the dataset registry, formula AST validator, immutable run manifest,
   and current galaxy/lens measurement engines behind a local FastAPI service.
2. Add endpoints to list real systems, generate a seeded observation-matched
   galaxy, validate a dimensioned formula, submit a run, read job status, and
   retrieve content-addressed artifacts.
3. Never execute submitted Python. Accept a dimension-aware expression tree
   for ordinary formulas; reserve advanced code for signed, network-disabled,
   resource-limited containers.
4. Put the browser and short authenticated control-plane calls on Vercel.
   Send numerical work through a queue to Modal, Cloud Run Jobs, AWS Batch, or
   Kubernetes workers with the scientific Python stack and versioned data.
5. Store datasets and artifacts in S3/R2, metadata and parameter accounting in
   Postgres, and return hashes, seeds, code revision, solver tolerances, and
   comparator versions with every result.
6. Add quotas, timeouts, memory/CPU limits, audit logs, formula complexity
   limits, private-by-default runs, and explicit dataset-license checks.
7. Launch only after local and hosted runs produce byte-identical manifests
   and statistically identical scores for the frozen benchmark suite.

Vercel is appropriate for the UI and gateway, not for multi-minute 3D field
solves or million-run batches. Those belong on asynchronous scientific workers.

## Reproduction

```powershell
python scripts/run_p0715_sky_lensing_engine_validation.py
python scripts/run_p0716_spent_arc_structure_deficit.py
python scripts/run_p0717_screened_contrast_transfer.py
python scripts/run_p0718_componentwise_summation_transfer.py
python -m pytest tests/test_sky_lensing.py tests/test_thin_lens.py tests/test_componentwise_mond_lensing.py tests/test_p0715_p0718_results.py -q
```
