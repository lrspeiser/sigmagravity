# Axisymmetric raw multiple-image lensing milestone

Date: 2026-08-02

## Outcome

The formula-independent local worker can now evaluate raw strong-lensing image
positions directly from a photon- or `both`-typed acceleration field solved on
an axisymmetric cylindrical `(r,z)` grid. It uses the same projection contract
as the v0.27 photon-map adapter, then profiles source positions, finds global
lens-equation roots, assigns observed images one-to-one, and scores image-plane
residuals without adding a gravity parameter.

The Vercel control plane publishes and validates this contract. Heavy execution
remains in the local reference worker until durable production compute is
connected.

## Explicit composition

The raw target declares:

- `axisymmetricInclinationDeg`, `skyShape`, and `lineOfSightSamples`;
- the solved origin `[0,z0]` and immutable `axisOrder=["r","z"]`;
- the lens angular-diameter distance and intrinsic Cartesian `skyCenterM`;
- a bounded root-search square and numerical closure/deduplication controls;
- one or more source families, each with a distance ratio, observed
  east/north image positions, and per-image positional uncertainty.

Cartesian storage-axis indices and axisymmetric projection controls are
mutually exclusive. A decoupled observation job must repeat `gridOriginM`, and
preflight requires it to match the solved field exactly.

The worker projects one distance-ratio-one map:

```text
alpha_1 = -(2/c^2) integral(a_photon,perp dl)
alpha_family = (D_ls/D_s)_family alpha_1
beta_family = mean(theta_observed - alpha_family(theta_observed))
```

It archives `alpha_1` once, counts the two profiled source coordinates per
family as observational nuisance parameters, and uses the existing global
root and minimum-cost assignment machinery. Missing observed multiplicity
produces `incomplete_topology`, not a finite partial-fit likelihood.

## Finite-support invariant

Cylindrical photon maps retain unsupported rays as non-finite cells. The raw
interpolator requires a finite rectangle, so the worker applies a strict gate:

1. expand the requested root bound by the declared Jacobian finite-difference
   step;
2. identify every map node that can enter bilinear interpolation in that
   expanded square;
3. require every one of those east/north deflection nodes to be finite;
4. only then zero-fill unsupported cells outside the verified root region.

The result records
`outsideSupportInterpolationFill=zero_only_outside_verified_root_support` and
the exact verified node ranges. A search crossing the projected cylinder
silhouette is rejected rather than silently using the fill.

## Acceptance evidence

1. A face-on cylindrical constant-inward-radial field produces the analytic
   SIS deflection and recovers images at `-0.8` and `1.2 arcsec` from a profiled
   source at `0.2 arcsec` with image-plane RMS below `1e-3 arcsec`.
2. The same fixture rejects a wider root square whose corners cross unsupported
   rays.
3. Cartesian sky axes, mismatched origins, swapped cylindrical axes, and
   excessive path cost are rejected in worker and hosted preflight tests.
4. The legacy Cartesian SIS, topology-failure, distance-ratio, and axis-
   permutation tests remain unchanged and pass after the ratio-one refactor.
5. An immutable field job solves a smooth cored-isothermal cylindrical
   potential, archives photon maps and roots, and scores its analytic outer
   images with zero per-object gravity parameters.
6. A real local asynchronous HTTP run completed upload, queue, worker, events,
   artifact download, and rehash verification. Its raw cylindrical case had:

   - field relative L2 error: `0.0014789730022103532`;
   - image-plane RMS: `0.001692053225097455 arcsec`;
   - photon sampler: `axisymmetric_cylindrical_ray_integral`;
   - artifacts: 13, all downloaded hashes valid;
   - per-object gravity parameters: 0.

These are analytic software and normalization fixtures, not real cluster fits.

## Scientific boundary

- Profiling a source is a declared nuisance fit, not an independent source
  measurement.
- Extra predicted roots are disclosed but need an explicit detectability and
  selection model before they can contribute to a likelihood.
- The adapter does not infer redshift distances or a cosmology.
- Axisymmetry cannot represent mergers, lopsided mass, member substructure, or
  general line-of-sight structure.
- Domain, grid, sky, path-sample, root-grid, and boundary sensitivity remain
  mandatory.
- Real advancement requires registered raw image positions with uncertainties,
  baryonic posterior ensembles, untouched cluster holdouts, and comparison to
  published dark-matter lens models with parameter counts.
- Weak-lensing catalogs, magnification selection, time delays, and cylindrical
  nonlocal kernels remain outside this milestone.
