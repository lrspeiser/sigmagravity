# Axisymmetric photon-lensing milestone

Date: 2026-08-02

## Outcome

The formula-independent local worker can now project a photon- or
`both`-typed acceleration field solved on an axisymmetric cylindrical `(r,z)`
grid into sky deflection and lensing-invariant maps. The same confirmed field
may therefore be evaluated against massive-tracer motion and photon data
without materializing a Cartesian proxy or introducing a second gravity
parameter.

The Vercel API publishes and validates the target contract. Heavy execution
remains in the local reference worker until production compute is connected.

## Explicit projection contract

The target declares:

- `axisymmetricInclinationDeg` in `[0,90]`;
- the two-dimensional `skyShape`;
- `lineOfSightSamples`;
- `distanceRatio` and `lensAngularDiameterDistanceM`;
- the solved field origin `[0,z0]` and immutable `axisOrder=["r","z"]`;
- optional deflection and reduced-shear observations, uncertainties and mask.

Cartesian `northAxis`, `eastAxis` and `lineOfSightAxis` indices are rejected on
this path. A separately submitted observation job must repeat `gridOriginM`,
and preflight requires it to match the solved field.

For inclination `i`, intrinsic Cartesian coordinates are reconstructed from
sky north `n`, east `e`, and path coordinate `l` as

```text
x = -n cos(i) + l sin(i)
y = e
z =  n sin(i) + l cos(i)
r = sqrt(x^2 + y^2)
```

The stored `(a_r,a_z)` vector becomes

```text
a_x = a_r x/r
a_y = a_r y/r
a_north = -cos(i) a_x + sin(i) a_z
a_east = a_y
```

with the regular radial contribution set to zero at `r=0`. Each ray is clipped
to the exact intersection of `r <= r_max` and `z_min <= z <= z_max`, then
integrated with

```text
alpha_perp = -(2 distanceRatio / c^2) integral(a_perp dl).
```

The archive includes east/north deflection in radians and arcseconds,
convergence, two shear components, reduced shear, rotation, Jacobian
determinant and eigenvalues, and absolute magnification. Deflection and
reduced-shear residuals retain separate score channels and units.

## Acceptance evidence

Independent fixtures cover different parts of the calculation:

1. A face-on harmonic radial field produces an exact affine deflection,
   constant convergence and zero shear.
2. An edge-on uniform vertical field produces the analytic cylindrical chord
   length `2 sqrt(r_max^2-e^2)`.
3. An axisymmetric point mass recovers the GR normalization
   `4GM/(c^2 R)` with median and 95th-percentile finite-domain errors below
   2% and 4%, matching the Cartesian fixture's gates. Its median path error
   decreases monotonically at 17, 33, 65 and 129 samples and is converged to
   below `1e-5` absolute change between 129 and 257 samples.
4. Exact deflection and reduced-shear maps score independently at numerical
   zero.
5. Swapped axes, a nonzero radial origin, Cartesian axis indices, mismatched
   decoupled origins and excessive path cost are rejected.
6. A real local asynchronous HTTP job solved one field, scored its rotation
   curve and photon map, downloaded and rehashed all 11 artifacts, and used
   zero per-object gravity parameters. Its known-answer results were:

   - field relative L2 error: `3.4364145737847694e-15`;
   - circular-speed RMSE: `4.220673123283083e-15 m/s`;
   - photon-deflection RMSE: `5.490987717737826e-26 arcsec`;
   - photon sampler: `axisymmetric_cylindrical_ray_integral`.

These are software normalization tests, not fits to real galaxies or clusters.

## Scientific boundary

- The solved field is finite. The projection integrates only inside its
  cylinder and treats the field outside as zero. Domain, grid and path-sample
  sensitivity are mandatory for a scientific result.
- Distances and inclination are inputs; the adapter does not infer a cosmology
  or fit them.
- A deflection/shear map is not a raw multiple-image, source-position,
  magnification, weak-shear-catalog or time-delay likelihood.
- Axisymmetry cannot represent bars, spiral arms, mergers, lopsided clusters,
  substructure or general line-of-sight structure.
- The next lensing step is to feed cylindrical deflection maps into the raw
  multiple-image and weak-lensing likelihoods, then test untouched systems.
