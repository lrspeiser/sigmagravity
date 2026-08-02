# P0734 typed photon-lensing adapter

Date: 2026-08-02

## Outcome

P0734 adds a formula-neutral observation path from a solved three-dimensional
photon acceleration field to coordinate-safe sky-lensing maps. It is a
different adapter from the circular-speed and resolved-velocity paths used for
massive tracers. A photon target cannot silently read a massive-only
observable, and a massive-tracer target still cannot read a photons-only
observable.

The public target kind is `photon_lensing_map`. It requires:

- a Cartesian 3D vector observable in `m/s^2` targeted to `photons` or `both`;
- an explicit permutation naming the north, east, and line-of-sight axes;
- an explicit positive distance ratio `D_ls/D_s`;
- an explicit positive lens angular-diameter distance used to convert physical
  transverse grid spacing into angular spacing; and
- provenance and license records for every observed map.

No redshift-to-distance relation or cosmology is inferred by the adapter.

## Mapping

For the submitted photon acceleration field, the adapter evaluates

```text
alpha_perp_rad = -(2 * distanceRatio / c^2) * integral(a_photon_perp dl)
```

and writes a deterministic, content-hashed NPZ archive containing named east
and north deflection maps in radians and arcseconds, convergence, both shear
components, shear magnitude, both reduced-shear components, rotation, the
Jacobian determinant and eigenvalues, and absolute magnification.

The fixed `2/c^2` factor is the declared weak-field zero-slip projection for an
acceleration-like photon observable. It is not a derivation of a relativistic
field equation. A researcher remains responsible for submitting a model whose
photon observable has the intended physical meaning.

## Score separation

Optional paired deflection observations are scored in an `arcsec` channel.
Optional paired reduced-shear observations are scored in a dimensionless
channel. Velocity remains a third `m/s` channel. Each channel has its own point
count, residual sum, RMSE, inverse-variance score, chi-square, degrees of
freedom, and Gaussian log likelihood.

No scalar aggregate combines residuals with different units. The legacy batch
velocity fields remain as backward-compatible aliases of only the `velocity_m_s`
channel; a photon-only batch returns no velocity RMSE.

## Frozen acceptance result

The preregistration is
[`../configs/p0734_typed_photon_lensing_adapter.json`](../configs/p0734_typed_photon_lensing_adapter.json).
The immutable result is
[`../results/p0734_typed_photon_lensing_adapter/report.json`](../results/p0734_typed_photon_lensing_adapter/report.json).

All 16 frozen gates passed:

- uniform-field normalization error: `2.22e-16`;
- distance-ratio and path-length scaling error: exactly `0` at recorded
  precision;
- maximum affine convergence/shear/Jacobian error: `4.44e-16`;
- gradient-field rotation RMS: `3.75e-17`;
- finite-grid point-mass `4GM/(c^2 b)` median relative error: `0.974%`;
- point-mass p95 relative error: `2.741%`;
- exact synthetic deflection RMSE: `0 arcsec`;
- exact synthetic reduced-shear RMSE: `0`;
- photon-only evaluation produced no velocity RMSE;
- integrated and separately cached observation jobs emitted byte-identical
  score JSON and photon-map NPZ artifacts;
- deterministic NPZ archives were byte-stable across repeated writes;
- hosted preflight repeated observable type, unit, geometry, axis, shape,
  distance, observation-triple, and mask checks;
- a composed batch retained deflection and reduced-shear aggregates separately;
- observation evaluation added zero gravity parameters;
- 68/68 hosted tests passed; and
- the production static build passed with all 175 SPARC systems.

The point-mass comparison uses impact parameters from 4 through 32 grid cells
inside a line-of-sight half-length of 128 cells. Its remaining error is the
expected finite-path truncation plus discretization; the frozen 2% median and
4% p95 limits were satisfied.

## Reproduce

```powershell
python scripts/run_p0734_typed_photon_lensing_adapter.py
```

## Scientific boundary and next work

P0734 proves that a submitted photon field can be projected and reported with
the declared normalization and coordinate convention. It does not show that
any particular baryonic gravity equation fits real cluster lensing.

The map channels can ingest published weak-lensing or reconstructed map
products, but such products may inherit assumptions from their original lens
models. Raw multiple-image positions are more direct and remain a separate
adapter milestone. That next layer must add source-position profiling,
image-plane root finding, critical curves, covariance, time delays where
available, and blinded holdouts without changing the P0734 photon projection.
Public execution also still needs durable metadata, object storage, isolated
container workers, authentication, quotas, retries, and monitoring.
