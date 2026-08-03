# Axisymmetric galaxy-observation adapter milestone

Date: 2026-08-02

## Outcome

The local formula-independent worker can now carry one confirmed stationary
field model through this complete path:

```text
baryonic source on (r,z)
  -> solve submitted elliptic field law
  -> evaluate the declared massive-tracer acceleration
  -> sample radial acceleration at the declared midplane
  -> predict circular speeds or a resolved line-of-sight velocity map
  -> score against observations with uncertainties
  -> publish content-addressed artifacts
```

No Cartesian disk proxy, azimuthal averaging, per-galaxy gravity parameter, or
theory-name branch is used.

## Physics and coordinate convention

For an axisymmetric equilibrium circular orbit,

```text
v_c(r)^2 = r * [-a_r(r,z_midplane)].
```

The target must declare `centerM=[0,z_midplane]`. The field grid must retain
`axisOrder=["r","z"]` and `origin=[0,z0]`. Cartesian `planeAxes` and
`azimuthalSamples` are rejected because an `(r,z)` grid does not contain two
Cartesian disk-plane axes and already asserts azimuthal symmetry.

For a resolved disk-plane coordinate map `(x_major,y_minor)`, the adapter uses

```text
r = sqrt(x_major^2 + y_minor^2)
v_los = handedness * sin(inclination) * v_c(r) * x_major/r.
```

The existing observation machinery then applies declared emission and score
masks, intensity weighting, beam convolution, velocity zero point,
uncertainties or covariance, and nuisance-parameter accounting.

When observations are evaluated in a separate immutable job, the target must
bind `gridOriginM=[0,z0]`. This prevents an observation bundle's pixel geometry
from silently redefining the already-solved field coordinates.

## Acceptance evidence

The analytic fixture uses

```text
Phi(r,z) = 0.5 * omega^2 * r^2
laplacian(Phi) = 2 * omega^2
a_r = -omega^2 r
v_c = omega r.
```

Python acceptance covers off-grid radial interpolation, off-grid midplane
interpolation, circular curves, resolved velocity maps, full scoring, and
negative coordinate cases. The immutable field-job acceptance writes and
rehashes the prediction artifacts.

The real local asynchronous HTTP acceptance reports:

| Quantity | Result |
|---|---:|
| Field relative L2 error | `3.4364145737847694e-15` |
| Circular-speed RMSE | `4.220673123283083e-15 m/s` |
| Downloaded artifacts | `10` |
| Artifact hashes valid | `true` |
| Per-object gravity parameters | `0` |
| Sampling mode | `axisymmetric_midplane_direct` |

The same smoke run retains the prior Cartesian 2D and 3D cases, so adding the
cylindrical observation path does not replace them.

## Scientific boundary

- The analytic acceptance validates software normalization, not agreement with
  a real galaxy.
- `v_c^2=r(-a_r)` assumes circular equilibrium. Pressure support, asymmetric
  drift, turbulence and non-circular flows require explicit forward modeling.
- An axisymmetric field cannot represent bars, spiral streaming, lopsidedness,
  mergers or general warps.
- Distance, inclination, mass-to-light and baryonic-structure uncertainty must
  come from the gravity-independent reconstruction layer and be propagated as
  ensembles.
- This milestone does not add axisymmetric photon lensing or cylindrical
  nonlocal convolution.
- Heavy execution remains local until durable isolated production workers are
  connected.
