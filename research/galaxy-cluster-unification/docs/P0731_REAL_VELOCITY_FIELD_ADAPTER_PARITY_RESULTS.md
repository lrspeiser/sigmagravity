# Real velocity-field adapter parity results (P0731)

Date: 2026-08-02

## Outcome

P0731 passed every frozen engineering gate. The formula-neutral
`line_of_sight_velocity_field` adapter evaluated four unchanged P0723
acceleration-field models against real LITTLE THINGS moment maps for all 13
eligible galaxies: 52 model--galaxy evaluations, with no per-galaxy gravity
parameter.

The production adapter and an independent implementation of the frozen P0712
projection, beam, mask, and score operations agreed to:

| Gate | Result | Frozen limit | Outcome |
|---|---:|---:|---|
| Maximum prediction parity RMS | `1.96e-10 m/s` | `1e-6 m/s` | pass |
| Maximum absolute pixel difference | `1.79e-8 m/s` | `1e-5 m/s` | pass |
| Maximum weighted-score difference | `5.82e-11 m/s` | `1e-6 m/s` | pass |
| Minimum scored pixels per evaluation | `6,115` | `100` | pass |
| Exact valid-pixel support | 52/52 | required | pass |
| Field and observation hashes valid | 52/52 | required | pass |
| Per-object gravity parameters | 0 | 0 | pass |

This is a real-data engineering commissioning result on a previously opened,
project-spent dwarf sample. It is not a new blind theory validation.

## What was tested

Each run reused the immutable P0723 `33 x 33 x 9` generated baryonic replica
and its solved acceleration field. No field equation or gravity parameter was
changed. P0731 then:

1. packaged the real moment-0, moment-1, and moment-2 inputs, sky-to-disk
   coordinates, beam, masks, uncertainty map, and metadata into a hashed
   observation bundle;
2. projected the declared massive-tracer acceleration into circular speed and
   then line-of-sight velocity;
3. convolved the predicted velocity moment with the measured intensity and
   radio beam;
4. scored the real velocity pixels with the frozen intensity and
   inverse-variance weighting; and
5. reproduced the same operation through an independent SciPy/Astropy path.

The public default excludes nonpositive inward acceleration. The frozen P0712
operation assigned it zero circular speed, so P0731 explicitly declares
`nonPositiveInwardPolicy=zero_speed` for parity rather than changing the public
default. Separate emission and score masks make the order of beam convolution
and pixel selection explicit. Intensity maps may retain physical flux units,
because the scale cancels in the normalized weighted score.

## Spent-sample real-map scores

Lower equal-galaxy weighted RMSE is better:

| Rank | Fixed field manifest | RMSE | Universal gravity parameters | Per-galaxy gravity parameters |
|---:|---|---:|---:|---:|
| 1 | QUMOND simple-nu | `17.764 km/s` | 2 | 0 |
| 2 | AQUAL simple-mu | `18.368 km/s` | 2 | 0 |
| 3 | Refracted Gravity published fixture | `19.048 km/s` | 4 | 0 |
| 4 | Newtonian baryons | `19.341 km/s` | 1 | 0 |

These are useful diagnostics, not a ranking gate. They use a circular-equilibrium
mapping on gas-rich dwarfs and do not model pressure support, asymmetric drift,
warps, radial flow, bars, or outflows. They also differ from P0712 because the
underlying 3D reconstruction and numerical grid differ; the apparent score
changes are not evidence that the adapter improved a theory.

The aggregate hides substantial system-to-system variation. The individual
winner count is Newtonian 5, AQUAL 3, Refracted Gravity 3, and QUMOND 2. The
lowest and highest winning errors are `4.288 km/s` for DDO210 and
`39.513 km/s` for NGC1569. QUMOND's best aggregate therefore does not mean it
is uniformly preferred galaxy by galaxy.

![P0731 score and parity summary](../results/p0731_real_velocity_field_adapter_parity/score_and_parity_summary.png)

## What this establishes

- One observation contract works on acceleration fields produced by four
  different formula manifests without model-name or galaxy-name branches.
- Real 2D velocity maps can be evaluated after a 2D or 3D gravity solve without
  fitting a gravity setting to each galaxy.
- The production projection and scorer reproduce an independent frozen
  implementation far inside the preregistered tolerance.
- Acceleration solves and observation evaluation are separable scientific
  stages. That is important for the next cached observation-only job endpoint.

## What remains open

- The sample contains only 13 gas-rich dwarfs and is already spent. It cannot
  establish performance across spirals, bulges, high-surface-brightness disks,
  warps, interacting systems, or a sealed holdout.
- Circular equilibrium is an incomplete forward model for several LITTLE
  THINGS systems. Kinematic nuisance physics must be declared and tested rather
  than absorbed into gravity parameters.
- The coarse P0723 fields are commissioning products, not a frozen
  production-resolution standard.
- This massive-tracer adapter says nothing about photon lensing. A separately
  typed ray/deflection adapter and raw cluster observables are still required.
- The current local batch worker needs durable object storage, job metadata,
  isolated computation, authentication, quotas, and monitoring before public
  execution.

## Reproduce

The frozen field jobs must be present in `tmp/p0723-http-2`:

```powershell
python scripts/run_p0731_real_velocity_field_adapter_parity.py
```

The deterministic report, model table, per-galaxy table, observation-bundle
manifest, and plot are under
`results/p0731_real_velocity_field_adapter_parity/`.
