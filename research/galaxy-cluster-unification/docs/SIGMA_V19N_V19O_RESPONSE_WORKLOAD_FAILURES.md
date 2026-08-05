# Sigma V19N/V19O response-workload failures

## Outcome

Neither workload protocol authorizes response extraction.  Both stopped before
constructing a PHA, ARF or RMF and before fitting any thermodynamic or gravity
quantity.

Both independently enumerated the same provisional workload:

| Cluster | Admitted regions | Provisional response cells | Four-file products | Conservative storage envelope |
|---|---:|---:|---:|---:|
| Bullet Cluster | 366 | 3,812 | 15,248 | 93.1 GiB |
| Abell 2146 | 128 | 1,270 | 5,080 | 31.0 GiB |
| Total | 494 | 5,082 | 20,328 | 124.1 GiB |

The 25-MB-per-cell storage figure is deliberately conservative engineering
headroom, not a scientific threshold.

## V19N: missing detector-support filter

V19N assigned registered event coordinates directly to the V19M bin map.  It
passed every region-coverage, CCD and scaled-background check.  Abell 2146 also
matched the frozen broad image exactly.  Bullet assigned 775,284 events where
the frozen image contains 775,283, so its exact science-conservation gate
failed by one event.

A full-image diagnostic found two registered Bullet rows absent from the
`flux_obs` count image, one inside a contbin label.  `flux_obs` had also applied
each observation's aspect-projected detector mask and bad-pixel support; V19N
had not.  The event rows themselves have valid energy, grade and status.

## V19O: a repro FOV is not the `flux_obs` detector mask

V19O copied and astrometrically translated each frozen repro FOV, then filtered
science and blank-sky events through it.  All 20 translations passed their hash
and uniqueness gates, and task membership remained exactly 5,082 cells.

The exact image-count gate still failed:

- Bullet retained the same one-event excess; and
- Abell 2146 retained 341,811 events versus 346,066 in the frozen broad image,
  a deficit of 4,255.

Thus the geometric repro FOV is not equivalent to the support used internally
by `flux_obs` after aspect, detector-mask and bad-pixel processing.  It is not a
valid substitute merely because both are called field-of-view products.

## Next implementation

The source-map scratch namespace retains the exact per-observation
`flux_obs` products: `OBS_ID_0.5-7.0_thresh.img`, the corresponding exposure
map, and `OBS_ID.fov`.  V19P should inventory and hash those products, then:

1. validate source counts per observation inside only the 494 admitted regions,
   which are the actual extraction scope;
2. use the exact `flux_obs` per-observation support product rather than a
   translated approximation;
3. preserve the frozen region, observation and CCD membership rule; and
4. prove that the resulting manifest is byte-stable before constructing one
   commissioning response.

This is not permission to relax a one-photon tolerance.  It changes the support
object to the one that actually generated the frozen source map.  The stable
5,082-cell result is useful capacity planning, but remains provisional until
V19P passes.
