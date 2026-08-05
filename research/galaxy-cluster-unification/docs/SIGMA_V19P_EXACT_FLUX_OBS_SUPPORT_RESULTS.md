# Sigma V19P exact `flux_obs` support result

V19P was frozen and pushed before any per-observation `flux_obs` FITS array or
support region was opened.  Its inputs comprise 100 hash-locked files: the 20
registered science events, 20 blank-sky events, and the exact count image,
exposure map, and FOV produced for each observation by the already-frozen V19H
`flux_obs` call.

## Outcome

The gate **failed closed**, but localized the discrepancy completely.

| Check | Bullet Cluster | Abell 2146 |
|---|---:|---:|
| Admitted regions | 366 | 128 |
| Response task keys | 3,812 | 1,270 |
| Task keys changed from V19N | 0 | 0 |
| Per-observation images sum pixelwise to combined image | pass | pass |
| Exact-FOV event minus image count | +1 | 0 |
| Frozen blank-sky conservation | pass | pass |

The exact `flux_obs` FOV corrected V19O's erroneous 4,255-event loss in Abell
2146.  It did **not** remove Bullet's one extra event, proving that the remaining
difference is not the geometric field-of-view footprint.

The post-failure pixel diagnostic found exactly one discrepant pixel: Bullet
ObsID 554, region 24, CCD 3.  The registered event is valid in energy, grade and
status and is geometrically inside the exact `flux_obs` FOV, but its matching
`flux_obs` exposure-map value is exactly zero.  The count image therefore has
zero at that pixel while coordinate/FOV assignment retains one event.

## Consequence

The next materially justified support rule is not a tolerance and does not
delete an event by identity.  It is the same universal pixel rule for all 20
observations: an event contributes to a regional response task only when the
matching exact `flux_obs` broad-band exposure pixel is positive.  That rule
reproduces the support actually used to make the frozen science image.  It must
be frozen and rerun across both clusters before response extraction is
authorized.  That follow-up was frozen as V19Q and passed every gate: it
preserved all 5,082 task keys and achieved exact science-count conservation in
both clusters.

No spectrum, response, temperature, density, Mach number, lensing target, or
gravity formula was constructed or changed in V19P.
