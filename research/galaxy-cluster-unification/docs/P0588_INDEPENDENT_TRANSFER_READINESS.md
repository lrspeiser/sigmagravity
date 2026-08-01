# P0588 independent transfer readiness

## Result

The repository did not contain a strict fresh cluster test for the P0586D
continuous baryonic metric. It contained several products that are valuable but
answer different questions:

- the four P0586/P0587 lenses have raw multiple-image positions and partial
  physical baryon maps, but they are spent formula-development systems;
- the CCCP/MENeaCS table contains weak-lensing mass summaries, not per-source
  shear likelihoods;
- the RELICS radial kappa products contain useful map covariance, but kappa is a
  standard-lens reconstruction rather than a raw observable;
- RXJ2129 has complete XMM response products, but the stellar BCG/ICL split and
  the required HST centroid covariance failed their frozen gates.

The strict count is therefore **zero fresh systems out of zero available raw
weak-lensing likelihoods**. This is a data-readiness result, not a failure of a
gravity formula.

## Best next target: MACS J0416

MACS J0416 was selected without reading a P0586 formula residual. It is a fresh,
two-core merging cluster with unusually rich observable coverage:

| Product | Local state | What it contributes | Remaining limitation |
|---|---:|---|---|
| Spectroscopic multiple images | 237 images, 88 sources | fresh strong-lens geometry | published error scale was rescaled through the standard fitted model |
| BUFFALO photometric catalog | 18,801 objects | multiband light and redshifts | catalog is ICL-subtracted; the removed ICL map is not retained in the table |
| Frozen member selection | 247 candidates | member-light field independent of the lens residual | stellar M/L and membership-window sensitivity remain to be propagated |
| Chandra-derived gas model | four elliptical dPIE components | independent gas geometry | full parameter covariance is absent from the quoted table |
| Raw shear/magnification | not local | would provide a second lensing channel | public model maps are not a substitute for the raw likelihood |

The downloaded catalog and readme are in
`data/raw/p0588_macs0416_transfer/`, with URLs, sizes, hashes, and license in
`provenance.json`. The 247-row frozen member candidate table is
`data/derived/p0588_macs0416_spectroscopic_member_candidates.csv`.

## Next experiment

Before viewing any new formula score:

1. Build a registered member-light field from the 247 candidates, varying only
   the predeclared universal M/L and membership-window sensitivities.
2. Render the four published gas dPIE components on the same grid and vary their
   normalizations coherently to expose the missing covariance.
3. Locate the released diffuse-ICL model or reconstruct a total-light residual
   from a small HST cutout; test zero, nominal, and doubled ICL as a bounded
   sensitivity rather than fitting it to the arcs.
4. Freeze the P0586D signed metric at its existing setting and score all 237
   images. Use 0.43 arcsec only as a descriptive scale, explicitly not as an
   independent covariance.
5. Hold out entire source families and both spatial lobes. A useful transfer
   must improve both lobes and held-out families; a gain confined to one core is
   evidence for geometry mismatch, not universality.

This test will tell us whether the signed continuous metric transfers to a
fresh, highly non-spherical cluster. It will not by itself establish a valid
weak-lensing law or beat dark matter statistically.
