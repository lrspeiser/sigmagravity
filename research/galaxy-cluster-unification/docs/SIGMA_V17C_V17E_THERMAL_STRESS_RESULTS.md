# Sigma v17C-v17E thermal-stress results

## Decision

The independently measured projected gas thermal-stress source is **not** a
transferable explanation of the full apparent cluster halo. It predicts the
approximate radial extent of the missing one-metric field, but it does not
predict enough of its amplitude, detailed spatial structure, or shear
orientation. The frozen v17E gate therefore fails and v17F is not authorized.

This is a useful separation of the halo-size problem from the halo-strength
problem. The result says that gas density and temperature contain information
about the scale over which the apparent halo is distributed. They do not, by
themselves, determine the field that strong lensing requires.

## Independent measurement chain

The source was constructed before either spent lensing target was opened:

1. Public Chandra observations were reprocessed, cleaned, registered, and
   divided into frozen signal-to-noise regions.
2. Background-subtracted spectra and detector responses were extracted for
   298 of 299 planned detector-region cells: 193 of 194 for AS295 and 105 of
   105 for PLCK G287.
3. One AS295 cell (region 12, observation 16524, CCD 3) produced valid source
   and background spectra but CIAO returned the exact empty calibrated-response
   error `max() iterable argument is empty`. Three nearby out-of-bound detector
   controls produced complete responses. The cell was quarantined under a
   frozen exact-signature rule; no temperature, lensing value, or new physical
   threshold was used to classify it.
4. The surviving cells were combined into 29 AS295 and 21 PLCK G287 spectra.
   The frozen snapshot contains 250 products totaling 120,479,937 bytes.
5. Every regional temperature fit completed and had a finite best fit. AS295
   had 28 of 29 individual quality passes and PLCK G287 had 21 of 21; both
   cluster-level regional gates passed. The integrated temperatures were
   9.6036 keV and 14.3962 keV, respectively.
6. The temperature maps were multiplied by the registered gas convergence to
   form the dimensionless, target-blind source

   $$
   q_{\rm total}(\boldsymbol x)=
   \kappa_{\rm gas}(\boldsymbol x)
   {k_B T_X(\boldsymbol x)\over \mu m_p c^2},
   $$

   together with the temperature-contrast source

   $$
   q_{\rm contrast}(\boldsymbol x)=
   \kappa_{\rm gas}(\boldsymbol x)
   {k_B[T_X(\boldsymbol x)-T_{\rm global}]\over \mu m_p c^2}.
   $$

   Every smoothed scalar generated convergence and both shear components from
   one E-mode potential. There was no independent lensing multiplier, fitted
   orientation, cluster scale, or cluster normalization.

## Frozen transfer result

The selected nested source was `thermal_component` with ridge value 1.0. It
was selected by symmetric leave-one-cluster-out transfer on the already-spent
pair, not promoted to a physical constant.

| Gate or diagnostic | Frozen requirement | Result | Decision |
|---|---:|---:|---|
| Symmetric full-field NRMSE | at most 0.500 | 0.785960 | fail |
| Static baseline NRMSE | control | 0.817941 | -- |
| Relative improvement over static | at least 10% | 3.910% | fail |
| Residual power closed, AS295 to PLCK | at least 0.25 | 0.2903 | pass |
| Residual power closed, PLCK to AS295 | at least 0.25 | -0.0602 | fail |
| Residual shear alignment, AS295 to PLCK | at least 0.50 | 0.4726 | fail |
| Residual shear alignment, PLCK to AS295 | at least 0.50 | 0.0528 | fail |
| Maximum R50/R80 error, AS295 to PLCK | at most 25% | 7.18% | pass |
| Maximum R50/R80 error, PLCK to AS295 | at most 25% | 9.53% | pass |
| Maximum doubled-resolution change | at most 2% | 1.009% | pass |

The directional full-field NRMSE values were 0.6450 for AS295 to PLCK G287
and 0.9052 for PLCK G287 to AS295. The asymmetry is physical evidence against
treating projected thermal gas stress as the missing universal state variable:
the source transfers some PLCK morphology but adds essentially no correct
power or shear structure to AS295.

## What we learned about apparent halo size

The radius test is amplitude-independent. Multiplying the predicted field by
any nonzero scalar does not change its field-energy radii. Passing it therefore
cannot be an artifact of fitting one stronger gravity constant.

For each transfer direction, the required and predicted residual triplets were
reduced to

$$
u(\boldsymbol x)=
\Delta\kappa^2+\Delta\gamma_1^2+\Delta\gamma_2^2,
$$

and independently centered before measuring the radii enclosing 50% and 80%
of the field energy. The four radius errors were between 5.35% and 9.53%.
That is the clearest current evidence that apparent halo extent is related to
the spatial extent of dynamically active baryons, rather than being an
arbitrary radius unrelated to them.

It is not yet a root equation. The same maps miss the centroid and directional
curvature strongly enough to fail the full-field and shear gates. A viable
equation must preserve the scale information while obtaining its amplitude
and tensor direction from another baryon-forced state.

## Mechanism consequence

The following constitutive choices are now prohibited on this evidence:

- inserting gas pressure or temperature alone into the gravitational response;
- rescuing the result with a cluster-specific pressure coefficient or length;
- running v17F after the parent v17E gate failed; and
- interpreting the passed extent diagnostic as a detected dark-halo radius.

The authorized next branches are collisionless baryonic stress or a genuinely
causal state carrying history or transport information. Either successor must
be different from the already-retired local tensor, diffusion, instantaneous
nonlocal, preferred-clock, and extra spin-two carrier families. It must retain
one metric and derive both dynamics and lensing from the same field equations.

## Reproducibility

| Artifact | SHA-256 |
|---|---|
| v17C regional spectra report | `8eed0131b74f6d24ece8e8a4f9ab34eb7ba574603a1415aeb78d29db1e5d8e91` |
| Runtime response-support config | `75fe99de5b2e4df2e64791e63090fcbb70ee947bc6910495a82c555710807987` |
| Runtime response-support report | `4311d87c37768f60eff0ae6d7f3ac4738dbe625a0690e8db0c587f8a9fe06b89` |
| v17C regional temperatures report | `fd3baf8285bb40de8589a28e5abab13053d6de64669b15d29b3c309e0d4e8362` |
| v17D thermal-map report | `a0f93505aa25a34fbd6c356c9cfedfa696b5982e03a9b810fe6dd4f8ee88a3da` |
| AS295 target-blind thermal source | `1d641edefd9d8ab92f5fb4fd78974c57715791f601aed6f61aa91c65678db6ec` |
| PLCK G287 target-blind thermal source | `e45dc2f9490bb3ef64aaf305d2191e957ea8aeb3591d4be2ace58d23dcf449c3` |
| v17E transfer report | `ec8bf5c62f73839f0ba5eca30fda57fcea688cfac962d4f9ca6127e7632765b7` |

The authoritative executable artifacts are under
`results/sigma_v17c_regional_spectra`,
`results/sigma_v17c_regional_temperatures`,
`results/sigma_v17d_thermal_stress_maps`, and
`results/sigma_v17e_thermal_stress_transfer`.
