# P0576B-P0578 source-plane degeneracy and two-cluster raw results

## Outcome

The impressive fractional-propagator source-plane result was a mass-sheet
false positive. A mass-sheet-resistant image diagnostic and a second raw
cluster reject a universal fractional power. Ordinary baryonic broadening is
more interpretable but still improves only one of two clusters.

The useful result is a parameter hierarchy and a corrected testing rule, not a
new force law.

## P0576B-P0576C: why the source-plane score failed

The original grid ended at `p=1.5`. Extending the same formula to `p=2.6`
continued to lower source-plane RMS with no interior minimum:

| Diagnostic at `p=2.6` | Value |
|---|---:|
| Held-out source-plane RMS | 0.087 arcsec |
| Apparent improvement versus local | 93.24% |
| Family splits apparently improved | 6/6 |
| Global affine/mass-sheet `R^2` | 0.999925 |
| Inferred-source/no-lens radius | 0.00915 |
| Family-mean source dispersion | 0.107 arcsec |

The high-power field was nearly proportional to image position. Choosing the
amplitude near criticality then mapped almost every image toward the same
source coordinate. Source-plane scatter rewarded that collapse even though it
did not predict distinct image roots.

This invalidates the interpretation of the P0576 `p=1.5` improvement. It does
not reject every non-Poisson theory; it rejects this metric as a selector for
long-wavelength propagators.

## P0576D: Jacobian-aware image residuals

P0576D converted each source residual back to a local image residual:

\[
\delta\theta_i=J_i^{-1}\delta\beta_i,
\qquad
J_i=I-A{D_{ls}\over D_s}{\partial\alpha\over\partial\theta}.
\]

A declared singular-value floor of 0.02 prevented exact criticality from
receiving a zero residual. The same 42-candidate grid selected `p=1.75, f=1`.
It improved aggregate held-out RMS by 15.70%, but only one of two families and
retained mass-sheet `R^2=0.9976`. It failed the lock gate.

The processed Lenstool reference scored 2.300 arcsec on held-out SMACS images,
versus 5.382 for the selected field and 6.385 for local light. The absolute
values are linearized-image statistics, not the paper's nonlinear image-plane
RMS.

## P0577: independent SPT0615 response

The second raw table came from Paterno-Mahler et al. (2018), ApJ 863 154,
arXiv:1805.09834v2. The frozen sample contains 17 secure positions:

- z=1.358 knot subfamilies 1, 10, 11, and 12;
- secure z=4.013 images 3.1, 3.2, and 3.5;
- systems 1 and 10 calibrated the amplitude;
- systems 11, 12, and 3 were held out.

SPT's internal grid again selected the upper boundaries, `p=2, f=1`, but
improved held-out RMS by only 6.15%, below the 10% gate. The SMACS-selected
`p=1.75, f=1` transferred poorly:

| SPT held-out field | RMS (arcsec) | Change versus local |
|---|---:|---:|
| Local B100 | 7.569 | — |
| SPT internally selected | 7.104 | 6.15% better |
| SMACS-locked `p=1.75` | 12.186 | 61.0% worse |
| Processed Lenstool reference | 3.243 | 57.1% better |

The SMACS coordinate helped only one of three SPT subfamilies. There is no
cross-cluster evidence for a universal fractional exponent.

## P0578: ordinary baryonic broadening

P0578 removed fractional propagation and used

\[
\Sigma=(1-f_bH)B_{20}+f_bH B_w,
\qquad
\nabla^2\psi=2\Sigma.
\]

The symmetry gate makes the broad component exactly absent for the Solar
point-source and deprojected axisymmetric-SPARC limits. Thirty-two universal
width/fraction candidates were selected by equal-cluster calibration.

The winner was `w=125 kpc, f_b=1`:

| Cluster | B100 held-out RMS | Selected RMS | Change |
|---|---:|---:|---:|
| SMACS J0723 | 6.385 | 6.702 | 4.98% worse |
| SPT-CL J0615 | 7.569 | 5.583 | 26.24% better |
| Equal-cluster mean | 6.977 | 6.143 | 11.95% better |

Only one cluster and 40% of held-out subfamilies improved, so universal
broadening failed. The parameter impacts are still informative:

| Coordinate | Calibration main-effect span |
|---|---:|
| Broad fraction, 0.25--1.0 | 2.254 arcsec |
| Width, 40--250 kpc | 0.777 arcsec |

Width has a broad mean minimum around 60--100 kpc, while full replacement by a
broad component is the larger coordinate. The selected 125-kpc interaction is
cluster-dependent and too affine in SMACS.

## Current universal truths

1. Normalized convergence morphology and raw deflection are not interchangeable.
2. Source-plane scatter is unsafe for long-wavelength formula selection unless
   mass-sheet collapse is explicitly audited.
3. A fractional exponent is highly impactful but not transferable; impact alone
   is not universality.
4. Broad radial organization matters more than small activation-power changes.
5. One physical smoothing width cannot represent both tested clusters.
6. The next formula should derive its scale from observable baryonic structure
   such as member separation, concentration, gas extent, or multiple radial
   scales—without reading the lens target.

## Reproduce

```powershell
python scripts/run_p0576b_fractional_boundary_extension.py
python scripts/run_p0576c_source_plane_degeneracy_audit.py
python scripts/run_p0576d_linearized_image_plane.py
python scripts/run_p0577_spt0615_raw_response.py
python scripts/run_p0578_two_cluster_baryon_broadening.py
python -m pytest -q tests/test_p0576b_p0578_two_cluster_results.py
```
