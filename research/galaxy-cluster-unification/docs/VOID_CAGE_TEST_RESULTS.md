# Void-cage hypothesis: completed test

## Outcome

The tested void-cage models fail the frozen galaxy prediction gates. A literal
external harmonic tide is worse than the same radial model without measured
void information, and the screened conversion does not improve held-out
rotation curves when its amplitude is tied to the independently reconstructed
void field. Fixed RAR remains much more accurate under the same fixed baryonic
inputs.

This rejects the formulas and mappings tested here. It does not prove that every
possible negative-gravity or boundary-pressure theory is false.

## What was tested

The exterior underdensity field was frozen from Cosmicflows-4 before any SPARC
velocity or gravity residual entered the geometry calculation. For a repulsive
source with force magnitude proportional to `1/d^p`, the isotropic shell limit
near the galaxy center is

```
Delta a = -kappa r,
kappa = A Q (p - 2) / (3 D^(p + 1)).
```

This makes `p=2` an analytic null: an isotropic inverse-square exterior shell
does not compress its interior. The numerical alternatives were a `p=3` force
and a finite-range repulsive Yukawa kernel. Their direct galaxy-scale effect was
tested as `Delta v^2 = kappa_i r^2`.

A separate screened conversion allowed a flat outer contribution while still
requiring the void map to predict its galaxy-to-galaxy strength:

```
v_pred^2 = v_bar^2
           + V0^2 E_i^m r^2 / [r^2 + (c_R R_d)^2].
```

`V0`, `m`, and `c_R` were universal training-fold parameters. `E_i` was the
residual-blind exterior compression score. The nested control used the same
radial formula with `E_i=1`, so only genuine predictive information in the void
map could make the cage win.

## Geometry result

The residual-blind geometry covered 175 SPARC catalog galaxies. The primary map
was the grouped CF4 64-cube with a Yukawa range of 15.625 `h100^-1 Mpc`; the
ungrouped 64- and 128-cubes were reconstruction checks.

| Geometry diagnostic | Grouped 64 | Ungrouped 64 | Ungrouped 128 |
|---|---:|---:|---:|
| Fully compressive 3D fraction | 2.29% | 1.71% | 2.86% |
| Median compressive directions | 2 | 2 | 2 |
| Median anisotropy | 1.53 | 1.85 | 1.63 |

The mapped exterior field is therefore not usually an all-sides scalar cage.
It normally compresses along two principal axes and expands along one. The
primary score rank correlation is 0.779 between the grouped and ungrouped
64-cubes, but only 0.528 between the grouped 64- and ungrouped 128-cubes. The
environment ordering is reconstruction-sensitive enough that robustness had to
be part of the pass condition.

## Held-out galaxy prediction

All radii from each galaxy were held out together. The sample contains 131
galaxies and 3,034 rotation-curve points in five folds. Disk and bulge
mass-to-light ratios, distances, inclinations, and the velocity-error rule were
the same for all models.

| Model | Held-out RMSE (km/s) | Chi squared / point |
|---|---:|---:|
| Fixed RAR | 23.085 | 21.340 |
| Screened radial control, no environment | 42.707 | 103.734 |
| Screened primary void cage | 42.819 | 104.993 |
| Direct harmonic control, no environment | 45.383 | 122.960 |
| Direct harmonic primary void cage | 46.966 | 138.300 |
| Newtonian catalog baryons | 60.721 | 206.779 |

The screened cage is 0.113 km/s worse in RMSE than its identical radial control.
Its candidate-minus-control chi-squared difference is +1.258 per point, with a
95% paired-galaxy interval of +0.247 to +2.691. Only 0.076% of 100,000 bootstrap
resamples favor the environmental cage on chi squared. Against fixed RAR, its
RMSE is 19.734 km/s worse and none of the bootstrap resamples favor it.

The environment exponent reached the zero bound in four of five primary folds.
Only 26.6% of 64 shuffled environment assignments were worse than the real map;
the preregistered requirement was at least 95%. The ungrouped 128 reconstruction
put the exponent at zero in all five folds. Every substantive success gate
failed.

## Gate decisions

| Frozen gate | Result |
|---|---|
| At least 5% better RMSE than radial control | Fail |
| At least 5% better RMSE than fixed RAR | Fail |
| At least 0.95 bootstrap probability vs control | Fail |
| At least 0.95 bootstrap probability vs RAR | Fail |
| Positive, non-bound environment exponent in every fold | Fail |
| Real map beats at least 95% of permutations | Fail: 26.6% |
| Response sign stable across CF4 reconstructions | Fail |
| No lensing-only normalization | Pass |

The literal external cage and the screened environmental conversion are both
rejected as currently formulated. Per-galaxy amplitudes, void centers, ranges,
or galaxy/cluster switches were not added after failure.

## Lensing and unification gate

No lensing response was fit. That is the scientifically informative action at
this stage: the galaxy gate failed, and the current package has zero systems
with both resolved dynamics and a raw theory-neutral lensing likelihood. The 84
CLASH acceleration summaries are GR/NFW-deprojected products, not raw shear or
image-coordinate likelihoods that can fairly test an alternative metric.

The existing audit found 20 CLASH systems but zero raw/forward-ready systems,
34 BCG dynamics summaries but zero resolved likelihoods, and zero strict-ready
same systems against a target of ten. A future unification pass must freeze the
dynamics constants and predict raw shear or image coordinates on the same
objects with `Phi=Psi` and no lensing normalization.

## What this establishes, and what it cannot

The run establishes four negative results for this implementation:

1. An inverse-square isotropic repulsive shell cannot supply the proposed
   inward cage because its interior compression cancels analytically.
2. Faster-falloff and finite-range exterior fields do produce tides, but the CF4
   field is usually anisotropic rather than fully compressive.
3. The measured scalar compression does not predict held-out SPARC rotation
   speeds under either a direct harmonic effect or the tested screened radial
   conversion.
4. The real environment map is not distinguished from shuffled assignments and
   is unstable across the available reconstructions.

The run cannot determine whether a disk aligned with the two compressive axes
has a directional velocity signature. SPARC provides one-dimensional rotation
curves, not two-dimensional residual velocity maps with a tested projection of
the CF4 Hessian into each disk plane. CF4 also resolves roughly 7.8
`h100^-1 Mpc`, not a galaxy-edge transition, and is a Wiener-filtered
cosmographic reconstruction rather than a theory-neutral direct measurement of
void gravity.

## Next falsifiable branch

The only branch directly suggested by the geometry, rather than by residual
tuning, is an anisotropic tensor test. It should be preregistered before looking
at velocity residuals:

1. Cross-match galaxies with public two-dimensional H-alpha or HI velocity maps
   to an independent local-density reconstruction.
2. Rotate the precomputed exterior Hessian into each measured disk plane and
   predict the phase and sign of the quadrupolar/dipolar velocity residual, with
   no galaxy-specific force amplitude.
3. Hold out whole sky regions so neighboring galaxies cannot leak the same void
   reconstruction into train and test folds.
4. Require the real tensor orientation to beat randomized sky rotations and
   randomized environment assignments, and require sign stability across
   density reconstructions.
5. Stop again if it fails. Only a galaxy pass authorizes acquisition and replay
   of raw same-system lensing likelihoods.

The locally downloaded MaNGA DR17 asset is the DRPall summary catalog; the
DynPop and GEMA assets are also catalogs. They are useful for selecting objects
but do not contain the required per-spaxel velocity maps. The next branch thus
needs explicit MAPS-cube or resolved-HI acquisition plus a documented
cross-match before it can be run honestly.
