# NBP0-M1 disk-versus-bulge morphology result

## Outcome

The axisymmetric calculation confirms that a flattened disk and a spherical
bulge generate different three-dimensional fields.  It does **not** confirm the
specific prediction that a scalar void-permittivity boundary consistently gives
disk-dominated galaxies a larger inward acceleration at their outer stars.

This is not a single-parameter miss.  The test varied ten source and constitutive
quantities over broad ranges, used 128 exactly matched disk/bulge environments,
and then tested the frozen sign against SPARC without fitting a parameter to an
individual galaxy.  The synthetic sign gate and every primary empirical gate
failed.

![NBP0 morphology summary](../results/nbp0_morphology_summary.png)

## Tested weak-field equations

The tested scalar model was

\[
\nabla\!\cdot[\epsilon(X)\nabla\Phi]=4\pi G\rho_b,
\qquad
(1-L_X^2\nabla^2)X=\rho_b,
\]

with

\[
\epsilon(X)=\epsilon_0+(1-\epsilon_0)
\frac{(X/\rho_c)^Q}{1+(X/\rho_c)^Q}.
\]

High-density regions therefore approach ordinary gravity, while low-density
regions approach a lower permittivity and a stronger effective field.  This is a
weak-field constitutive benchmark, not a covariant theory or a derived lensing
law.

The baryonic source combined:

- a double-exponential stellar disk;
- a spherical Hernquist bulge; and
- an independently extended double-exponential gas disk.

The finite-volume solver works in axisymmetric cylindrical coordinates.  It
therefore resolves the radial field in the midplane and both the radial and
vertical fields above it; the calculation is no longer a one-dimensional slab
proxy.

## Frozen variation

The run contains 1,023 synthetic cases:

| Family | Cases | Purpose |
|---|---:|---|
| Baseline | 1 | Declared reference case |
| One-at-a-time | 54 | Direct response to each varied number |
| Disk/bulge geometry factorial | 200 | Bulge fraction x disk thickness x bulge scale |
| Sobol global sweep | 512 | Space-filling ten-dimensional variation |
| Matched environments | 256 | 128 environments, each solved as a pure disk and pure bulge |

The varied ranges were:

- stellar bulge fraction: 0 to 1;
- disk height / disk scale: 0.0625 to 0.8;
- bulge scale / disk scale: 0.05 to 0.8;
- gas fraction: 0 to 0.7;
- gas radial scale / disk scale: 1.5 to 4;
- gas height / disk scale: 0.0625 to 0.3;
- minimum permittivity: 0.03 to 1;
- dimensionless critical-density logarithm: -5 to +1;
- transition sharpness: 0.5 to 8; and
- smoothing length / disk scale: 0 to 3.

The synthetic source mass and disk scale are dimensionless.  Varying the
critical density through six orders of magnitude spans the corresponding source
density rescalings; it is not a fitted physical value.

## Numerical validation

All 12 focused tests pass.  They cover mass normalization, the exact constant-
permittivity limit, analytic spherical and disk-like Miyamoto--Nagai forces,
Helmholtz smoothing, the constitutive limits, and pure disk/pure bulge solutions.

Seven selected cases were repeated on a 128 x 160 reference grid after the
72 x 96 sweep.  The median acceleration-enhancement change was 0.053%, versus a
5% gate.  The maximum was 9.90% in the thinnest, sharpest transition case,
versus a 12% gate.  Both convergence gates pass.

## What changing the numbers does

### Overall amplitude

The minimum permittivity is the dominant outer-amplitude control.  Its Sobol
rank correlations with acceleration enhancement are -0.72, -0.86, and -0.94 at
4, 6, and 8 disk scales.  The critical density is next, with correlations
+0.54, +0.35, and +0.20.  Across all cases, the median enhancements are 4.77,
5.97, and 7.81 at those radii.

Most of that large number is the nearly uniform far-field factor
\(1/\epsilon_0\), not disk-specific focusing.  For example, the baseline has
\(\epsilon_0=0.1\) and outer enhancements near 10.  Multiplying out that global
factor leaves a geometry-only response close to unity.

### Morphology and radius

The baseline bulge-fraction scan gives:

| Stellar B/T | Enhancement at 1 Rd | 4 Rd | 6 Rd | 8 Rd | Change in 4--8 Rd speed slope |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.423 | 8.597 | 10.944 | 10.518 | +0.0605 |
| 0.3 | 1.310 | 9.522 | 10.816 | 10.447 | +0.0080 |
| 0.7 | 1.232 | 10.344 | 10.532 | 10.300 | -0.0250 |
| 1.0 | 1.162 | 10.183 | 10.203 | 10.127 | -0.0085 |

The disk is favored at 1, 6, and 8 disk scales in this one parameter slice, but
the bulge is favored around 4 disk scales.  Thus the formula moves the transition
radius rather than creating one stable disk-greater-than-bulge ordering.

The full matched-environment result is more stringent:

| Radius | Fraction disk > bulge | Fraction disk > bulge by at least 1% |
|---:|---:|---:|
| 1 Rd | 78.1% | 57.8% |
| 2.2 Rd | 35.2% | 24.2% |
| 4 Rd | 33.6% | 17.2% |
| 6 Rd | 54.7% | 26.6% |
| 8 Rd | 71.9% | 35.2% |

Only 37/128 environments (28.9%) favor the disk at 4, 6, and 8 disk scales
simultaneously, versus the frozen 80% requirement.  Thirty-three favor the bulge
at all three radii; another 52 flip from bulge at 4 disk scales to disk at one or
both larger radii.  The median disk-minus-bulge difference is effectively zero
at every outer radius.

Disk wins are more common for thinner disks, less gas, sharper transitions, and
shorter smoothing.  No setting makes the prediction parameter-independent.  In
particular, the all-radius disk-win fraction falls from 43.8% in the lowest
smoothing-length quartile to 6.2% in the highest.  Among the 11 matched
environments in which both disk and bulge rotation curves are descriptively
flat (absolute 4--8 disk-scale speed slope below 0.1), only 3 favor the disk at
all three outer radii.

### Above-plane force

At a probe near \((R,z)=(4,1)\) disk scales, the raw vertical-to-radial force
ratio is larger for the disk in 93.0% of matched environments.  Ordinary
Newtonian gravity gives the disk the larger ratio in 100% of them.  After dividing
out that Newtonian source-geometry effect, the constitutive change is larger for
the disk in 71.9% of cases, but only 39.8% exceed the bulge by at least 1%.

The strong raw directional difference is therefore real, but primarily ordinary
disk geometry rather than a new void-boundary amplification.

## Frozen SPARC test

The morphology catalog was built without inspecting observed rotation speeds.
It fits an exponential disk and Hernquist bulge only to SPARC's published
baryonic component fields.  The audit retains 113/175 systems, of which 17 have
a measured bulge.  Requiring at least two observed points beyond three disk
scales leaves 96 systems and 16 bulges.

For each galaxy the target is the median outer residual
\(\log_{10}(g_{obs}/g_{RAR,fixed})\).  The baseline controls baryonic mass, disk
scale, effective surface brightness, gas fraction, median baryonic acceleration,
and sampled radius.  The test then adds baryonic bulge fraction and fitted bulge
scale.  All folds hold out complete galaxies.

For the primary disk/bulge mass-to-light ratios 0.5/0.7:

- held-out baseline RMSE: 0.1182 dex;
- held-out RMSE with morphology: 0.1265 dex;
- relative improvement: -7.03%, versus a required +10%;
- one-sided whole-galaxy permutation p-value: 0.953, versus at most 0.05;
- standardized partial bulge coefficient: +0.0082, opposite the predicted
  negative sign; and
- median matched bulge-minus-disk residual: +0.1036 dex, with only 4/16 pairs in
  the predicted negative direction.

The result is insensitive to plausible stellar masses.  Across all nine
combinations of disk mass-to-light 0.3, 0.5, 0.7 and bulge mass-to-light 0.5,
0.7, 0.9:

- morphology worsens held-out RMSE in every case, by 4.46% to 7.48%;
- permutation p ranges from 0.901 to 0.961;
- no combination passes the predictive or significance gate; and
- the matched-pair direction ranges from only 12.5% to 50% support.

## Decision

NBP0-M1, the **scalar density-dependent permittivity morphology mechanism**, is
not supported.  Tuning its amplitude, transition density, sharpness, nonlocal
length, gas geometry, disk thickness, or bulge scale cannot turn its
radius-dependent sign into the required stable population prediction.  This is
best described as a structural-formula failure, not merely a failure to find the
right number.

The calculation does not reject every geometry-aware basin theory.  It says the
next theory would need a new directional degree of freedom--for example a tensor
constitutive response or a shape-dependent boundary field--rather than another
scalar retuning.  Such a branch must be frozen before looking at residuals and
must predict, from the same parameters:

1. radial rotation curves across disk and bulge morphologies;
2. vertical forces from gas flaring or stellar vertical kinematics;
3. polar-ring or off-plane dynamics; and
4. lensing amplitude and ellipticity without a separate lensing multiplier.

Those extra directional predictions are safeguards: an anisotropic term should
not be advanced merely because it can be chosen to increase radial acceleration
in a disk.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_nbp0_sparc_morphology.py
python scripts/run_nbp0_morphology_sweep.py
python scripts/run_nbp0_sparc_morphology_test.py
python scripts/plot_nbp0_morphology_results.py
python -m pytest -q tests/test_axisymmetric_permittivity.py tests/test_sparc_morphology.py tests/test_permittivity_morphology.py
```

Machine-readable outputs are in
`results/nbp0_morphology_sweep/`,
`results/nbp0_sparc_morphology_test/`, and
`data/derived/nbp0_sparc_morphology.csv`.
