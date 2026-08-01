# P0593-P0598 field-physics update

## Outcome

The tests isolate one repeatable observation: the *spatial placement* of the
apparent extra response contains information about the scale-free radial shape
of the baryons. A conservative broadening layer can improve both galaxy
rotation residuals and cluster lens-map morphology. It cannot yet replace the
scalar RAR relation in galaxies, and the cluster work has not yet tested
absolute lensing amplitude.

The current candidate is

\[
C={R_{50,b}\over R_{80,b}},\qquad
H(C)={1\over1+\exp[-(C-0.6)/0.1]},
\]

\[
S_4={1\over1+[g_b(R_{80})/a_0]^4},\qquad
f_{\rm eff}=0.3\,H(C)S_4,\qquad \ell=3R_{80,b},
\]

\[
\rho_{\rm eff}=(1-f_{\rm eff})\rho_b
 +f_{\rm eff}\,({\cal G}_{\ell}*\rho_b).
\]

Here $R_{50,b}$ and $R_{80,b}$ enclose 50% and 80% of the baryonic proxy,
${\cal G}_{\ell}$ is a positive normalized Gaussian propagator, and $a_0$ is
the fixed RAR scale. The mixture conserves total source weight. In the galaxy
diagnostic, the force-equivalent spherical mass from this spatial layer is
passed through the empirical RAR scalar relation. In cluster maps, the same
spatial layer acts on normalized baryon morphology.

## What each test changed

| Test | Concrete result | Interpretation |
|---|---:|---|
| P0593 conservative spatial response | Best no-scalar galaxy RMSE 68.87 km/s versus 10.35 for fixed RAR | Spatial redistribution alone does not supply the missing galaxy amplitude. |
| P0593B first formula holdout | 6.57% equal-galaxy gain on 40 unseen galaxies | A small spatial residual transferred on that split. |
| P0594 galaxy-locked cluster replay | 40.2% held-cluster morphology gain; 53.7% of cluster-tuned gain retained | The same spatial direction helps clusters, but clusters prefer stronger mixing. |
| P0595 extended whole-galaxy CV | 1.04% gain; 3/5 folds; 64.4% bootstrap probability | Ungated width/fraction tuning was unstable and remained at a width boundary. |
| P0596 radial-shape gate CV | 3.26% gain over fixed RAR and 1.82% over the no-shape family; 5/5 folds | $R_{50}/R_{80}$ carries transferable information, but only 55.7% of galaxies improved and the 89.5% bootstrap probability narrowly missed the frozen 90% gate. |
| P0597 one fixed post-hoc law | 6.08% galaxy gain; 19.5% cluster gain; 18.5% GLAFIC gain | One simple spatial law moves both domains favorably, but it was selected after disclosure. |
| P0598 acceleration screen | $n=4$ galaxy gain 6.41%; Solar proxy activation $4.94\times10^{-50}$; MACS J0416 activation 0.999976 | The acceleration gate is essential for stellar interiors and nearly inactive as a penalty in a measured low-acceleration cluster. |

The P0597 cluster replay used the low-acceleration limit $S=1$. P0598's
physically normalized MACS J0416 calculation changes that to 0.999976 for
$n=4$, so screening has a negligible effect there. Physical mass
normalizations are still unavailable for the other nine relative-light maps.

## What was learned about the parameters

The scalar completion is by far the dominant galaxy control. In P0593 its
median RMSE impact span was 61.14 km/s, while spatial strength changed the
median by 2.50 km/s and geometry by 1.95 km/s. That is why the present work has
not escaped empirical RAR.

Within the spatial residual, radial shape matters. P0595 improvement correlated
with $R_{50}/R_{80}$ ($\rho=0.233$, FDR $q=0.0245$), gas fraction, low bulge
fraction, lower surface brightness, and size. P0596 retained only the
cross-domain variable $R_{50}/R_{80}$ and improved every held fold. Larger
$R_{50}/R_{80}$ should be called a radial-shape ratio rather than simply
"greater concentration," because that interpretation depends on profile
family.

The acceleration exponent barely affects galaxy error but is not dispensable.
The ungated radial kernel substantially alters a uniform Solar interior proxy.
$n=2$ and $n=4$ suppress that effect below numerical precision, and $n=4$ has
the best galaxy score of the tested safe gates. This is an algebraic screening
test, not a solar or stellar-evolution model.

## What this does not establish

- The formula does not explain galaxy rotation without RAR/MOND-like scalar
  phenomenology. RAR is being used as an observational baseline, not derived.
- Normalized cluster-map similarity does not test the absolute amount of light
  bending, time delays, shear, or magnification.
- The ten-cluster baryon proxy omits hot gas, intracluster light, and
  stellar-population mass variation. MACS J0416 shows gas is the dominant
  baryonic component in its physical source field.
- The spherical force-equivalent galaxy calculation is not a three-dimensional
  disk solution.
- The Gaussian propagator is an endpoint rule. There is no action, stress-energy
  accounting, causal propagation equation, metric, or PPN limit yet.
- P0596-P0598 are discovery-stage analyses on already disclosed systems. Their
  constants need a new object sample.

## Next observations that discriminate the idea

1. Freeze the screened shape law and test a new rotation-curve sample with
   independently measured gas and stellar maps. Refit nuisance parameters in a
   nested holdout, not with the current inherited values.
2. Build several cluster baryon maps with physical hot-gas, stellar, and ICL
   masses. Calculate $C$, $g_b(R_{80})$, and therefore $f_{\rm eff}$ before
   reading their lensing scores.
3. Test absolute strong-plus-weak lensing: convergence amplitude, shear,
   magnification, image positions, and time delays. Normalized morphology alone
   is insufficient.
4. Replace the one-dimensional Solar proxy with a three-dimensional response in
   a standard solar model and compare against helioseismology and stellar
   structure.
5. Derive or falsify a scalar field equation that produces the galaxy amplitude
   without inserting RAR. Until this step succeeds, the work is an observed
   spatial correction to RAR, not new field physics replacing MOND.

## Reproduction

```powershell
python scripts/run_p0593_diffusion_cross_domain.py
python scripts/run_p0593b_diffusion_formula_holdout.py
python scripts/run_p0594_galaxy_locked_cluster_replay.py
python scripts/run_p0595_diffusion_boundary_cv.py
python scripts/run_p0596_radial_shape_gate_cv.py
python scripts/run_p0597_simple_shape_law_cross_domain.py
python scripts/run_p0598_stellar_interior_screen.py
python -m pytest tests/test_conservative_diffusion.py tests/test_p0593_diffusion_cross_domain_results.py tests/test_p0593b_diffusion_formula_holdout_results.py tests/test_p0594_galaxy_locked_cluster_replay_results.py tests/test_p0595_diffusion_boundary_cv_results.py tests/test_p0596_radial_shape_gate_cv_results.py tests/test_p0597_simple_shape_law_cross_domain_results.py tests/test_p0598_stellar_interior_screen_results.py -q
```
