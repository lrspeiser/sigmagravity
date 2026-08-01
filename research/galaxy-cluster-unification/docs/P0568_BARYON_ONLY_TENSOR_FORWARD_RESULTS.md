# P0568: baryon-only tensor forward results

## Question

P0567 showed that a positive local tensor could *represent* the relation
between baryon-sourced and lensing-inferred fields over most of 13 RELICS maps.
That inverse result was not predictive because it used the lens map to infer
the required response.

P0568 asks the harder question: can a tensor calculated only from baryonic
member-light geometry predict a lensing convergence map that it has not used
to construct itself?

The first-order field screen was

\[
\nabla_i[(\delta^{ij}+tQ_b^{ij})\nabla_j\Phi]=\Sigma_b,
\]

\[
\nabla^2\Phi=\Sigma_b-t\,\nabla_i
(Q_b^{ij}\nabla_j\Phi_N)+O(t^2).
\]

Every tensor was rescaled to spectral radius at most one and
\(|t|\leq0.9\), keeping the local operator positive-definite. No cluster had a
custom gravity parameter.

## Frozen formula variations

Nine baryon-only operators were tested:

1. Newtonian-field aligned;
2. surface-density-gradient aligned;
3. full normalized tidal Hessian;
4. tidal Hessian weakened in high-density regions;
5. isotropic response weakened in high-density regions;
6. tidal/gradient blend;
7. noncircular tidal contrast;
8. noncircular gradient contrast; and
9. noncircular blend.

Each was crossed with four source widths from 20 to 100 kpc and thirteen
couplings from -0.9 to +0.9. Selection used only seven development clusters.
Three P0568 holdouts were scored after selection. Those clusters had appeared
in older project analyses, so this is a new-analysis holdout rather than an
untouched-project sample.

## Primary P0568 result

The selected setting was

\[
Q_b=\frac{B_{50}}{B+B_{50}}\widehat{T}_b,
\qquad t=-0.15,
\qquad w=100\ {\rm kpc},
\]

where `T_b` is the normalized traceless Hessian of the baryonic potential and
`B50` is a fixed within-map median of the baryon proxy.

| Metric | Result |
|---|---:|
| Development mean JS | 0.07720 |
| Locked holdout mean JS | 0.03571 |
| Holdout gain versus development-selected local-light null | 8.01% |
| Holdout gain versus development-selected central Gaussian | 30.91% |
| Holdout mean Pearson | 0.789 |
| Maximum negative first-order raw fraction | 1.05% |

The 8.01% local-light gain missed the frozen 10% gate. The central-halo
comparison is favorable but weak: a broad central Gaussian is intentionally a
simple null, not a fitted dark-matter solution.

The direction transferred under the independent GLAFIC reconstruction. On the
three holdouts, mean JS was 0.03779 for the tensor, 0.04193 for local light,
and 0.06629 for the central null. The 100 Lenstool realizations per cluster
also retained the same qualitative result.

## Which changes mattered

The most important parameter was not tensor orientation. It was baryonic
extent.

| Local smoothing | Development JS | Holdout JS |
|---:|---:|---:|
| 20 kpc | 0.24293 | 0.17994 |
| 50 kpc | 0.11220 | 0.04689 |
| 75 kpc | 0.08636 | **0.03379** |
| 100 kpc | **0.07884** | 0.03882 |

Changing only the smoothing scale from 20 to 100 kpc improved development JS
by 67.5%. The best tensor improved the development-selected 100 kpc local map
by only 2.08% in the primary screen.

Only two operator families selected a nonzero coupling:

| Family | Selected t | Development gain vs local |
|---|---:|---:|
| low-density tidal | -0.15 | 2.08% |
| low-density isotropic | +0.15 | 0.66% |
| all other directional/noncircular families | 0 | 0% |

Large couplings were highly consequential but harmful. Across the frozen
amplitude grid, the development JS span was 0.248 for gradient alignment,
0.185 for noncircular gradient, 0.173 for the tidal/gradient blend, and 0.110
for the full tidal tensor. Their minima were nevertheless at zero. This is an
important parameter-impact distinction: a parameter can strongly change the
answer while having no useful setting.

## Boundary refinements P0568B and P0568C

The 100 kpc development choice hit the original width boundary. A frozen
refinement expanded widths to 200 kpc and refined the two surviving operators.
Development selected low-density tidal response at 125 kpc and `t=-0.30`:

| Metric | Refined result |
|---|---:|
| Development JS | 0.07441 |
| Development gain vs refined local width | 5.01% |
| Descriptive holdout JS | 0.04122 |
| Descriptive gain vs refined 125 kpc local | 13.48% |

A final interaction grid extended the coupling back to -0.9. The optimum
stabilized at `t=-0.30`, so it was not a boundary artifact. It still failed the
transfer stability rule: the refined holdout JS of 0.04122 was worse than the
original P0568 value of 0.03571. The selected tensor's ten-system score vector
was closest to an ordinary 125 kpc smoothed local map, with RMS separation
0.00719 in JS. Thus width and tidal coupling are partly degenerate.

The development and holdout samples prefer different effective extents:
development chooses 125 kpc, while the descriptive holdouts choose an ordinary
75 kpc local map. This sample-dependent scale is a stronger obstacle to a
universal law than the exact choice between the tested tensors.

## Galaxy and Solar transfer

The selected cluster coupling was transferred to 968 outer points in 131
SPARC galaxies through the same acceleration-screened spherical radial proxy.

| Model/proxy | SPARC outer RMSE |
|---|---:|
| Newtonian baryon-only | 72.40 km/s |
| P0568 selected `t=-0.15` | 65.59 km/s |
| Refined `t=-0.30` | 56.98 km/s |
| Fixed RAR comparator | 10.35 km/s |

The tensor direction improves a purely baryonic Newtonian proxy but remains
far from galaxy data. Explicitly noncircular operators preserve an
axisymmetric null, but then revert to the 72.40 km/s Newtonian prediction.
This is the central conflict: angular cluster redistribution does not supply
the radial galaxy force.

Solar screening is easy. For both selected couplings, the maximum fractional
force change from 1.6 Solar radii to Saturn is below (7\times10^{-13}), Earth
is at numerical-zero scale, and the Mercury precession proxy is effectively
zero. Passing Solar tests therefore does not distinguish the cluster
operators; the acceleration gate simply suppresses all of them in a strong
field.

## Numerical audit

- Maximum tensor spectral radius: 1.0.
- Maximum correction-integral fraction: (2.31\times10^{-4}).
- Eighteen P0567/P0568 and existing tidal-response tests pass.
- The explicitly noncircular tidal tensor is nearly zero on the numerical
  circular source. Gradient-derived noncircular maps retain discretization
  residuals under the square FFT grid; because every such family selected
  `t=0`, they do not affect a reported physical score. They should be rebuilt
  with a polar or finite-volume solver before any nonzero use.

## Universal observations from this cycle

1. **Inverse compatibility is much easier than forward prediction.** P0567's
   95.8% local tensor feasibility did not compress into a strong universal
   baryon-only operator.
2. **Effective baryonic extent is the dominant cluster-map coordinate.** Its
   impact is tens of percent, versus a few percent for the useful tensor term.
3. **Only a density-gated response transferred at all.** Pure orientation,
   full tidal, gradient, and noncircular variants selected zero.
4. **Strong anisotropy has high impact but the wrong sign or morphology.** It
   can radically change a map without improving it.
5. **Solar safety is not the bottleneck.** A universal acceleration screen
   makes these weak-field operators negligible in the Solar System.
6. **Galaxy amplitude is the bottleneck.** The same tensor remains 5.5--6.3
   times worse than fixed RAR on SPARC even after improving Newtonian gravity.
7. **The member-light map is incomplete.** A 75--125 kpc effective width may
   be absorbing hot gas and intracluster light rather than measuring a new
   spacetime propagation length.

## Best next observation

Do not add another member-only orientation parameter. Replace the phenomenological
75--125 kpc smoothing with registered measured baryons:

- X-ray gas surface density;
- BCG and intracluster light;
- member-galaxy stellar mass; and
- their measurement covariance.

Then ask whether the preferred smoothing collapses toward the instrumental
resolution and whether the low-density tidal coupling remains nonzero. If it
does not, P0568 measured missing baryonic extent. If it does, the tensor has a
more defensible physical target for raw lensing tests.

## Reproduce

```powershell
python scripts/run_p0568_baryon_only_tensor_forward.py
python scripts/run_p0568b_tensor_width_refinement.py
python scripts/run_p0568c_width_coupling_interaction.py
python -m pytest -q tests/test_p0567_baryon_flux_tensor_backtrack_results.py tests/test_p0568_baryon_only_tensor_forward_results.py
```
