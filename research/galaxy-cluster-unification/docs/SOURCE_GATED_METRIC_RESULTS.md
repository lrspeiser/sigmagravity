# Source-gated metric test

## Result

The tested source-distribution gate fails. It is safe in the Solar System and
preserves the fixed-RAR galaxy score when placed only in the lensing potential,
but it makes held-out cluster radial predictions substantially worse than a
plain universal metric-slip amplitude.

The tested gate was

$$
F(r)=
\left({a_\dagger\over a_\dagger+2g_{\rm bar}(r)}\right)^2
[1-C(r)]^2
$$

or, more compactly, $F(r)=D(r)^2[1-C(r)]^2$.

Here

$$
C(r)={1\over 1+d\ln M_{\rm bar}(<r)/d\ln r}
$$

is near one outside a compact central source and smaller while baryonic mass is
still being added locally. No galaxy or cluster label enters the equation.

The metric-slip version was

$$
\Phi=\Phi_N+\phi_{\rm RAR},
\qquad
\Psi=\Phi_N+[1+2\kappa F(r)]\phi_{\rm RAR},
$$

so that

$$
g_{\rm lens}
=g_{\rm RAR}
+\kappa F(r)[g_{\rm RAR}-g_{\rm bar}].
$$

The single-potential control applied the same right-hand side to both matter
and light.

## Held-out results

The run used 131 SPARC galaxies (2,066 inner fitting points and 968 outer
points) and 20 CLASH clusters with 72 lensing-derived radial points. Complete
objects, not individual rows, were held out in five folds.

| Formula | SPARC outer RMSE | CLASH equal-cluster RMSE | Median fitted $\kappa$ |
|---|---:|---:|---:|
| Fixed-RAR matter + source-gated slip | 10.348 km/s | 0.203 dex | 25.877 |
| Fixed-RAR matter + constant slip | 10.348 km/s | 0.103 dex | 3.969 |
| Source gate applied to matter and light | 10.426 km/s | 0.522 dex | 0.039 |

The gated-slip error is 1.98 times the constant-slip error. A paired bootstrap
over complete clusters gives a gated-minus-constant RMSE difference of
+0.101 dex, with a 95% interval of +0.058 to +0.144 dex. None of 10,000
bootstrap draws favors the gate.

The single-potential formula also loses to fixed RAR on galaxy rotation:
+0.096 km/s in equal-galaxy RMSE, with a 95% interval of +0.050 to
+0.149 km/s. Galaxy inner data independently choose $\kappa=0$, whereas the
cluster radial data choose $\kappa=26.37$. This is the central incompatibility.

## Why it fails

The concentration idea changes the response in the wrong places for this data.
Across the 72 cluster points, the gate and the pointwise extra amplitude needed
to reach the target have correlation -0.675. The target needs the largest
extra response where this gate is smallest. One shared $\kappa$ consequently
over-boosts some parts of a cluster and under-boosts others.

Only 4 of 20 clusters improve relative to the constant-slip comparator. The
constant amplitude is also far more stable across folds:

- constant slip: $\kappa=3.87$ to $4.10$;
- source-gated slip: $\kappa=25.44$ to $28.89$.

The gate therefore does not explain the cluster-to-cluster or radial
differences. It suppresses a comparatively successful universal amplitude.

## Scope of the conclusion

This rejects the exact radial statistic $D^2(1-C)^2$ in either of the two tested
placements. It does not reject gravitational slip as a formal concept, nor
does it prove that stars and photons must have separate fundamental laws.

The CLASH target is derived from spherical GR plus per-cluster NFW fits, rather
than a raw shear or image-position likelihood. The concentration estimate is a
radial proxy, not a full three-dimensional, multi-centre baryonic field.
Nevertheless, the proxy failed its cheaper radial advancement test so a new
raw image-plane optimization was not run.

The next defensible direction is not another scalar concentration multiplier.
It is a direction-sensitive field sourced by the measured two-dimensional
baryonic geometry, with the same metric governing matter and light and any
slip derived from that field. That requires member-galaxy and gas maps; without
them, a tensor term would mostly absorb missing baryonic structure.

## Reproducible artifacts

- `configs/source_gated_metric_protocol.json`
- `src/voidscreen/source_gated_metric.py`
- `scripts/run_source_gated_metric.py`
- `tests/test_source_gated_metric.py`
- `results/source_gated_metric/report.json`
- `results/source_gated_metric/predictions.csv`
- `results/source_gated_metric/fold_fits.csv`
- `results/source_gated_metric/source_gated_metric.png`
