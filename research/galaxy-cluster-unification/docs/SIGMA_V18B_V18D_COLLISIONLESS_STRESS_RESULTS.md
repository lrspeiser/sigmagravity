# Sigma v18B-v18D collisionless-stress results

## Decision

The tested **instantaneous projected collisionless-member stress** is retired
as the missing universal cluster field. The source was constructed without
lensing targets, used one shared stellar-mass rule and one adaptive estimator,
and was transferred unchanged in both directions between the already-spent
MACS J0416 and PLCK G287 clusters. It is numerically stable, but it fails every
physics gate except resolution stability.

This is a source-mechanism failure, not a rejection of baryonic assembly
history. Present-day projected member positions and line-of-sight velocity
variance do not determine a universal apparent-halo amplitude, convergence
pattern, or shear orientation.

No holdout was opened.

## V18B: replacement-pair readiness

The inherited minimum was 50 secure spectroscopic members inside 1,800 kpc and
within 4,500 km/s of the cluster median. Both spent clusters pass without
weakening that gate:

| Cluster | Input rows | Retained members | Main rejection |
|---|---:|---:|---|
| MACS J0416 | 247 | 231 | 15 outside the aperture and 1 outside the velocity window |
| PLCK G287 | 129 | 129 | none |

Both catalogs use the same declared F160W conversion,

\[
M_\star/M_\odot
=0.8\,10^{-0.4(M_{\rm AB}-4.55)},
\]

with no cluster-specific mass normalization.

## V18C: target-blind source construction

For member \(i\), the adaptive width is half the projected distance to its
eighth nearest selected member:

\[
\sigma_i={d_{i,8}\over2},
\qquad
K_i(\mathbf x)={\exp[-|\mathbf x-\mathbf x_i|^2/(2\sigma_i^2)]
\over2\pi\sigma_i^2}.
\]

The local velocity mean and dimensionless random-stress source are

\[
\bar v(\mathbf x)
={\sum_i M_iK_i(\mathbf x)v_i\over\sum_iM_iK_i(\mathbf x)},
\]

\[
q_{\rm member}(\mathbf x)
={\sum_iM_iK_i(\mathbf x)[v_i-\bar v(\mathbf x)]^2/c^2
\over\Sigma_{\rm critical}}.
\]

The neighbor rank was frozen before either map or lensing target was opened.
Ranks 6 and 12 are reported only as source-estimator sensitivities; they were
not selected by lensing.

| Cluster | Median \(\sigma_i\) | Source \(R_{50}\) | Source \(R_{80}\) |
|---|---:|---:|---:|
| MACS J0416 | 56.94 kpc | 169.31 kpc | 336.56 kpc |
| PLCK G287 | 56.03 kpc | 235.08 kpc | 305.29 kpc |

## V18D: one-coefficient transfer

The public GLAFIC convergence maps were used only after the v18C source maps
were frozen. Their documented \(D_{LS}/D_{OS}=1\) normalization was converted
to the frozen source redshift \(z_s=2\). The baryon-only GR convergence has
coefficient one. One nonnegative coefficient multiplying the frozen member
source was trained on one spent cluster and transferred unchanged to the
other; then the direction was reversed. Convergence and both shear components
were always derived from one E-mode potential.

| Quantity | MACS J0416 to PLCK G287 | PLCK G287 to MACS J0416 |
|---|---:|---:|
| Transferred coefficient | 5,680,669 | 397,967 |
| Full-field NRMSE | 2.26197 | 1.02966 |
| Residual-field NRMSE | 2.34820 | 0.98153 |
| Residual power closed | -4.51405 | 0.03659 |
| Residual shear alignment | 0.07802 | 0.44046 |
| Convergence correlation | 0.15783 | 0.48222 |
| \(R_{50}\) fractional error | 29.56% | 2.77% |
| \(R_{80}\) fractional error | 7.67% | 8.80% |

The two inferred coefficients differ by 1.15455 dex, or about a factor of
14.3. That is incompatible with one universal source strength.

The symmetric baryon-only baseline NRMSE is 0.99689. Adding and transferring
the member-stress source gives 1.75737, a 76.29% worsening rather than the
required improvement.

## Frozen gates

| Gate | Requirement | Result | Pass? |
|---|---:|---:|:---:|
| Symmetric full-field NRMSE | at most 0.500 | 1.75737 | no |
| Improvement over baryon baseline | at least 10% | -76.29% | no |
| Residual power, each direction | at least 0.25 | -4.514 / 0.0366 | no |
| Shear alignment, each direction | at least 0.50 | 0.0780 / 0.4405 | no |
| Every \(R_{50},R_{80}\) error | at most 25% | maximum 29.56% | no |
| Directional coefficient agreement | at most 0.15 dex | 1.15455 dex | no |
| Doubled-resolution change | at most 2% | 1.485% | yes |

## What this teaches about halo size

The source has a plausible cluster-scale spatial support, and three of four
transferred residual-field radii are within 10%. Combined with v17E's thermal
result, this says that the spatial reach of hot gas and member motion contains
information about the *extent* of the conventional halo-like field.

But extent alone is not a root equation. A successful law must also determine
the field strength, centroid, and two shear components. Here, a coefficient
can change strength but cannot move the source into the missing phase, and the
coefficient itself fails universality by a factor of 14.3.

The immediate inference is therefore narrow:

> Current-time local baryonic stress is a useful halo-size tracer, but it is
> not the universal gravitational source tested here.

The next branch may examine a measurable causal assembly state, but it cannot
simply add another scalar memory or smoothing length. Those mechanisms have
already been tested in v3-v6 and v11. A new branch must first identify a
history-dependent baryonic observable that adds information absent from both
the thermal and collisionless snapshots.

## Claim boundary and provenance

- These GLAFIC maps are conventional inverse reconstructions, not raw image
  positions and not an untouched validation sample.
- Only line-of-sight velocities are observed; transverse velocities and
  line-of-sight member positions are not reconstructed.
- The adaptive kernel is a measurement estimator, not a fundamental
  propagation law.
- No galaxy, cluster, or Solar-System success is claimed from this test.

The GLAFIC archive convention is documented by the
[STScI Frontier Fields archive](https://stdatu.stsci.edu/prepds/frontier/lensmodels/)
and its
[MACS J0416 GLAFIC v4 readme](https://stdatu.stsci.edu/missions/hlsp/frontier/macs0416/models/glafic/v4/hlsp_frontier_model_macs0416_glafic_v4_readme.pdf).

## Reproduction and integrity

```powershell
python scripts/audit_sigma_v18b_replacement_pair_readiness.py
python scripts/build_sigma_v18c_collisionless_stress_maps.py
python scripts/run_sigma_v18d_collisionless_stress_transfer.py
python -m pytest -q tests/test_sigma_v18b_replacement_pair_readiness.py tests/test_sigma_v18c_collisionless_stress_maps.py tests/test_sigma_v18d_collisionless_stress_transfer.py
```

Report SHA-256 values:

- v18B: `56ebe3b0ecb780b2523955b77af6dcbd9d49be336f7bc1e261b27b02293fb315`
- v18C: `ff7a4ff915f8796c801c015609ce88e626dbbbe1dff5eba52da1ba88ea8d288b`
- v18D: `73b3d136b040cc0aca64b179dde5d4c8f366fa5a75db0ba5ca3a3bf5e15ce2cd`
