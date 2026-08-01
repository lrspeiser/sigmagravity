# P0586-P0586D continuous baryonic field metric

## Result in plain language

A continuous baryon-built metric is more useful than the retired endpoint
route, but it is not yet a joint theory. A signed, broad tidal response
preserved every nonlinear image root and reduced the four-cluster exact RMS by
2.01%. It helped MACS0329 and MACS1931 materially, but slightly worsened
MACS0429 and MACS1115, remained 1.76 times the compact-halo comparator, and
failed a mass-sheet audit in MACS1115.

The strongest new observation is structural: cluster lensing responds mainly
to the **signed tidal orientation**, then to scalar field strength and spatial
reach. The acceleration-transition scale and transition exponent were almost
irrelevant in these physical cluster maps because their projected,
smoothed-acceleration gate was already saturated. Galaxy rotation behaves in
the opposite way: only the scalar branch changes SPARC, while the selected
cluster branch sets that scalar response exactly to zero.

## Formula tested

The weak-field constitutive equation is

\[
\partial_i\!\left(K_b^{ij}\partial_j\Phi\right)=4\pi G\rho_b,
\]

with a positive baryon-built metric

\[
K_b=\epsilon(S)\exp\!\left[\tau S H Q_b\right],
\qquad
\epsilon(S)=1-(1-\epsilon_0)S,
\]

and low-acceleration activation

\[
S={1\over 1+(g_b/a_0)^n}.
\]

`Q_b` is the normalized trace-free Hessian direction of the smoothed baryonic
potential. The global baryonic quadrupole supplies a circularity gate,

\[
H={q_b^4\over q_b^4+0.05^4}.
\]

The matrix exponential keeps both eigenvalues positive for either sign of
`tau`. The lens calculation uses the first-order response

\[
\nabla^2\delta\Phi
=-\partial_i\!\left[(K_b-I)^{ij}\partial_j\Phi_b\right].
\]

It is solved as one scalar potential, so the correction is curl-free. Its
circular monopole is removed before adding it to the inherited raw-lens
parent, whose radial field is already fitted. This is therefore an angular
closure, not yet a standalone cluster mass prediction.

## Physical inputs

The four raw CLASH lenses use the same baryonic construction throughout:

- ACCEPT electron-density shells projected along the line of sight;
- square-root Chandra morphology with annular mass preservation;
- registered F160W starlight normalized to the published BCG stellar mass;
- no target-lens map in the construction of `K_b`; and
- no cluster-specific gravity parameter.

The state-variable smoothing length is `eta R80`, where each `R80` is measured
from that system's stars-plus-gas map. The selected `eta=0.8` corresponds to
roughly 215--259 kpc across the four clusters.

## P0586: five-coordinate factorial

The first screen crossed 324 universal combinations:

- `epsilon0 = 0.25, 0.5, 0.75, 1`;
- `a0 = 0.6, 1.2, 2.4e-10 m/s^2`;
- `n = 1, 2, 4`;
- `tau = -0.6, 0, +0.6`; and
- `eta = 0.2, 0.35, 0.5`.

The two selection clusters chose `epsilon0=1`, `tau=+0.6`, `eta=0.5`, with
`a0=2.4e-10` and `n=4`. Its fixed-geometry gain was only 0.473%. On the two
validation clusters the exact result reversed to a 0.476% loss, although every
root survived.

Near that selected point, the response hierarchy was:

| Coordinate | Selected-neighborhood RMS span |
|---|---:|
| Signed anisotropy `tau` | **0.12414 arcsec** |
| Minimum permittivity `epsilon0` | 0.05499 arcsec |
| Reach `eta R80` | 0.04820 arcsec |
| Transition power `n` | 0.00381 arcsec |
| Acceleration scale `a0` | 0.00047 arcsec |

For the selected cluster fields, `S` was 0.999--1.000 over nearly the entire
scored aperture. Thus `a0` and `n` were observationally unidentifiable in this
construction. This is a data result, not evidence that the particular locked
`a0` is fundamental.

The spherical galaxy limit gave a separate result. The best scalar grid point,
`epsilon0=0.25`, `a0=0.6e-10`, `n=2`, reduced SPARC outer RMS from 72.399 to
17.100 km/s and passed Solar, Earth, and Mercury controls. It remained far
behind fixed RAR at 10.348 km/s. A bounded far-field enhancement can supply
much of the amplitude but not the observed radial shape.

## P0586B-P0586C: boundary and sign response

The positive extension reached `tau=+1.2` and `eta=0.8`. Its selection winner
used `tau=+1.2`, `eta=0.65`, and no scalar response. It improved three fixed
geometries but worsened MACS1115 by 8.88%; no positive candidate improved all
four. Its apparent 0.60% exact validation gain lost a MACS1931 training root.

The signed audit then crossed 108 combinations over `epsilon0=0.25--1`,
`tau=-1.2--+1.2`, and `eta=0.2,0.5,0.8`. Four candidates improved all four
fixed-geometry scores. They all had:

- `epsilon0=1`, so no scalar permeability boost;
- `eta=0.8`, the broadest tested reach; and
- negative `tau=-0.3,-0.6,-0.9,-1.2`.

The locked common candidate `tau=-1.2` improved the fixed screen by 0.63%,
0.10%, 6.16%, and 3.01% in MACS0329, MACS0429, MACS1115, and MACS1931.

That compromise hides real conflict. Independent diagnostic optima were
negative for MACS0329 and MACS1115 but positive for MACS0429 and MACS1931.
Across all 108 candidates, MACS1115's response correlation was -0.572 with
MACS0429 and -0.677 with MACS1931. The common negative branch exists because
its small losses and gains balance, not because all systems prefer one tidal
orientation.

## P0586D: exact signed replay

The P0586C common candidate was locked before the exact scores. Zero and four
negative strengths were independently refit with 12 deterministic starts per
cluster and model.

| `tau` | Four-cluster exact RMS | Gain vs zero | All roots |
|---:|---:|---:|---:|
| 0 | 17.9844 arcsec | -- | yes |
| -0.3 | 17.9201 arcsec | 0.357% | yes |
| -0.6 | 17.8388 arcsec | 0.809% | yes |
| -0.9 | 17.6692 arcsec | 1.753% | yes |
| **-1.2 primary** | **17.6237 arcsec** | **2.006%** | **yes** |

The response is monotonic across the tested negative strengths and does not
owe its finite score to missing roots. The system-level primary result is:

| Cluster | Zero RMS | Primary RMS | Change |
|---|---:|---:|---:|
| MACS0329 | 19.9988 | 19.1116 | **4.44% better** |
| MACS0429 | 14.6391 | 14.7352 | 0.66% worse |
| MACS1115 | 24.6353 | 24.6443 | 0.04% worse |
| MACS1931 | 8.5204 | 7.2566 | **14.83% better** |

The primary remains far from the compact-halo comparator: 17.624 versus 9.989
arcsec, a ratio of 1.764.

## Numerical and degeneracy audits

- every primary training and held-out image root converged;
- maximum normalized curl was numerically zero;
- the minimum metric eigenvalue was 0.320, safely positive; and
- maximum affine-vector `R2` over observed image locations was **0.9924**, above
  the allowed 0.95.

The affine failure is localized to MACS1115:

| Cluster | Affine `R2` | Correction RMS | Baryonic asymmetry gate |
|---|---:|---:|---:|
| MACS0329 | 0.119 | 0.0738 arcsec | 0.548 |
| MACS0429 | 0.307 | 0.0396 arcsec | 0.413 |
| MACS1115 | **0.992** | 0.2392 arcsec | 0.949 |
| MACS1931 | 0.527 | 0.1832 arcsec | 0.898 |

Thus the strong response is not globally a mass sheet, but one cluster's
sampled correction is nearly affine. A next formula should separate local
baryonic tidal structure from long-wavelength affine modes using a
baryon-defined aperture, then repeat exact roots. It must not fit or subtract
an affine component at the observed image positions.

## Universal observations retained

1. A continuous positive metric is numerically cleaner and more effective than
   depositing redirected gravity at discrete endpoints.
2. Signed tidal orientation is the dominant cluster control in this branch.
3. Baryonic spatial reach remains important; the common signed response needs
   a broad scale near `0.8 R80`, not a fixed number of kiloparsecs.
4. The projected cluster gate saturates, so `a0` and `n` cannot be inferred
   from these maps.
5. A bounded scalar permeability greatly improves Newtonian galaxy curves but
   still misses RAR's radial behavior.
6. The best cluster candidate explicitly turns that scalar branch off, so the
   present formula does not unify cluster lensing and galaxy rotation.
7. Source-plane agreement can identify a useful direction, but exact roots,
   system-by-system signs, and affine-mode audits remain decisive.

## Decision

The continuous metric branch remains worth studying, but P0586's formula is
not promoted. The next bounded change is a **baryon-defined local/tidal
high-pass response**: remove only the correction's constant and symmetric
affine potential modes over a fixed fraction of measured `R80`, without using
image positions or a lensing target. That test will show whether the 2% exact
gain survives after the MACS1115 mass-sheet-like component is structurally
forbidden.

## Reproduction

```powershell
python scripts/run_p0586_continuous_baryonic_metric.py
python scripts/run_p0586b_metric_boundary_response.py
python scripts/run_p0586c_signed_metric_response.py
python scripts/run_p0586d_signed_metric_exact.py
python -m pytest -q tests/test_baryonic_metric.py tests/test_p0586_continuous_metric_results.py
```

Machine-readable outputs are in:

- `results/p0586_continuous_baryonic_metric/`;
- `results/p0586b_metric_boundary_response/`;
- `results/p0586c_signed_metric_response/`; and
- `results/p0586d_signed_metric_exact/`.
