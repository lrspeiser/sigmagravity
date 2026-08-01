# P0587 baryon-defined high-pass metric

## Outcome

Removing the broad affine part of the P0586D tidal correction does not isolate
a stronger local lensing signal. The predeclared high-pass field preserves all
nonlinear roots and remains 1.12% better than zero, but it is 0.23% worse than
the raw continuous metric. It also fails to remove MACS1115's image-sampled
affine degeneracy.

The failure is informative. The high-pass operator reduces the symmetric
affine fit over the baryonic `R80` grid aperture to about `1e-9` in `R2`, yet
the same field is still 0.9919 affine at MACS1115's sparse image positions.
The apparent mass sheet is therefore not a global long-wavelength component of
the baryonic correction. It is a local sampling coincidence or caustic
geometry effect. Subtracting it using the image positions would be lens-target
overfitting and is not a permissible theory construction.

## Target-independent formula

P0587 locks the P0586D base metric,

\[
K_b=\exp[-1.2\,S H Q_b],
\qquad L=0.8R_{80},
\]

with `epsilon0=1`. Inside a circular aperture derived only from the baryonic
centroid and `R80`, it fits either:

- a trace-only affine deflection; or
- a constant plus fully symmetric affine deflection.

The symmetric fit is the gradient of

\[
\psi_A={1\over2}A_{xx}x^2+A_{xy}xy+{1\over2}A_{yy}y^2+b_xx+b_yy.
\]

Its potential is multiplied by a cosine window equal to one inside the
aperture and zero at twice the aperture radius. Subtracting the gradient of
that windowed scalar potential keeps the response curl-free. No image
position, source position, lensing residual, or dark-matter map defines the
fit.

The predeclared primary removes the full symmetric affine field inside
`1.0 R80`. A 17-field diagnostic grid also varies trace versus symmetric mode,
aperture `0.75--1.5 R80`, and half versus full removal.

## Exact comparison

Zero, the raw P0586D metric, and the predeclared high-pass field were each
independently refit with 12 starts on all four clusters.

| Formula | Four-cluster exact RMS | Change vs zero | All roots |
|---|---:|---:|---:|
| Zero | 17.8740 arcsec | -- | yes |
| Raw signed metric | **17.6327 arcsec** | **1.35% better** | yes |
| Symmetric high-pass primary | 17.6737 arcsec | 1.12% better | yes |

The high-pass field is 0.233% worse than the raw metric and remains 1.769 times
the compact-halo comparator.

| Cluster | Zero | Raw metric | High-pass | High-pass vs zero |
|---|---:|---:|---:|---:|
| MACS0329 | 19.5989 | 19.3374 | 19.2879 | 1.59% better |
| MACS0429 | 14.6392 | 14.4820 | 14.7438 | 0.71% worse |
| MACS1115 | 24.6353 | 24.6443 | 24.6446 | 0.04% worse |
| MACS1931 | 8.5204 | 7.2552 | 7.2576 | 14.82% better |

The raw metric's exact aggregate gain differs from P0586D's 2.01% because the
12-start seed ensemble found a different MACS0329 nuisance basin. The
within-run raw-versus-high-pass comparison is the authoritative P0587 result.

## Why the affine audit survives

For the primary, the baryon-grid symmetric affine `R2` falls from roughly
0.018--0.030 before subtraction to `2e-9--7e-9` afterward. The construction
does exactly what it declares over the baryonic aperture.

Nevertheless, affine `R2` evaluated only at observed image locations is:

| Cluster | Raw metric | High-pass primary |
|---|---:|---:|
| MACS0329 | 0.119 | 0.106 |
| MACS0429 | 0.307 | 0.284 |
| MACS1115 | **0.9924** | **0.9919** |
| MACS1931 | 0.527 | 0.495 |

MACS1115's images sample a small part of a globally non-affine field where the
vectors happen to look linear. A global, baryon-defined high-pass cannot
remove that coincidence. An image-defined subtraction could, but it would
encode the test target in the formula.

## Parameter impacts

The fixed-geometry main-effect spans are small:

| Coordinate | Mean RMS span |
|---|---:|
| Aperture radius | 0.01859 arcsec |
| Trace versus symmetric removal | 0.01337 arcsec |
| Half versus full removal | 0.00425 arcsec |

The diagnostic screen slightly prefers full trace removal at `1.0 R80`, with
2.220% source-plane gain versus 2.213% for the raw metric. That difference is
0.00088 arcsec and does not replace the predeclared primary or justify another
exact fit.

## Decision

The simple affine high-pass branch is closed:

1. it does not beat the raw metric;
2. it does not remove the image-sampled affine warning;
3. it retains the same two-good/two-unfavorable exact cluster pattern;
4. it remains far from the compact-halo comparator; and
5. with `epsilon0=1`, it still leaves spherical SPARC at the Newtonian score.

The retained observation is narrower: a broad negative tidal metric produces
a root-safe, nonzero exact response in MACS0329 and MACS1931. The next useful
test is not another projection parameter. It needs either independently
measured missing baryons (satellite stellar mass and ICL) in the same field
operator, or an independent lens observable such as weak shear/magnification
that does not share the strong-lens image sampling geometry.

## Reproduction

```powershell
python scripts/run_p0587_baryonic_highpass_metric.py
python -m pytest -q tests/test_baryonic_metric.py tests/test_p0587_baryonic_highpass_results.py
```

Machine-readable outputs are in
`results/p0587_baryonic_highpass_metric/`.
