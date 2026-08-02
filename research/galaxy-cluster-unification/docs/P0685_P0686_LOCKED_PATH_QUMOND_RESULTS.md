# P0685-P0686 locked path-QUMOND field and raw topology

Protocols frozen before scores: 2026-08-02  
Verdict: 3D numerical field **passes**; spent raw-image topology **fails**; no robustness or sealed validation

## Tested equation

The P0684 diagnostic generator was locked without further tuning:

\[
\nabla^2\Phi=\nabla\!\cdot\left[
\nu_0\!\left({|\nabla\Phi_N|\over a_0}\right)^{p(\chi_b,\eta_b)}
\nabla\Phi_N\right],
\]

\[
p=1+2{\chi_b^4\over\chi_b^4+(10^{-6})^4}
[\max(\eta_b,1)]^{-1/2},\qquad
\eta_b={|\Phi_N|\over r|\nabla\Phi_N|}.
\]

It has no RX J2129-specific gravity parameter and no fitted photon amplitude.
The 3D source is the registered P0670 HST-plus-Chandra baryonic map. Photon
deflection uses the same zero-slip diagnostic closure as P0671.

## P0685 3D field result

The finite-volume implementation passes its unit tests, including exact
reduction to fixed-RAR QUMOND when the two extra channels are removed and
covariance under Cartesian-axis permutation.

| Field metric | Result | Frozen gate | Verdict |
|---|---:|---:|---|
| Newtonian normalized residual | `8.89e-14` | `<=1e-10` | pass |
| Candidate normalized residual | `2.66e-14` | `<=1e-10` | pass |
| Exponent min / median / max | `1.052 / 2.632 / 2.873` | `[1,3]` | pass |
| Strong-lens median physical deflection | `10.12 arcsec` | `>=1 arcsec` | pass |
| Candidate/scalar-AQUAL deflection RMS | `3.336` | `[1.5,20]` | pass |
| Normalized deflection curl | `4.34e-16` | `<=1e-8` | pass |

All P0685 gates pass. This establishes that the equation is numerically
solvable and produces a strong curl-free field. It does not establish that the
field has the correct spatial structure.

## P0686 spent raw-image result

P0686 was frozen before the P0685 field was calculated. It fits only four
ordinary nuisances: a two-coordinate map center and two external-shear
components. Seven source coordinates are profiled from the training images.
There are zero fitted gravity parameters and zero fitted photon amplitudes.

| Raw-lens metric | Result | Frozen gate | Verdict |
|---|---:|---:|---|
| Training exact roots | `14 / 15` | `15 / 15` | fail |
| Spent-heldout exact roots | `6 / 7` | `7 / 7` | fail |
| Training / heldout RMS | `infinite / infinite` | each `<=3 arcsec` | fail |
| Missing-multiplicity families | `3 / 7` | `0` | fail |
| Exact-multiplicity families | `4 / 7` | acceptable total `>=5` | fail |
| Parity-diverse families | `5 / 7` | `7 / 7` | fail |
| Critical-curve families | `7 / 7` | `7 / 7` | pass |
| Observable-surplus families | `0 / 7` | `<=2` | pass |
| Nuisances near bounds | `2` | `0` | fail |

Both shear components land at `-0.25`, their hard bounds. The equation does
produce critical curves, unlike several earlier weak candidates, but it does
not reproduce the complete image topology or finite predictive residuals.
The candidate is not advanced to resolution or baryonic-map robustness.

## What failed physically

The local path coordinate creates a hollow response. In the P0685 midplane,
the baryonic potential remains finite while `r |grad Phi_N|` goes to zero near
the center. Therefore `eta_b` becomes large and the inverse-square-root factor
turns off the extra channels where central strong-lensing convergence is most
needed.

| Projected radius | Median physical deflection | Median midplane `p` | Median midplane `eta_b` |
|---:|---:|---:|---:|
| `0-15 kpc` | `4.27 arcsec` | `1.28` | `51.2` |
| `15-30 kpc` | `6.17 arcsec` | `1.47` | `18.0` |
| `30-50 kpc` | `8.10 arcsec` | `1.69` | `8.33` |
| `50-75 kpc` | `10.78 arcsec` | `1.94` | `4.47` |
| `75-100 kpc` | `13.37 arcsec` | `2.17` | `2.87` |
| `100-150 kpc` | `16.76 arcsec` | `2.44` | `1.83` |
| `150-225 kpc` | `20.11 arcsec` | `2.71` | `1.16` |

The radial screens rewarded the total annular amplitude but could not test
this two-dimensional inner-versus-outer topology. P0686 therefore supplies a
new necessary design constraint: a cluster enhancement must be strong without
making the channel exponent increase outward across the strong-lens region.

## Next formula generator

Do not tune the local singularity with an RX J2129 core radius. That would add
an object-sensitive scale. Instead test a nonlocal but baryon-computed system
coordinate:

\[
r_* = \underset{r}{\arg\max}\,[r g_b(r)],\qquad
\eta_*={|\Phi_b(r_*)|\over r_* g_b(r_*)},
\]

and replace local `eta_b(r)` with the single derived `eta_*`:

\[
p(r)=1+2{\chi_b(r)^4\over\chi_b(r)^4+(10^{-6})^4}
[\max(\eta_*,1)]^{-1/2}.
\]

`eta_*` is calculated from each baryonic field rather than fit to its outcome.
It retains cross-system concentration information but removes the central
coordinate singularity. It must first re-pass the spent galaxy, multi-cluster
radial, and Solar screens. Only then may a new frozen 3D topology run be made.

## Reproduction

```powershell
python scripts/run_p0685_locked_path_qumond_3d_field.py
python scripts/run_p0686_locked_path_qumond_raw_topology.py
python -m pytest tests/test_spatial_qumond_3d.py tests/test_field_solvers.py tests/test_metric_lensing_3d.py -q
```

Artifacts are in `results/p0685_locked_path_qumond_3d_field/` and
`results/p0686_locked_path_qumond_raw_topology/`.

## Claim boundary

RX J2129 is fully spent mechanism-development data. The P0685 numerical pass
and P0686 topology failure are not external validation. The zero-slip photon
closure is not a covariant metric theory. P0633 and P0640 remain sealed.
