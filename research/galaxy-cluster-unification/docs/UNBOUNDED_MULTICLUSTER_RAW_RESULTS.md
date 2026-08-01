# Locked unbounded-running laws on multiple raw cluster image catalogs

Status: completed failure investigation, 2026-07-29.

## Decision

Neither locked unbounded curvature-running law survives the multi-cluster raw
image-position transfer. The failure is large, is not repaired by the far-field
cutoff control, and is not consistently repaired by allowing a forbidden
per-cluster amplitude. Do not proceed directly to a covariant completion of
either exact weak-field formula.

The next theory-development target should represent the full spatial vector
sum of BCG, intracluster light, gas, satellite galaxies, and external structure.
The present spherical radial field plus one pseudo-elliptical deformation does
not contain enough angular information to test the user's multiple-source
gravity-vector idea in clusters.

## Frozen test

The two laws were copied unchanged from the galaxy/cluster sensitivity cycle:

- balanced control: `curvature_power_p2`;
- lensing-favored control: `curvature_additive_alpha10`.

No gravity amplitude or gravitational slip was fit. For each cluster, six
ordinary geometry parameters were fit on training images: ellipticity,
orientation, two center offsets, and two external-shear components. Source
positions were profiled independently for each source family.

For every source family with at least three images, the lexicographically last
image was held out. This produced 11 held-out images from four previously
unscored raw coordinate likelihoods:

- MACS J0329: 3 held-out images;
- MACS J0429: 2;
- MACS J1115: 2;
- MACS J1931: 4.

The image coordinates and spectroscopic source redshifts are raw observables,
but the systems are not fully external: all four contributed Tian-derived
lensing-profile rows to the earlier bridge that selected the universal
constants. RX J1347 is descriptive because every usable family has only two
images. RX J2129 is a spent prior diagnostic.

## Predictive result

| Model | Equal-system held-out RMS | Pooled reduced chi-square | Relative status |
|---|---:|---:|---|
| compact object-specific halo | **9.048 arcsec** | **142.62** | best declared control, still inadequate |
| additive curvature, `alpha=10` | 18.165 arcsec | 612.36 | fails |
| curvature power, `p=2` | 18.630 arcsec | 663.32 | fails |
| baryons in GR | 27.439 arcsec | 1495.23 | fails |
| fixed simple MOND | 28.188 arcsec | 1535.57 | fails; one root did not converge |

Per-system held-out RMS:

| System | `p=2` | `alpha=10` | compact halo |
|---|---:|---:|---:|
| MACS J0329 | 25.750 | 25.279 | 11.167 |
| MACS J0429 | 5.672 | 6.122 | 1.797 |
| MACS J1115 | 24.623 | 24.622 | 14.057 |
| MACS J1931 | 9.321 | 6.093 | 1.401 |

The two candidate laws beat fixed simple MOND but are about twice as poor as
the compact halo in aggregate, miss the absolute 0.75-arcsec gate by more than
an order of magnitude, and drive geometry parameters to their bounds. No
candidate survives.

## Far-field closure

The literal isolated-field integral was evaluated to 1 Gpc because an
unbounded effective coupling has no native finite cutoff. The declared control
truncated the isolated cluster at 3 Mpc.

- `p=2`: 18.630 to 20.730 arcsec, an 11.3% change;
- `alpha=10`: 18.165 to 18.092 arcsec, a 0.4% change.

Both remain within the predeclared 20% robustness threshold. The far tail is a
real theoretical ambiguity, but it is not the cause of this raw-image failure.

## Forbidden amplitude diagnostic

A post-failure grid allowed one multiplicative deflection amplitude per cluster,
selected on training images only. This cannot be a theory survivor; it asks
whether the main failure is merely normalization.

| Model | Selected amplitudes across four systems | Held-out RMS | Result |
|---|---|---:|---|
| `p=2` | 1.4, 1.0, 1.4, 0.7 | 10.340 arcsec | all roots, but insufficient rescue |
| `alpha=10` | 1.4, 1.4, 1.4, 0.7 | 7.153 arcsec on converged subset | one root fails; no rescue |

The required amplitudes span a factor of two and therefore are not universal.
Even the optimized diagnostic remains far above the absolute image-position
gate. This rules out a simple universal renormalization as the missing piece.

## What failed physically

The bridge asks only for the correct radial acceleration at a few spherical
radii. Raw multiple images also demand the correct angular deflection field.
The tested lens closure compresses every baryonic component into one radial
profile and then deforms that profile into one ellipse plus constant shear.
That loses:

- separate deflection vectors from cluster member galaxies;
- multiple cluster-scale mass concentrations;
- offsets between gas, BCG, intracluster light, and total baryons;
- localized perturbations near individual images;
- line-of-sight structures and correlated image covariance.

This explains why a law can look close on a derived radial cluster curve yet
fail raw positions. It also directly supports investigating the user's earlier
idea that several galaxy-scale gravity vectors interfere or add differently
inside a cluster. The current test did not implement that idea; it showed that
the information it would require cannot be discarded.

## Next data and equation target

Before adding another scalar exponent or deriving a full covariant action, build
a baryon-only two-dimensional source model for at least two clusters containing:

1. HST BCG and intracluster-light surface density with a declared stellar
   population model;
2. member-galaxy positions and luminosities under one universal stellar
   mass-to-light relation;
3. projected X-ray gas density and covariance;
4. independent multiple-image coordinates, redshifts, and measurement covariance;
5. a vector/tensor superposition rule that reduces to Newtonian/GR summation in
   the Solar System and has no fitted lensing-only multiplier.

Only if that spatial model predicts held-out images should its weak-field rule
be promoted into a covariant action and full PPN calculation.

## Reproducible artifacts

- `configs/unbounded_running_multicluster_raw_protocol.json`
- `scripts/run_unbounded_running_multicluster_raw.py`
- `results/unbounded_running_multicluster_raw/report.json`
- `results/unbounded_running_multicluster_raw/predictions.csv`
- `configs/unbounded_running_multicluster_failure_diagnostic.json`
- `scripts/run_unbounded_running_multicluster_failure_diagnostic.py`
- `results/unbounded_running_multicluster_failure_diagnostic/report.json`
