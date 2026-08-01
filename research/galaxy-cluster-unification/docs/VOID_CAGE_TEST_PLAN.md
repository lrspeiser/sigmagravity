# Void-cage hypothesis: frozen test plan

## Question

Can independently mapped underdense structure surrounding a galaxy predict an
inward differential acceleration that is small in the center and increases for
exterior tracers?

The proposal is not tested by adding an independently fitted halo-like term to
each galaxy. It is tested only if the amplitude of the term is predicted by a
void field constructed without rotation velocities or gravity residuals.

## Mechanisms

### C0: inverse-square shell null

For a repulsive source with force magnitude proportional to `1/d^p`, an
isotropic exterior shell produces, near its center,

```
Delta a = -kappa r,
kappa = A Q (p - 2) / (3 D^(p + 1)).
```

The inverse-square case `p=2` has zero isotropic compression. It is retained as
the analytic shell-theorem null.

### C1: faster-falloff external cage

The primary power-law alternative fixes `p=3`. The measured CF4 underdensity
cells between 7.8125 and 62.5 `h100^-1 Mpc` from each galaxy define the unit
Hessian. One universal positive coupling scales that Hessian. At galaxy scale
this predicts a harmonic addition `Delta v^2 = kappa_i r^2`.

### C2: finite-range Yukawa cage

The primary finite range is 15.625 `h100^-1 Mpc`, with one-, four-, and
eight-voxel ranges as robustness checks. The full exterior-source Hessian is
calculated. Its trace is the isotropic compression and its trace-free part is a
predicted anisotropy.

### C3: screened galaxy-scale conversion

Because an Mpc-scale Hessian is harmonic across a kpc-scale disk and cannot by
itself make a flat outer curve, the separate screened hypothesis is

```
v_pred^2 = v_bar^2
           + V0^2 E_i^m r^2 / [r^2 + (c_R R_d)^2].
```

`E_i` is the independently calculated cage compression divided by the median
of the training galaxies. `V0`, `m`, and `c_R` are global. The nested control
sets `E_i=1` and has the identical radial flexibility.

## Data and split

- CF4 grouped 64-cube: primary surrounding-void geometry.
- CF4 ungrouped 64- and 128-cubes: reconstruction robustness.
- SPARC: the existing quality/inclination cuts retain 131 whole galaxies and
  3,034 radial points.
- Five folds are balanced by the primary cage score. All radii of a galaxy are
  held out together.
- Stellar mass-to-light ratios are fixed at 0.5 for disks and 0.7 for bulges;
  the velocity uncertainty includes the existing 2 km/s floor.

## Benchmarks and controls

1. Newtonian observed baryons.
2. Fixed RAR at `a0=1.2e-10 m/s^2`.
3. Harmonic term with no environment information.
4. Screened radial term with no environment information.
5. The primary score randomly permuted among galaxies.
6. Alternative CF4 reconstructions, Yukawa ranges, and the `p=3` kernel.

The void origin is supported only if the real score improves prediction beyond
the same radial formula with no score and beyond shuffled scores.

## Frozen success conditions

The primary screened cage must simultaneously:

1. Reduce heldout RMSE by at least 5% relative to the screened no-environment
   model and fixed RAR.
2. Have at least 0.95 paired-galaxy bootstrap probability of improving chi
   squared against both.
3. Fit `m>0` away from its bounds in all five folds.
4. Beat at least 95% of 64 environment permutations.
5. Preserve the sign of the response in both ungrouped CF4 reconstructions.
6. Use no lensing-only normalization.

Failure is not repaired with per-object amplitudes, ranges, centers, or
galaxy/cluster switches.

## What each stage can establish

- C1/C2 can test the literal external-pressure geometry. A pass would show
  that a measured exterior void distribution predicts a compressive tide.
- C3 can test whether that external geometry predicts the strength of a
  galaxy-scale radial transition. A pass would establish predictive
  environmental phenomenology, not its microscopic cause.
- A lensing replay with the same potential can test whether the response bends
  light, but only raw shear/image likelihoods are theory-neutral enough to
  identify that response.

## Known ceiling

The current public-data audit has no strict-ready same system with complete
resolved dynamics, baryons, and theory-neutral lens covariance. Therefore this
run can falsify galaxy-scale cage formulas and audit lensing feasibility, but it
cannot establish galaxy-cluster unification. A population claim remains frozen
until ten same-system packages exist.
