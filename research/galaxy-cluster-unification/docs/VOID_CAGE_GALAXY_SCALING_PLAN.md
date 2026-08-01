# Galaxy-dependent void screening: frozen test plan

## Purpose

The first scalar void-cage test allowed galaxy size to set the transition radius
through `cR Rd`, but it did not allow total baryonic mass, surface density,
concentration, or local positive gravity to change the transition. This extension
tests those missing dependencies without assigning a fitted amplitude to each
galaxy.

The scientific question has two nested parts:

1. Does residual-independent galaxy structure improve whole-galaxy predictions?
2. After that structure is included, does the independently mapped void field
   add predictive information?

An improvement from galaxy structure alone is not evidence for negative gravity.

## Residual-blind predictors

All predictors come from SPARC Table 1 and exclude `Vflat` and every resolved
rotation velocity:

```
M = [0.5 L3.6 + 1.33 MHI] / training median
Sigma = [0.5 SBdisk] / training median
C = [Rdisk / Reff] / training median
E = CF4 exterior compression / training median
```

The 131-galaxy frozen sample has finite positive values for all four quantities.

## Model families

The existing size-only control is

```
v^2 = vbar^2 + V0^2 r^2 / [r^2 + (cR Rd)^2].
```

### Local positive-gravity screening

```
S(gbar) = 1 / [1 + (gbar/gstar)^n]
v^2 = vbar^2 + V0^2 S(gbar)                 [internal control]
v^2 = vbar^2 + V0^2 E^m S(gbar)             [void candidate]
```

This lets every galaxy cross the transition where its own baryonic acceleration
falls through one universal `gstar`.

### Mass amplitude and surface-density transition

```
rt = cR Rd Sigma^beta
v^2 = vbar^2 + V0^2 M^eta r^2/(r^2+rt^2)        [internal]
v^2 = vbar^2 + V0^2 M^eta E^m r^2/(r^2+rt^2)    [void]
```

### Concentration transition robustness

The same pair is repeated with `rt = cR Rd C^gamma`. This distinguishes a
surface-density transition from a structural-concentration transition without
combining every catalog quantity into one high-dimensional fit.

## Validation and decisions

All radii of a galaxy remain in the same one of the five previously frozen
folds. Parameters are fitted only on the other four folds. Predictors are
renormalized using training galaxies only. Fixed RAR, Newtonian baryons, and the
existing size-only screened model are retained as references.

A galaxy-dependent family is retained only if its internal-only version reduces
held-out RMSE by at least 5% relative to the size-only model and has at least
0.95 paired-galaxy bootstrap probability of improving chi squared.

A void origin is retained only if the void version independently:

1. Reduces RMSE by at least 5% relative to its matching internal-only model.
2. Has at least 0.95 paired-galaxy bootstrap probability of improvement.
3. Keeps `m` positive and away from both bounds in all five folds.
4. Beats at least 95% of 64 shuffled void assignments.
5. Preserves the response sign in both ungrouped CF4 reconstructions.

The fixed RAR comparison is reported but is not a pass condition. No failed
model will be repaired with per-galaxy strengths, favorable subsamples, or new
parameter bounds.
