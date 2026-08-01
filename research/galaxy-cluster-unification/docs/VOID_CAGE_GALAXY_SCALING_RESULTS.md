# Galaxy-dependent void screening: completed result

## Outcome

Galaxy baryonic mass predicts the required outer velocity scale extremely well,
but the CF4 exterior-void score adds no measurable predictive information. A
separate isolation check also finds no evidence that mass, surface density, or
concentration needs to move the transition radius beyond its existing scaling
with disk size.

The retained empirical relation is therefore an internal mass-amplitude law,
not a detected void or negative-gravity effect.

## Frozen design

The test used the same 131 SPARC galaxies, 3,034 rotation points, five
whole-galaxy folds, fixed mass-to-light ratios, and CF4 cage scores as the first
void-cage test. No `Vflat`, resolved observed velocity, residual, or fitted
per-galaxy missing-gravity amplitude entered the galaxy predictors.

The residual-blind predictors were

```
M     = [0.5 L3.6 + 1.33 MHI] / training-fold median
Sigma = [0.5 SBdisk] / training-fold median
C     = [Rdisk/Reff] / training-fold median
E     = CF4 cage compression / training-fold median.
```

The void candidate always had a matching internal-only control with identical
radial and galaxy-scaling flexibility.

## Primary results

| Model | Held-out RMSE (km/s) | Chi squared / point |
|---|---:|---:|
| Fixed RAR | 23.085 | 21.340 |
| Mass amplitude + surface transition | 23.189 | 22.885 |
| Mass amplitude + concentration transition | 23.363 | 22.815 |
| Existing size-only screened model | 42.707 | 103.734 |
| Local-acceleration screening | 48.571 | 133.450 |
| Newtonian catalog baryons | 60.721 | 206.779 |

The mass/surface model improves RMSE by 45.7% relative to the size-only model;
the mass/concentration model improves it by 45.3%. Both have bootstrap
probability 1.000 of improving chi squared against the size-only model. They are
close to, but do not outperform, fixed RAR over the full rotation curves.

At each galaxy's outermost measured point, the concentration model reaches
21.287 km/s RMSE and the surface model 21.394 km/s, compared with 21.374 km/s for
fixed RAR.

The tested local-acceleration switch

```
S(gbar) = 1/[1+(gbar/gstar)^n]
```

fails. Its screening power reaches the upper bound in four folds and its
held-out RMSE is worse than the existing size-only model.

## What drives the successful scaling

The isolation check gives every candidate the same formula

```
v^2 = vbar^2 + V0^2 M^eta r^2/[r^2+rt^2].
```

The mass-amplitude-only control uses `rt=cR Rd` and achieves 23.126 km/s RMSE.
Its fold parameters are highly stable:

| Parameter | Five-fold range | Mean |
|---|---:|---:|
| `V0` at the training-median mass | 97.1–101.1 km/s | 98.5 km/s |
| `cR` | 1.29–1.45 | 1.40 |
| Mass exponent `eta` in added `v^2` | 0.525–0.557 | 0.537 |

At large radius this means

```
v_extra proportional to M^(eta/2) approximately M^0.269.
```

That is close to the quarter-power velocity scaling of the baryonic
Tully-Fisher relation. The run has recovered a familiar mass-velocity
regularity in an explicit radial form; it has not identified its microscopic
cause.

## Transition isolation

| Transition driver added to the mass-amplitude model | RMSE (km/s) | Change from mass-only | Bootstrap probability of chi-squared improvement | Pass |
|---|---:|---:|---:|---|
| None: `rt=cR Rd` | 23.126 | — | — | Control |
| Mass: `rt=cR Rd M^alpha` | 23.187 | +0.061 | 0.096 | No |
| Surface density: `rt=cR Rd Sigma^beta` | 23.189 | +0.063 | 0.305 | No |
| Concentration: `rt=cR Rd C^gamma` | 23.363 | +0.237 | 0.505 | No |

The mass-transition exponent changes sign across folds. The surface exponent is
positive but very small, 0.009–0.064. The concentration exponent is consistently
negative, -0.342 to -0.165, but it does not improve held-out prediction. No
transition driver passes the frozen 5% improvement and bootstrap gates.

The supported transition scale is therefore approximately `1.4 Rd`, without a
detected additional dependence on total mass, central surface density, or the
tested concentration proxy.

## Void increment

Adding the primary CF4 void score to the local-acceleration, mass/surface, or
mass/concentration families produces no material improvement. In the primary
local family, the void exponent is exactly zero in all five folds. In both
catalog families it is also zero in all five folds.

For the primary family, only 73.4% of 64 shuffled void assignments are worse
than the real map, below the frozen 95% threshold. The response is not stable
across the two alternative CF4 reconstructions. Every incremental-void pass
condition fails.

The tiny numerical chi-squared advantage reported for the mass/surface void
model corresponds to less than one millionth of a km/s in RMSE and an exponent
of zero. It is optimizer-level numerical noise, not a physical effect.

## Scientific interpretation

The data support this empirical galaxy-scale predictor:

```
v_pred^2 = vbar^2
           + Vref^2 (Mbar/Mref)^0.537
             r^2/[r^2+(1.40 Rd)^2],
```

where `Mref` is approximately the sample median, `8.4e9 solar masses`, and
`Vref` is approximately 98.5 km/s under the frozen catalog mass definition.
Those rounded values summarize cross-validation fits; they are not a new
fundamental-law calibration.

A negative-gravity theory could attempt to derive why an exterior response is
converted with this mass scaling. The present data do not supply that link:
the measured void field can be removed from the formula without degrading a
single substantive prediction. Until an independent environmental observable
predicts deviations around the mass law, the conservative interpretation is a
baryonic mass-velocity relation rather than void pressure.

The next valid theory step is derivational, not another flexible galaxy fit:
derive the approximate `M^0.537` contribution and `1.4 Rd` transition from a
field equation, then freeze its constants and test independent galaxies or raw
same-system lensing. Adding per-galaxy amplitudes or more catalog exponents would
erase the predictive result.
