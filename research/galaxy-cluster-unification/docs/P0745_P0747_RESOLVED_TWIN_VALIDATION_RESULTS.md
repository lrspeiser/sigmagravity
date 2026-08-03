# P0745-P0747: frozen resolved-twin validation

These stages test whether a fake galaxy built from the same baryonic observations as a real galaxy preserves a gravity formula's speed prediction, and whether either prediction matches the real two-dimensional H I velocity field.

The tests keep three errors separate:

1. **Galaxy fidelity:** did the synthetic gas-plus-star map reproduce the real baryonic map?
2. **Formula transport:** did the same fixed equation give nearly the same velocity field on the real map and the fake twin?
3. **Observed accuracy:** did the equation's prediction match the measured velocity field?

## Frozen sequence and leakage boundary

The development-selected method and all gates were committed before the validation pixels were opened. The validation systems are NGC3521 and NGC6946. The two holdout systems, NGC2841 and NGC7331, remain sealed.

- P0745A opened only THINGS moment-0, SINGS 3.6-micron light, and SINGS weights. It opened no velocity or dispersion target and used no gravity parameter.
- P0745B applied the already-frozen SINGS/AllWISE fusion. It opened no velocity target and used no gravity residual to choose an image calibration.
- P0745C applied the development-selected Haar-256 representation without trying 128 or 512 on validation.
- P0746 was committed before opening validation moment-1 and moment-2 arrays. It used the same fixed Newtonian and simple-MOND comparators as P0744, with no formula, halo, M/L, distance, inclination, pressure-support, bar, warp, or streaming fit.
- P0747 is explicitly post-reveal. It diagnoses viewing geometry with one shared kinematic-axis direction per galaxy, fitted independently of the gravity formula. It fits no velocity amplitude and no gravity parameter, and it does not replace the P0746 validation score.

## Baryonic-map and twin results

The SINGS-only registration failed its predeclared 90% footprint gate because NGC6946 covered only 83.2% of the H I disk; NGC3521 covered 96.6%. The frozen WISE fusion then reached 100% coverage for both. SINGS and WISE morphology agreed with Spearman correlations of 0.997 for NGC3521 and 0.812 for NGC6946.

The unchanged Haar-256 twin passed the total-map gates but failed the overall validation protocol because NGC6946's clumpy gas was under-resolved and its two-dimensional formula transport was too large.

| Galaxy | Total-map normalized error | Total pixel correlation | Fixed-MOND radial transport | Fixed-MOND 2D transport |
|---|---:|---:|---:|---:|
| NGC3521 | 4.86% | 0.9988 | 0.86 km/s | 2.74 km/s |
| NGC6946 | 14.58% | 0.9917 | 5.08 km/s | 11.65 km/s |

NGC6946's gas-only normalized error was 39.71%, above the frozen 35% gate. This failure is preserved rather than repaired with validation-driven coefficient tuning.

## Frozen raw velocity comparison

P0746 scored 31,029 validation H I pixels. RMSE is gas-weighted across the full two-dimensional field; the error ratio is RMSE divided by the declared RMS gas-dispersion-plus-channel uncertainty.

| Galaxy | Formula | Real baryons RMSE | Fake twin RMSE | Declared uncertainty RMS | Raw band |
|---|---|---:|---:|---:|---|
| NGC3521 | Newtonian baryons | 54.31 km/s | 54.65 km/s | 34.44 km/s | close |
| NGC3521 | Fixed simple MOND | 24.23 km/s | 24.51 km/s | 34.44 km/s | consistent |
| NGC6946 | Newtonian baryons | 66.01 km/s | 65.33 km/s | 12.26 km/s | miss |
| NGC6946 | Fixed simple MOND | 71.15 km/s | 70.02 km/s | 12.26 km/s | miss |

For NGC3521 the answer to the simulator question is yes: the velocity-blind twin and the real baryonic map produce nearly identical formula scores, and fixed MOND matches the raw field within its declared scatter. Newtonian baryons do not.

For NGC6946 the fake twin is not accurate enough under the frozen 2D transport gate, but it is not the main source of the very large raw residual: real-map and twin formula scores are similarly poor. The residual map shows a coherent orientation mismatch rather than random generator noise.

## Post-reveal geometry diagnosis

The formula-independent first-harmonic diagnostic found that the image-only and kinematic axes differed by only 0.21 degrees for NGC3521 but by 60.03 degrees for the nearly face-on NGC6946. The first harmonic explains 97.1% and 92.5% of their weighted velocity variance, respectively.

Using only the corrected direction, while keeping inclination and every formula amplitude fixed, changed the NGC6946 scores to:

| Formula | Frozen raw RMSE | Kinematic-axis diagnostic RMSE | Diagnostic error ratio | Band |
|---|---:|---:|---:|---|
| Newtonian baryons | 66.01 km/s | 39.14 km/s | 3.19 | miss |
| Fixed simple MOND | 71.15 km/s | 29.64 km/s | 2.42 | miss |

The viewing angle explains much of the NGC6946 failure and reverses the raw ranking: fixed MOND becomes better than Newtonian baryons. It still remains outside the declared uncertainty band, so geometry is not a complete explanation.

## What the simulator can now show honestly

The hosted evidence page exposes six resolved galaxies and 107,211 scored velocity pixels. For each selected galaxy and fixed comparator it shows:

- the real-versus-fake baryonic-map score;
- the formula's real-map-versus-twin transport error;
- the twin prediction versus the observed velocity field;
- the underlying radial curve;
- the original two-dimensional field atlas;
- for the validation galaxies, the post-reveal viewing-axis diagnostic and its explicit non-blind status.

The next defensible step is not to tune Haar coefficients or gravity parameters on these validation systems. It is to freeze a better observation-geometry policy on development plus validation, then open NGC2841 and NGC7331 once as the final holdout.
