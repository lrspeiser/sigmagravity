# Adaptive resolved-galaxy policy and final holdout protocol

## What changed after validation

The frozen 256-coefficient Haar twin transferred well for NGC3521 but failed to preserve enough of NGC6946's resolved gas structure. P0748 varied only the number of baryonic-map coefficients and never used a velocity target or gravity residual for selection. NGC6946 first crossed the predeclared gas-map error threshold at 768 coefficients per component.

P0749 converted that observation into one universal rule: begin at 256 coefficients per component and increase through 384, 512, 768, and 1024 until fixed baryonic morphology, mass, 3D projection, and compression gates pass. The rule selected 256 for NGC2403, NGC3198, NGC5055, NGC7793, and NGC3521; it selected 768 for NGC6946. Formula transport was calculated only after each selection.

P0750 then applied one formula-independent first-harmonic kinematic-axis estimator to all six open galaxies. The fitted harmonic amplitude was discarded; only its unit phase was shared by every formula. Fixed simple MOND's registered-baryon median field error ratio was 1.15, with a maximum of 2.42. NGC6946 improved greatly but remained a miss, showing that observation geometry explained part, not all, of its original discrepancy.

## Frozen final holdout

NGC2841 and NGC7331 remain the final holdout. The following stages are committed before any of their pixel arrays are opened:

1. P0751A opens only THINGS moment 0 and SINGS IRAC1/weight arrays and performs velocity-blind registration.
2. P0751B applies the already-frozen SINGS/AllWISE fusion and opens no velocity target.
3. P0751C applies the P0749 adaptive Haar rule. Candidate selection may use only baryonic-map reconstruction, mass conservation, 3D projection replay, and compression.
4. P0752 opens THINGS moment 1 and moment 2 once, extracts systemic velocity and one kinematic phase identically for every formula, discards the fitted harmonic amplitude, and scores fixed Newtonian and fixed simple MOND predictions on both registered baryons and adaptive twins.

The P0752 field error ratio is prediction RMSE divided by declared measurement-plus-dispersion uncertainty. A ratio at or below 1 is consistent, 1 to 2 is close, and above 2 is a miss. Strict formula success requires both holdout galaxies at or below 1. A formula is competitive but incomplete only when its median and maximum are both at or below 2. Any galaxy above 2 is a holdout failure for the present raw 2D formulation.

Simulator fidelity is separate from formula accuracy. The adaptive twin must remain within 12 km/s RMSE of the registered-baryon prediction for every fixed formula. No distance, inclination, mass-to-light ratio, Newton constant, MOND acceleration scale, pressure-support term, warp, halo, or formula-specific parameter is fitted.

The holdout result may fail. Its method and labels must not be altered afterward to turn a failure into holdout evidence.
