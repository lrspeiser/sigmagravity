# P0554 subcritical route-strength results

## Outcome

Reducing the angular gravity-route amplitude separates a small smooth position
effect from its image-creating caustic effect, but not by enough to make the
present formula competitive. The best topology-preserving grid value, eta =
0.30, improves the MACS1931 equal-family assigned-position RMS by only 0.468%.
The full-strength route improves it by 24.855%, but almost all of that larger
gain occurs after the model has begun predicting additional lensed images.

This is evidence about a mechanism, not a fitted or promoted law.

## Formula and frozen test

The existing conservative angular route was scaled without changing its shape:

$$
\boldsymbol{\alpha}_{\eta}
= \boldsymbol{\alpha}_{\rm radial}(s_{\gamma}=0.98)
+ \eta\,s_{\rm route}\,\delta\boldsymbol{\alpha}_{\rm A0279},
\qquad 0\leq\eta\leq1.
$$

In plain language, eta is a volume knob. Eta = 0 turns off the proposed
sideways redirection of baryonic gravity; eta = 1 restores its original
strength. The scan froze eleven values from 0.0 through 1.0 before seeing their
scores. At every value it independently refit the same six ordinary lens
geometry quantities from eight starts, profiled source positions, and searched
globally for every image of all seven MACS1931 source families.

No new per-cluster gravity parameter was fitted. Eta was scanned on already
examined MACS1931 data, so the best value is descriptive and must be transferred
unchanged before it can be treated as predictive.

## Results

| eta | Assigned RMS (arcsec) | Change vs eta=0 | Family 2 roots | Family 3 roots | Observable-surplus roots | Topology preserved? |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 21.662 | 0.000% | 3 | 5 | 2 | yes |
| 0.1 | 21.629 | +0.151% | 3 | 5 | 2 | yes |
| 0.2 | 21.579 | +0.380% | 3 | 5 | 2 | yes |
| 0.3 | 21.560 | **+0.468%** | 3 | 5 | 2 | yes |
| 0.4 | 21.584 | +0.357% | 3 | 5 | 2 | yes |
| 0.5 | 21.934 | -1.257% | 3 | 5 | 2 | yes |
| 0.6 | 21.942 | -1.295% | 3 | 7 | 4 | no |
| 0.7 | 21.455 | +0.951% | 3 | 7 | 4 | no |
| 0.8 | 21.486 | +0.811% | 3 | 8 | 5 | no |
| 0.9 | 21.406 | +1.180% | 3 | 7 | 4 | no |
| 1.0 | 16.278 | **+24.855%** | 5 | 7 | 6 | no |

The first root-count change occurs at eta = 0.60 in family 3. Family 2 does not
cross its relevant caustic until eta = 1.00. This corrects the earlier rough
impression that the transition began near 0.4.

The strict held-out observed-seed score remains infinite for eta below 1 because
only three of four held-out roots converge. The output also records an explicitly
partial RMS over converged roots (8.129 arcsec at eta = 0 and 8.078 at eta =
0.3), but this is not counted as success and cannot compensate for the missing
fourth root.

## What the data teach us

The route is doing two different things:

1. A weak smooth displacement exists below the caustic. Its best measured
   benefit is only about one-half of one percent.
2. A much stronger apparent fit improvement appears when the mapping creates
   new solution branches. Those branches raise the predicted count of
   potentially visible surplus images from two to six.

The attractive full-strength positional score is therefore largely entangled
with topology. Simply lowering the route strength does not recover most of the
benefit while avoiding the companion-image liability. A better next equation
needs to change the spatial shape or physical source of redirection, not merely
its amplitude.

Because this route is an angular, zero-monopole addition, every eta retains the
same SPARC rotation-curve, CLASH radial-profile, Mercury, and Solar proxy values
as the photon-softness 0.98 radial parent. That is a preservation result by
construction; it is not an independent success on those data.

## Limits and next falsifiable step

All eleven nuisance-geometry fits touch at least one allowed bound, the lens is
simplified, the root visibility screen is relative rather than a calibrated
completeness model, and MACS1931 is spent evidence. No formula is promoted.

The next clean test is to freeze eta = 0.30 and transfer it unchanged to the
other four raw clusters, then repeat the 27-family multiplicity audit. Passing
requires no new root-count liability and a consistent position improvement.
Failure would retire amplitude-only tuning and direct work toward a localized
route shape tied to independently measured baryonic morphology.

## Reproduction

```powershell
python scripts/run_p0554_subcritical_route_scan.py
python scripts/run_p0554_subcritical_route_scan.py --postprocess-only
python -m pytest tests/test_p0554_subcritical_route_scan.py -q
```

Machine-readable products are in `results/p0554_subcritical_route_scan/`.
