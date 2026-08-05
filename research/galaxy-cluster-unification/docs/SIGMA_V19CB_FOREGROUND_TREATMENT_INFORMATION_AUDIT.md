# Sigma V19CB foreground-treatment information audit

V19CB asks whether the Gaia foreground evidence found in V19CA actually makes
the H I/optical association decisive. It is explicitly post-source
exploration: the complete V19CA result was inspected before these branches
were written. No kinematic, lensing or gravity target was inspected.

Every one of the 711 release maps is evaluated under four counterfactual
treatments: retain every candidate, multiply quality-controlled foreground
objects by 0.1, set those objects to zero, or set every five-sigma astrometric
foreground object to zero. The last two are diagnostic masks only. They are
not authorized data edits because a foreground star can overlap a galaxy.

For every treatment, V19CB repeats all four H I beam-kernel rankings and
reports the smallest top-to-second margin and top-identity stability. No
treatment, candidate, counterpart or galaxy sample is selected.

## Result

All six execution gates passed and the output contains all 2,844
release/treatment combinations. Foreground handling improves the count of
kernel-stable 3:1 margins, but the strongest counterfactual reaches only 41 of
711 release maps (5.8%):

| Treatment | Robust 3:1 maps | Fraction |
|---|---:|---:|
| Retain all | 3 | 0.42% |
| Weight quality-controlled foreground by 0.1 | 34 | 4.78% |
| Diagnostic quality-controlled mask | 35 | 4.92% |
| Diagnostic mask of any five-sigma astrometry | 41 | 5.77% |

Even the most aggressive diagnostic leaves only 3 of 144 Norma maps robust,
and it leaves one Hydra map with no positive candidate at all. The median
minimum margin remains only 1.063. Foreground astrometry therefore explains
crowding and improves it somewhat, but does not identify the H I galaxy.

No mask branch is promoted. Optical pixels, survey masks, deblending evidence,
an independent source-only validation set and mixture propagation remain
necessary.

```powershell
python scripts/run_sigma_v19cb_foreground_treatment_information_audit.py
python -m pytest tests/test_sigma_v19cb_foreground_treatment_information_audit.py -q
```
