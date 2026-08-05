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

```powershell
python scripts/run_sigma_v19cb_foreground_treatment_information_audit.py
python -m pytest tests/test_sigma_v19cb_foreground_treatment_information_audit.py -q
```
