# P0593B: galaxy-level formula holdout

P0593 found a small in-sample gain when a conservative spatial redistribution
was applied before the empirical RAR scalar relation. P0593B asks whether that
gain survives on whole galaxies excluded from choosing the redistribution
settings.

The split is deterministic: the first eight hexadecimal characters of each
galaxy name's SHA-256 digest are converted to an integer; remainder zero modulo
four is the 40-galaxy formula holdout and the other 91 galaxies are discovery.
All 160 RAR-completed P0593 spatial candidates are ranked by discovery
equal-galaxy outer RMSE. Only the winning formula is then interpreted on the
holdout. This is a formula holdout, not a fresh nuisance-parameter holdout.

Run:

```powershell
python scripts/run_p0593b_diffusion_formula_holdout.py
python -m pytest tests/test_p0593b_diffusion_formula_holdout_results.py -q
```
