# SPARC Statistical Validation

This isolated research package tests the submitted Σ-Gravity rotation-curve result with galaxy-grouped statistics and frozen nuisance diagnostics. It does not modify the manuscript or production regression code.

Run from the repository root:

```powershell
python research/sparc_statistical_validation/run_validation.py
python -m pytest research/sparc_statistical_validation/tests -q
```

The design was frozen in `PREREGISTRATION.md` before the new statistics were computed. Generated tables, the decision record, and the diagnostic figure are written to `results/`.
