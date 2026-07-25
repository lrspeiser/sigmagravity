# SPARC photometric scale-length sensitivity

This folder contains the frozen outputs for the no-refit test of the candidate

\[
W(r,R_d)=\frac{r}{R_d/(2\pi)+r},\qquad B_{R_d}=A_0W(r,R_d).
\]

The calculation uses the catalog SPARC photometric scale length for each galaxy, the same baryonic assumptions as the locked manuscript comparison, and zero fitted parameters. The candidate replaces the endogenous factor in the locked predictor; it is not multiplied by that factor.

Run from the repository root:

```powershell
python "Publications/Frontiers/scripts/run_sparc_scale_length_sensitivity.py"
python -m pytest -q "Publications/Frontiers/scripts/test_sparc_scale_length_sensitivity.py"
```

The primary 164-galaxy results are:

- locked predictor mean RMS: 16.3657 km/s;
- catalog-scale-length candidate mean RMS: 16.5126 km/s;
- acceleration-only mean RMS: 16.8823 km/s;
- tested MOND prescription mean RMS: 16.0563 km/s; and
- catalog-scale-length minus locked paired mean: +0.1468 km/s, with galaxy-bootstrap 95% interval [−0.0881, +0.3854] km/s.

In 2,000 permutations of the catalog \(R_d\) assignments, the one-sided probability that a random assignment performs at least as well as the actual assignment is 0.7111. A common median \(R_d=2.3\) kpc gives mean RMS 16.4125 km/s. The measured galaxy-to-galaxy scale lengths therefore do not add detectable information through this particular window.

Files:

- `summary.json`: design, primary results, paired intervals, negative controls, and all-valid-points results;
- `per_galaxy_primary.csv`: galaxy-level residual statistics and provenance fields;
- `rdisk_permutation.csv`: frozen permutation distribution; and
- `bulge_threshold_sensitivity.csv`: 20%, 30%, 40%, and all-valid-point sample checks.
