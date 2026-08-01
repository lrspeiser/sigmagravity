# P0595: diffuse-boundary whole-galaxy cross-validation

P0593B selected the largest tested diffuse width and smallest tested mixing
fraction. P0595 extends both boundaries while retaining the fixed empirical RAR
scalar completion. It tests 216 combinations of width, fraction, and
low-acceleration gate steepness.

Each galaxy is assigned to one of five deterministic SHA-256 folds. A candidate
is chosen using four folds and predicted on the excluded fold. The combined
out-of-fold score therefore measures formula-family flexibility without using a
galaxy to select the formula that predicts it. It is not a fully nested nuisance
refit because the parent SPARC nuisance values are held fixed.

The out-of-fold improvement is also compared to measured galaxy properties:
baryonic mass, gas and bulge fractions, surface brightness, Hubble type,
force-equivalent concentration and size, and characteristic acceleration.
Those correlations are exploratory and use Benjamini-Hochberg FDR labels.

Run:

```powershell
python scripts/run_p0595_diffusion_boundary_cv.py
python -m pytest tests/test_p0595_diffusion_boundary_cv_results.py -q
```
