# P0599: bounded potential amplitude

P0598 supplies a screened spatial redistribution but not the absolute cluster
field amplitude. P0599 adds the smallest bounded amplitude carrier suggested by
the existing parameter ranking:

`P(chi) = 1 / (1 + (chi_t/chi)^p)`, where `chi = Phi_b/c^2`.

The prediction is `RAR(g_spatial) * [1 + A S4 P(chi) W]`. Five choices for `W`
test whether potential acts alone, through radial shape, through the bounded
potential path length, or through their product. The P0598 spatial layer is
also compared with leaving the baryonic profile local.

All settings are universal. Within each of five whole-object folds, selection
minimizes training-cluster error subject to training-galaxy RMSE no more than
2% above fixed RAR. Held galaxies and held clusters then receive the selected
formula unchanged.

The 84 CLASH points include measured gas, BCG, and cluster-galaxy baryons and a
total acceleration derived from strong lensing, weak shear, and magnification.
The target is nevertheless a spherical NFW deprojection rather than a raw
lensing likelihood, and the table omits radial covariance.

Run:

```powershell
python scripts/run_p0599_bounded_potential_amplitude.py
python -m pytest tests/test_p0599_bounded_potential_amplitude_results.py -q
```
