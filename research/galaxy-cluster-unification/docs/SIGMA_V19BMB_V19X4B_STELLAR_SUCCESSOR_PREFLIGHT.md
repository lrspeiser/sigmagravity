# Sigma V19BMB V19X4B stellar-successor preflight

V19BMB preserves the complete V19BM stellar-morphology calculation while
changing only its future common-grid authority from the historical V19X4 chain
to V19X4B. A separately named successor is necessary because V19BM and V19BP
are already hash-bound; redirecting either file in place would invalidate the
preregistered evidence chain.

The successor retains all 4,096 member-posterior draws, the exact 241 by 241
physical grid, 50 and 100 kpc smoothing, cloud-in-cell deposition, unit-light
normalization and within-draw region percentile ranks. It still forbids
cross-filter amplitude comparisons and stellar-mass inference.

The preflight is target sealed. V19BMB cannot be frozen or executed until
V19X4B has passed every gas-posterior and common-grid gate with 12 hash-bound
products. No terminal gas value, stellar result, I4/I5 source score, lensing,
halo, galaxy, gravity parameter or holdout is opened here.

After terminal V19X4B and V19BMB both pass, a separately named V19BQ executor
may apply the already frozen V19BP source decision. A stellar-control pass is a
data-pipeline authorization, not evidence for modified gravity.

## Reproduction

```powershell
python scripts/check_sigma_v19bmb_v19x4b_stellar_successor_preflight.py
python -m pytest tests/test_sigma_v19bmb_v19x4b_stellar_morphology_control.py -q
```

The frozen preflight report is
`results/sigma_v19bmb_v19x4b_stellar_successor_preflight/report.json`.
