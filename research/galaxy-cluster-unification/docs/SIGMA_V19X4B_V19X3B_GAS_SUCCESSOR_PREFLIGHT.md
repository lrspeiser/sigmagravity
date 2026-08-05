# Sigma V19X4B V19X3B gas-successor preflight

## Decision

The gas-posterior stage is now prepared for the V19W5-authorized regional
pipeline. V19X4B remains closed until V19X3B has passed all 494 regional fits,
but its implementation preflight passes now without opening a temperature,
gas, source-invariant, lensing, halo or gravity result.

This is a separately named successor. The original V19X4 configuration and
runner remain byte-preserved because they are parents of the frozen V19BP
source decision.

## Exact scientific inheritance

V19X4B mechanically copies and canonically hashes all eight scientific sections
from V19X4:

- the APEC emission-measure definition and corrected slab algebra;
- physical composition and unit-conversion constants;
- the two cluster geometries and all 494 accepted regions;
- 4,096 scrambled-Sobol posterior draws per region;
- the $-0.9$, $0$ and $+0.9$ temperature--normalization dependence branches;
- the line-of-sight depth prior and failed-profile retention rule;
- the 241-by-241, 10-kpc common physical grid;
- both 50- and 100-kpc smoothing branches, mass conservation and all runtime
  gates.

No scientific section can be changed by the future freezer. The only changed
input authority is the terminal regional report: V19X3B replaces the historical
V19X3 path.

## Terminal gates

The future freezer requires a hash-exact V19X3B config, runner, freezer and
report. The report must contain exactly 494 regions across Bullet and Abell
2146, every gate must pass, gas-source construction must be authorized and the
lensing/halo seal must remain false.

The resulting V19X4B run will produce six regional posterior products and six
common-grid products: two clusters times three dependence branches. Every
product is size- and SHA-256-bound. A pass authorizes a separately named
source-only scoring successor; it does not authorize an action or gravity fit.

## Verification

```powershell
python scripts/check_sigma_v19x4b_v19x3b_gas_successor_preflight.py
python -m pytest tests/test_sigma_v19x4b_v19x3b_gas_state_posterior.py tests/test_run_sigma_v19x4_gas_state_posterior.py -q
```

The frozen report is
`results/sigma_v19x4b_v19x3b_gas_successor_preflight/report.json`.
