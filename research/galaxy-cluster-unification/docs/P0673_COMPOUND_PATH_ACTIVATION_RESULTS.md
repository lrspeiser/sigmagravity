# P0673 compound-path activation results

## Frozen result: pass

All 18 preregistered gates pass without adding a constant or fitting an
object-specific setting:

- registered galaxy/cluster nominal median activation:
  `5.33408e-5 / 0.555829`;
- nominal cluster/galaxy separation: `10,420x`;
- weakest separation under the three frozen mass sensitivities: `9,042x`;
- spent RX J2129 mass-weighted activation: `0.180394`;
- spherical, co-centered radial-null activation: `2.79e-21`; and
- global maximum activation/minimum constitutive-eigenvalue proxy:
  `0.999999 / 1.55905e-7`.

The elementary RX J2129 probability is only `0.00501`, but the measured path
contains a mass-weighted `69.8` coherence opportunities. Retaining the
unrouted fraction at each opportunity therefore builds a nonperturbative
cluster response while the much shorter galaxy paths remain almost inactive.

## What this establishes—and does not

P0673 fixes P0672's coefficient-level failure: the same baryonic geometry can
now enter the constitutive tensor strongly enough to deserve a new field
solve. It preserves exact spherical cancellation, acceleration screening,
positive definiteness, one universal setting, and zero per-object gravity
parameters.

It does **not** show that the resulting field converges, becomes stronger in
the needed direction, develops a critical curve, or reproduces any image. No
new lensing score was computed and both P0633 and P0640 remain sealed. The
compound survival rule is still a constitutive hypothesis rather than a
derivation from a covariant action or microscopic field dynamics.

## Reproduction

```powershell
python scripts/run_p0673_compound_path_activation.py
python -m pytest tests/test_p0673_compound_path_activation.py -q
```
