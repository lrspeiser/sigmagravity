# P0597: one simplified shape law across current domains

P0597 freezes a deliberately simple representative of the P0596 basin:

- `C = R50/R80`
- `H = logistic((C-0.6)/0.1)`
- `f_eff = 0.3 H`
- `ell = 3 R80`

The acceleration gate is removed because it was P0596's weakest parameter.
Galaxies apply the fixed empirical RAR scalar relation after this conservative
spatial redistribution. Clusters apply the same spatial law to normalized
member-baryon maps. The finite-source Solar check uses the same spatial law.

This is a post-hoc simplification: P0596 results informed all four constants.
It is useful for measuring whether one clean law points in the right direction
in every current domain, but it is not a fresh formula holdout.

P0598 subsequently showed that removing the acceleration gate is not safe for
the interior of a Solar-density source, even though exterior planetary forces
remain unchanged. The screened `n=4` version supersedes the ungated law as the
physical candidate; P0597 remains the exact cluster-map replay of its
low-acceleration spatial limit.

Run:

```powershell
python scripts/run_p0597_simple_shape_law_cross_domain.py
python -m pytest tests/test_p0597_simple_shape_law_cross_domain_results.py -q
```
