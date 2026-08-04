# Sigma v3F scale-homology structural results

## Decision

The preregistered Sigma v3F Gram interaction is **retired**.  It passes every
algebraic, screening, per-mass, and resolution gate, but its median
distributed-to-compact response ratio is `7.0402`, below the frozen requirement
of `10` and below v3E's `8.9789`.

No observational holdout was opened.  This is a synthetic pre-action failure,
not a third raw cluster-topology failure.

## Frozen result

| Equal mass normalization | `65^3` ratio | `81^3` ratio | Per-mass gate |
|---:|---:|---:|---:|
| 0.3 | 5.4154 | 5.4147 | at least 2: pass |
| 1.0 | 34.9979 | 34.9224 | at least 2: pass |
| 3.0 | 7.0402 | 7.1333 | at least 2: pass |
| **Median** | **7.0402** | **7.1333** | primary at least 10: **fail** |

The resolution change is only `1.32%`.  The proportional null is
`3.07e-16`; the fixed nonproportional response is `0.14045`; the maximum random
potential is `0.97772`; rotation and analytic-gradient relative errors are
`2.99e-15` and `7.08e-8`; the quartic-onset error is `2.78e-6`; and the
high-field screen at `g/a_sigma=1e5` is `1e-20`.  All are inside their frozen
gates.

![Sigma v3F frozen structural audit](../results/sigma_v3f_scale_homology_action_audit/scale_homology_audit.png)

## Interpretation

The full five-dimensional STF shape difference adds response to both fixtures.
It helps the low-mass distributed fixture relative to v3E (`5.42` versus
`3.13`) but reduces the high-mass and median separation.  A compact oblate
galaxy is not scale-homologous throughout its interior, so the broader Gram
invariant activates where the orientation-only commutator remained quieter.

The useful part of the signal is therefore specifically scale-dependent
eigenframe rotation, not every departure from proportionality.  More
importantly, three distinct bounded scalar invariants have now failed the same
frozen morphology gate.  The project stop rule applies: do not add a fourth
scalar blend or tune the gate.

## Reproduction

```powershell
python scripts/check_sigma_v3f_scale_homology.py
python -m pytest tests/test_sigma_v3f_scale_homology.py -q
python -m ruff check src/voidscreen/sigma_scale_homology.py scripts/check_sigma_v3f_scale_homology.py tests/test_sigma_v3f_scale_homology.py
```
