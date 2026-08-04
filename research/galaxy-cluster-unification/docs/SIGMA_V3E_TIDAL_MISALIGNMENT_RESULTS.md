# Sigma v3E tidal-misalignment structural results

## Decision

Sigma v3E is the strongest nonlinear structural candidate tested so far, but
the exact preregistered potential is **retired** because it narrowly misses the
primary morphology-separation gate.

Its median distributed-to-compact response ratio is `8.9789`, compared with
the frozen requirement of `10`.  Every individual mass normalization passes
the weaker factor-two gate, and the median changes by only `1.85%` on the
higher-resolution grid.  The miss is therefore small but real under the
declared rule; the threshold is not changed after seeing it.

No observational holdout was opened, and this synthetic failure does not count
as a third raw cluster-topology failure.

## Frozen result

| Equal mass normalization | `65^3` ratio | `81^3` ratio | Per-mass gate |
|---:|---:|---:|---:|
| 0.3 | 3.1318 | 3.1316 | at least 2: pass |
| 1.0 | 26.4558 | 26.4286 | at least 2: pass |
| 3.0 | 8.9789 | 9.1449 | at least 2: pass |
| **Median** | **8.9789** | **9.1449** | primary at least 10: **fail** |

All mathematical proxy gates pass:

| Check | Result | Gate | Decision |
|---|---:|---:|:---:|
| Commuting-tensor null | `3.34e-34` | at most `1e-12` | pass |
| Fixed noncommuting activation | `0.002168` | at least `1e-4` | pass |
| Maximum random potential | `0.6981` | at most 1 | pass |
| Rotation relative error | `3.85e-15` | at most `1e-10` | pass |
| Analytic-gradient relative error | `1.85e-8` | at most `3e-6` | pass |
| Quartic-onset relative error | `2.78e-6` | at most `1e-4` | pass |
| High-field screen at `g/a_sigma=1e5` | `1e-20` | at most `1e-18` | pass |
| Maximum field trace residual | `1.42e-14` | at most `1e-10` | pass |
| Resolution change | `1.85%` | at most `20%` | pass |

![Sigma v3E frozen structural audit](../results/sigma_v3e_tidal_misalignment_action_audit/tidal_misalignment_audit.png)

## What we learned

The improvement over v3D is substantial.  The v3D eigenvalue-discriminant
median was only `1.1167`; directly comparing local and scale-averaged tidal
eigenframes raises it to `8.9789`.  This supports the v3C conclusion that the
missing cluster information is associated with spatial phase and orientation,
not merely wavelength or local triaxiality.

The remaining failure is concentrated at the ends of the acceleration range:

- At mass `0.3`, both compact and distributed fixtures are mostly unscreened.
  A compact oblate disk still changes its tidal eigenframe with scale, so the
  ratio is only `3.13`.
- At mass `1`, the compact high-field region is screened while the distributed
  system remains responsive, producing a strong factor `26.46`.
- At mass `3`, both systems develop screened regions and the ratio settles near
  nine.

Thus eigenframe rotation is a real discriminator but not a complete one.  It
ignores changes in the *shape* of the tidal tensor when the local and memory
tensors happen to share eigenvectors.

## Next invariant: scale homology in STF tensor space

A symmetric trace-free tensor is a point in a five-dimensional vector space.
Instead of testing only whether two such tensors commute, the next candidate
will test whether they are proportional.  Define

\[
I_E=\operatorname{tr}(\widehat{\mathcal E}^2),\qquad
I_M=\operatorname{tr}(\mathcal M^2),\qquad
J=\operatorname{tr}(\widehat{\mathcal E}\mathcal M).
\]

The Gram determinant

\[
\mathcal G=I_E I_M-J^2
\]

is nonnegative by Cauchy--Schwarz and vanishes exactly when the local and
memory tides are proportional.  A bounded quartic candidate is

\[
\mathcal V_{\rm hom}=
\mathcal S(g/a_\Sigma)
{I_E I_M-J^2\over(1+I_E)(1+I_M)}.
\]

It retains the desirable v3E properties—boundedness, quartic onset, local
Solar screening and no object label—but responds to eigenvalue-pattern changes
as well as eigenframe rotation.  It is a new candidate, not a post-hoc rescue
of v3E, and must receive its own committed protocol before calculation.

## Reproduction

```powershell
python scripts/check_sigma_v3e_tidal_misalignment.py
python -m pytest tests/test_sigma_v3e_tidal_misalignment.py -q
python -m ruff check src/voidscreen/sigma_triaxial_memory.py src/voidscreen/sigma_tidal_misalignment.py scripts/check_sigma_v3e_tidal_misalignment.py tests/test_sigma_v3e_tidal_misalignment.py
```
