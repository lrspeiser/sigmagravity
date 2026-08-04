# Sigma v3D triaxial-memory structural results

## Decision

The preregistered Sigma v3D discriminant interaction is **retired as the frozen
structural mechanism**.  It passes every algebraic, high-field-screening, and
resolution gate, but it does not reliably distinguish an equal-mass compact
galaxy fixture from a distributed cluster fixture over the declared field
volume.

No observational holdout was opened.  This is a synthetic pre-action failure,
so it does not count as a third raw cluster-topology failure.

## Frozen result

For the `65^3` primary calculation, the distributed-to-compact integrated
response ratios were:

| Equal mass normalization | Ratio | Per-mass gate |
|---:|---:|---:|
| 0.3 | 1.1167 | at least 2 |
| 1.0 | 3.5205 | at least 2 |
| 3.0 | 0.9599 | at least 2 |

The median was `1.1167`, compared with the frozen primary requirement of `10`.
The `81^3` median was `1.0707`; the `4.13%` resolution change passes the `20%`
stability gate and confirms that the negative result is not a coarse-grid
artifact.

The invariant itself behaved correctly:

| Check | Result | Gate | Decision |
|---|---:|---:|:---:|
| Maximum STF trace residual | `2.78e-17` in the fields | `1e-10` | pass |
| Rotation-invariance relative error | `1.85e-14` | `1e-10` | pass |
| Analytic-gradient relative error | `4.20e-8` | `2e-6` | pass |
| Largest axisymmetric null value | `7.51e-16` | `1e-12` | pass |
| Fixed overlapping-tide response | `0.17044` | at least `1e-4` | pass |
| High-field screen at `g/a_sigma=1e5` | `1e-20` | at most `1e-18` | pass |
| Median morphology ratio | `1.1167` | at least `10` | **fail** |
| Minimum per-mass ratio | `0.9599` | at least `2` | **fail** |

![Sigma v3D frozen structural audit](../results/sigma_v3d_triaxial_memory_action_audit/triaxial_memory_audit.png)

## What failed physically

The discriminant correctly detects three distinct eigenvalues at a point, but
that is not the same as detecting a distributed cluster over an extended
field.  A compact oblate source is not pointlike throughout its interior and
therefore has triaxial tidal regions.  Conversely, the far field of a
multi-component cluster becomes increasingly pointlike.  Integrating the
sixth-order potential over the full central volume mixes both effects.

The result is also nonmonotonic in field strength.  At mass normalization one,
the high-acceleration screen suppresses the compact fixture enough to produce a
factor `3.52`.  At low mass neither fixture is substantially screened.  At high
mass the action integral receives enough response outside the compact core
that the ratio returns to approximately one.  The invariant is therefore not a
universal substitute for the earlier galaxy/cluster coherence label.

## Frozen post-failure diagnostics

A separately frozen diagnostic varied screen power, memory length, whether the
screen acts before or after memory, and the scored volume.  Across 216
combinations, 34.7% produced a ratio above ten, including extremely large
central-volume ratios when the compact response was driven numerically close
to zero.  This does not rescue v3D:

- screen, memory length, ordering, and volume each changed the ratio by more
  than a factor of three;
- the largest ratios were dominated by a vanishing compact denominator rather
  than a stable whole-field relation; and
- at the original full half-width of two, the frozen `p=4`, `L=1` ratios remain
  of order one to four depending on mass and ordering.

The useful clue is that applying a local screen to the *effect* of a memory
field produces much stronger central separation than screening the memory
source before it propagates.  That is a mechanism-selection observation, not
an accepted change to the failed formula.

![Sigma v3D post-failure diagnostics](../results/sigma_v3d_post_failure_diagnostics/post_failure_diagnostics.png)

## Next materially different invariant

The next structural candidate should measure rotation of tidal axes across a
physical scale rather than degeneracy of the eigenvalues at one scale.  Let

\[
\widehat{\mathcal E}_{ab}={\mathcal E_{ab}\over\mathcal E_*},
\qquad
\mathcal M_{ab}=(1-L_\Sigma^2\Box_{\rm ret})^{-1}
\widehat{\mathcal E}_{ab}.
\]

The commutator

\[
\mathcal C=[\widehat{\mathcal E},\mathcal M]
\]

vanishes when the local and scale-averaged tidal tensors share eigenvectors,
as they do for an ideal isolated spherical source.  It becomes nonzero when
nearby baryonic components rotate the tidal axes across `L_sigma`.  A bounded
quartic candidate is

\[
\mathcal V_{\rm mis}=
{\operatorname{tr}(\mathcal C^T\mathcal C)
\over
2(1+\operatorname{tr}\widehat{\mathcal E}^2)
(1+\operatorname{tr}\mathcal M^2)}.
\]

This potential begins at fourth order, leaves the quadratic GR/Sigma-v1
propagator unchanged, and directly targets the phase/orientation failure
measured in Sigma v3C.  It has not yet been run and is not an accepted action.

## Reproduction

```powershell
python scripts/check_sigma_v3d_triaxial_memory.py
python scripts/diagnose_sigma_v3d_failure.py
python -m pytest tests/test_sigma_v3d_triaxial_memory.py -q
python -m ruff check src/voidscreen/sigma_triaxial_memory.py scripts/check_sigma_v3d_triaxial_memory.py scripts/diagnose_sigma_v3d_failure.py tests/test_sigma_v3d_triaxial_memory.py
```
