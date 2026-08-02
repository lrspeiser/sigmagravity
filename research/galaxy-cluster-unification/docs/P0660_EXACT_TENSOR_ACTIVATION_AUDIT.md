# P0660 exact tensor activation audit

## Why this audit was necessary

P0659 correctly established the mathematics of the projected tensor-AQUAL
solver, but its quoted `18.655x` cluster/galaxy lever came from P0643's
component-cancellation proxy. P0659's proposed constitutive equation instead
uses a transverse component tensor. P0660 therefore froze a direct test of
the actual coefficient before opening any galaxy velocity or lensing target.

The tested coefficient was

\[
\sigma={a_0\over a_0+|g_N|}
\left[{2\sqrt{|g_\star||g_g|}\over |g_\star|+|g_g|}
\left(1-{(g_\star\cdot g_g)^2\over |g_\star|^2|g_g|^2}\right)\right]
\left(1-e^{-\ell/10\,{\rm kpc}}\right).
\]

## Frozen result

P0660 fails one and only one of its 17 preregistered gates:

- galaxy nominal median weighted `sigma`: `0.00891466`;
- cluster nominal median weighted `sigma`: `0.0732417`;
- exact cluster/galaxy separation: `8.21588x`;
- required separation: at least `10x`;
- weakest mass-map separation: `7.13247x` at the high stellar scale;
- maximum domain-median resolution change: `33.609%`, below the frozen `35%` limit;
- radial aligned activation: `2.49e-15`;
- minimum constitutive eigenvalue proxy: `1.31e-5`; and
- rotation and direction-reversal errors: numerically zero.

The candidate does not advance to real-map field solves. The `10x` threshold
was not relaxed after the result.

## What the failure teaches us

The tensor geometry is not absent. On these baryonic maps it is roughly eight
times more active in clusters than galaxies, and the reason is visible without
using target outcomes: the median coherence length is about `177 kpc` in the
clusters but only `1.23 kpc` in the galaxies. The transverse component mismatch
itself is similar in the two domains. The scale separation therefore comes
primarily from persistence, not from an intrinsically different local angle.

This points to a legitimate next question: whether persistence should be a
memoryless exponential survival law, as P0660 assumed, or a variance/phase
accumulation law that is quadratic at short path length. That question can be
tested entirely against structural gates while velocities and lensing remain
sealed.

## Claim boundary

P0660 does not test rotation curves or lens topology and cannot support an
observational gravity claim. The calculation remains projected and uses
thin-sheet Newtonian fields to construct the coefficient. The inherited 10 kpc
length remains phenomenological.

## Reproduction

```powershell
python scripts/run_p0660_exact_tensor_activation_audit.py
python -m pytest tests/test_tensor_activation.py tests/test_p0660_exact_tensor_activation_audit.py -q
```
