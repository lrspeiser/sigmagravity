# P0562-P0563 direct tensor response

## Outcome

A direct fixed-geometry diagnostic initially appeared spectacular: a common
`t=-3.75` reduced a Jacobian-mapped local residual by **86.8-87.8%** in two
optimizer-seed ensembles. That is **not a gravity result**. P0563 showed that
the statistic was dominated by critical-curve matrix inversion.

When the same images, geometry ensembles, baryonic tensor, and coupling grid
are evaluated using unweighted source-plane closure—without ever inverting the
Jacobian—the common gain collapses to **0.169-0.178%**. Three clusters prefer a
positive perturbation near zero and one prefers negative. No formula is
promoted.

## What P0562 isolated

P0562 fitted the six zero-field lens-geometry parameters on training images in
two independent 12-start ensembles. It then held those parameters fixed,
reprofiled each training source at every `t`, and applied the source unchanged
to the held-out image. Unlike P0560/P0561, it did not refit lens geometry or
solve for an inverse-lens root.

Its local image residual used

$$
\Delta\theta_{\rm local}=A^{-1}\Delta\beta,
$$

where `A` is the image-to-source Jacobian. This is a standard first-order
linearization only when `A` is adequately conditioned. Strong-lensing images
deliberately sit near critical curves where an eigenvalue of `A` can approach
zero.

The raw P0562 common result was:

| Geometry ensemble | Zero local RMS | Best `t` | Best local RMS | Apparent gain |
|---|---:|---:|---:|---:|
| seed 1 | 113.96 | -3.75 | 14.99 | 86.85% |
| seed 2 | 122.73 | -3.75 | 15.01 | 87.77% |

MACS1931 alone had a zero-field local RMS of 226-244 arcsec, much larger than
its exact P0559 residual. That discrepancy was the warning that triggered the
frozen P0563 conditioning audit.

## Conditioning audit

P0563 computes the singular values of `A` at every held-out image and compares
the local residual to the raw source-plane separation. Across the grid:

- maximum inverse-Jacobian gain: **3,966x**;
- maximum local/source RMS ratio: **207x**;
- correlation between their base-10 logarithms: **0.877**;
- MACS1931 zero-field inverse gain: **331-357x**;
- MACS1931 zero-field local/source ratio: **60-65x**.

This directly explains the 87% P0562 effect: the tensor moves images into and
out of nearly singular Jacobian regimes. It does not demonstrate that it
predicts their observed positions.

## Conditioning-robust source-plane result

P0563 estimates each source as the simple mean of its training images after
ray shooting, then measures the held-out source separation. This is not a
final likelihood, but it answers the narrow directional question without
critical-curve amplification.

| Geometry ensemble | Zero source RMS | Best common `t` | Best source RMS | Gain |
|---|---:|---:|---:|---:|
| seed 1 | 11.5598 | +2.50 | 11.5403 | **0.169%** |
| seed 2 | 11.5407 | +2.75 | 11.5201 | **0.178%** |

The response is smooth and the two geometry ensembles are close, but the
optimum differs by one grid step and the gain is smaller than the exact-fit
basin variability observed in P0560/P0561.

Per-system directions are stable across both geometry ensembles:

| Cluster | Near-zero sign | Best spent-grid `t` | Individual gain |
|---|---|---:|---:|
| MACS0329 | positive | +2.75 / +3.00 | 0.15-0.17% |
| MACS0429 | **negative** | -6.00 boundary | 3.32% |
| MACS1115 | positive | +6.00 boundary | 1.91% |
| MACS1931 | positive | +3.00 | 1.58% |

The opposing MACS0429 sign and two boundary optima prevent a universal
interpretation.

## General lessons

1. **Near-critical image-plane linearization is not a safe screening metric.**
   It can manufacture order-unity or larger gains from a small source-plane
   change.
2. **Exact roots remain mandatory for final lens claims.** Source-plane closure
   is useful for direction, while exact image positions are needed for
   prediction.
3. **The physical tensor contains a weak, stable directional tendency, not a
   universal law.** Three signs agree, one does not, and the aggregate effect
   is about two parts per thousand.
4. **Percent-level exact gains must clear both basin and conditioning audits.**
   The current tensor clears neither at a useful universal strength.
5. **A different structural observable is required.** More amplitude tuning of
   this same tensor is unlikely to resolve the sign conflict.

Galaxy and Solar results remain unchanged only by the tensor's cluster-only
scope, not because P0562/P0563 derive a covariant environmental gate.

## Reproduce

```powershell
python scripts/run_p0562_accept_tensor_direct_response.py
python scripts/run_p0563_accept_tensor_source_plane_response.py
python -m pytest tests/test_p0562_p0563_direct_response_results.py -q
```

Machine-readable outputs are under
`results/p0562_accept_tensor_direct_response/` and
`results/p0563_accept_tensor_source_plane_response/`.
