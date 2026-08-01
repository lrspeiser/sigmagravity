# P0610 dual-component misalignment driver

## Outcome

The P0609 transfer result is not evidence for a universal gas-directed gravity
route. Its entire aggregate improvement came from MACS0429. P0610 asks a
narrower question: is that response associated with hot-gas morphology pointing
away from both the discrete cluster members and the continuous stellar light?

There is a sharp same-data pattern, but it is dominated by MACS0429. Across the
four systems with finite raw held-out responses, the exploratory dual-
misalignment score has Pearson $r=0.963$ ($p=0.037$) and Spearman
$\rho=0.800$ ($p=0.200$). Omitting MACS0429 lowers Pearson $r$ to 0.230. These
numbers generate a frozen future prediction; they do not validate a field law.

## Candidate observable and gate

Let $c_{gm}$ be the luminosity-weighted directional cosine between the gas and
member-galaxy route fields, and $c_{gs}$ the corresponding cosine between gas
and continuous-starlight fields. Define

$$
A_{\rm dual}=\sqrt{\max(0,1-c_{gm})\max(0,1-c_{gs})}.
$$

This is large only when gas disagrees with *both* stellar tracers. A deliberately
sharp candidate gate is

$$
H(A_{\rm dual})={A_{\rm dual}^4\over A_{\rm dual}^4+0.3^4},
\qquad s_{\rm eff}=0.0025H.
$$

The threshold and fourth power were proposed after seeing the P0609 outcome.
They must therefore remain fixed on a new cluster and cannot be counted as a
successful fit to the present sample.

| System | $A_{\rm dual}$ | $H$ | Raw held-out improvement |
|---|---:|---:|---:|
| MACS0329 | 0.0359 | 0.0002 | -0.2490% |
| MACS1115 | 0.0622 | 0.0018 | -0.0002% |
| MACS1931 | 0.1428 | 0.0488 | unavailable (lost root) |
| RX J2129 | 0.1943 | 0.1496 | -0.1080% |
| MACS0429 | 0.5912 | 0.9378 | +41.1139% |

## Physical interpretation worth testing

The useful possibility is not that more gas produces a stronger route. It is
that separated baryonic components create a non-spherical field geometry in
which a small nonlocal redistribution can matter. In the proposed field
language, $A_{\rm dual}$ is an environmental observable inside the baryonic
kernel $K(\mathbf y\mid\mathbf x,E_b)$; it is not a new mass component and not
an object-by-object fitted amplitude.

That distinction matters. A universal gate makes a risky prediction: a fresh
cluster with $A_{\rm dual}\ll0.3$ should show essentially no route correction,
while one with $A_{\rm dual}\gg0.3$ should activate nearly the same
$s_{\rm eff}\simeq0.0025$ correction. Failure of that ordering on independent
raw lens data rejects this particular gate.

## Limits

- Only four systems have finite matched responses.
- MACS0429 supplies nearly all of the dynamic range and has only two held-out
  images.
- The HST and masked Chandra maps are morphology proxies, not calibrated
  stellar- and gas-mass maps.
- The direction statistic says nothing about arc height, propagation time, or
  the absolute field multiplier.
- A correlation found after inspecting the outcome is not independent evidence.

The first chronologically prospective transfer is now complete. A383 activates
the gate at effectively zero; MS2137 activates it at 0.595, but the latter has
no material valid response and fails one training root under both variants.
The gate fails all advance criteria and is not promoted. See
[`P0611_FROZEN_DUAL_MISALIGNMENT_TRANSFER_RESULTS.md`](P0611_FROZEN_DUAL_MISALIGNMENT_TRANSFER_RESULTS.md).

## Reproduction

```powershell
python scripts/run_p0610_dual_component_misalignment_driver.py
python -m pytest tests/test_p0610_dual_component_misalignment_driver_results.py -q
```

Machine-readable results are in
`results/p0610_dual_component_misalignment_driver`.
