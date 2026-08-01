# P0609 locked gas-route transfer to four raw clusters

## Outcome

The RX J2129 gas-directed route does not transfer as a universal raw-lensing
law. With no route parameter retuned, the standard single-plane version
improves the matched three-cluster aggregate from 21.386 to 19.893 arcsec
(6.98%), but only one of three comparable systems improves. MACS1931 still
loses one of four held-out roots, and the aggregate remains nearly ten times
the predeclared 2-arcsecond ceiling.

The whole aggregate gain comes from MACS0429, whose two held-out images improve
from 16.932 to 9.971 arcsec. MACS0329 worsens from 21.886 to 21.940 arcsec and
MACS1115 is unchanged at 24.624 arcsec. A one-cluster response with only two
held-out images is a morphology clue, not evidence for a universal field law.

## Locked formula

Every cluster uses its own observed Chandra morphology to define a direction,
but no score changes the formula:

$$
\boldsymbol\alpha=
\boldsymbol\alpha_{0599}+
0.0025\,\delta\boldsymbol\alpha_{\rm gas}
\left[{\beta(z_s)\over\beta(z_{\rm ref})}\right]^\gamma.
$$

The route length is $0.25R_{80}$, its landing width is $0.50R_{80}$, and the
angular field carries only the P0599 response above baryons-only GR. Both
$\gamma=0$ and the standard single-plane $\gamma=1$ were locked. The
convergence mean in every annulus and the circular radial-deflection mean are
removed, so the calculation does not add a radial halo.

| Variant | Complete systems | Equal-system held-out RMS | Matched improvement | Systems improved |
|---|---:|---:|---:|---:|
| P0599, no route | 3/4 | 21.386 arcsec | baseline | -- |
| Gas route, gamma=0 | 3/4 | 19.950 | 6.72% | 1/3 |
| Gas route, gamma=1 | 3/4 | 19.893 | 6.98% | 1/3 |

The gamma values are practically indistinguishable in transfer, consistent
with the P0608 non-identification result.

## The useful observation

MACS0429 is the only cluster where this tiny gas-directed correction matters.
Its gas direction was already one of the least aligned with its member-galaxy
direction in the earlier registered-map screen. This suggests a narrow
conditional hypothesis:

> A gas-direction term may become visible when gas and stellar/member
> structures are substantially misaligned and the available images cross that
> angular mode.

That hypothesis is not the current formula. A legitimate next test must define
the activation from baryonic data alone, for example

$$
A_{\rm mis}=1-\left\langle
\hat{\mathbf d}_{\rm gas}\cdot\hat{\mathbf d}_{\star}
\right\rangle_w,
$$

freeze a gate such as $H=A_{\rm mis}^n/(A_{\rm mis}^n+A_0^n)$ on historical
clusters, and score it on a different cluster without using its lens
residuals. The gate must not merely turn on for MACS0429 after seeing this
result.

## What fails and what remains open

This test rejects the simplest transferable statement that a fixed small
fraction of the excess field is redirected toward X-ray gas in every cluster.
It does not reject:

- a component-misalignment or merger-state gate fixed independently;
- a component-separated mass kernel using temperature-corrected gas surface
  density rather than X-ray brightness;
- a path law responding to the full baryonic tidal tensor rather than one
  attraction direction; or
- an off-plane field tested with time delays and outer shear.

The P0599 scalar parent itself is also a poor raw multi-cluster lens here:
three complete-system RMS values are tens of arcseconds. A tiny angular route
cannot repair a wrong radial/topological parent. Future arc tests need a raw
baseline that already produces all image families before angular physics is
judged.

## Reproduction

```powershell
python scripts/run_p0609_gas_route_multicluster_raw_transfer.py
python -m pytest tests/test_p0609_gas_route_multicluster_raw_transfer_results.py -q
```
