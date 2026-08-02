# P0672 spent RX J2129 absolute raw topology results

## Frozen result: fail

The tensor field fails eight of 16 progression gates:

- scalar/tensor training RMS: `7.0869 / 7.0862 arcsec`;
- scalar/tensor spent-heldout RMS: `17.8373 / 17.8346 arcsec`;
- tensor training improvement: `0.0105%`, below the frozen `0.5%` minimum;
- tensor/compact-halo heldout RMS ratio: `7.032`, above `1.25`;
- all seven tensor families have missing multiplicity;
- each family has only one global root rather than the observed three or four;
- no tensor family has both parities or a critical-curve sign change; and
- all four ordinary nuisance parameters reach a bound.

The held-out stability gate passes because tensor and scalar are almost
identical, but that is not a success: both are far too weak and topologically
subcritical. No photon amplitude or gravity parameter was fitted.

## What the failure isolates

P0670 showed that the real baryonic multipoles activate the coefficient.
P0671 showed that the tensor field is numerically nonzero. P0672 now shows that
the constitutive form `mu(I-sigma h h)` uses that information too
perturbatively: a `0.168%` field change cannot turn the single-root scalar map
into the required multi-image topology.

The next admissible candidate must therefore change how a coherent path builds
the constitutive response. It must not lower these gates, rescale the photon
field, increase the baryonic mass after seeing the images, or fit a cluster
amplitude. One parameter-free possibility is to treat the measured tidal path
as repeated coherent opportunities:

\[
\epsilon=A_{\rm multipole}S_{a}C_\perp,\quad
N=(\ell/L_c)^2,\quad
\sigma_{\rm compound}=1-(1-\epsilon)^N.
\]

This replaces the saturating survival multiplier with a compound survival law.
It is exactly zero for a radial/co-centered system, remains screened at high
acceleration, and can be nonperturbative only where a long coherent path and
baryonic component mismatch coexist. It requires a new frozen coefficient
audit before another field or raw-lens calculation.

## Reproduction

```powershell
python scripts/run_p0672_spent_rxj2129_absolute_raw_topology.py
python -m pytest tests/test_p0672_spent_rxj2129_absolute_raw_topology.py -q
```
