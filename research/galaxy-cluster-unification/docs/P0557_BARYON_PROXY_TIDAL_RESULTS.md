# P0557 registered-baryon proxy tidal results

## Outcome

Registered stellar and X-ray morphology contains a **small transferable angular
clue**, but not the missing cluster-gravity law. A prospectively selected
75%-starlight + 25%-square-root-X-ray contrast tensor with one universal
coupling, `t=+0.3`, reduced the held-out exact image-position RMS on two
different clusters from **18.4322 to 18.3927 arcsec** (**0.215%**). Every image
root remained present.

The gain is much too small to advance. The compact-halo comparator is
**9.9891 arcsec**, so the selected proxy remains **1.841 times** its error. It
also did not improve both validation systems: MACS1931 improved by about 2.16%,
while MACS1115 worsened very slightly. No formula is promoted.

## Equation tested

The locked radial starting point remains fixed RAR for matter and scalar metric
slip `s=5` for photons. The new field equation was

$$
\partial_i\left[(\delta^{ij}+tQ_{\rm proxy}^{ij})
\partial_j\phi_\Sigma\right]=S_\Sigma .
$$

In plain language, `phi_Sigma` is the extra radial field already required by
the galaxy law. `Q_proxy` describes preferred directions in the observed
baryons. The single universal number `t` says how strongly those directions
alter propagation. The first-order correction was solved as

$$
\nabla^2\delta\phi=-\partial_i
(Q_{\rm proxy}^{ij}\partial_j\phi_\Sigma),
$$

so the lens deflection remains the gradient of one scalar potential rather
than an arbitrary rotated force.

Each proxy tensor was normalized to have eigenvalue magnitude no greater than
one. Two operators were tested:

- `full` retains both the circular/radial tensor and non-circular structure;
- `contrast` removes the circular average and retains only departures from a
  round distribution.

## Data and prospective search

The registered maps were constructed without reading the new lens scores:

- CLASH F160W light, with known multiple images masked;
- point-source-masked Chandra 0.7-2.0 keV count-rate morphology;
- discrete cluster-member positions and relative light weights.

X-ray brightness is proportional to emissivity, not gas mass. It was tested
both linearly and after a square root. The square root is only a deliberately
naive density-like stress test under constant depth and emissivity. Every
component was normalized before mixing, so a brighter X-ray image could not
silently add more gravity.

The frozen factorial contained 9 morphologies, 2 tensor operators, and 6
nonzero values of `t`, for **108 fixed-source screens**. MACS0329 and MACS0429
selected four exact-root candidates. Only after that choice were MACS1115 and
MACS1931 scored for transfer. There were no per-cluster gravity parameters.

## Results

All four fixed-source finalists used the contrast tensor and `t=+0.3`. The
exact selection refit then gave:

| Candidate | Exact selection training RMS | All selection roots? |
|---|---:|---|
| Linear X-ray contrast | 3.735 arcsec | **no** |
| 75% stars + 25% sqrt-X-ray contrast | **5.783 arcsec** | yes |
| Zero tensor | 5.786 arcsec | yes |
| 50% stars + 50% sqrt-X-ray contrast | 5.790 arcsec | yes |
| Square-root X-ray contrast | 5.794 arcsec | yes |

The attractive 3.735-arcsec linear-X-ray fit is invalid because at least one
required exact image root disappeared. Among complete candidates, the chosen
mixture improved the discovery score by only **0.043%**.

Transfer to the two different clusters was:

| System | Zero tensor | Selected proxy tensor | Change |
|---|---:|---:|---:|
| MACS1115 | 24.6353 | 24.6391 | slightly worse |
| MACS1931 | 8.5204 | 8.3364 | about 2.16% better |
| Equal-system aggregate | 18.4322 | 18.3927 | **0.215% better** |

The 10, 20, and 30 kpc softening diagnostics preferred the frozen 20 kpc
choice. Moving to either neighbor worsened the fixed-fit validation diagnostic
by roughly 6-8%, so even the small signal is somewhat scale-sensitive.

## What the data teach us

1. **Non-circular structure is the only useful part of this construction.**
   Full tensors produced unit-coupling RMS corrections of roughly 12-13
   arcsec; contrast tensors stayed below 0.7 arcsec. The circular radial term
   overwhelms the perturbative channel instead of repairing the raw lenses.
2. **Smooth baryonic morphology may contain a weak angular clue.** Every
   discovery finalist used gas or gas+starlight rather than the member-only
   tensor, and the selected mixture transferred with complete roots.
3. **The clue is not universal at useful strength.** Almost the entire
   validation gain came from MACS1931, with no improvement in MACS1115.
4. **A local residual is not a valid lensing score.** The numerically best
   exact selection candidate lost an image root, repeating an important lesson
   from the earlier member-tensor tests.
5. **The brightness-to-density transformation is not identified.** Linear
   and square-root X-ray maps were nearly tied in the screen. Current data do
   not tell us which, if either, traces the physical environmental tensor.

## Galaxy and Solar controls

The fixed-RAR galaxy outer error remains **10.3485 km/s**, and the inherited
Solar proxy has maximum `|eta-1|=0` from the limb through Saturn. These values
are unchanged because P0557 explicitly defines `Q_proxy` as an external
cluster-environment tensor. That is preservation by scope, not a new success:
a final theory must define this environmental activation covariantly and show
why an isolated disk or the Solar System does not receive the same correction.

## Decisive next test and its disposition

P0559 subsequently used public ACCEPT electron-density shells to project a
physical `Sigma_gas` profile for all four lenses, modulated angularly by the
registered Chandra maps. That test worsened the aggregate by 1.41% and retained
the same mixed response signs. It also found a factor 4-14 disagreement between
the ACCEPT-integrated and Tian central gas masses.

The instrumental XMM X4 response package for RX J2129 already passes, but the
project's frozen terminal data-disposition gate does not authorize a new X5
population claim. If reopened with new data, X5 should resolve the gas-catalog
discrepancy and provide covariance; it should not be used to tune this already
spent tensor.

That would replace the arbitrary 75/25 morphology mixture with measured
baryonic weights. If the gain remains at the sub-percent level or changes sign
across clusters, this tensor route should be retired. If it becomes consistent
and much larger, the next task is a covariant environmental gate tested on
cluster lenses, isolated galaxies, non-axisymmetric disks, and the Solar
System.

## Reproduce

```powershell
python scripts/run_p0557_baryon_proxy_tidal.py
python -m pytest tests/test_p0557_baryon_proxy_tidal_results.py -q
```

Machine-readable artifacts are in `results/p0557_baryon_proxy_tidal/`.
