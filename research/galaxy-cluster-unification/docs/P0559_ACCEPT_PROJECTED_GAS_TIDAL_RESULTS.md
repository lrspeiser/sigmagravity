# P0559 projected ACCEPT gas tensor

## Outcome

A tensor built from a physically projected X-ray gas-density profile does not
repair the four raw cluster lenses. The held-out exact image-position RMS
changes from **17.8810 to 18.1325 arcsec**, a **1.41% worsening**. Every image
root remains present, so the failure is not numerical.

The equation and coupling were inherited unchanged from P0557:

$$
\partial_i\left[(\delta^{ij}+0.3Q_{\rm contrast}^{ij})
\partial_j\phi_\Sigma\right]=S_\Sigma .
$$

There are no new or per-cluster gravity parameters. The new input is a
baryonic surface-density map rather than an arbitrary star/gas morphology
mixture.

## Physical map construction

For every cluster, the published ACCEPT electron-density shells were treated
as piecewise constant and projected exactly along the line of sight:

$$
\Sigma_{\rm gas}(R)=2\mu_e m_p\sum_k n_{e,k}
\left[\sqrt{r_{k,\rm out}^2-R^2}
-\sqrt{\max(r_{k,\rm in}^2-R^2,0)}\right] .
$$

The circular radial mass profile comes entirely from measured electron
density; ACCEPT's hydrostatic `Mgrav` column is never used. A smoothed,
point-source-masked Chandra image supplies only angular contrast. Its mean is
forced to one in every annulus, so it cannot change the measured radial mass
profile. Registered F160W light is normalized to the published BCG stellar
mass and added to the gas map.

The locked primary uses square-root X-ray contrast, the P0557 coupling
`t=0.3`, 20 kpc softening, and a 512-by-512 tensor grid.

## Exact results

| Universal construction | Four-cluster RMS | Change vs zero | Roots complete? |
|---|---:|---:|---|
| Zero tensor | **17.881** | baseline | yes |
| Absolute ACCEPT gas + stars | 18.132 | **-1.41%** | yes |
| ACCEPT rescaled to Tian central gas mass | 18.145 | **-1.48%** | yes |

The primary response is sign-incoherent across systems:

| Cluster | Zero | Physical gas tensor | Change |
|---|---:|---:|---:|
| MACS0329 | 19.624 | 20.304 | 3.46% worse |
| MACS0429 | 14.639 | 14.990 | 2.39% worse |
| MACS1115 | 24.635 | 24.630 | 0.02% better |
| MACS1931 | 8.520 | 8.458 | 0.73% better |

On the prior validation pair alone the gain is only **0.096%**, and its RMS is
still **1.843 times** the compact-halo comparator.

## Independent mass cross-check

The ACCEPT profile and Tian central gas anchor disagree far beyond their stated
errors:

| Cluster | ACCEPT / Tian gas mass at Tian radius | Tian rescale needed |
|---|---:|---:|
| MACS0329 | 0.0838 | 11.93x |
| MACS0429 | 0.0693 | 14.43x |
| MACS1115 | 0.0732 | 13.65x |
| MACS1931 | 0.2371 | 4.22x |

This makes the earlier P0558 central fractions insecure as local physical
weights. It also supplies a useful sensitivity test: changing gas amplitude by
4-14x moves the aggregate score only 0.013 arcsec and does not change which
clusters improve. The dominant problem is therefore the **direction and shape
of the tensor**, not the absolute gas normalization.

## What the data teach us

1. **A physically weighted baryonic tensor still has mixed signs.** The two
   discovery clusters worsen, while the two prior validation clusters improve
   slightly.
2. **Gas amplitude is not the missing universal control.** The enormous
   ACCEPT-to-Tian rescaling barely affects exact predictions.
3. **Angular structure matters more than spherical mass.** Removing angular
   contrast in fixed-source diagnostics is much more destructive than changing
   normalization, although those local diagnostics are not valid predictive
   scores.
4. **The radial ACCEPT profile alone cannot create a dark-matter-like offset.**
   The contrast operator intentionally removes its circular mean; only
   non-circular baryonic structure can redirect the field in this test.
5. **The weak P0557 transfer gain was not evidence for a universal 75/25 mass
   mixture.** Replacing the proxy with physical radial gas mass changes its
   magnitude but not the cluster-to-cluster conflict.

## Limits and next diagnostic

ACCEPT publishes diagonal electron-density errors but not full shell
covariance. Chandra brightness supplies angular shape, not local gas mass. The
stellar map lacks resolved mass-to-light ratios and separate satellite/ICL
normalizations. These limitations prevent a definitive rejection of every
baryonic tensor, but the current formula fails its frozen gates.

The next diagnostic should sweep the already-spent coupling through positive
and negative values for each physical map. That cannot validate a formula; it
can answer a narrower causal question: do clusters prefer a common sign and a
roughly common dimensionless response, or would the model require individually
tuned field reversal? A common optimum would justify a prospectively frozen
holdout. Opposite optima would retire this tensor geometry.

## Reproduce

```powershell
python scripts/run_p0559_accept_projected_gas_tidal.py
python -m pytest tests/test_gas_surface_density.py tests/test_p0559_accept_projected_gas_tidal_results.py -q
```

Machine-readable outputs are in `results/p0559_accept_projected_gas_tidal/`.
