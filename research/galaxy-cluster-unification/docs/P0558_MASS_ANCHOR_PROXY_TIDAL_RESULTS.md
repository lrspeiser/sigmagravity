# P0558 measured mass-anchor proxy tensor

## Outcome

Replacing the fixed 75/25 star/gas morphology mixture with each cluster's
published central baryon ratio makes the result **worse**, not better. The
four-cluster held-out exact image-position RMS changes from **17.8924 to
18.1648 arcsec**, a **1.52% worsening**. All primary image roots remain present,
so this is a physical transfer failure rather than a solver artifact.

The locked equation was unchanged from P0557:

$$
\partial_i\left[(\delta^{ij}+0.3Q_{\rm contrast}^{ij})
\partial_j\phi_\Sigma\right]=S_\Sigma .
$$

There are no new gravity parameters and no per-cluster gravity settings.
Object-to-object differences enter only through observed baryonic masses.

## Measured input that was added

Tian et al.'s archived table supplies a BCG stellar mass and a Chandra gas mass
inside a central 14.3-22.2 kpc aperture. P0558 used

$$
f_{\rm gas}=\frac{M_{\rm gas}}{M_{\star}+M_{\rm gas}}
$$

to weight separately normalized F160W and square-root-X-ray morphology maps.
The measured fractions are:

| Cluster | Central gas fraction | Conservative observational range |
|---|---:|---:|
| MACS0329 | 60.7% | 57.7-63.9% |
| MACS0429 | 36.1% | 32.0-40.4% |
| MACS1115 | 65.9% | 63.0-68.9% |
| MACS1931 | 17.5% | 16.0-19.3% |

These are measured per-object inputs, not fitted gravitational freedom. The
important limitation is that the stellar value is a BCG total while the gas
value is aperture-cumulative, and the registered starlight map also contains
satellites/ICL. It is therefore a physical mass anchor, not a complete 2-D
surface-density map.

## Exact scores

| Universal diagnostic | Four-cluster RMS | Change vs zero | Roots complete? |
|---|---:|---:|---|
| Zero tensor | **17.892** | baseline | yes |
| Half published gas scale, sqrt X-ray | 18.023 | -0.73% | yes |
| Published masses, sqrt X-ray | 18.165 | -1.52% | yes |
| Published masses, linear X-ray | 18.177 | -1.59% | yes |
| Double published gas scale, sqrt X-ray | 20.443 | -14.26% | **no** |

The primary per-cluster changes were:

| Cluster | Zero | Measured-mass tensor | Direction |
|---|---:|---:|---|
| MACS0329 | 19.666 | 20.539 | 4.44% worse |
| MACS0429 | 14.639 | 14.944 | 2.08% worse |
| MACS1115 | 24.635 | 24.635 | essentially unchanged |
| MACS1931 | 8.520 | 8.232 | 3.38% better |

On the prior validation pair, the measured-mass result is still **1.839 times**
the compact-halo error.

## What this clarifies

1. **The P0557 75/25 mixture was not uncovering the measured universal baryon
   ratio.** Real central gas fractions range from 18% to 66%, and inserting
   them degrades aggregate prediction.
2. **MACS1931 drove the positive signal because it is the low-gas system.** Its
   measured 17.5% fraction is close to the earlier 25% mixture, and even the
   half-gas sensitivity improves it more. The same rule worsens MACS0329 and
   MACS0429.
3. **Gas fraction alone is not the environmental control variable.** MACS1115
   and MACS0329 have similarly high gas fractions but respond very differently.
4. **Increasing gas influence is actively harmful in this tensor.** The
   half-gas version is the least-bad nonzero model; the measured version is
   worse; doubling gas loses an image root.
5. **Linear versus square-root brightness remains unresolved.** Their aggregate
   errors differ by only 0.013 arcsec, far below the model shortfall.

The useful surviving clue is narrower: non-circular baryon morphology can move
individual cluster predictions in the right direction, but neither a universal
mixture nor the published central mass ratio determines its correct strength.

## Subsequent independent catalog audit

P0559 later integrated the published ACCEPT electron-density shells at these
same central radii. The ACCEPT cumulative gas masses are only **6.9%-23.7%** of
the Tian table values, a discrepancy of factors **4.2-14.4**. Therefore the
P0558 fractions should no longer be described as secure local gas fractions;
they are a cross-catalog stress test whose exact aperture/quantity mismatch is
unresolved. This strengthens rather than weakens P0558's negative conclusion:
neither the nominal values nor large changes in their amplitude produce a
universal lens improvement.

## Next decisive observation

The missing observable is not another global gas fraction. It is a registered
posterior over the **radial and two-dimensional gas surface density**, with
temperature/emissivity, PSF mixing, background, and covariance propagated.
P0559 used the already public ACCEPT density profiles to make this projection
without hydrostatic, lensing, dark-matter, or new-gravity priors. Its physical
tensor still worsens the four-cluster aggregate by 1.41%. A new X5 posterior
would now be valuable primarily to resolve the ACCEPT/Tian disagreement and
propagate full covariance, not to repeat the same locked tensor geometry.

Only then can the tensor be built from physical local mass at every position.
If that still gives mixed signs across clusters, the baryonic-tensor path should
be retired rather than given a fitted per-cluster amplitude.

## Reproduce

```powershell
python scripts/run_p0558_mass_anchor_proxy_tidal.py
python -m pytest tests/test_p0558_mass_anchor_proxy_tidal_results.py -q
```

Machine-readable outputs are in `results/p0558_mass_anchor_proxy_tidal/`.
