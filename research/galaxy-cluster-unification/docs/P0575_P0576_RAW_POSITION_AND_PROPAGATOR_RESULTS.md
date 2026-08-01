# P0575-P0576 raw image positions and fractional routed propagation

> **Subsequent correction (P0576B-P0577):** the apparent P0576 fractional
> improvement is a source-plane/mass-sheet degeneracy, not validated evidence
> for `p=1.5`. Extending the scan drove the score monotonically to `p=2.6`,
> where the field was 99.9925% affine in image position and collapsed inferred
> sources to 0.9% of their unlensed radius. A Jacobian-aware image-plane metric
> removed the spectacular gain, and the resulting SMACS coordinate worsened a
> second cluster by 61%. Read the P0576 result below as the discovery of a
> metric failure and a useful radial-potential warning, not a promoted law.

## What changed

The normalized-map success did not survive the ordinary lens equation. It did
survive after one explicitly new-physics change: letting the symmetry-gated
routed component contribute more strongly through long-wavelength potential
modes than an ordinary Poisson field.

At this stage it looked like the most promising clue, but later diagnostics
showed that its numerical gain was dominated by the mass-sheet degeneracy.

## Raw data and fair comparison

P0575 parsed Table 1 of Caminha et al. (2022), A&A 666 L9,
arXiv:2207.07567v3. The frozen subset has 12 pre-JWST image positions from the
four families with exact spectroscopic redshifts: 1, 2, 5, and 19.

For any fixed deflection map,

\[
\beta_i=\theta_i-A{D_{ls}\over D_s}\alpha(\theta_i).
\]

One nonnegative amplitude `A` was fitted on families 1 and 2. Each family's
unknown source position was its analytic mean inferred position. The amplitude
was held fixed for families 5 and 19. No halo, shear, ellipticity, core radius,
or per-family strength was fitted.

The metric is source-plane RMS, not image-plane RMS. The paper's published
0.39 arcsec image-plane result from a ten-mass-parameter Lenstool model is
therefore context, not a directly comparable score.

## Ordinary Poisson failure

| Held-out field | Source-plane RMS (arcsec) |
|---|---:|
| Local 100-kpc member light | 1.288 |
| P0573 no-gate arrival | 1.449 |
| P0574 symmetry-gated arrival | 1.374 |
| Processed Lenstool-map reference | 0.647 |

P0574 is 6.63% worse than local light and improves neither held-out family.
This is not a fragile partition result:

- P0574 beats local in 0/6 two-family calibration partitions;
- its median change is 7.84% worse;
- two-, three-, and four-times FFT padding give changes of -6.63%, -6.66%,
  and -6.67%;
- the processed Lenstool reference is best in 6/6 partitions.

Matching the reconstructed convergence morphology is therefore insufficient.
The raw positions constrain the gradient and radial organization of the
potential, not merely where positive map density sits.

## Fractional routed propagator

P0576 kept the P0574 baryonic destination `D60` and symmetry gate fixed. The
local field uses the ordinary Poisson response. Only the routed component was
changed:

\[
\alpha_{D,p}(k)= {-2i\mathbf{k}\,D_{60}(k)\over k^2}
\left({k\over k_0}\right)^{2(1-p)},
\quad k_0={2\pi\over60\ {\rm kpc}}.
\]

`p=1` is the ordinary Poisson limit. `p>1` weights long wavelengths more
strongly. The complete field is

\[
\alpha=(1-f_\alpha H)\alpha_B+f_\alpha H\alpha_{D,p},
\quad H={Q_{90}^4\over Q_{90}^4+0.05^4}.
\]

The frozen 5-by-5 grid selected `p=1.5` and `f_alpha=1` on calibration
families only. Held-out results were:

| Held-out field | Source-plane RMS (arcsec) |
|---|---:|
| Local ordinary field | 1.285 |
| P0574 ordinary routed field | 1.371 |
| P0576 selected fractional field | 0.709 |
| Processed Lenstool-map reference | 0.646 |

The selected field appeared to improve local by 44.83% and both held-out
families in the source-plane statistic. P0576B-P0576C later showed that this
comparison was not physically discriminating: higher powers made the field
nearly affine and collapsed all sources, allowing an arbitrarily small score.

## Cross-domain meaning

The fractional response is multiplied by the same quarter-turn gate. In the
Solar point-source and deprojected axisymmetric-SPARC limits, `Q90=0`, so the
routed field disappears exactly and the ordinary local field remains. Thus
P0576 does not change any of the 175 SPARC rotation curves and does not alter
the Solar/Mercury checks.

That safety is structural, but it also means P0576 still does not explain
galaxy rotation. A separate radial law remains necessary unless both branches
can be derived from one covariant environmental operator.

## What is learned, and what is not

The strongest empirical hierarchy is now:

1. a 50--60 kpc baryonic destination scale robustly predicts normalized
   cluster-map morphology;
2. a weakly sensitive symmetry factor separates clumpy clusters from circular
   galaxy/Solar environments;
3. ordinary Poisson propagation of that destination fails raw image positions;
4. long-wavelength-enhanced propagation can artificially recover or exceed
   standard-map source-plane consistency through the mass-sheet degeneracy.

The required extended scan and second-cluster test were completed in
P0576B-P0577. They rejected a universal fractional exponent. Absolute
strength, nonlinear image-plane roots, weak shear, gas, and ICL remain absent.

## Reproduce

```powershell
python scripts/run_p0575_smacs0723_raw_position.py
python scripts/run_p0575b_raw_position_robustness.py
python scripts/run_p0576_fractional_routed_propagator.py
python -m pytest -q tests/test_p0575_p0576_raw_propagator_results.py
```
