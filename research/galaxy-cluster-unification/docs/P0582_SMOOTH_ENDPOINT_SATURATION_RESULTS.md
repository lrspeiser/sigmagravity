# P0582 smooth endpoint saturation results

## Result in plain language

The lens data prefer a **window** of redirected angular response. Too much
correction loses an image in MACS0329; too little fails to recover images in
MACS1931. A smooth tanh response at nominal scale 20 was the only smooth curve
tested that stayed inside both limits and found all 11 held-out images.

This does not rescue the model as a theory. The four clusters were already
opened, lens geometry was not refitted for these variants, and the all-cluster
residual remains about 19 arcsec. The result identifies a useful formula
property: caustic stability depends on the entire nonlinear response curve,
not simply its maximum value or whether it is smooth.

## Controlled formula change

Everything except the contrast transformation was held fixed at P0581:

- ordinary baryonic gravity remained local;
- the P0554 scalar excess remained the carrier;
- concentration gate, endpoint-only residence, and centroid destination were
  unchanged;
- return length remained `0.36 R80` and width `0.23 R80`;
- all lens geometry and source positions remained at the P0581 fitted values;
- every angular correction retained exact zero annular monopole and came from
  one potential.

For the nonnegative endpoint-map ratio `x`, P0582 tested

\[
T_A^{\rm hard}(x)=\min(x,A),
\]

\[
T_A^{\tanh}(x)=A\tanh(x/A),
\]

\[
T_A^{\exp}(x)=A[1-e^{-x/A}],
\qquad
T_A^{\rm rat}(x)={Ax\over A+x},
\]

at `A = 3, 5, 7.5, 10, 15, 20`. After transformation, each annulus was divided
by its carrier-weighted mean. This preserves the zero-monopole condition.

The label `A` needs care: it bounds the transform before annular normalization,
not the final field weight. Sparse annuli can acquire much larger normalized
weights. The largest recorded weight was about 730, while the actual
convergence and deflection fields remained finite.

## Results across all four clusters

| Mode | Best nominal `A` by frozen diagnostic rule | Complete systems | Roots / 11 | Equal-complete-system RMS |
|---|---:|---:|---:|---:|
| Hard | 5 | 4 | 11 | 19.040 arcsec |
| Tanh | 20 | 4 | 11 | 19.159 arcsec |
| Exponential | 15 | 3 | 9 | 20.982 arcsec on complete systems |
| Rational | 3 | 3 | 9 | 20.570 arcsec on complete systems |

Five of 24 variants completed all clusters: hard `A=5, 7.5, 10, 15` and tanh
`A=20`. The hard diagnostic winner is not automatically the better physical
formula: it has a derivative kink and won by only 0.119 arcsec over tanh-20 on
already-opened clusters.

At the same nominal scale `A=20`, root counts were:

| Transform | Converged roots | Complete systems |
|---|---:|---:|
| Tanh | 11 | 4 |
| Hard | 10 | 3 |
| Exponential | 9 | 3 |
| Rational | 8 | 3 |

This rejects both simple interpretations. Smoothness alone cannot explain the
result because two smooth transforms fail. Nominal cap alone cannot explain it
because four transforms at the same `A` give four different root counts.

## Which clusters impose the window

| Cluster | Complete variants / 24 | Root-count range | Relationship to correction RMS |
|---|---:|---:|---|
| MACS0329 | 23 | 2--3 | the single strongest variant fails |
| MACS0429 | 24 | 2--2 | insensitive over this grid |
| MACS1115 | 24 | 2--2 | insensitive over this grid |
| MACS1931 | 6 | 1--4 | generally needs a stronger correction |

For MACS0329, complete variants had correction-field RMS from 0.587 to 0.722
arcsec; hard-20 at 0.725 arcsec was the only failure. For MACS1931, root count
had Spearman correlation `rho=0.788` with correction RMS, and complete variants
occupied 4.519--4.634 arcsec, though incomplete and complete ranges overlap.
These thresholds are descriptive and cluster-specific, not universal constants.

Tanh-20 works because it makes a modest smooth reduction relative to hard-20:
enough to move MACS0329 below its upper caustic boundary, while leaving enough
of the response to recover all four MACS1931 images. Exponential and rational
curves compress the response too strongly for MACS1931 in this range.

## Audits and cross-domain meaning

Across 96 fields:

- maximum annular monopole leakage was `2.59e-16`;
- maximum normalized curl RMS was `3.30e-17`;
- Solar point-source and axisymmetric galaxy corrections remained exactly null.

The galaxy and Solar nulls are consistency controls, not new fits. This angular
channel cannot explain SPARC rotation by itself; P0580 already showed that a
separate universal scalar amplitude law is required.

## Decision

The most elegant form worth freezing for another raw-cluster test is

\[
T(x)=20\tanh(x/20),
\]

followed by exact carrier-weighted annular normalization. Hard cap 5 remains a
useful control. The next validation should compare only those two forms plus
the hard-20 parent, use exact image roots, and choose no coefficient on the new
cluster.

## Reproduction

```powershell
python scripts/run_p0582_smooth_endpoint_saturation.py
pytest -q tests/test_p0582_smooth_endpoint_saturation_results.py
```

Machine-readable outputs are in
`results/p0582_smooth_endpoint_saturation/`.

## Subsequent transfer result

P0583 later froze tanh-20 and tested it on RX J2129, which had not entered any
K0338 or saturation selection. It retained all seven roots after refitting
ordinary geometry but worsened held-out RMS from 1.256 to 14.130 arcsec. Thus
P0582 remains a useful caustic-stability diagnostic, but tanh-20 did not
transfer as an accurate angular-lensing formula.
