# P0581 locked endpoint exact-root results

## Result in plain language

The endpoint-return idea did **not** validate as a complete four-cluster lens
formula. It recovered a missing held-out image in MACS1931 and produced a tiny
improvement in MACS1115, but it lost an image in MACS0329 and slightly worsened
MACS0429. On the two systems where both the scalar parent and endpoint formula
found every held-out image, the endpoint formula was 0.70% worse.

The experiment nevertheless identified an important field-design parameter:
**the endpoint contrast transform controls image topology**. Reducing the
pre-normalization contrast cap
from 20 to 5 or 10 recovered all 11 held-out roots in the one-at-a-time replay,
while changing the common-system RMS by only 0.034 arcsec. A residual-only
analysis would have missed this effect.

This is not evidence for a successful theory. It is evidence that a physical
return-field law must be smoothly bounded; very concentrated redirected
gravity can move a lens across a caustic and create or destroy images even when
the positions of surviving images barely change.

## Formula tested

P0581 kept ordinary baryonic gravity at the observed baryon locations. It took
the excess deflection of the frozen P0554 scalar parent,

\[
\boldsymbol\alpha_X=\boldsymbol\alpha_{P0554}
-\boldsymbol\alpha_b,
\]

and used P0579's baryon-derived endpoint map to redistribute only the angular
shape of that excess. The locked endpoint geometry was

\[
C={R_{50}\over R_{80}},\qquad
s(C)={1\over1+\exp[-4.77637(C-0.648526)]},
\]

with return length `0.36 R80`, endpoint width `0.23 R80`, endpoint-only
residence, and the member-light centroid as destination. A contrast field made
from this normalized map modulated the excess convergence. Its annular mean
was removed exactly, and a potential was solved so that

\[
\boldsymbol\alpha_{\rm test}
=\boldsymbol\alpha_{P0554}+\nabla\psi_{\rm endpoint},
\qquad
\nabla^2\psi_{\rm endpoint}=2\,\Delta\kappa_{\rm endpoint}.
\]

Thus the correction is curl-free and has zero radial monopole. It tests where
the already-defined scalar excess appears, not whether endpoint routing alone
can supply the missing total cluster strength.

## Exact nonlinear lens test

The four systems contain 282 member sources, 44 training images, and 11
held-out images. None entered the inverse analysis that selected K0338, though
all have been used by other formula families in this project. Six ordinary
lens-geometry variables were refitted per cluster; no endpoint-field parameter
was fitted to these clusters.

| Cluster | Scalar roots / required | Endpoint roots / required | Scalar RMS | Endpoint RMS | Outcome |
|---|---:|---:|---:|---:|---|
| MACS0329 | 3/3 | 2/3 | 23.412 arcsec | undefined | lost a root |
| MACS0429 | 2/2 | 2/2 | 14.774 arcsec | 15.172 arcsec | 2.69% worse |
| MACS1115 | 2/2 | 2/2 | 24.624 arcsec | 24.619 arcsec | 0.021% better |
| MACS1931 | 3/4 | 4/4 | undefined | 10.921 arcsec | recovered a root |

Both formulas are complete in three of four systems, but they fail in
different systems. Therefore their apparent finite-system aggregate RMS values
(21.391 versus 17.847 arcsec) are not a valid head-to-head comparison. On the
matched complete systems MACS0429 and MACS1115, scalar RMS is 20.306 arcsec and
endpoint RMS is 20.449 arcsec: a 0.70% endpoint loss.

The endpoint model completed both historical validation systems, but its
19.044-arcsec validation RMS is 1.91 times the 9.989-arcsec compact-halo
comparator. It improved or recovered a root in only two of four clusters,
below the frozen three-cluster gate. Every performance gate failed.

## Root topology versus residual sensitivity

Each one-at-a-time variant was evaluated at the locked primary fit and fitted
source positions. These opened-data sensitivities are diagnostic, not new
predictive results.

| Parameter | Levels | Total converged roots | Complete systems | Common-system RMS span |
|---|---|---|---|---:|
| Gate | none / soft / standard | 7 / 10 / 10 | 1 / 3 / 3 | 6.159 arcsec |
| Endpoint width / `R80` | 0.18 / 0.23 / 0.28 | 9 / 10 / 8 | 2 / 3 / 3 | 0.188 arcsec |
| Routed fraction | 0.50 / 0.75 / 1.00 | 8 / 9 / 10 | 2 / 2 / 3 | 0.079 arcsec |
| Contrast ceiling | 5 / 10 / 20 | 11 / 11 / 10 | 4 / 4 / 3 | 0.034 arcsec |
| Return length / `R80` | 0.30 / 0.36 / 0.42 | 8 / 10 / 8 | 2 / 3 / 2 | 0.021 arcsec |

The gate has the largest overall topology span: removing it loses three roots
and reduces complete systems from three to one. The return length has a sharp
interior root-count maximum at the inverse-derived `0.36 R80`, even though the
surviving-image RMS changes by only 0.021 arcsec. Full routing similarly
maximizes root count in this construction.

The most actionable result is the pre-normalization contrast cap. Caps 5 and 10 are the
only tested levels that complete all four systems. Because these values were
observed after opening the exact-root results, they must be frozen and tested
on another cluster before receiving predictive status.

## Physical consistency audits

The numerical construction behaved as designed:

- route-map normalization error: `2.22e-16` maximum;
- annular monopole leakage: `4.18e-16` maximum;
- normalized curl RMS: `3.24e-17` maximum;
- Solar point-source and axisymmetric zero-monopole controls: exact pass.

These checks show that the failure is not numerical leakage, creation of total
mass, or a non-potential deflection field. It is a failure of the currently
locked angular response to transfer consistently between clusters.

## What was learned and what comes next

1. The attractive P0579 linearized gain does not survive as a four-cluster
   exact-root prediction.
2. Redirecting only the scalar excess is a coherent cross-domain construction,
   but K0338 with contrast cap 20 is too topologically fragile.
3. Lens-image count is a more sensitive discriminator than RMS for sharp
   angular field changes.
4. The concentration gate is necessary in this formula; an ungated response
   destroys roots.
5. The inverse-derived `0.36 R80` length remains interesting because it is an
   interior maximum in root completeness, not because it improves residuals.
6. The next small formula change should replace hard pre-normalization contrast clipping with a
   smooth bounded saturation, preserve the exact annular mean, and freeze its
   scale before testing another untouched raw cluster.

## Reproduction

```powershell
python scripts/run_p0581_locked_endpoint_exact_root.py
pytest -q tests/test_p0581_locked_endpoint_exact_root_results.py
```

Machine-readable outputs are in
`results/p0581_locked_endpoint_exact_root/`.
