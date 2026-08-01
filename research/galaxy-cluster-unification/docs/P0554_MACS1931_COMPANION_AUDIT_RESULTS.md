# P0554 MACS1931 companion-image audit

## Outcome

The route-induced 3-to-5-root transition does not survive its first direct
image-level consequence. In all 11 five-root formulas, the two roots near
MACS1931 image 2c have opposite parity and nearly equal predicted brightness.
If one is image 2c, the other should be a readily separated counterimage.

No additional family-2 image is present in the published 19-image MACS1931
catalog. In the local CLASH F160W mosaic:

- all 11 unregistered absolute companion coordinates are formally blank;
- after shifting each formula's anchor root exactly onto observed image 2c,
  ten companion predictions fall on clean blank sky;
- the remaining prediction is contaminated by neighboring light but has no
  centered counterpart and is not catalogued as family 2; and
- no plausible centered uncatalogued counterimage is visible.

This strongly disfavors the extra-pair realization that made the route formulas
look “complete.” It does not reject every possible gravity-routing law, and it
does not make the three-root baseline accurate. It removes the apparent
topology advantage of these particular five-root variants.

## Why a second, registered test was necessary

The first audit used the model's absolute root coordinates. They were all blank,
but the supposed anchor root itself missed observed image 2c by 1.95--5.88
arcseconds. That makes the absolute companion coordinate too harsh a test.

Before inspecting any shifted positions, a second protocol froze the unique
translation

$$
\boldsymbol\theta_{\rm companion,reg}
=\boldsymbol\theta_{2c,\rm observed}
+\left(\boldsymbol\theta_{\rm companion}
-\boldsymbol\theta_{\rm anchor}\right).
$$

No rotation, scale, shear, lens parameter, or position-specific adjustment was
allowed. This cancels only the anchor's local offset and preserves the formula's
predicted pair separation and direction.

## Prediction strength

| Quantity | Result |
|---|---:|
| Five-root formulas | 11 |
| Pair parity | opposite in 11/11 |
| Companion/anchor absolute magnification ratio | 0.917--1.251 |
| Registered pair separation | 1.782--10.859 arcsec |
| F160W pixel scale | 0.065 arcsec/pixel |
| Minimum pair separation | 27.4 pixels |
| Observed 2c formal aperture S/N | 118.8 |
| Expected companion formal S/N | 108.9--148.6 |

The minimum separation is far larger than a pixel and the predicted companion
is not a faint central image. These formulas predict a fold-like pair with
roughly equal brightness. A clean blank is therefore genuinely problematic.

## Absolute and anchor-registered audits

| Audit | Catalogued family-2 matches | Formal sources | Formal blanks | Visual result |
|---|---:|---:|---:|---|
| Absolute model coordinates | 0/11 | 0/11 | 11/11 | all exact coordinates clean |
| Anchor registered to 2c | 0/11 | 2/11 | 9/11 | 10 clean blanks; 1 contaminated/inconclusive |

The two formal registered “detections” are not counterimage confirmations:

- `combined_fraction_095` has aperture S/N 5.2 but only a 2.98-sigma peak and
  no centered visible source; diffuse background drives the sum.
- `combined_lens_099` has aperture S/N 39.4 because its coordinate lands on
  the flank of neighboring compact/diffuse light. That object is not centered
  on the prediction, does not resemble 2c in this band, and is not a published
  family-2 image. It is conservatively labelled contaminated/inconclusive.

The 11 registered formula positions collapse to five spatial groups, so this
is not 11 fully independent blank-sky trials. It is still a direct test of all
11 frozen formulas.

## What we learned about the formula

1. **Counting a root near a held-out seed is not enough.** The 18/18 score hid
   an unobserved extra image pair.
2. **The caustic margin was diagnostically successful but selected the wrong
   side of the boundary.** It perfectly identified the 3-to-5 transition; HST
   says that transition is likely unwanted.
3. **Strong-lensing topology supplies stricter data than RMS alone.** Every
   predicted root needs a parity, relative magnification, and observable
   counterpart—not just a low coordinate residual.
4. **The route term should not be rewarded for MACS1931 completeness.** The
   continuous photon-softness result remains separable, but the present route
   configurations lose their claimed topology advantage.
5. **A future baryonic field law must predict only the observed multiplicity.**
   It must move the existing three roots toward their images without crossing
   the caustic that creates an equal-brightness extra pair.

## Limits

MACS1931 and these formulas are spent evidence. The lens is simplified and has
large residuals; F160W alone cannot prove that an arbitrary source is or is not
at redshift 1.8347; drizzle noise and intracluster light complicate formal S/N;
and the contaminated `combined_lens_099` location remains inconclusive. The
supported conclusion is limited to the frozen local extra-pair realizations.

A subsequent frozen global search across all 27 source families shows that the
route changes root count only in MACS1931 families 2 and 3, adding a pair in
each, while leaving the other 25 families unchanged. Route-only improves
equal-family positional RMS but increases potentially observable surplus roots
from eight to twelve at the primary threshold. See
[`P0554_MULTIFAMILY_MULTIPLICITY_RESULTS.md`](P0554_MULTIFAMILY_MULTIPLICITY_RESULTS.md).

## Reproduction

```powershell
python scripts/run_p0554_macs1931_companion_audit.py
python scripts/run_p0554_macs1931_relative_companion.py
python -m pytest tests/test_p0554_macs1931_companion_audit.py -q
python scripts/run_p0554_multifamily_multiplicity.py --postprocess-only
python -m pytest tests/test_p0554_multifamily_multiplicity.py -q
```

Results, annotations, and figures are under
`results/p0554_macs1931_companion_audit/` and
`results/p0554_macs1931_relative_companion/`.
