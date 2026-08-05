# Sigma V19AL FORS1 shared-PSF astrometry plan

## Why this is materially different

V19AI, V19AJ and V19AK treated every star as an isolated patch and changed only its local centroid estimator. They established that B can be solved accurately, compact center-of-light works best overall, and R contains band-dependent, non-single-peak structure. Repeating aperture sizes would be outcome tuning.

V19AL instead estimates one point-spread width from the ensemble of fixed foreground stars in each filter. It then fits every star with that shared imaging response, a per-star amplitude and a planar local background. A robust soft-L1 objective reduces the leverage of contaminated pixels. The per-filter PSF widths describe the telescope/atmosphere, not gravity, and are never shared with the physics model.

## Frozen two-stage fit

1. On every unchanged V19AH association, fit a circular Gaussian width, position, amplitude and planar background inside radius 4 pixels. Require at least 20 quality-passing preliminary stars per filter.
2. Take the median accepted width separately in B, I and R.
3. Refit every association with its filter width fixed. Reject optimizer failures, fits within 0.02 pixel of the two-pixel coordinate bound, amplitude S/N below 5 or normalized residual RMSE above 5.
4. Fit the TAN WCS and exact leave-one-star-out predictions with no clipping.

The circular PSF is intentionally the lowest-complexity field model. An elliptical or spatially varying PSF is not introduced unless this frozen version shows a specific, independently testable failure.

## Evidence gates and boundary

The 30-star minimum per filter, 20 shared-ID minimum, fitted/leave-one-out residual ceilings, center consistency and R improvement requirement are identical to V19AI. Detection, rematching, member/candidate inspection, science photometry/deblending, baryonic inference, lensing/halo inputs, gravity changes and holdouts remain prohibited.

A pass means only that later baryonic mapping may use these coordinates. It provides no support by itself for the long-wavelength Sigma premise.
