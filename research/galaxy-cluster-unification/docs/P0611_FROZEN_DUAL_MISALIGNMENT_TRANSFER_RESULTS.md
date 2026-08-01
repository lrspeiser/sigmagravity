# P0611 frozen dual-misalignment transfer

## Outcome

The exact P0610 gate does not transfer to A383 and MS2137. These systems are
chronologically prospective for this formula—their gate activations and route
responses were unread when P0611 was frozen—but they are not pristine
project-wide holdouts because other formula families used them earlier.

The baryonic maps produced the contrast the test needed. A383 has almost
perfect gas/member and gas/star directional agreement, so its frozen activation
is effectively zero. MS2137 disagrees with both and activates the gate at
0.595. Despite that contrast, the high-activation system gains only 0.026% in
a held-out-only diagnostic and fails one of eight training roots under both the
baseline and route. The low-activation system's separate refits differ by
-0.415% on held-out RMS, which is an optimizer-basin effect at an actual route
strength of only $1.2\times10^{-10}$, not a physical response.

| System | Members | $c_{gm}$ | $c_{gs}$ | $H$ | $s_{\rm eff}$ | Raw result |
|---|---:|---:|---:|---:|---:|---|
| A383 | 35 | 0.9955 | 0.9956 | $4.8\times10^{-8}$ | $1.2\times10^{-10}$ | complete; 9.459 to 9.499 arcsec |
| MS2137 | 89 | 0.6824 | 0.6567 | 0.5949 | 0.001487 | invalid full score; 7/8 training roots for both variants |

Every predeclared advancement gate fails: exact-root completeness, response
ordering, both-systems-not-worse, 1% matched improvement, and the 2-arcsecond
absolute target.

## What this teaches us

The important negative result is not merely that one threshold failed. The
dual-misalignment observable can separate baryonic geometries, but the proposed
fourth-power gate does not turn that separation into a useful raw-lensing
correction. A high value of $A_{\rm dual}$ is therefore not sufficient evidence
that gravity should be routed toward the gas morphology.

This narrows the next field search:

1. Do not promote or retune the P0610 threshold, exponent, or amplitude on
   these two systems.
2. Separate *where a route is allowed* from *where its endpoints land*.
   Misalignment may be a regime label while the successful destination field
   depends on tidal saddles, caustics, or member-to-member paths rather than
   the gas-attraction direction.
3. Treat exact-root topology as a prerequisite. A small RMS change among the
   roots that exist cannot rescue a formula that misses an observed image.
4. Use common-start or fixed-geometry diagnostics whenever a gate is near zero;
   otherwise ordinary lens-geometry basins can appear larger than the field
   perturbation.

## Scope and limits

The member maps use a frozen CLASH BPZ interval cut, F160W weights, and a
300-kpc aperture. HST light and Chandra emissivity remain morphology proxies,
not complete calibrated baryonic mass maps. The P0599 parent was also developed
partly from CLASH reconstructed profiles. P0611 rejects this projected gated
gas-route implementation; it does not reject baryon-sourced nonlocal gravity in
general.

## Reproduction

```powershell
python scripts/run_p0611_frozen_dual_misalignment_raw_transfer.py
python -m pytest tests/test_p0611_frozen_dual_misalignment_raw_transfer_results.py -q
```

Machine-readable results are in
`results/p0611_frozen_dual_misalignment_raw_transfer`.
