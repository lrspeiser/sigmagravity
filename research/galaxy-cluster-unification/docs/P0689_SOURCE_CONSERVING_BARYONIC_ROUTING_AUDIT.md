# P0689 source-conserving baryonic routing audit

Frozen before metrics: 2026-08-02  
Verdict: mathematical and numerical audit **passes**; no observational score computed

## Motivation

P0684-P0688 showed that changing a single scalar QUMOND exponent cannot retain
the spent cross-cluster radial result and also produce a nonhollow strong-lens
field. P0689 starts a different branch: change where the effective extra source
is placed while preserving total added source and the fixed-RAR far-field
monopole.

## Operator

Start with fixed-RAR QUMOND and the retired local-path generator:

\[
S_0=\nabla\!\cdot[\nu_0\nabla\Phi_N],
\]

\[
\delta S_{\rm loc}=\nabla\!\cdot
[(\nu_0^{p_{\rm local}}-\nu_0)\nabla\Phi_N].
\]

Use only the positive strength of that generator,

\[
A_+=\int\max(\delta S_{\rm loc},0)dV,
\]

and route it onto the observed baryonic morphology. Place an equal negative
polarization source on the existing potential transition:

\[
W_t=4S_4(1-S_4)|\nabla\chi_b|,
\]

\[
S_{\rm route}=S_0+A_+{\rho_b\over\int\rho_b dV}
-A_+{W_t\over\int W_t dV}.
\]

The two added terms integrate to equal and opposite strengths. The operator
therefore changes the internal source distribution while conserving the
far-field monopole. `A_plus`, the baryonic route, and the compensation shell
are all calculated from the baryonic field. No amplitude, core radius, shell
radius, or object-specific gravity setting is fit.

## Dimensional audit

| Quantity | Units |
|---|---|
| `S_0`, `S_route` | `s^-2` |
| `A_plus` | `m^3 s^-2` |
| `rho_b / integral rho_b` | `m^-3` |
| `W_t / integral W_t` | `m^-3` |
| integrated added source | `m^3 s^-2`, zero by construction |

## Frozen result

The audit uses the spent registered P0670 RX J2129 baryonic map but is forbidden
from calculating photon deflections, radial targets, image roots, or topology.

| Metric | Result | Frozen gate | Verdict |
|---|---:|---:|---|
| Newtonian residual | `8.89e-14` | `<=1e-10` | pass |
| Routed-field residual | `2.47e-14` | `<=1e-10` | pass |
| Boundary mismatch | `0` | `<=1e-14` | pass |
| Positive/negative strength mismatch | `1.57e-16` | `<=1e-12` | pass |
| Net added source / positive strength | `9.29e-17` | `<=1e-12` | pass |
| Transition-shell positive cells | `35,937` | `>=20` | pass |
| Interior transition-shell weight | `0.781` | `>=0.25` | pass |
| Positive route on baryon-positive cells | `1-1e-16` | `>=1-1e-12` | pass |
| New universal constants | `0` | `0` | pass |
| Per-object gravity / photon amplitudes | `0 / 0` | `0 / 0` | pass |

All frozen gates pass. Thirteen targeted unit tests also pass across the source
routing, 3D spatial QUMOND, and potential-channel modules.

## Numerical precision note

Directly subtracting the large `S_0` arrays from `S_route` loses precision.
The conservation gate therefore integrates the explicitly constructed added
pair `positive_route - negative_shell`, which is the quantity defined by the
equation. The large-array subtraction is retained only as a diagnostic and is
not used to claim source nonconservation.

## What this proves

The operator is dimensionally consistent, source conserving at floating-point
precision, finite, resolved on the current grid, compatible with the
fixed-RAR boundary, and solvable by the validated Poisson machinery. The
positive route follows baryonic multipoles rather than a nearly circular
potential exponent.

## What it does not prove

- No galaxy rotation or cluster radial-deflection score has been computed.
- No photon field, image position, multiplicity, parity, or critical curve has
  been computed.
- The negative term is an effective polarization source; no microscopic energy
  condition, covariant action, or causal transport law is derived.
- One 33-cell spent map does not establish resolution or map robustness.

## Next frozen empirical test

Before viewing any outcome, preregister both stages:

1. **Spherical transfer:** 131 spent galaxies, six spent cluster profiles, and
   Solar proxies. Require galaxy error `<=1.05x` fixed RAR, all-five and
   reliable-three cluster RMS `<=0.20 dex`, and at least 75% fixed-RAR gap
   closure.
2. **Registered 3D topology:** if the transfer stage passes, calculate the
   zero-slip field on RX J2129 with the existing four ordinary nuisances and
   the P0686 root/multiplicity/parity/critical-curve gates. No gravity or photon
   amplitude may be fit.
3. **Robustness:** a topology pass must retain source conservation, field sign,
   root count, and residual class across frozen grid resolutions and stellar/
   gas map sensitivities.

Only that survivor may open the sealed P0633 galaxy and P0640 cluster outcomes.

## Reproduction

```powershell
python scripts/run_p0689_source_conserving_baryonic_routing_audit.py
python -m pytest tests/test_source_routing_qumond.py tests/test_spatial_qumond_3d.py tests/test_potential_channel_qumond.py -q
```

Artifacts are in
`results/p0689_source_conserving_baryonic_routing_audit/`.
