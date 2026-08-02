# P0688 monotone-envelope QUMOND

Frozen before scores: 2026-08-02  
Verdict: integrity **passes**; fixed monotone primary **does not advance**; local-path exponent family retired from further shape patches

## Tested operator

P0688 applies the smallest pointwise repair that removes the central hollow
found by P0686 while never weakening the P0684 local radial response. For an
ordered baryonic profile,

\[
p_{\rm env}(r)=\max_{s\ge r}p_{\rm local}(s),
\]

where

\[
p_{\rm local}(r)=1+2{\chi_b(r)^4\over\chi_b(r)^4+(10^{-6})^4}
[\max(\eta_b(r),1)]^{-1/2}.
\]

The envelope is the minimal pointwise majorant of `p_local` that cannot
increase outward. It adds no constant and no per-object fitted gravity value.

## Frozen result

| Variant | Galaxy/RAR | All-five cluster RMS | Reliable-three RMS | Gap closed | Nonincreasing majorant |
|---|---:|---:|---:|---:|---|
| inward monotone majorant | **1.030** | **0.216 dex** | **0.271 dex** | **63.0%** | yes |
| local P0685 control | 1.030 | 0.145 dex | 0.165 dex | 75.1% | no |

The primary passes the galaxy, Solar, monotonicity, pointwise-majorant,
no-new-constant, and parameter-accounting gates. It fails the frozen
`0.20 dex` all-five and reliable-three cluster gates and the 75% gap-closure
gate. It does not advance to 3D.

## Failure anatomy

The monotone fill fixes the under-bending systems but over-bends the systems
that the original local law already matched.

| Cluster | Mean log10(prediction / target) |
|---|---:|
| MACS0329 | -0.043 |
| MACS0429 | -0.032 |
| MACS1115 | +0.092 |
| MACS1931 | +0.288 |
| RXJ1347 | +0.307 |
| RXJ2129 | +0.351 |

This is not merely a threshold miss. A universal inward fill moves the two
most underpredicted clusters in the right direction while making three other
clusters much too strong. Scalar exponent reshaping alone lacks the morphology
or source-location information needed to distinguish these cases.

## Family-level conclusion

P0684-P0688 jointly establish:

1. the local path exponent can match spent galaxy and cluster radial
   amplitudes but creates a hollow 3D response and fails raw topology;
2. a global baryonic concentration coordinate removes the hollow but loses the
   cluster amplitude match; and
3. the minimal nonhollow majorant also loses the cluster match, in opposite
   directions for different systems.

Further core radii, fractional powers, or blends would add constants to patch
a fully spent system. The local-path exponent family is therefore retired from
additional topology fixes.

## Next source-operator branch

The next mechanism should change **where the effective source is placed**, not
only the scalar boost. A parameter-free source-conserving routing candidate is:

\[
S_0=\nabla\!\cdot[\nu_0\nabla\Phi_N],
\]

\[
\delta S_{\rm loc}=
\nabla\!\cdot[(\nu_0^{p_{\rm local}}-\nu_0)\nabla\Phi_N],
\qquad
A_+=\int \max(\delta S_{\rm loc},0)\,dV,
\]

\[
S_{\rm route}=S_0+A_+{\rho_b\over\int\rho_b dV}
-A_+{W_t\over\int W_t dV},
\]

where `W_t` is a nonnegative shell weight fixed by the existing transition,
for example `S_4(1-S_4)|grad chi_b|`. The positive extra source is routed onto
the observed baryonic morphology; an equal negative compensator lies on the
natural `chi_b=chi_t` transition shell. The integral of the added source is
zero, so the far-field monopole is conserved. Its amplitude and shell are
calculated from the locked baryonic field rather than fitted.

This is a hypothesis generator, not yet a frozen equation. The next work is to
audit its units, signs, boundary behavior, spherical limit, and Solar limit;
then freeze a small set of exact shell-weight definitions before looking at
any new score.

## Reproduction

```powershell
python scripts/run_p0688_monotone_envelope_qumond.py
python -m pytest tests/test_potential_channel_qumond.py tests/test_spatial_qumond_3d.py -q
```

Artifacts are in `results/p0688_monotone_envelope_qumond/`.

## Claim boundary

P0688 is spent mechanism development. The envelope is nonlocal and has no
covariant action or causal derivation. No new raw-image score was calculated,
and P0633/P0640 remain sealed.
