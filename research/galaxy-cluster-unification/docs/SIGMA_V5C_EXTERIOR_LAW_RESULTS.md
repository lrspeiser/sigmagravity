# Sigma v5C exterior-law result

## Decision

The fixed canonical massive-scalar v5C row is **retired before its full field
variation and before any observational fit**. It satisfies the luminal DHOST
degeneracy identities, but it cannot produce the exterior acceleration law
required by flat galaxy rotation curves.

This is a physical-feasibility failure, not a mathematical-health failure:

- in its screened DHOST regime, the exterior metric is exactly GR once the
  enclosed baryonic mass is constant; and
- in its unscreened linear regime, it is Newton plus an attractive Yukawa
  scalar, whose acceleration falls at least as fast as \(1/r^2\).

A flat circular-speed curve instead requires approximately

\[
g(r)\propto {1\over r}.
\]

No galaxy, cluster, Solar, or other observational array was opened.

## Screened exterior

For the luminal Class-Ia DHOST Vainshtein branch, the published spherical
weak-field potentials are

\[
{d\Phi\over dr}
={G_NM(r)\over r^2}+\Xi_1G_NM''(r),
\]

\[
{d\Psi\over dr}
={G_NM(r)\over r^2}
+\Xi_2{G_NM'(r)\over r}
+\Xi_3G_NM''(r).
\]

These relations are derived in
[Langlois et al.](https://arxiv.org/abs/1711.07403). Outside the baryonic
source,

\[
M(r)=M_b,
\qquad
M'(r)=M''(r)=0.
\]

Therefore

\[
\boxed{
\Phi'=\Psi'={G_NM_b\over r^2}.
}
\]

The same result holds for massive tracers and the Weyl potential. The Hessian
operators can change gravity inside an extended baryonic distribution, but on
this screened branch they cannot maintain extra exterior acceleration after
the enclosed mass has saturated.

That is useful for Solar screening and fatal for the intended outer-galaxy
mechanism.

## Unscreened exterior

At small scalar invariant the fixed v5C coefficient begins as

\[
A_3\propto \widehat X^2,
\]

so the higher-derivative DHOST operators vanish at leading order. The remaining
canonical massive conformal scalar gives the most favorable attractive
point-source exterior

\[
g(r)={GM_b\over r^2}
\left[1+\alpha(1+x)e^{-x}\right],
\qquad
x={r\over L_\Sigma},
\qquad
\alpha\ge0.
\]

Its exact logarithmic slope is

\[
\boxed{
{d\ln g\over d\ln r}
=-2-{alpha x^2e^{-x}
\over1+\alpha(1+x)e^{-x}}
\le-2.
}
\]

Changing the scalar amplitude or range cannot make the slope approach the
approximately `-1` value needed for a flat curve. The Yukawa term only makes
the transition steeper.

For a circular orbit, \(v_c^2=rg\). Even the shallowest allowed exterior gives

\[
{v_c(10r)\over v_c(r)}\le{1\over\sqrt{10}}=0.316228,
\]

whereas the frozen flat-curve adequacy interval is `0.9--1.1`.

## Executable result

The analytic scan covers

- `1e-8 <= r/L <= 1e8`;
- scalar strengths from zero through `1e6`; and
- both the screened and linear exterior branches.

| Quantity | Best v5C value | Required |
|---|---:|---:|
| Shallowest acceleration slope | `-2.000` | `-1.1` to `-0.9` |
| Longest adequate flat-slope span | `0.000 dex` | at least `1.000 dex` |
| Largest speed ratio across a radial decade | `0.316228` | `0.9` to `1.1` |
| Screened vacuum differs from GR | no | yes |

All three exterior gates fail.

## Why a hidden branch is not accepted as a repair

One could posit a nonperturbative source-free scalar profile outside the
baryons. That would no longer be the declared regular retarded branch. Its
amplitude, center, or initial profile would have to be supplied in addition to
the baryonic source unless a new action proved a unique matching theorem.

Such a profile would be functionally similar to assigning a halo state and
would violate the present goal's no-per-object gravitational initial data
rule. No such branch is present in the fixed v5C construction, and inventing
one after the exterior failure would define a new theory.

## Scope of the rejection

This result rejects the **fixed v5C row**:

- canonical massive `P`;
- conformal curvature source;
- the selected signed-safe `A3` shape; and
- its dependent luminal Class-Ia coefficients.

It does not prove that every DHOST theory is incapable of galaxy
phenomenology. A different deep-field `P(X)` can create a shallower exterior,
but the project already established the corresponding obstacle: a pure static
derivative screen with `P_X` increasing toward large spacelike gradients has a
superluminal parallel characteristic under the strict causality gate.

The combined lesson is narrower and useful:

> A local one-scalar theory cannot merely combine a canonical/Yukawa exterior,
> Vainshtein recovery of GR, and Hessian corrections inside matter. That set of
> ingredients contains no causal, long-range exterior channel capable of
> producing flat galaxy curves.

The next action must change that physical channel rather than choosing another
v5C coefficient tail. The remaining serious direction is a constrained causal
orientation/memory carrier whose baryon-forced response persists through
vacuum without an independently assigned halo state.

## Reproduction

```powershell
python scripts/check_sigma_v5c_exterior_law.py
python -m pytest tests/test_sigma_degenerate_action.py -q
python -m ruff check src/voidscreen/sigma_degenerate_action.py scripts/check_sigma_v5c_exterior_law.py tests/test_sigma_degenerate_action.py
```
