# Sigma v12A aligned finite-wave-vector Dirac correction

## Decision

Both signs of `lambda_D` survive the aligned constant-background Dirac gate.
The aligned calculation does not select a sign, and v12A is not yet a viable
theory.

Two earlier reductions were incomplete:

1. the clock-only calculation omitted the conformal metric component of the
   Class-Ia primary null direction; and
2. the first correction restored that metric component but held the
   longitudinal aether velocity fixed.

The complete constant-background reduction cancels both apparent spatial
terms. The aligned bracket is

$$
\boxed{\Delta/F_0=-4K_2,}
$$

independent of wave number and of the sign of `lambda_D`.

## First incomplete block: clock only

For an aligned scalar clock `q`, normal Hessian `V_*`, metric trace `K`, and
spatial clock gradient `D_iq`,

$$
L_3=-q^2V_*^2-q^3KV_*,
$$

$$
L_4=-q^2V_*^2+q^2|Dq|^2,
\qquad
L_5=q^4V_*^2.
$$

Keeping only the direct `A4 q^2|Dq|^2` term gives

$$
\Delta_{\rm clock}/F_0
=-[4K_2+2r^2\bar A_4\bar k^2].
$$

For `lambda_D=+1`, `r=2`, and `K2=2`, this incomplete block vanishes at
`k_bar=3.5678747366`. That numerical zero is retained as a regression fixture,
not as a physical result.

## Second incomplete block: metric restored, aether fixed

The Class-Ia primary moves the conformal spatial metric with the clock:

$$
\delta\zeta=\eta\,\delta r,
\qquad
\eta=-{r^3\bar A_3\over4}.
$$

The direct DHOST, Einstein cross, and Einstein metric terms are

$$
r^2\bar A_4,
\qquad
-{4\eta\over r},
\qquad
2\eta^2.
$$

The luminal Class-Ia relation makes their sum vanish identically:

$$
\boxed{
r^2\bar A_4-{4\eta\over r}+2\eta^2=0.
}
$$

If the aether is then held fixed, the Maxwell sector appears to leave
`K_B/r^2`. This was the result reported by the preceding version of this
document. It is still not the full Dirac reduction.

## Complete aligned reduction: dynamical aether

The longitudinal Maxwell sector is a perfect square. Schematically,

$$
L_M=K_B|\dot A_L+i k\,\delta N|^2.
$$

Its direct lapse-gradient term is positive, but the canonical aether momentum
contains the same combination. Performing the Legendre/Schur reduction gives

$$
{K_B\over r^2}
+\left(-{K_B\over r^2}\right)=0.
$$

Thus the full aligned gradient coefficient is zero. The ordinary AeST clock
curvature remains:

$$
\Delta/F_0=-4K_2.
$$

At `K2=2`, it is `-8`. The result is nonzero for every finite nonzero clock
whenever `K2>0`.

## Executable audit

The 50,000-row signed logarithmic scan:

- reproduces the original clock-only zero;
- verifies the Class-Ia metric cancellation;
- verifies the dynamical-aether Maxwell cancellation;
- obtains the exact positive core `4K2=8` for both signs at every wave number;
- keeps all observational-data flags closed.

The aligned conclusion is unchanged: neither sign is falsified or selected.
What changed is the reason the bracket remains nonzero.

## Scope

This result is exact only for a local constant background with scalar gradient
and aether aligned. The separate tilted audit extends the constraint bracket
to arbitrary constant aether tilt. Neither calculation includes background
scalar Hessian, aether gradient, extrinsic curvature, spacetime curvature, or
the reduced physical characteristic and energy matrices.

No observational data or holdout were opened.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_aligned_finite_k.py
python -m pytest -q tests/test_sigma_v12a_aligned_finite_k.py
```

Machine-readable evidence is in
`results/sigma_v12a_aligned_finite_k/report.json`.
