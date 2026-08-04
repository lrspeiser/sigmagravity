# Sigma v12A aligned finite-wave-vector Dirac subgate

## Decision

The frozen positive-orientation row `lambda_D=+1` is falsified before data.
On an aether-aligned, zero-spatial-scalar-gradient auxiliary background with
`q/q_sigma=2`, its
primary-secondary bracket vanishes at the finite wave number
`k/q_sigma=3.5678747366`. This is a constraint-rank surface, not a poor data
fit.

The covariant mechanism is not yet retired. The same calculation derives an
analytic sign condition under which the aligned symbol cannot vanish:

$$
{-8\over\sqrt{1+x_0^2}}\le\lambda_D\le0.
$$

For the frozen `x0=-1` normalization this is
`-4 sqrt(2)<=lambda_D<=0`. The nonzero sentinel `lambda_D=-1` passes this
aligned subgate and advances to the fully tilted and anisotropic principal
matrix. This is a theory-health restriction made without opening data, not a
refit.

## Exact aligned reduction

Take a constant background with the scalar first gradient and AeST aether
aligned with the ADM normal, and use the standard DHOST auxiliary covector
before lapse gauge fixing. Write

$$
q=\nabla_n\phi,\qquad
V_*=\phi_{nn},\qquad
u_i=D_iq,
$$

and let `K` be the trace of the extrinsic curvature. Restoring the spatial
normal-clock gradient in the covariant Hessian invariants gives

$$
L_3=-q^2V_*^2-q^3KV_*,
$$

$$
\boxed{L_4=-q^2V_*^2+q^2|Dq|^2},
$$

$$
L_5=q^4V_*^2.
$$

The velocity terms retain the exact Class-Ia degenerate square derived in the
homogeneous audit. After removing that null velocity, the new spatial term is

$$
L_{\rm spatial}=A_4q^2|Dq|^2.
$$

The AeST Maxwell term contains a spatial derivative of the aether/lapse, not
of the independent auxiliary coordinate `q`, and therefore does not alter
this principal coefficient in the ungauge-fixed primary-secondary bracket.
Only after imposing unitary gauge is `q` identified with the inverse lapse;
making that identification before constructing the Dirac matrix would mix the
second-class pair with the lapse gauge condition and give the wrong bracket.
The reduced AeST scalar function supplies the positive zeroth-order clock
susceptibility `4K2` derived in the preceding audit.

## Fourier symbol

Use

$$
r={q\over q_\Sigma},\qquad
\bar k={k\over q_\Sigma},\qquad
x=-r^2,
$$

and remove the common coefficient factors through

$$
\bar A_3={q_\Sigma^4A_3\over F_0}
=\lambda_D\mathcal A(x,x_0),
$$

$$
\bar A_4=-\bar A_3-{x^2\bar A_3^2\over8},
\qquad
\bar C=r^2\bar A_4.
$$

Linearizing around a constant zero-momentum background gives

$$
\boxed{
{\Delta(\bar k)\over F_0}
=-\left(4K_2+2\bar C\bar k^2\right).
}
$$

If `A4<0`, this necessarily vanishes at

$$
\bar k_*^2=-{2K_2\over\bar C}.
$$

For the selected positive row, `r=2`, `x0=-1`, and `K2=2`, the exact numbers
are

| quantity | value |
|---|---:|
| activation | `0.0690268489963` |
| `A4_bar` | `-0.0785562607610` |
| `C_bar` | `-0.314225043044` |
| `k_*/q_sigma` | `3.56787473663` |
| normalized symbol residual | below `1e-15` |

Thus the positive row cannot retain one regular primary-secondary pair over
its admitted phase space.

## Why the negative sign survives this branch

Define the non-negative activation weight

$$
B(x)=x^2\mathcal A(x,x_0)
={x^2(x-x_0)^2\over
[1+(x-x_0)^2]^{3/2}\sqrt{1+x^2}}.
$$

It obeys the analytic bound

$$
B(x)
\le {|x|\over\sqrt{1+(x-x_0)^2}}
\le\sqrt{1+x_0^2}.
$$

Writing `z=lambda_D B`, the dependent coefficient is

$$
\bar A_4=-\bar A_3\left(1+{z\over8}\right).
$$

Consequently `A4_bar>=0` for all finite `x` whenever

$$
-{8\over\sqrt{1+x_0^2}}\le\lambda_D\le0.
$$

Together with `K2>0`, this makes the aligned symbol strictly nonzero for every
finite wave number. Fifty thousand signed clock and wave-number trials over
`10^-6` through `10^6` test the analytic inequalities with `lambda_D=-1`.

## Scope and next kill gate

This result is exact only on the constant, zero-momentum, aether-aligned
branch. It is enough to falsify `lambda_D=+1`, because one admissible rank-zero
background is sufficient. Passing with `lambda_D=-1` is narrower: aether tilt,
nonzero scalar spatial first gradient, anisotropic metric/aether perturbations,
and arbitrary wave-vector orientation could still create a zero eigenvalue.

The next gate is the full negative-branch principal matrix in those variables.
No astronomical data or holdout was opened, and the project still marks the
theory as not viable.

The covariant degeneracy relations and primary-secondary construction follow
the published [DHOST Hamiltonian analysis](https://arxiv.org/abs/1512.06820).
The aether/lapse derivative distinction follows the published
[AeST ADM action](https://arxiv.org/abs/2307.15126). The aligned sign bound and
counterexample above are project calculations; no novelty claim is made.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v12a_aligned_finite_k.py
python -m pytest -q tests/test_sigma_v12a_aligned_finite_k.py
```

Machine-readable evidence is in
`results/sigma_v12a_aligned_finite_k/report.json`.
