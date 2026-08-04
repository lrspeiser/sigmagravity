# Sigma v10D exponential kinetic selection

> **Superseded gate result (2026-08-04):** the exponential removes v10C's
> finite-amplitude vector ghost, but it does not enter the exact spin-two
> sector. An anisotropic carrier gives
> `c_TT^2=1+c_P^2(p_parallel-p_perp)^2`, outside the physical metric cone.
> Exact v10D is retired and the aether-tidal family is reset. See
> [`SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md`](SIGMA_V10D_TENSOR_CONE_FALSIFICATION.md).

## Decision

Sigma v10D passes a narrow, theory-only successor-selection gate. It removes
v10C's finite-amplitude physical vector ghost with a fixed covariant matrix
function, adds no physical constant, and leaves the successful zero-background
static response and characteristic cones unchanged.

V10D advances only to the complete nonlinear ADM and arbitrary-background
characteristic gates. No astronomical fitting is authorized, and no
observational product or holdout was opened.

## Construction from the v10C failure

V10C's reduced aether kinetic matrix was

$$
K_BI-\beta P=K_B(I-X),
\qquad X={\beta\over K_B}P.
$$

V10D adds the covariant spatial matrix function

$$
\boxed{
\Delta\mathcal L_{\exp}
=K_BJ_\mu
\left[\exp(X)^\mu{}_{\nu}-q^\mu{}_{\nu}\right]
J^\nu
}.
$$

The exponential is defined on the three-dimensional aether-spatial bundle by
its convergent power series. Since `P` is self-adjoint with respect to the
spatial metric, all eigenvalues of `X` are real.

After the constraint-induced v10C term is included, the completed physical
vector kinetic matrix is

$$
\boxed{
K_{\rm vec}(P)=K_B\left[\exp(X)-X\right].
}
$$

For each real eigenvalue `x`,

$$
f(x)=e^x-x,
\qquad f'(x)=e^x-1,
\qquad f''(x)=e^x>0.
$$

The unique global minimum is therefore

$$
\boxed{f(0)=1}.
$$

Thus every completed vector kinetic eigenvalue is at least `K_B>0` for every
finite carrier amplitude. The audit samples 200,001 scalar values on
`[-20,20]` and random symmetric matrices with norm scales through 100; the
analytic result, rather than the finite scan, is the proof.

## Why this does not add a fit parameter

The matrix exponential has no adjustable shape or saturation constant. Its
argument is fixed by the already selected ratio `beta/K_B`; `beta^2/K_B=2/11`
still follows from the static-capacity and cone equations. The physical
parameter list remains

$$
\{a_\Sigma,\mu_\Sigma,K_B,K_2,L_P\}.
$$

No carrier cutoff, object label, or lensing-only coefficient is introduced.

The small-`P` expansion is

$$
K_B(e^X-I)=\beta P+{\beta^2\over2K_B}P^2+\cdots.
$$

Its leading `+beta P J J` term cancels the nonlinear kinetic term that retired
v10C. Because this addition is cubic in perturbations around `P=J=0`, it does
not change v10C's quadratic static response, flat mode count, or selected cone
coefficients.

## Nonzero-amplitude local cone check

Let `f>=1` be an eigenvalue factor of `exp(X)-X`. In a constant-`P`, zero-`J`
principal channel, the mixed characteristic equation becomes

$$
(u-fy)(s-y)-qy=0,
$$

with

$$
u={3\over4},\qquad s={3\over11},\qquad q={2\over11}
$$

for the longitudinal channel and `q/2` transversely. At `f=1` this is exactly
v10C, giving longitudinal squared speeds `9/44` and `1`. Scanning `f` from one
through `10^8` keeps both roots positive and no greater than one; increasing
the completed kinetic coefficient moves the upper cone inward.

The corresponding static block is

$$
\begin{pmatrix}
f&-\sqrt q\\
-\sqrt q&s
\end{pmatrix}.
$$

It is positive for every `f>=1`; its weakest point is the already passed
zero-background v10C block.

## What this selection does not prove

The exponential completion depends on the aether projector and on `P`, so the
full tilted metric--aether--carrier velocity Hessian must still be derived.
Background `J` can activate derivatives of the matrix exponential and change
the principal symbol. V10D therefore still needs:

1. the complete nonlinear ADM primary/secondary constraint chain;
2. tilted, nonzero-`J`, nonzero-`P`, separated-source, and FLRW characteristics;
3. the expanded Hilbert stress tensor and one-metric weak potentials;
4. PPN, Solar, compact-source, and cosmological limits;
5. a convergent PDE solver and only then the frozen observational gates.

Failure at any one retires exact v10D. The v10C scores are not inherited as
observational evidence.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10d_exponential_kinetic_selection.py
python -m pytest -q tests/test_sigma_v10d_exponential_kinetic.py
```

Machine-readable evidence is in
`results/sigma_v10d_exponential_kinetic_selection/report.json`.
