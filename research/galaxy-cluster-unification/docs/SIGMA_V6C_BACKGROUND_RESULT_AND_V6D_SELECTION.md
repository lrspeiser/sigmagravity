# Sigma v6C background result and v6D cubic-orientation selection

## Outcome

Sigma v6C's detuned tensor memory remains useful, but its placement inside
$\chi=X(1+\lambda_\Sigma C_Q)$ is retired. On any fixed nonzero orientation
background it changes the quadratic term required to cancel the Einstein weak
action and creates a negative deep-field constitutive coefficient.

Sigma v6D retains the bounded, detuned, retarded tensor memory but moves its
effect into the cubic MOND term. A 100,000-point scan over 32 decades in the
gradient invariant finds positive constitutive and parallel ellipticity
coefficients for $0\le q\le0.999$, where
$q=\lambda_\Sigma C_Q$. No observational data were accessed.

## Necessary weak scalar normalization

Write the scalar weak-field surrogate as

$$
\ell(X)=X-f(X),
$$

where the first $X$ represents the relevant Einstein quadratic term and the
leading $X$ in $f$ cancels it in the deep regime. This is a necessary reduction
of the full metric theory, not a substitute for the pending covariant variation.

For v6C,

$$
f_C=f((1+q)X),\qquad q=\lambda_\Sigma C_Q>0.
$$

The total constitutive coefficient is

$$
\mu_C={d\ell_C\over dX}
=1-(1+q)e^{-\sqrt{(1+q)X}}
\left(1-\frac12\sqrt{(1+q)X}\right).
$$

Therefore

$$
\boxed{\lim_{X\to0}\mu_C=-q.}
$$

Every nonzero orientation background has a negative-response interval. Detuning
the tensor pole does not fix this sign error.

## Selected v6D placement

Keep the v6C tensor memory,

$$
Q_{\mu\nu}
=\operatorname{STF}_h[(\Box_R-m_\Sigma^2)^{-1}T_{\mu\nu}],
$$

$$
C_Q={Q_{\mu\nu}Q^{\mu\nu}
\over Q_{\rho\sigma}Q^{\rho\sigma}+\phi_\Sigma^2},
\qquad q=\lambda_\Sigma C_Q,
$$

but freeze the correction as

$$
\boxed{
f_D(X,C_Q)=e^{-\sqrt X}
\left(X+\lambda_\Sigma C_Q X^{3/2}\right).
}
$$

At small $X$,

$$
f_D=X-(1-q)X^{3/2}+O(X^2),
$$

so

$$
\ell_D=X-f_D=(1-q)X^{3/2}+O(X^2).
$$

The quadratic cancellation is exact for every $C_Q$. Orientation changes only
the coefficient of the cubic deep-field law.

The exact constitutive coefficient is

$$
\mu_D=1-e^{-s}
\left[1+{(3q-1)s\over2}-{q s^2\over2}\right],
\qquad s=\sqrt X.
$$

Its deep limit is

$$
\mu_D={3\over2}(1-q)s+O(s^2).
$$

The parallel ellipticity coefficient is
$\mu_D+2X\,d\mu_D/dX$. Both coefficients remained positive in the scan for
$q\le0.999$. The strict action domain is frozen to

$$
0\le\lambda_\Sigma<1,
$$

because $0\le C_Q\le1$.

## Physical meaning

In the spherical deep-field surrogate, the equation has the scaling

$$
(1-q){g^2\over a_\Sigma}\propto g_{\rm bar}.
$$

At fixed baryonic source, orientation therefore changes the acceleration relative
to the $q=0$ MOND-like control by

$$
{g(q)\over g(0)}={1\over\sqrt{1-q}}.
$$

For example, $q=0.99$ permits a factor-ten deep-field enhancement without making
the kinetic coefficient negative. This is a capability statement, not a fitted
cluster result. Whether the covariantly calculated $C_Q$ is near zero in coherent
galaxies and near one in complex clusters remains an empirical prediction to test
only after the action gates pass.

## Complexity

The four provisional universal constants remain

$$
\{a_\Sigma,\lambda_\Sigma,\phi_\Sigma,L_\Sigma\}.
$$

No object label, per-object force parameter, lensing multiplier, or freely chosen
homogeneous memory state is introduced.

## Required next gate

v6D now requires the complete closed-time-path influence action and variation.
The next report must derive the diagonal diffeomorphism Ward identity with the
metric-built timelike direction, include every reciprocal $U_R$ and $Q_{\mu\nu}$
term, and compute the kinetic matrix on nonzero scalar and tensor-memory
backgrounds. Positivity of this reduced scalar surrogate is necessary but not
sufficient.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/select_sigma_v6d_cubic_orientation.py
python -m pytest -q tests/test_sigma_v6_metric_memory.py
```

Machine-readable evidence is in
`results/sigma_v6d_cubic_orientation_selection/report.json`.
