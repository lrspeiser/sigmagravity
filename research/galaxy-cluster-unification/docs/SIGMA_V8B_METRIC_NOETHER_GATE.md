# Sigma v8B metric Euler tensor and Noether gate

## Result

The v8B completion now has explicit Euler derivatives for all three dynamical
field types: scalar, vector, and metric. Its off-shell diffeomorphism identity
is also explicit. This completes the variation and conservation subgate for the
new operator, but not the Hamiltonian degree-of-freedom or characteristic-cone
gates for the combined AeST theory.

## Metric convention

Take the contravariant vector `A^mu` as an independent field. Then

$$
Q=A^\mu\nabla_\mu\phi
$$

does not change under a metric variation at fixed `A^mu` and `phi`. The metric
dependence of

$$
\mathcal L_C=C B^2q^{\alpha\beta}\nabla_\alpha\nabla_\beta\phi
$$

comes from the volume element, `g^mn` inside `q^mn`, and the connection in the
scalar Hessian.

## Exact metric Euler tensor

Define `E_g^{mn}` as the coefficient of `delta g_mn` in
`delta(sqrt(-g)L_C)`. Direct variation and integration by parts give

$$
\boxed{
\begin{aligned}
{E_g^{\mu\nu}\over C}={}&
{1\over2}g^{\mu\nu}B^2H_\perp
-B^2\nabla^\mu\nabla^\nu\phi\\
&+\nabla_\rho\!\left[
B^2q^{\rho(\mu}\nabla^{\nu)}\phi
\right]
-{1\over2}\nabla_\rho\!\left[
B^2q^{\mu\nu}\nabla^\rho\phi
\right].
\end{aligned}
}
$$

If the completion is moved to the right-hand side of an Einstein-form metric
equation, its effective stress tensor is `T_C^{mn}=-2E_g^{mn}` with the common
overall gravitational-action normalization restored.

The algebraic part of this tensor was checked independently by perturbing a
general constant Lorentzian metric in a symmetric random direction. The
analytic contraction and centered finite difference agree to better than
`1e-9` relative error.

## Off-shell diffeomorphism identity

For independent fields `{g_mn,A^m,phi}`, diffeomorphism invariance gives

$$
\boxed{
-2\nabla_\mu E_g^{\mu}{}_{\nu}
+E_\phi\nabla_\nu\phi
+E_{A^\mu}\nabla_\nu A^\mu
+\nabla_\mu(E_{A_\nu}A^\mu)=0.
}
$$

On the scalar and vector equations, this reduces to

$$
\nabla_\mu E_g^{\mu}{}_{\nu}=0.
$$

Because all ordinary matter is minimally coupled to the same `g_mn`, its
stress-energy conservation follows from the combined metric equation and the
matter equations. No separate photon or galaxy conservation law is introduced.

The unit constraint contribution from `lambda(A^2+1)` must be included with the
published AeST vector and metric Euler derivatives when applying the identity to
the full action; the completion identity above is its independently covariant
piece.

## What this establishes

- The new operator is not merely a prescribed static equation.
- Its scalar principal third derivatives cancel.
- Its vector equation introduces no new aether derivative.
- Its metric stress tensor is fixed by the same action.
- The scalar, vector, and metric equations obey one covariance identity.
- Matter and photons still use one conserved physical metric.

## What remains open

Second-order-looking field equations and a Noether identity do not by themselves
prove the correct number of degrees of freedom. The next gate must compute the
nonlinear kinetic/constraint rank, including tilted aether and `Q` away from
`Q0`. It must then build the full coupled metric-vector-scalar characteristic
determinant. Only after that may v8B proceed to Solar/PPN solutions.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v8b_metric_noether.py
python -m pytest -q tests/test_sigma_v8b_covariant_variation.py
```

Machine-readable evidence is stored in
`results/sigma_v8b_metric_noether_gate/report.json`.
