# Sigma v10D exact tensor-cone falsification and mechanism reset

## Decision

Exact Sigma v10D is retired before observational data. On a nonzero
axisymmetric carrier background, its two transverse-traceless metric modes
propagate outside the null cone of the one physical metric:

$$
\boxed{c_{TT}^2=1+c_P^2(p_\parallel-p_\perp)^2}.
$$

The result is an exact symmetry-sector characteristic, not a random scan or a
weak empirical preference. Every allowed nonzero carrier anisotropy widens the
cone. The exponential completion that repaired v10C's vector ghost affects the
aether acceleration block and cannot cancel this carrier spatial-connection
term.

V10B, v10C, and v10D are now three materially different closures of the same
aether-tidal carrier idea that fail the common mathematical
causality/stability gate. Under the preregistered rule, this mechanism family
is reset rather than receiving a v10E counterterm. No astronomical product or
holdout was opened.

## Exact invariant sector

At one event, use the aether-rest local frame and choose an axisymmetric
spatial carrier and a wave along its symmetry axis:

$$
\bar P_{ij}=\operatorname{diag}(p_\perp,p_\perp,p_\parallel),
\qquad k_i=k\,\delta_{iz}.
$$

The background retains rotations about the wave axis. The plus and cross
metric polarizations have spin two under this symmetry. Lapse and scalar
perturbations have spin zero, while shift and aether-vector perturbations have
spin one. They therefore cannot mix with this sector. The two TT polarizations
and the spin-two carrier perturbations form a closed principal block.

This removes the usual ambiguity of diagnosing a submatrix while omitted
constraints might change its eigenmodes.

## Time derivative

For a spatial carrier, the projected covariant time derivative is

$$
W_{ij}=\delta\dot P_{ij}
-{1\over2}(\dot h_i{}^k\bar P_{kj}
+\bar P_i{}^k\dot h_{kj}).
$$

Define

$$
r_{ij}=\delta P_{ij}
-{1\over2}(h_i{}^k\bar P_{kj}
+\bar P_i{}^k h_{kj}).
$$

Then `W_ij=dot(r)_ij`. This is the same unit-determinant velocity
transformation that made the ADM Legendre-rank gate pass. It gives no extra TT
time stiffness.

## Spatial derivative

The spatial covariant derivative does not transform in the same way. Its
linearized principal part is

$$
\delta(D_\ell P_{ij})=ik[n_\ell r_{ij}+R_{\ell ij}],
$$

with

$$
R_{\ell ij}=-{1\over2}\left[
(n_i h_{\ell k}-n_k h_{\ell i})\bar P^k{}_j
+(n_j h_{\ell k}-n_k h_{\ell j})\bar P_i{}^k
\right].
$$

For either normalized TT polarization on the axisymmetric background,

$$
n^\ell R_{\ell ij}=0,
\qquad
R_{\ell ij}R^{\ell ij}
={(p_\parallel-p_\perp)^2\over2}h_{ij}h^{ij}.
$$

The first identity means that this residual is orthogonal to the ordinary
carrier plane-wave derivative `n_l r_ij`: the pure TT mode with `r_ij=0` is an
exact eigenmode. The second identity gives an additional positive spatial
gradient cost but no corresponding time kinetic term.

## Characteristic speed

The Einstein-Hilbert TT principal Lagrangian plus the carrier gradient term is

$$
\mathcal L_{TT}^{(2)}
={1\over4}\left[\dot h:\dot h-k^2h:h\right]
-{c_P^2\over2}k^2R:R.
$$

Consequently,

$$
\mathcal L_{TT}^{(2)}
={1\over4}\left[
\dot h:\dot h
-k^2\{1+c_P^2(\Delta p)^2\}h:h
\right],
$$

where `Delta p=p_parallel-p_perp`. Both polarizations therefore have

$$
\boxed{\omega^2/k^2=1+c_P^2(\Delta p)^2}.
$$

At the frozen value `c_P^2=3/11`, even `Delta p=10^-6` gives

$$
{c_{TT}\over c}-1=1.3636\times10^{-13},
$$

which exceeds the declared `10^-15` tolerance. The tolerance would require

$$
|\Delta p|\le
\sqrt{{2(10^{-15})+(10^{-15})^2\over3/11}}
=8.56\times10^{-8}.
$$

That number is illustrative, not an inferred astronomical amplitude. The
stronger theory failure is that the action admits arbitrary nonzero
anisotropy and is superluminal on every such background. A potential that
makes large carrier values energetically expensive does not remove those
states or make the metric cone characteristic.

## Why the other terms do not repair this sector

- The aether acceleration is exactly zero for TT perturbations in local
  synchronous gauge, so the v10D matrix-exponential `J.exp(X).J` term vanishes.
- The tidal interaction `P^{ij}D_iJ_j` also vanishes in this spin-two sector.
- Scalar and lapse fields cannot mix by the surviving axial symmetry.
- The carrier potential and background curvature are lower derivative and do
  not enter the high-frequency characteristic.
- Adjusting `c_P^2` cannot help while it is positive: every positive value
  produces the same sign. Setting it to zero destroys hyperbolic spatial
  propagation and returns toward the v10B causal failure.

## Three-closure mechanism audit

| Closure | Intended repair | Exact failure |
|---|---|---|
| v10B auxiliary carrier | Positive static tidal response | Physical equal-preferred-time Yukawa tail; no causal front |
| v10C hyperbolic carrier | Replace instantaneous response with retarded waves | Finite-amplitude aether-vector kinetic zero and ghosts |
| v10D exponential completion | Make the vector kinetic matrix globally positive | Anisotropic carrier widens the exact physical TT cone |

These are not three parameter choices. They are auxiliary, hyperbolic, and
nonlinearly completed actions. All fail Action 12's mathematical
causality/stability gate before observational scoring. The next candidate must
change the mechanism, not add another function of `P` to the same kinetic
architecture.

## Consequence for the research program

The empirical RAR plus coherence-gated refracted-gravity bridge remains a
useful description of the target behavior, but it is not evidence for this
aether-tidal field family. The next mechanism selection should preserve the
lessons that survived:

1. one physical metric must determine both dynamics and lensing;
2. geometry beyond a local scalar acceleration is needed for cluster shear;
3. all added tensor structure must leave the physical metric null cone exact
   on nonzero backgrounds, not merely on Minkowski vacuum;
4. nonlinear regularity must be analytic rather than enforced by a fitted
   carrier cutoff;
5. no observational holdout should be opened until the replacement passes
   the action, constraint, characteristic, and Solar/PPN gates.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10d_tensor_cone.py
python -m pytest -q tests/test_sigma_v10d_tensor_cone.py
```

Machine-readable evidence is in
`results/sigma_v10d_tensor_cone/report.json`.
