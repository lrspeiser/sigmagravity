# Sigma v10D anisotropic source-block characteristics

## Decision

V10D passes the arbitrary-carrier-orientation characteristic gate for its
fixed-metric aether--carrier source block. The result covers noncommuting
carrier and wave-direction matrices, all finite carrier eigenvalues, nonzero
background aether acceleration at principal order, and Lorentz-tilted views of
the resulting cones.

This is not the complete metric--aether--scalar--carrier ADM or characteristic
proof. V10D advances only to that full coupled gate. No observational product
or holdout was opened.

## Arbitrary spatial carrier background

Write the completed dimensionless aether kinetic matrix as

$$
F=e^X-X,\qquad X={\beta\over K_B}P.
$$

The v10D selection proved

$$
F\succeq I.
$$

For a unit wave direction `n`, the divergence of a symmetric spatial carrier
maps its six orthonormal components into the three aether components. The Gram
matrix of that map is

$$
\boxed{R={1\over2}(I+nn^{\mathsf T})},
$$

so

$$
{1\over2}I\preceq R\preceq I.
$$

This formulation does not require `F` and `R` to commute.

## Static positivity

Eliminating the carrier in the high-frequency static block gives the aether
Schur complement

$$
S_{\rm static}=F-{q\over s}R.
$$

At the selected values `q=2/11` and `s=3/11`,

$$
\boxed{
S_{\rm static}\succeq I-{2\over3}I={1\over3}I.
}
$$

Thus no anisotropic orientation can reproduce v10C's kinetic singularity or
destroy the static elliptic block. The weakest case is the already tested
zero-carrier longitudinal alignment.

## Full source-block characteristic polynomial

After the three sourced carrier combinations are eliminated, the six squared
characteristic speeds are the roots of

$$
\boxed{
M(y)=Fy^2-(sF+uI+qR)y+suI,
\qquad y={\omega^2\over k^2}.
}
$$

The other three carrier polarizations retain `y=s=3/11`.

For any unit vector `z`, define

$$
f=z^{\mathsf T}Fz\ge1,
\qquad
r=z^{\mathsf T}Rz\in[1/2,1].
$$

The projected scalar polynomial is

$$
f y^2-(sf+u+qr)y+su.
$$

Its coefficients give two positive roots. At the physical metric cone,

$$
z^{\mathsf T}M(1)z
=(1-s)(f-u)-qr
\ge(1-s)(1-u)-q=0.
$$

The last equality is exactly the zero-background v10C cone-saturation
condition. Because the product of the two roots is `su/f<1`, the roots cannot
both lie beyond one; the nonnegative value at one places both inside or on the
metric cone. Positive kinetic and static energy make the conservative
quadratic system hyperbolic with real frequencies.

The machine audit solves the noncommuting quadratic matrix eigenproblem for
2,000 random carrier tensors and wave directions. More than 95% deliberately
have noncommuting `F` and `R`. Every root is real, positive, and no greater
than one.

## Tilted frames and nonzero J

The characteristic calculation is made in the local aether-rest tetrad. A
rest-frame signal speed `c` with `|c|<=1` transforms under a one-dimensional
metric boost `v` as

$$
c'={c+v\over1+cv},
$$

which also has `|c'|<=1`. The numerical audit samples boosts through
`|v|=0.999`. Thus changing coordinates or viewing the aether as tilted does
not move a cone outside the physical metric cone.

The completion is quadratic in

$$
J_\mu=A^\nu\nabla_\nu A_\mu.
$$

Its derivative Hessian is `F(P)` and is independent of the background value
of `J`. A nonzero `J` activates derivatives of `F` in lower-order terms, but it
does not change this fixed-metric principal block. The first-order carrier
coupling is likewise bilinear in field derivatives with background-independent
principal coefficient `beta`.

## Remaining decisive gate

The metric is dynamical, and `J` contains the metric connection. The full ADM
kinetic matrix must therefore include metric--aether mixing, the AeST scalar
clock, lapse/shift constraints, the carrier spatiality multiplier, and the
matrix-exponential projector dependence. The current proof does not substitute
for that calculation.

V10D still requires:

1. the full nonlinear ADM primary and secondary constraint chain;
2. the complete metric--aether--scalar--carrier principal determinant on
   inhomogeneous and FLRW backgrounds;
3. weak metric potentials, PPN/Solar limits, and cosmological stability;
4. a convergent PDE implementation before any observational fit.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v10d_anisotropic_characteristics.py
python -m pytest -q tests/test_sigma_v10d_anisotropic_characteristics.py
```

Machine-readable evidence is in
`results/sigma_v10d_anisotropic_characteristics/report.json`.
