# Sigma v17J flat-vacuum kinetic falsification

## Result

The frozen v17H/v17I susceptibility action fails before observational testing.
At its declared (c_U=1), the transverse aether kinetic coefficient vanishes
while its spatial-gradient coefficient remains nonzero. The spin-1
characteristic is therefore singular. Independently, the aether spin-0
combination vanishes, leaving a zero-speed or singular scalar characteristic.

Changing only (c_U) cannot repair the action. The exact interval analysis is:

- (c_U<0): wrong-sign Maxwell gradient and spin-1 energy numerator;
- (c_U=0): degenerate transverse gradient;
- (0<c_U<1): negative kinetic coefficient and negative (s_1^2);
- (c_U=1): zero kinetic coefficient and singular spin-1 characteristic; and
- (c_U>1): positive transverse quadratic coefficients, but
  (s_1^2=1+1/(c_U-1)>1), outside v17H's frozen physical-metric cone.

This retires the **frozen kinetic completion**, not the pressure-source idea or
the goal of deriving an apparent halo scale from baryonic fields.

## Where the earlier check went wrong

V17H computed the positive Hessian of

\[
F_A(Z)=\sqrt{1+Z}-1
\]

as a static function of the acceleration magnitude. It then described the
Maxwell term as adding a positive floor. But the Lorentzian action contains

\[
-M_{\rm Pl}^2{a_\Sigma^2\over c^4}F_A(Z).
\]

The Hessian of (F_A) is positive, yet its contribution to the time-dependent
Lagrangian has the **minus sign**. A positive static function Hessian is not a
positive kinetic matrix.

No original result file is rewritten. This v17J report explicitly supersedes
only v17H's limited reduced-health interpretation.

## Direct quadratic expansion

Take flat spacetime with signature ((-+++)), background
(U_\mu=(-1,0,0,0)), (A_\mu=0), (X=0), and no matter. For a transverse
spatial perturbation (u_T),

\[
F_{0i}=\dot u_i,
\qquad
A_i=\dot u_i
\]

to linear order. Since (F_A(Z)=Z/2+O(Z^2)),

\[
\mathcal L_{F}^{(2)}
={M_{\rm Pl}^2c_U\over2}
\left(|\dot u_T|^2-k^2|u_T|^2\right),
\]

while

\[
\mathcal L_A^{(2)}=-{M_{\rm Pl}^2\over2}|\dot u_T|^2.
\]

Therefore

\[
\boxed{
\mathcal L_T^{(2)}={M_{\rm Pl}^2\over2}
\left[(c_U-1)|\dot u_T|^2-c_Uk^2|u_T|^2\right].
}
\]

At the frozen (c_U=1), the time Hessian is zero. The executable report also
differentiates the exact square-root Lagrangian numerically and recovers this
coefficient within the fixed tolerance.

## Full aether--metric cross-check

At quadratic order the action maps to the standard Einstein--aether invariant
basis as

\[
c_1=c_U,\qquad c_2=0,\qquad c_3=-c_U,\qquad c_4=-1.
\]

Thus

\[
c_{13}=0,\qquad c_{14}=c_U-1,\qquad c_{123}=0.
\]

The established flat-background squared mode speeds are

\[
s_2^2={1\over1-c_{13}},
\]

\[
s_1^2={2c_1-c_1^2+c_3^2\over
2c_{14}(1-c_{13})},
\]

and

\[
s_0^2={c_{123}(2-c_{14})\over
c_{14}(1-c_{13})(2+c_{13}+3c_2)}.
\]

Substitution gives

\[
s_2^2=1,
\qquad
s_1^2={c_U\over c_U-1},
\qquad
s_0^2=0
\]

whenever the denominators are regular. At (c_U=1), both (c_{14}=0) and
(c_{123}=0), so the corresponding characteristics are singular rather than
healthy propagating limits. The separate canonical (X) scalar does not repair
the aether scalar in vacuum: at (X=0) and without matter, there is no quadratic
mixing that can generate the missing aether gradient.

Because (c_{13}=0), the tensor cone remains exactly luminal. That success is
not enough to rescue the singular vector and scalar sectors.

The mode formulas and energy analysis are established Einstein--aether prior
art: [Jacobson and Mattingly](https://arxiv.org/abs/gr-qc/0402005),
[Eling](https://arxiv.org/abs/gr-qc/0507059), and
[Jacobson's review](https://arxiv.org/abs/0801.1547). This project claims no
novelty for those formulas; it applies them to the frozen Sigma coefficient
mapping.

## Why the sign-flipped control is not a rescue

For diagnosis only, reversing the Born--Infeld acceleration sign would give

\[
\mathcal L_{T,+}^{(2)}={M_{\rm Pl}^2\over2}
[(c_U+1)|\dot u_T|^2-c_Uk^2|u_T|^2]
\]

and (s_{1,+}^2=c_U/(c_U+1)). That repairs the transverse sign for (c_U>0),
but it is a materially different theory. It also leaves (c_{123}=0), hence
does not repair the aether scalar sector. At (c_U=1), it gives (c_{14}=2),
where the standard Newtonian coupling denominator vanishes.

Advancing a related carrier would require adding and preregistering independent
expansion/shear operators that make (c_{123}\ne0), then repeating the complete
PPN, constraint, characteristic, and nonlinear-background analysis. Those
operators cannot be selected after looking at cluster targets.

## Consequence for the halo-size program

The attractive part of v17H was the proposed output relation

\[
(1-L_\Sigma^2\nabla_\perp^2)s_\Sigma
\propto\chi(Z)\mathcal J_T,
\qquad
\Delta W\propto\alpha\chi(Z)s_\Sigma,
\]

under which baryonic stress extent, local acceleration, and one common
propagation length would determine an apparent halo radius. V17J shows that the
particular field carrier used to realize that relation is unhealthy. It does
not show that baryonic stress cannot determine halo size.

The next candidate must preserve the empirical question while changing the
field completion materially. In particular, it must generate positive
kinetic/gradient matrices for all modes before its halo-radius predictions are
compared with data.
