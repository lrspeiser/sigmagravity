# Sigma v17N decreasing metric-screen no-go

## Result

The whole differentiable decreasing-screen class is retired under the stated
assumptions. All five representative curves had a finite positive-density
kinetic zero, and their exact matter Hessians agreed with five-point finite
differences to a maximum normalized error of \(4.00\times10^{-8}\).

The quartic soft start illustrates the general result. Although
\(\chi'(0)=0\), at \(z=1\)

\[
\chi'(1)=-0.11463,
\qquad
\widehat\rho_{\rm crit}=0.01454
\quad (q_b=10^{-5},\ p=\rho).
\]

It therefore relocates the v17M failure rather than removing it. Trying more
decreasing curve shapes inside the same physical metric would not constitute a
new physical mechanism.

## Why this gate exists

V17M falsified

\[
\chi(z)=(1+z)^{-1/4},\qquad z={c^4A^2\over a_\Sigma^2},
\]

because active matter can drive the transverse aether kinetic coefficient
through zero. A tempting response is to make the susceptibility flatter near
\(z=0\), for example \((1+z^2)^{-1/8}\). V17N asks whether any smooth curve
change can actually solve the problem.

## General transverse identity

Let

\[
q=q_b\chi(z),\qquad z=v_iv_i,qquad q_b>0.
\]

At a background \(v_i\), perturb in a perpendicular direction \(\delta v_T\).
Then

\[
z(\delta v_T)=z_0+\delta v_T^2,
\]

so

\[
{\partial q\over\partial\delta v_T}=0,
\qquad
{\partial^2q\over\partial\delta v_T^2}
=2q_b\chi'(z_0).
\]

For any explicit matter action with

\[
\widehat{\mathcal J}={\partial\widehat L_m\over\partial q}>0,
\]

the exact transverse matter Hessian is

\[
\boxed{
\Delta K_T=2q_b\widehat{\mathcal J}\chi'(z_0).
}
\]

There is no \(L_{qq}\) term because the first transverse derivative of \(q\)
vanishes. If the bare gravitational coefficient is finite and positive,

\[
K_T=K_b(z_0)+2q_b\widehat{\mathcal J}\chi'(z_0).
\]

Whenever \(\chi'(z_0)<0\), the zero occurs at

\[
\boxed{
\widehat{\mathcal J}_{\rm crit}
={K_b(z_0)\over-2q_b\chi'(z_0)}.
}
\]

That value is finite and positive. Canonical matter amplitude scales
\(\widehat{\mathcal J}\), so an allowed matter background always reaches it.

## Why a soft start cannot work

A quartic soft start can arrange \(\chi'(0)=0\). But a useful screen must be
smaller at some larger acceleration. By the mean value theorem, if

\[
\chi(z_2)<\chi(z_1),\qquad z_2>z_1,
\]

then \(\chi'(z_*)<0\) somewhere between them. The kinetic zero simply moves
from \(z=0\) to \(z_*\).

The executable audit tests the exact canonical-matter Hessian for the original,
quartic-soft, exponential, rational, and nonzero-asymptote screens. These are
not five theories or a parameter sweep; they are numerical witnesses to the
same analytic result.

## Scope

The no-go applies when all of the following are true:

- the susceptibility is differentiable and decreases somewhere;
- it is placed inside the physical metric as a function of \(A^2\);
- the bare transverse kinetic coefficient is finite and independent of local
  matter amplitude;
- an allowed matter action has positive reciprocal source.

It does not reject the luminal aether carrier or the baryon-derived halo goal.
It says the environmental response must be moved out of derivative dependence
of the physical matter metric. A constant or increasing susceptibility would
avoid this sign proof but would not provide the intended high-acceleration
screening.
