# Sigma v17M active-pressure kinetic gate

## Result

The acceleration-dependent susceptibility **fails** this necessary kinetic
gate. The analytic Hessian agreed with the five-point finite difference to a
maximum normalized error of \(2.13\times10^{-8}\), so the failure is not a
differentiation artifact. Of 19,845 preregistered canonical-matter backgrounds,
7,836 had \(c_{14}^{\rm eff}\le0\) and hence a non-positive spin-1 speed
squared.

At \(q=10^{-5}\), even the dust-like case \(p/\rho=10^{-5}\) reaches the
singular surface at

\[
\widehat\rho_{\rm crit}=399.9824,
\qquad p_{\rm crit}=3.43\times10^{-14}\ {\rm Pa}.
\]

For the explicit stiff canonical witness
\((q,\widehat\rho,p/\rho)=(10^{-5},0.01,1)\),

\[
c_{14}^{\rm eff}=-5.00\times10^{-8},
\qquad s_1^2=-2.00,
\]

while its matter kinetic coefficient is positive. This isolates the failure in
the gravity sector.

The required decision is therefore to remove \(\chi(A^2)\) from the physical
matter metric. The healthy vacuum luminal aether carrier can remain available
for a materially different coupling.

## Question

V17L made the equations local and reciprocal. It did not answer the harder
question: when pressure is a dynamical matter field rather than an imposed
number, does the acceleration-dependent physical metric keep a positive kinetic
matrix?

V17M answers that question in the transverse sector with an explicit canonical
matter action. This sector is enough for a decisive necessary test because
rotational symmetry prevents a transverse aether perturbation from mixing with
the scalar matter perturbation.

## Explicit matter realization

Use

\[
S_\phi=-\int d^4x\sqrt{-\widetilde g}
\left({1\over2}\widetilde g^{\mu\nu}
\partial_\mu\phi\partial_\nu\phi+V\right).
\]

At one event choose a local rest frame with

\[
g_{\mu\nu}=\eta_{\mu\nu},\qquad U^\mu=(1,0,0,0),
\qquad \phi=y t,
\]

and let \(q_0=\alpha X\). The physical metric is

\[
d\widetilde s^2=e^{2q}
[-(1-2q)dt^2+d\mathbf x^2].
\]

It remains Lorentzian for \(q<1/2\). The exact local matter Lagrangian is

\[
L_\phi(q)={y^2e^{2q}\over2\sqrt{1-2q}}
-Ve^{4q}\sqrt{1-2q}.
\]

At \(q=0\),

\[
\rho={y^2\over2}+V,\qquad p={y^2\over2}-V.
\]

Differentiating the actual matter action, rather than holding a stress tensor
fixed, gives

\[
\begin{split}
\mathcal J(q)={}&{(\rho+p)e^{2q}(3-4q)
\over2(1-2q)^{3/2}}\\
&-{(\rho-p)e^{4q}(3-8q)
\over2\sqrt{1-2q}}.
\end{split}
\]

This reproduces \(\mathcal J(0)=3p\), but it also reveals a previously hidden
nonlinear term:

\[
\boxed{\mathcal J(q)=3p+q(2\rho+9p)+O(q^2).}
\]

Thus cold rest energy cancels only at exactly \(q=0\). Once the Sigma field is
active, ordinary rest density returns as a source.

## Exact transverse kinetic coefficient

Write

\[
v_i={c^2A_i\over a_\Sigma},
\qquad q(v)=q_0(1+v^2)^{-1/4}.
\]

Near zero aether acceleration,

\[
q(v)=q_0\left(1-{v^2\over4}+O(v^4)\right).
\]

Define

\[
\widehat\rho={8\pi G\rho\over a_\Sigma^2},\quad
\widehat p={8\pi Gp\over a_\Sigma^2},\quad
\widehat{\mathcal J}={8\pi G\mathcal J\over a_\Sigma^2}.
\]

The exact matter Hessian is

\[
{\partial^2\widehat L_\phi\over\partial v_i\partial v_j}
=-{q_0\widehat{\mathcal J}(q_0)\over2}\delta_{ij}.
\]

The mixed \(v_i\)-\(\dot\phi\) Hessian is zero at \(A_i=0\), while the matter
scalar kinetic term is positive:

\[
K_\phi={e^{2q_0}\over\sqrt{1-2q_0}}>0.
\]

Consequently the matter term changes the Einstein--aether coefficient as

\[
c_{14}^{\rm eff}=\varepsilon
-{q_0\widehat{\mathcal J}(q_0)\over2},
\]

and the reduced spin-1 speed is

\[
s_1^2={\varepsilon\over c_{14}^{\rm eff}}.
\]

For \(p=w\rho\), the singular density is finite whenever the response is
positive:

\[
\boxed{
\widehat\rho_{\rm crit}
={2\varepsilon\over q_0F(q_0,w)},\qquad
F={\widehat{\mathcal J}\over\widehat\rho}.
}
\]

Above this surface, \(c_{14}^{\rm eff}<0\) and \(s_1^2<0\). The canonical
matter field itself remains healthy, so this is a negative transverse gravity
direction rather than a matter ghost or a frozen-source artifact.

## Preregistered decision

The derivative susceptibility survives only if no allowed canonical-matter
background with

\[
0<q<1/2,qquad \rho>0,qquad 0\le p\le\rho
\]

has a zero or negative \(c_{14}^{\rm eff}\). If a finite positive-density zero
surface exists, the term \(\chi(A^2)\) must be removed from the physical metric.
Changing the already frozen \(\varepsilon\) is not a solution: every finite
value merely relocates the zero surface.

This gate does not test observations and does not reject the published luminal
Einstein--aether carrier. It tests only the proposed location of the Sigma
susceptibility inside the matter metric.
