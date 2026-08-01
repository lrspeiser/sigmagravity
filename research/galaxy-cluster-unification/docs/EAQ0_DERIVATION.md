# EA-Q0 local-action derivation and pre-fit decision

Status: completed pre-fit derivation checkpoint. The minimal local coupling is
retired because the reciprocal Aether source changes the environment field by
far more than the frozen 5% limit. No new astrophysical fit was performed.

## Frozen action

Use signature $(-,+,+,+)$ and $c=1$ in this section. The physical metric is
$g_{ab}$ itself; every matter field and light ray couples through the same
$S_m[g_{ab},\psi_m]$. Define

$$
F(Q)=1+2\beta Q,
\qquad
a_Q(Q)=a_0e^{\eta Q},
\qquad
a^b=u^a\nabla_a u^b,
\qquad
Y=\frac{a^aa_a}{a_Q^2}.
$$

The action tested in the first complete local cycle is

$$
S_{\rm EAQ0}=\frac{M_{\rm Pl}^2}{2}\int d^4x\sqrt{-g}\left[
F(Q)R
-2\beta\left(\nabla_aQ\nabla^aQ+L_Q^{-2}Q^2\right)
-K^{ab}{}_{cd}\nabla_a u^c\nabla_bu^d
+a_Q^2\mathcal H_s(Y)
+\lambda(u^au_a+1)
\right]+S_m[g,\psi_m],
$$

$$
K^{ab}{}_{cd}=c_1g^{ab}g_{cd}+c_2\delta^a_c\delta^b_d
+c_3\delta^a_d\delta^b_c-c_4u^au^bg_{cd},
$$

$$
\mathcal F_s(Y)=\sqrt{Y(1+Y)}-\operatorname{asinh}\sqrt Y,
\qquad
\mathcal H_s(Y)=2\left[Y-\mathcal F_s(Y)\right].
$$

There are exactly five global physical parameters:
$\{\beta,L_Q,\eta,c_1,c_{14}\}$. The remaining Aether coefficients are fixed
by

$$
c_3=-c_1,
\qquad
c_2=\frac{c_{14}}{1-2c_{14}},
\qquad
c_4=c_{14}-c_1.
$$

Thus $c_{13}=0$ identically and the high-field preferred-frame combination
$\alpha_2$ vanishes. There are no per-object, object-class, or lensing-only
parameters. The standard $a_0=1.2\times10^{-10}\,{\rm m\,s^{-2}}$ is retained
from the frozen galaxy control.

The one-parameter positive environment response is fixed without a new fit.
It is required to reproduce the midpoint of the already frozen H7s target,
$a_Q(\chi_t)=a_0\sqrt F$, giving

$$
\eta=\frac{\ln\sqrt F}{\chi_t}
=5.11204\times10^5,
$$

with $F=100$ and $\chi_t=4.50424\times10^{-6}$. This does not widen $F$.

## Euler--Lagrange equations

Let

$$
J^a{}_b=K^{ac}{}_{bd}\nabla_cu^d,
\qquad
\mathcal B(Y)=\mathcal H_s-Y\mathcal H_{s,Y}.
$$

Variation with respect to $\lambda$ gives the exact unit constraint

$$
u^au_a=-1.
$$

Variation with respect to the Aether gives

$$
\nabla_aJ^a{}_b
+c_4a_c\nabla_bu^c
+\mathcal H_{s,Y}a_c\nabla_bu^c
-\nabla_a\left(\mathcal H_{s,Y}u^aa_b\right)
+\lambda u_b=0.
$$

Contracting this equation with $u^b$ fixes $\lambda$; it does not add a
propagating degree of freedom.

Variation with respect to $Q$ gives

$$
2\beta R+4\beta(\Box-L_Q^{-2})Q
+2\eta a_Q^2\mathcal B(Y)=0,
$$

or

$$
(\Box-L_Q^{-2})Q
=-\frac{R}{2}
-\frac{\eta a_Q^2}{2\beta}\mathcal B(Y).
\tag{1}
$$

The last term is required by reciprocity. Deleting it while retaining
$a_Q(Q)$ in the Aether equation cannot follow from this action.

The metric equation can be written without hiding a second matter metric as

$$
F G_{ab}=M_{\rm Pl}^{-2}T^{(m)}_{ab}
+T^{(Q)}_{ab}+\Theta^{(u)}_{ab}
+\nabla_a\nabla_bF-g_{ab}\Box F,
\tag{2}
$$

$$
T^{(Q)}_{ab}=2\beta\nabla_aQ\nabla_bQ
-\beta g_{ab}\nabla_cQ\nabla^cQ
-\beta L_Q^{-2}Q^2g_{ab},
$$

where the complete Aether Hilbert tensor is fixed, rather than separately
parameterized, by

$$
\Theta^{(u)}_{ab}=-\frac{1}{\sqrt{-g}}
\frac{\delta}{\delta g^{ab}}
\int d^4x\sqrt{-g}\left[
-K^{cd}{}_{ef}\nabla_cu^e\nabla_du^f
+a_Q^2\mathcal H_s(Y)+\lambda(u^2+1)
\right].
$$

Equations (1), (2), the Aether equation, and the constraint are all variations
of the same action. No dynamics/lensing split has been introduced.

## Conservation check

For Euler derivatives $E_{ab}$, $E_Q$, $E^{(u)}_a$, and $E_\lambda$, an
infinitesimal diffeomorphism gives the off-shell Noether identity

$$
2\nabla^aE_{ab}
+E_Q\nabla_bQ
+E_\lambda\nabla_b\lambda
+E^{(u)}_a\nabla_bu^a
+\nabla_a\left(E^{(u)}_bu^a\right)=0.
$$

Consequently $\nabla^aE_{ab}=0$ when the $Q$, Aether, and constraint equations
hold. Minimal matter coupling independently gives
$\nabla^aT^{(m)}_{ab}=0$ on the matter equations. This verifies conservation of
the total metric source. It also shows why a one-way, externally assigned
$Q$ would not be an allowed repair.

## Quasistatic and mode checks

For

$$
ds^2=-(1+2\Phi)dt^2+(1-2\Psi)d\mathbf x^2,
$$

the unit constraint gives $u^0=1-\Phi+O(\Phi^2)$ and
$a_i=\partial_i\Phi$. The nonlinear function obeys

$$
\mathcal H_{s,Y}=2(1-\mu_s),
\qquad
\mu_s(x)=\frac{x}{\sqrt{1+x^2}},
$$

so the leading static metric/Aether equations have the required form

$$
\nabla\!\cdot\!\left[
\mu_s(|\nabla\Phi|/a_Q)\nabla\Phi
\right]=4\pi G_N\rho_b+O(\beta,c_i).
$$

At $g_b/a_Q=10^{-3}$ the positive spherical solution differs from
$\sqrt{a_Qg_b}$ by $2.50\times10^{-4}$, below the 5% gate. At
$g_b/a_Q=10^5$ its fractional correction is $5.00\times10^{-11}$, below
$10^{-5}$. Both $\mu_s$ and $d(\mu_sg)/dg$ are positive.

For the declared high-field Aether subspace,

$$
c_T^2=1,
\qquad
c_V^2=\frac{c_1}{c_{14}}\ge1,
\qquad
c_S^2=\frac{c_2(2-c_{14})}{c_{14}(2+3c_2)}=1.
$$

The kinetic signs are positive for $0<c_{14}\le c_1$. The scalar-curvature
sector is ghost-free for $\beta>0$ and $F(Q)>0$. Its massless PPN shift is

$$
\gamma-1=-\frac{4\beta^2}{2\beta F+8\beta^2}.
$$

The frozen $|\gamma-1|\le2.3\times10^{-5}$ gate therefore requires, at the
local background $Q\simeq0$,

$$
\beta\le1.15005\times10^{-5}.
$$

## The reciprocal-source test

Without the Aether term, equation (1) has the desired quasistatic limit

$$
(\nabla^2-L_Q^{-2})Q=-\frac{4\pi G\rho_b}{c^2},
$$

up to $O(\beta)$ metric backreaction. A point source differs from its massless
$1/r$ potential by at most 5% only when

$$
L_Q/r\ge[-\ln(0.95)]^{-1}=19.496.
$$

The numerical test makes every choice in the direction that minimizes the
second source in (1):

1. set $\beta$ to its largest PPN-allowed value;
2. set $L_Q/r$ to its shortest 5%-allowed value separately at every point;
3. continue only the positive baryonic mass already enclosed at the measured
   radius as a point source; and
4. omit all exterior baryons.

Across the observed support, $a_Q^2\mathcal B(Y)$ increases with both enclosed
$g_b$ and $a_Q$. The constructed exterior source is therefore a lower bound for
a positive full baryon profile.

The resulting lower-bound fractional changes are

| Domain | Points | Minimum | Median | Maximum | Points passing 5% |
|---|---:|---:|---:|---:|---:|
| SPARC | 3,034 | $2.0\times10^{-15}$ | 1,089 | 10,659 | 8 |
| CLASH | 84 | 42.7 | 21,298 | 410,967 | 0 |
| SPIDERS--MaNGA BCG | 34 | 216 | 3,156 | 5,211 | 0 |

The largest coupling allowed by the 5% gate for *all* 34 BCGs is
$\eta=6.05$. It changes $a_Q$ at $\chi_t$ by only 0.0027%, whereas the frozen
bridge requires a factor of 10 there. The required coupling is 84,469 times
larger.

Increasing $\beta$ cannot repair this while keeping the physical metric viable.
Even the easiest BCG requires $\beta\ge0.0497$; making all BCGs pass requires
$\beta\ge1.20$, for which the predicted $\gamma-1=-0.414$.

## Decision

EA-Q0 passes parameter counting, universal metric, unit constraint,
conservation, high-field mode, and quasistatic constitutive checks. It fails the
environment-field gate by orders of magnitude once the reciprocal source
required by its own action is retained.

The local scalar-curvature EA-Q0 coupling is therefore retired before Stage 3.
No Stage 3 refit or Stage 4 replay is frozen. Dropping the reciprocal source,
widening $F$, or adding a screening/interpolation parameter would define a new
cycle and violate the checkpoint. The next declared control is the
environmental MOG family.

Machine-readable evidence is in `results/eaq0_derivation/report.json`; the
point audit is in `results/eaq0_derivation/feedback_points.csv`.
