# EMOG-Q0 environmental scalar--tensor--vector derivation

Status: **retired at the pre-fit checkpoint**. The action is internally usable in
its regular field domain, but its spherical force shape and environmental ordering
miss the frozen target by far more than 5%. No astrophysical parameter fit was
performed and no Stage 3 or Stage 4 configuration was frozen.

## Prior-art boundary

Scalar--tensor--vector gravity, a massive vector, and a repulsive Yukawa term are
not new here. Moffat's original STVG action made $G$, the vector coupling, and the
vector mass dynamical and derived the familiar weak-field form; it also stated that
the particular source strength used in that phenomenology was not then derivable
from the action ([Moffat 2005](https://arxiv.org/abs/gr-qc/0506021)). A later MOG
paper introduced density-dependent scalar screening for Solar-System and lensing
applications ([Moffat 2014](https://arxiv.org/abs/1410.2464)).

EMOG-Q0 is a deliberately smaller control. It asks whether one canonical
chameleon scalar and one positive-energy Proca vector can make the MOG mechanism
predictive with a conserved universal matter charge, one vector range, one physical
metric, and no object-class or lensing-only rule. We do not claim the field content,
screening mechanism, or Yukawa law as an original theory.

## Frozen action

Use signature $(-,+,+,+)$ and $c=\hbar=1$. The one metric $g_{ab}$ is the metric
used by ordinary matter and light. Define

$$
B_{ab}=2\nabla_{[a}\phi_{b]},\qquad
F(s)=e^{-2\beta s},\qquad
U(s)=\Lambda_s^2s^{-n},\quad s>0.
$$

The action is

$$
\begin{aligned}
S={}&\int d^4x\sqrt{-g}\left[
 {M_{\rm Pl}^2\over2}F(s)R
 -{M_{\rm Pl}^2\over2}(\nabla s)^2
 -M_{\rm Pl}^2U(s)
 -{1\over4}B_{ab}B^{ab}
 -{1\over2}\mu^2\phi_a\phi^a
 \right]\\
&+S_{m0}[g,\psi]-\int d^4x\sqrt{-g}\,\phi_aJ^a,
\end{aligned}
$$

where $M_{\rm Pl}^{-2}=8\pi G_N$ and

$$
J^a=\kappa j_m^a,\qquad \nabla_a j_m^a=0,\qquad
\kappa=\sqrt{4\pi G_N\alpha}.
$$

The overall charge sign is chosen so equal matter charges repel. The charge-to-mass
ratio is the same for every material body. The five and only five new global
physical parameters are

$$
\{\beta,\Lambda_s,n,\mu,\alpha\}.
$$

$s,F,\beta,n,$ and $\alpha$ are dimensionless; $\Lambda_s$ and $\mu$ have mass
(inverse-length) dimension one; $\phi_a$ has mass dimension one; and $\kappa$ has
mass dimension minus one. Every term inside the action density has mass dimension
four. There is no per-object, object-class, or lensing-only parameter.

## Equations from variation

Define the scalar and Proca stress tensors

$$
T^{(s)}_{ab}=M_{\rm Pl}^2\left[
 \nabla_as\nabla_bs-{1\over2}g_{ab}(\nabla s)^2-g_{ab}U
\right],
$$

$$
T^{(\phi)}_{ab}=B_{ac}B_b{}^c-{1\over4}g_{ab}B_{cd}B^{cd}
 +\mu^2\left(\phi_a\phi_b-{1\over2}g_{ab}\phi_c\phi^c\right).
$$

The metric equation is

$$
M_{\rm Pl}^2\left[
 FG_{ab}+(g_{ab}\Box-\nabla_a\nabla_b)F
\right]
=T^{(m)}_{ab}+T^{(\phi)}_{ab}+T^{(s)}_{ab},
$$

where $T^{(m)}_{ab}$ is the exact variational stress tensor of the ordinary-matter
and current-coupling sector. The other Euler--Lagrange equations are

$$
\boxed{\ \Box s-U_{,s}+{1\over2}F_{,s}R=0\ },
$$

$$
\boxed{\ \nabla_bB^{ba}-\mu^2\phi^a=J^a\ },
$$

and the matter equations derived from the same $S_{m0}-\phi\cdot J$ action. For a
point body these have the universal Lorentz-force form

$$
u^b\nabla_bu^a=\kappa B^a{}_bu^b,
$$

up to the already-fixed common charge convention. Taking the divergence of the
Proca equation and using $\nabla_aJ^a=0$ gives the nonpropagating constraint

$$
\nabla_a\phi^a=0\qquad(\mu\ne0).
$$

Thus the vector has the expected three physical polarizations rather than a fourth
ghost degree of freedom.

## Exact conservation identity

Let

$$
E_g^{ab}={2\over\sqrt{-g}}{\delta S\over\delta g_{ab}},\quad
E_s={1\over\sqrt{-g}}{\delta S\over\delta s},\quad
E_\phi^a={1\over\sqrt{-g}}{\delta S\over\delta\phi_a},
$$

and denote all matter Euler expressions collectively by $E_\psi$. Diffeomorphism
invariance, with no field equation imposed, gives

$$
-\nabla_aE_g{}^a{}_b
+E_s\nabla_bs
+E_\phi^aB_{ba}
-\phi_b\nabla_aE_\phi^a
+{\cal I}_b(E_\psi)=0.
$$

${\cal I}_b(E_\psi)$ is the standard Lie-derivative combination for the matter
fields. On the scalar, Proca, current, and matter equations this reduces exactly to
$\nabla_aE_g{}^a{}_b=0$. The same cancellation can be seen sector by sector:

$$
\nabla^aT^{(m)}_{ab}=B_{ba}J^a,
\qquad
\nabla^aT^{(\phi)}_{ab}=-B_{ba}J^a,
$$

while

$$
\nabla^aT^{(s)}_{ab}
=M_{\rm Pl}^2(\Box s-U_{,s})\nabla_bs
=-{M_{\rm Pl}^2\over2}RF_{,s}\nabla_bs.
$$

The last expression is exactly the divergence of the nonminimal $F(s)R$ terms on
the left of the metric equation. There is therefore no one-way external environment
field and no deleted reciprocal source. In contrast to EA-Q0, the Proca mass and
charge are constants, so varying $s$ creates no hidden vector-sector scalar source.

## Health and local limits

For $F>0$, transforming the gravitational and scalar sector to Einstein form gives
the dimensionless scalar kinetic coefficient

$$
K_E={1\over F}+{3\over2}\left({F_{,s}\over F}\right)^2
={1\over F}+6\beta^2>0.
$$

Also

$$
U_{,ss}=n(n+1)\Lambda_s^2s^{-n-2}>0
$$

for $n,\Lambda_s,s>0$. The canonical Maxwell--Proca signs and $\mu^2>0$ give a
positive local Hamiltonian. The tensor, scalar, and Proca principal characteristics
all have speed $c$ on a regular background; in particular $c_T=c$. This proves the
local linear kinetic, gradient, and causal checks. It is not a proof of global
nonlinear well-posedness across a hypothetical $F\rightarrow0$ boundary, which the
frozen field domain excludes.

In an unscreened background the Einstein-frame scalar coupling is

$$
\alpha_s^2={\beta^2\over F^{-1}+6\beta^2},\qquad
\gamma-1=-{2\alpha_s^2\over1+\alpha_s^2}.
$$

The frozen $|\gamma-1|\le2.3\times10^{-5}$ gate gives
$|\beta|\le0.0033913$ near $F=1$ unless the chameleon is screened. The inverse-power
potential can make the local scalar massive at high density, but EMOG-Q0 fails the
astrophysical structural gate before any thin-shell parameter solution can be
promoted.

There is also a vector constraint that scalar thin-shell screening cannot remove.
Any universal range capable of changing kpc dynamics has $\mu r_{\rm AU}\ll1$.
Writing the locally constant metric enhancement as $E=F^{-1}$, nonrelativistic
massive bodies measure $E-\alpha$ because of vector repulsion, whereas light
measures $E$. After calibrating the Cavendish/planetary value to $E-\alpha=1$,
the light-to-dynamics comparison has

$$
\gamma_{\rm eff}-1={2\alpha\over E-\alpha}=2\alpha.
$$

Thus the same $2.3\times10^{-5}$ gate requires
$\alpha\le1.15\times10^{-5}$. Choosing $E=1+\alpha$ can cancel the vector in
planetary dynamics but cannot hide the enhanced metric from photons. Choosing
$E\simeq1$ fixes Solar-System lensing but leaves an uncancelled repulsive force.
This is an action-level conflict between a universal long vector range and local
light/dynamics consistency.

## Environmental response

For nonrelativistic density $\rho$ and slowly varying fields,
$R\simeq\rho/(M_{\rm Pl}^2F)$. The scalar equation then has the adiabatic minimum

$$
s_{\min}(\rho)=
\left({n\Lambda_s^2M_{\rm Pl}^2\over\beta\rho}\right)^{1/(n+1)}.
$$

Consequently

$$
{1\over F}=\exp(2\beta s_{\min})
$$

must increase as baryonic density decreases for $\beta,n>0$. This ordering is a
prediction, not an interpolation choice. The numerical envelope grants the scalar
instantaneous tracking of this minimum at every radius, which is more responsive
than any finite-Compton-wavelength solution.

There is a second structural constraint. The metric attraction is $1/F(s)$, while
the repulsive-vector amplitude $\alpha$ is fixed by the one conserved universal
current. Short-distance Newtonian cancellation requires $1/F=1+\alpha$. A varying
$s$ can satisfy that equality at one background value only. Making $\alpha$ depend
on $s$ would destroy the frozen conserved-current construction and introduce the
very reciprocal matter/scalar source and extra coupling this checkpoint forbids.

## Spherical weak-field solution and lensing

For a constant background $s_0$ and a point mass $M$, vector stress is second order
in the weak field, so

$$
\Phi=\Psi=-{G_NM\over F_0r}.
$$

The same two potentials determine the Weyl potential for photons. Thus

$$
\widehat\alpha_{\rm light}(b)={4G_NM\over F_0bc^2},
$$

with no lensing normalization. The Proca equation gives

$$
\phi_0={\kappa M\over4\pi r}e^{-\mu r}.
$$

Massive matter feels metric attraction and vector repulsion:

$$
g_{\rm dyn}(r)={G_NM\over r^2}
\left[{1\over F_0}-\alpha(1+\mu r)e^{-\mu r}\right].
$$

When the one-background matching condition $F_0^{-1}=1+\alpha$ is imposed,

$$
{g_{\rm dyn}\over g_{\rm bar}}
=1+\alpha\left[1-(1+x)e^{-x}\right],\qquad x=\mu r.
$$

The short- and long-distance limits are

$$
{g_{\rm dyn}\over g_{\rm bar}}
=1+{\alpha x^2\over2}+O(x^3),\qquad
{g_{\rm dyn}\over g_{\rm bar}}\rightarrow1+\alpha.
$$

So Newtonian cancellation and large-radius attraction both work in that one matched
environment. However, the extra point-mass acceleration is constant at small $r$
and proportional to $r^{-2}$ at large $r$; it has an $r^{-1}$ slope only at
$\mu r=1.79328$. Even the loose slope interval $-1\pm0.05$ spans only
$1.68209<\mu r<1.90827$, a radial factor 1.134 or **0.0548 dex**. A single range
therefore cannot supply an approximately $1/r$ extra acceleration across extended
galaxy and cluster radial baselines.

The implementation also includes the exact spherical thin-shell Proca kernel, not
only the point-source expression. For a shell $dM$ at $r'$ its angle-averaged Yukawa
Green function is

$$
{e^{-\mu|\boldsymbol r-\boldsymbol r'|}\over
 |\boldsymbol r-\boldsymbol r'|}\bigg|_{\Omega'}
=\mu i_0(\mu r_<)k_0(\mu r_>)
={\sinh(\mu r_<)e^{-\mu r_>}\over\mu rr'}.
$$

Differentiating this expression gives both the exterior outward force and the
interior-shell inward Yukawa force used by `spherical_vector_acceleration_m_s2`.
Its $\mu\rightarrow0$ limit recovers the ordinary shell theorem.

## Frozen feasibility result

The action was tested on all 3,034 SPARC points, 84 CLASH points, and 34 independent
SPIDERS--MaNGA BCG points. The favorable envelope allowed the complete range

$$
10^{-6}\le z_{\rm ref}\le10^3,\quad
0.02\le p={1\over n+1}\le1,
$$

$$
10^{-3}\ {\rm kpc}\le\mu^{-1}\le10^5\ {\rm kpc},\quad
10^{-4}\le\alpha\le10^4,
$$

and repeated the global feasibility search with three deterministic seeds. These
effective values are diagnostics only; none is adopted as a physical fit.

The result fails well before a likelihood comparison:

- No point in any of the three domains is within 5% in the best joint minimax
  envelope.
- The best envelope's maximum fractional error is 0.9959, versus the 0.05 gate.
- More decisively, a CLASH point at lower mean baryonic density requires
  enhancement 4.217, while a higher-density point requires 16.368. Any monotone
  chameleon response must miss at least one of that pair by **59.0%**. This lower
  bound is analytic and independent of the optimizer or the power-law form.
- A constant-scalar MOG control also fails. Its universal Yukawa transition cannot
  reproduce the joint target.
- The favorable envelope uses $\alpha\simeq0.983$, about 85,500 times the maximum
  allowed by the Solar-System light/dynamics comparison for a universal kpc-range
  vector. This optimizer value is not adopted; the ratio records the independent
  incompatibility.

The 5% requirement is intentionally a hard structural target rather than an
uncertainty-weighted goodness of fit. The result does not prove that every possible
scalar--vector theory is false. It rejects this frozen five-parameter environmental
MOG action and, especially, the assumption that a monotone one-scalar environment
response plus one universal Yukawa range can unify these data.

## Decision

EMOG-Q0 is retired. It does not repeat EA-Q0's excessive reciprocal backreaction;
instead it fails because the consistent scalar response has the wrong cross-system
ordering, its changing metric strength cannot remain locked to a conserved
universal vector charge, a long-range vector cannot satisfy Solar-System light and
dynamics simultaneously, and one Yukawa transition resembles $1/r$ over only
0.055 dex. No Stage 3 cross-validation or Stage 4 lensing replay is authorized.

Per the preregistered stopping rule, the next task is a premise-level rethink of
the one-field environmental unification target. Adding another interpolation
function, assigning a range to each object, or introducing a lensing-only factor is
not an allowed response.

Reproduce with:

```powershell
python scripts/check_environmental_mog0.py
python -m pytest -q
```
