# Sigma v13C minimal khronon completion falsification

## Decision

The minimal one-metric khronon placement of the v13B Legendre carrier is
rejected before observational data. It preserves the intended static AQUAL
response and does not change the local transverse-traceless tensor kinetic
term, but it makes the shift-reduced scalar gravitational kinetic energy
negative throughout the required high-acceleration regime.

The failure is analytic and is not repaired by changing the positive carrier
weight. A constant trace counterterm can avoid the ghost interval only by
remaining order unity or far larger at infinite acceleration, which abandons
the GR high-field limit.

This is the third materially distinct post-v12 formulation to fail the
bounded-Hamiltonian gate. The project's three-failure stopping rule is
therefore triggered. The preferred-foliation clock/ADM-trace mechanism must be
reset rather than given another coefficient or interpolation function.

The standalone v13B theorem remains true: its isolated phase-space Hamiltonian
is convex. What fails is the minimal identification of its temporal Legendre
variable with the trace expansion of the gravitational foliation.

## Prior-art boundary

The static half of this construction is established literature, not a Sigma
Gravity invention. Blanchet and Marsat introduced a generally covariant
khronon whose normalized gradient defines a preferred foliation and whose
acceleration enters the action through a function $f(a)$. Their weak-field
limit gives a MOND-like modified Poisson equation, equal metric potentials,
and hence the same single modified potential for massive-particle dynamics
and light deflection:

- [Blanchet and Marsat, *Modified gravity approach based on a preferred time
  foliation*](https://arxiv.org/abs/1107.5264);
- [Blanchet and Marsat, *Relativistic MOND theory based on the Khronon scalar
  field*](https://arxiv.org/abs/1205.0400); and
- [Bonetti and Barausse, *Post-Newtonian constraints on Lorentz-violating
  gravity theories with a MOND phenomenology*](https://arxiv.org/abs/1502.05554).

The only candidate distinction tested here is to use the v13B positive
Hamiltonian/Legendre pair as a specific temporal completion of that known
static acceleration action. No novelty claim is made.

## Candidate covariant action

Let the khronon $T$ define the unit normal

$$
u_\mu=-{\nabla_\mu T\over
\sqrt{-g^{\alpha\beta}\nabla_\alpha T\nabla_\beta T}},
$$

and define its expansion and acceleration by

$$
\Theta=\nabla_\mu u^\mu,
\qquad
a_\mu=u^\nu\nabla_\nu u_\mu,
\qquad
a=\sqrt{a_\mu a^\mu}.
$$

Introduce an algebraic auxiliary momentum $p$ and the v13B Hamiltonian

$$
\mathcal H_{13B}(p,a)
={a_\Sigma^2\over2}F_\epsilon
\left({p^2+a^2\over a_\Sigma^2}\right).
$$

Its covariant first-order Legendre form is

$$
\mathcal L_C(p,\Theta,a)
=p\Theta-\mathcal H_{13B}(p,a).
$$

The algebraic $p$ equation is

$$
\Theta={\partial\mathcal H_{13B}\over\partial p},
$$

which has one solution everywhere because v13B is strictly convex. The
minimal same-weight completion tested here is

$$
\boxed{
S_{13C}={M_{\rm Pl}^2\over2}\int d^4x\sqrt{-g}
\left[
R+w\left(\mathcal L_C-\mathcal L_0\right)
-c_{\rm tr}\Theta^2
\right]+S_m[g,\psi]
}
$$

with

$$
\mathcal L_0={1\over2}(\Theta^2-a^2).
$$

The frozen minimal row has $w=1$ and $c_{\rm tr}=0$. The weight is fixed to
one by the demand that the GR reference plus the static modifier equal the
v13B AQUAL energy. Matter is minimally coupled to the single metric
$g_{\mu\nu}$, so massive particles and photons follow the same physical
geometry.

This action is invariant under spacetime diffeomorphisms and monotonic
reparameterizations $T\mapsto f(T)$. The auxiliary $p$ has no independent
initial data: it is fixed algebraically by $\Theta$ and $a$. Those facts are
necessary, but the kinetic calculation below shows they are not sufficient.

## Static limit and relationship to the published khronon action

On a static foliation, $\Theta=0$ and strict convexity sets $p=0$. Writing
$x=a/a_\Sigma$, the required published-action function is

$$
{f(a)\over a_\Sigma^2}
=F_\epsilon(x)-x^2
=-2(1-\epsilon)\left[x-\ln(1+x)\right].
$$

Its susceptibility is

$$
\chi(x)={f'(a)\over2a}
=-{1-\epsilon\over1+x}.
$$

Therefore

$$
1+\chi
=\epsilon+(1-\epsilon){x\over1+x}
=\mu_\epsilon(x),
$$

and the known weak-field khronon result becomes

$$
\boldsymbol\nabla\cdot
\left[\mu_\epsilon(|\boldsymbol\nabla\Phi|/a_\Sigma)
\boldsymbol\nabla\Phi\right]
=4\pi G\rho_b.
$$

The off-diagonal weak-field equation gives $\Psi=\Phi$, so both stellar
dynamics and lensing use this same modified potential. The frozen numerical
derivative of $f$ reproduces $\mu-1$ to 1.22e-8 absolute error.

This static success is why the candidate is worth testing. It is also why the
result must not be described as a new lensing mechanism: that property is the
published khronon construction.

## Where the ghost enters

At a static background $p=0$, the momentum direction is transverse to the
spatial acceleration in the radial v13B phase space. Its Hamiltonian curvature
is therefore $\mathcal H_{pp}=\mu(x)$. Legendre duality gives

$$
{\partial^2\mathcal L_C\over\partial\Theta^2}
={1\over\mathcal H_{pp}}
={1\over\mu(x)}.
$$

Relative to the canonical reference, the added trace-kinetic curvature is

$$
\boxed{
\delta(x)
=w\left({1\over\mu(x)}-1\right)
=w{1-\epsilon\over\epsilon+x}>0.
}
$$

In coordinates adapted to the foliation, $\Theta=K$, the trace of the ADM
extrinsic curvature. The quadratic kinetic action therefore has the form

$$
\mathcal L_{\rm kin}
=K_{ij}K^{ij}-\lambda_{\rm eff}K^2,
\qquad
\lambda_{\rm eff}
=1+c_{\rm tr}-{\delta\over2}.
$$

Now use a scalar perturbation $h_{ij}=(1+2\zeta)\delta_{ij}$ and scalar shift
$N_i=\partial_i B$. With $q=k^2B$, the complete time-derivative/shift block is

$$
\mathcal L_{\rm kin}^{(2)}
=(3-9\lambda)\dot\zeta^2
+(2-6\lambda)\dot\zeta q
+(1-\lambda)q^2.
$$

The shift is nondynamical. Eliminating it gives

$$
\boxed{
\mathcal K_\zeta
=2{1-3\lambda\over1-\lambda}.
}
$$

Thus

$$
{1\over3}<\lambda<1
\quad\Longrightarrow\quad
\mathcal K_\zeta<0.
$$

This is a negative kinetic-energy scalar gravitational mode. It is not a
choice of gauge: the shift has already been solved and substituted. The direct
Schur complement and the closed expression agree to 3.56e-15.

## Frozen selected result

For $\epsilon=10^{-6}$, $w=1$, and $c_{\rm tr}=0$, the ghost begins when

$$
\delta<{4\over3}
\quad\Longleftrightarrow\quad
{a\over a_\Sigma}>0.74999825.
$$

At the project's Solar/high-field screening sentinel,
$a/a_\Sigma=10^5$:

| Quantity | Result |
|---|---:|
| Static $\mu$ | 0.9999900001099988 |
| Fractional force correction $1/\mu-1$ | 9.9999900e-6 |
| Temporal excess $\delta$ | 9.9999900e-6 |
| $\lambda_{\rm eff}$ | 0.9999950000050001 |
| Shift-reduced scalar kinetic coefficient | -799994.8000134 |

This makes the physical conflict especially clear. The static force is just
small enough to satisfy the declared $10^{-5}$ high-field force criterion,
but the temporal completion is deeply on the wrong-sign kinetic branch.

The deep-field endpoint lies on the other healthy-sign branch:
$\delta(0)=999999$, $\lambda\ll1/3$. A continuous theory connecting that
endpoint to the GR limit must cross the ghost interval.

## Parameter and counterterm closure

### Changing the completion weight

For every finite $w>0$,

$$
\delta(x)=w{1-\epsilon\over\epsilon+x}
\longrightarrow0^+
\quad\text{as}\quad x\longrightarrow\infty.
$$

Hence $\lambda\to1^-$ when $c_{\rm tr}=0$, and every positive weight has a
high-field ghost. The frozen scan verified witnesses for

$$
w=10^{-6},10^{-3},1,100,10^6.
$$

Changing $w$ only moves the onset

$$
x_{\rm ghost}={3w(1-\epsilon)\over4}-\epsilon.
$$

It never removes the asymptotic problem.

### Adding a constant trace term

Let

$$
\delta_{\max}=w{1-\epsilon\over\epsilon}.
$$

Because $\lambda(x)$ is continuous and increases from
$1+c_{\rm tr}-\delta_{\max}/2$ to $1+c_{\rm tr}$, it avoids
$1/3<\lambda<1$ only in either of the domains

$$
c_{\rm tr}\le-{2\over3}
\qquad\text{or}\qquad
c_{\rm tr}\ge{\delta_{\max}\over2}.
$$

For the selected row, the positive boundary is

$$
{\delta_{\max}\over2}=499999.5.
$$

The negative domain leaves $\lambda_\infty\le1/3$; the positive domain leaves
$\lambda_\infty\ge500000.5$. Neither approaches the GR value
$\lambda_\infty=1$. Every counterterm small enough to preserve the high-field
limit lies between these domains and therefore crosses the ghost interval.

The frozen signed counterterm scan confirms this topology from -1000000
through +1000000. No constant succeeds at both requirements.

### Tensor cone

The modifier's kinetic Hessian is proportional to $h^{ij}h^{kl}$ because it
depends only on $K$. Its contraction with either transverse-traceless tensor
polarization vanishes exactly. The local quadratic tensor kinetic and $c_T$
therefore remain those of Einstein-Hilbert gravity. This pass does not rescue
the negative scalar kinetic term.

## Failure accounting

- Formulation rejected: v13C minimal one-metric khronon trace completion.
- Primary failed gate: positive reduced scalar kinetic / bounded physical
  Hamiltonian.
- Post-v12 materially distinct formulation failures: 3.
- Same bounded-Hamiltonian gate failures: 3.
- Three-failure mechanism reset: **triggered**.
- Observational data opened: no.
- Raw holdout opened: no.
- Theory viable: no.

The three failures are materially different:

1. v12A: constraint-solved AeST/DHOST modes retained a negative-energy branch;
2. v13A: an exact clock multiplier generated a free signed dust charge; and
3. v13C: the convex Legendre carrier placed in the gravitational ADM trace
   generated a negative shift-reduced scalar kinetic term.

The common lesson is not that a positive isolated Hamiltonian is impossible.
It is that repeatedly using a preferred gravitational clock or trace as the
carrier reintroduces an unhealthy physical mode once the metric constraints
are included.

## Required mechanism reset

Do not add another clock coefficient, acceleration-dependent trace
counterterm, or interpolation gate. The next cycle must return to the physical
postulates and place any new carrier outside the gravitational conformal/ADM
trace direction while retaining one physical metric and a baryon-determined
source.

A possible new question is whether an independently constrained spatial
constitutive field can change the metric's elliptic response without acquiring
an object-level charge or a propagating ghost. That is a new mechanism and
must begin with a fresh action and degree count, not with observational fitting.

## Reproduction

Run:

    python scripts/audit_sigma_v13c_khronon_completion.py
    python -m pytest -q tests/test_sigma_v13c_khronon_completion.py

Machine-readable evidence is in
results/sigma_v13c_khronon_completion/report.json.
