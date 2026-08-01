# Action-motivated Sigma gravity exploration

## Outcome

The most useful next candidate is a compact Sigma-refracted AQUAL equation.  It
keeps the finite-void screening field from the first exploration, reduces
exactly to Newtonian gravity when $\Sigma=0$, and derives its modified Poisson
law from an explicit AQUAL free function.  On the same idealized galaxy and RX
J2129 transfer test, its best universal row has a 0.2304 dex joint descriptive
score and an almost perfectly flat far-galaxy rotation curve.

The earlier bounded-permittivity model scored slightly closer at 0.2132 dex but
returned to a Keplerian far slope of $-0.481$.  The action-motivated candidate's
far slope is $-0.011$.  Thus this iteration materially improves the radial
physics at a cost of 0.017 dex in the two-object descriptive score.

Raw cluster lensing remains the unresolved part.  The universal joint row gives
3.853 arcsec heldout radial RMS.  A cluster-target-only row improves that to
2.279 arcsec, but its galaxy error rises to 0.463 dex.  This localizes the
remaining problem to the universal galaxy/cluster coupling rather than the
ability of the radial cluster profile to generate multiple-image roots.

## What is inherited rather than invented

The environmental equation

$$
L_\Sigma^2\nabla^2\Sigma=
\left(\frac{\rho_b}{\rho_s}-1\right)\Sigma+\Sigma^3
$$

is the normalized static equation of the symmetron screening mechanism: matter
restores the $\Sigma\rightarrow-\Sigma$ symmetry in dense regions, while a
low-density region develops a nonzero field.  We should therefore call this a
symmetron-like Sigma field, not claim the field equation as a new invention.

The nonlinear modified Poisson structure is AQUAL.  Bekenstein's relativistic
tensor-vector-scalar construction demonstrates one established route by which
the potential governing low-acceleration dynamics can also govern lensing.  Our
specific Sigma gating is exploratory, and a dedicated prior-art audit would be
required before making any novelty claim about that combination.

Primary references:

- [Hinterbichler and Khoury, symmetron screening](https://arxiv.org/abs/1001.4525)
- [Burrage et al., symmetron galactic dynamics and lensing boundary](https://arxiv.org/abs/1811.12301)
- [Bekenstein, relativistic tensor-vector-scalar gravity](https://arxiv.org/abs/astro-ph/0403694)

## Design rule

Only force laws with a compact action and a recognized limit were admitted.
The Sigma field was first solved from the baryonic density and held fixed as a
leading weak-backreaction environment.  Each force law then used one coupling
parameter in addition to the common $\rho_s$ and $L_\Sigma$.

This is not yet a fully coupled action: varying the complete action with respect
to $\Sigma$ will add a gravitational backreaction term to its field equation.
That derivation is deliberately the next step rather than another empirical
correction.

## Candidate 1: conformal symmetron

The standard even conformal coupling can be written

$$
A(\Sigma)=\exp\left(\frac{\alpha\Sigma^2}{2}\right).
$$

It gives massive matter the radial acceleration

$$
g=g_N+\alpha c^2\Sigma\frac{d\Sigma}{dr}.
$$

This is the force dictated most directly by the screened scalar action.  It is
nonzero only across a Sigma transition.  Its best universal result was:

- $\log_{10}\rho_s=-24.5$, $L_\Sigma=30$ kpc,
  $\log_{10}\alpha=-5$;
- galaxy RMSE 0.2713 dex;
- cluster RMSE 0.7773 dex;
- joint score 0.5822 dex;
- far velocity slope $-0.582$.

The gradient force makes a transition bump rather than a sustained flat curve.
Furthermore, a purely conformal metric transformation does not directly change
null trajectories, so this minimal force cannot by itself provide the required
extra lensing.  This agrees with the published symmetron lensing boundary rather
than constituting a new failure unique to this implementation.

## Candidate 2: Sigma-gated AQUAL

Let $y=g/a_0$.  The minimal screened AQUAL interpolation tested was

$$
\mu(y,\Sigma)=\frac{y}{y+\lambda\Sigma^2},
\qquad \nabla\cdot(\mu\nabla\Phi)=4\pi G\rho_b.
$$

It has transparent limits:

- $\Sigma=0\Rightarrow\mu=1$: Newtonian gravity;
- $\Sigma=1$, $\lambda=1$: the simple AQUAL/MOND interpolation;
- high acceleration: $\mu\rightarrow1$;
- deep low acceleration: $g\simeq\sqrt{g_Na_0\lambda\Sigma^2}$.

Its best universal row used $\log_{10}\rho_s=-23.5$, $L_\Sigma=3$ kpc, and
$\lambda=3$.  It scored 0.1933 dex on the galaxy, 0.2843 dex on the cluster, and
0.2431 dex jointly.  Its velocity slopes were 0.0228 over 10--50 kpc and 0.0020
over 100--250 kpc: flat at both scales.

## Candidate 3: Sigma-refracted AQUAL

The most promising combination retained one term from each successful limit:

$$
\mu(y,\Sigma)=
(1-\eta\Sigma^2)\frac{y}{y+\Sigma^2}.
$$

For spherical systems, $\mu g=g_N$ has the closed solution

$$
g=\frac{g_N+
\sqrt{g_N^2+4(1-\eta\Sigma^2)g_Na_0\Sigma^2}}
{2(1-\eta\Sigma^2)}.
$$

This is not an acceleration formula pasted onto Newtonian gravity.  Defining
$X=|\nabla\Phi|^2/a_0^2$, $s=\Sigma^2$, and
$\epsilon=1-\eta\Sigma^2$, it follows from the static AQUAL free function

$$
\mathcal F(X,\Sigma)=\epsilon\left[
X-2s\sqrt X+2s^2\ln\left(1+\frac{\sqrt X}{s}\right)
\right],
$$

because $\partial\mathcal F/\partial X=\mu$.  The $s\rightarrow0$ limit is
$\mathcal F=X$, recovering the Newtonian action.

The best joint row was

$$
\eta=0.6,\qquad \rho_s=10^{-23.5}\ {\rm g\,cm^{-3}},
\qquad L_\Sigma=3\ {\rm kpc}.
$$

Its results were:

| Quantity | Result |
|---|---:|
| Galaxy RAR RMSE, 5--50 kpc | 0.2375 dex |
| Galaxy velocity slope, 10--50 kpc | -0.0391 |
| Galaxy velocity slope, 100--250 kpc | -0.0110 |
| Gravity enhancement at 20 kpc | 5.76 |
| RX J2129 derived-field RMSE | 0.2230 dex |
| RX J2129 mean residual | -0.2076 dex |
| RX J2129 enhancement at 100 kpc | 4.20 |
| Joint descriptive score | 0.2304 dex |

The nearly zero far slope fixes the first model's structural Keplerian-return
problem.  The cluster amplitude is still low for the setting preferred by the
galaxy.

## Raw lensing diagnostic

The same-potential, no-amplitude-refit lensing propagation gave:

| Selection | Galaxy RMSE | Cluster derived RMSE | Training RMS | Spent-heldout RMS |
|---|---:|---:|---:|---:|
| Universal joint row | 0.2375 dex | 0.2230 dex | 1.030 arcsec | 3.853 arcsec |
| Cluster-target-only row | 0.4627 dex | 0.0699 dex | 0.424 arcsec | 2.279 arcsec |

Both rows recovered roots for all 15 training and seven heldout images.  No
gravity or lensing amplitude was fit to the coordinates.  The second row uses
$\eta=0.8$, $\rho_s=10^{-23.5}$ g cm$^{-3}$, and $L_\Sigma=30$ kpc.  It is a
diagnosis, not a universal solution: its stronger coupling helps the cluster and
lensing but over-amplifies the galaxy.

## Next elegant step

The next calculation should not add another interpolation factor.  It should
vary one complete action with respect to every field.  Two established routes
are worth deriving before choosing:

1. Include the symmetron stress-energy in the Einstein equations.  Its field
   energy then lenses automatically, but the model must acknowledge if that
   energy is functioning as a dark component.
2. Embed the Sigma-gated AQUAL free function in a tensor-vector-scalar physical
   metric so that stellar dynamics and photon deflection are derived together.

Either route must retain three limits: GR/Newton in dense regions,
$R_{\rm critical}=\pi L_\Sigma$ for finite empty regions, and the AQUAL flat
point-mass limit.  A new term is acceptable only if it is required by varying
the covariant action or by stability—not merely because it improves a score.

Before another cluster fit, the universal row should also be transferred to a
small ladder of galaxy masses and sizes.  That is the most direct test of whether
$\rho_s$ and $L_\Sigma$ create a genuine environmental scaling law.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_sigma_action_exploration.py
python -m pytest tests/test_sigma_actions.py tests/test_sigma_action_exploration_results.py -q
```

Machine-readable outputs are under `results/sigma_action_exploration/`.
