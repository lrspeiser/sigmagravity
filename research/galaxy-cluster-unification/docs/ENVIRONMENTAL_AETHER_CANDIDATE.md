# EA-Q0: environmental generalized-Aether candidate

Status: local EA-Q0 cycle completed and retired before any astrophysical fit.
The exact selected action, variations, conservation identity, spherical source
calculation, and failed 5% gate are in `docs/EAQ0_DERIVATION.md`.

## Pre-fit outcome

The minimal five-parameter realization used one universal, minimally coupled
metric, imposed $c_{13}=0$, and made $Q$ a local scalar-curvature field. Its
unit constraint, on-shell conservation identity, high-field mode speeds, and
standard-$\mu$ quasistatic limits pass their declared checks.

It fails for a structural reason. Once $a_Q(Q)$ appears in the Aether action,
variation with respect to $Q$ generates a reciprocal Aether source. Even a
lower-bound spherical calculation changes $Q$ by at least a factor 216 for
every frozen BCG, rather than the allowed 5%. A coupling small enough to keep
all BCGs within 5% changes $a_Q$ at the transition by only 0.0027%, while the
frozen bridge requires a factor of 10. Increasing the scalar coupling enough to
suppress this feedback violates the PPN $\gamma$ gate by orders of magnitude.

The local coupling is therefore retired. Removing the reciprocal source would
break the action derivation; adding a screening/interpolation term would be a
new cycle. The declared next control is environmental MOG.

## Why this candidate advances

The 34 frozen SPIDERS--MaNGA systems pass the Stage 4 scientific thresholds when
the environment variable is the full radial baryonic potential from BCG stars,
hot gas, and satellite stars. The point score is
$\chi^2/N=1.658$, RMS $=0.132$ dex, and mean residual $=-0.083$ dex. All 5,000
uncertainty realizations pass both frozen gates.

This supports one narrow statement: independently reconstructed baryonic
potential is worth retaining as the environmental state variable. It does not
validate H7s. H7s still has $F=100$ on its hard bound in three of five folds, so
the algebraic logistic closure is retired rather than widened.

## Prior-art boundary

A unit timelike dynamical vector, its four quadratic derivative invariants, and
a nonlinear kinetic function capable of MOND-like weak-field behavior are
generalized Einstein-Aether prior art. So are scalar-dependent couplings and a
universal conformal/disformal matter metric. Relevant starting points are
[Zlosnik, Ferreira & Starkman (2007)](https://arxiv.org/abs/astro-ph/0607411),
[Jacobson & Mattingly (2004)](https://arxiv.org/abs/gr-qc/0402005), and
[Foster & Jacobson (2006)](https://arxiv.org/abs/gr-qc/0509083).

The project-specific contribution to test is narrower: use an environment
field whose quasistatic solution is compared directly with the independently
reconstructed *full baryonic potential*, freeze it without BCG residuals, and
ask whether one physical metric can predict SPARC dynamics, BCG dynamics, and
CLASH lensing with no object-class switch or lensing multiplier.

## Minimal covariant envelope

The fields are an Einstein metric $g_{ab}$, a unit timelike Aether $u^a$, an
environment scalar $Q$, a constraint multiplier $\lambda$, and matter fields
$\psi_m$. All matter and light couple to one physical metric $\tilde g_{ab}$:

$$
S_{\rm EAQ}=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R
-\frac{M_A^2}{2}\mathcal F(\mathcal K,Q)
-\frac{M_{\rm Pl}^2Z_Q}{2}\nabla_aQ\nabla^aQ
-V(Q)+\lambda(u^au_a+1)
\right]+S_m[\tilde g_{ab},\psi_m],
$$

$$
\mathcal K=M_A^{-2}K^{ab}{}_{cd}\nabla_a u^c\nabla_bu^d,
$$

$$
K^{ab}{}_{cd}=
c_1g^{ab}g_{cd}+c_2\delta^a_c\delta^b_d
+c_3\delta^a_d\delta^b_c-c_4u^au^bg_{cd}.
$$

The provisional universal metric is

$$
\tilde g_{ab}=C(Q)g_{ab}+D(Q)u_au_b.
$$

This is an envelope, not a license to fit four arbitrary functions. The first
cycle must derive the lowest-order allowed forms of $\mathcal F$, $C$, and $D$
and count every independent coefficient. If the same empirical response can be
obtained with $C=1$ and $D=0$, minimal coupling is preferred.

Varying the envelope gives the required structure

$$
M_{\rm Pl}^2G_{ab}=T^{(m)}_{ab}+T^{(u)}_{ab}+T^{(Q)}_{ab},
$$

$$
\nabla_a\!\left(\mathcal F_{\mathcal K}J^a{}_b\right)
+\mathcal E_b^{(u)}+2\lambda u_b
=-\frac{1}{\sqrt{-g}}\frac{\delta S_m}{\delta u^b},
$$

$$
M_{\rm Pl}^2Z_Q\Box Q-V_Q-
\frac{M_A^2}{2}\mathcal F_Q
=-\frac{1}{\sqrt{-g}}\frac{\delta S_m}{\delta Q}.
$$

The exact stress tensors and Aether current $J^a{}_b$ must be derived from the
chosen functions; they may not be reverse-engineered independently for
dynamics and lensing.

## Quasistatic target, not an inserted answer

The first derivation must determine whether a stable parameter restriction has
the two coupled quasistatic limits

$$
\nabla\!\cdot\!\left[
\mu_A(|\nabla\Psi|,Q)\nabla\Psi
\right]=4\pi G\rho_b,
$$

$$
(\nabla^2-L_Q^{-2})Q=-\frac{4\pi G}{c^2}\rho_b+\text{controlled cosmological terms}.
$$

For $L_Q$ larger than a host, the second equation has the target solution

$$
Q(\mathbf x)\simeq\frac{G}{c^2}
\int d^3x'\frac{\rho_b(\mathbf x')}{|\mathbf x-\mathbf x'|}.
$$

For a spherical profile truncated at $R$, this is exactly the quantity tested
at Stage 4:

$$
Q(r)\simeq\chi_b(r)=\frac{G}{c^2}\left[
\frac{M_b(<r)}{r}+\int_r^R\frac{dM_b(s)}{s}
\right].
$$

This explains why galaxy-pair midpoints and a pointlike $-9.8\,{\rm m\,s^{-2}}$
void source are not used. Voids enter as a lower-density boundary condition or
large-scale perturbation of $Q$; the inward acceleration remains centered on
the measured baryons. The direct fifth force sourced by $Q$ must be negligible
in the Solar System. If obtaining the normalized $Q$ equation makes that
impossible or strongly coupled, EA-Q0 fails before an astrophysical fit.

## Frozen minimal parameter policy

The first complete EA-Q0 weak-field cycle has at most five global physical
parameters, no per-object force parameters, and no lensing-only parameters.

1. Impose $c_{13}=c_1+c_3=0$ before fitting so the tensor mode is luminal.
2. Use the Solar-System-compatible Aether subspace; do not scan all four $c_i$
   independently.
3. Allow at most one environment response coefficient and one range $L_Q$.
4. Use one fixed, action-derived nonlinear kinetic family. Do not add a second
   logistic transition, widen H7s $F$, or import an object-class label.
5. Predict both metric potentials $\Psi$ and $\Phi$ from the same solution.

The GW170817/GRB170817A constraint implies $|c_{13}|\lesssim10^{-15}$ in
Einstein-Aether theory; setting it to zero is therefore a structural condition,
not an astrophysical fit result. Preferred-frame PPN, spin-0/spin-1 speeds,
positive kinetic energy, and strong coupling remain independent failure modes.

## Next-stage outcomes and stop rules

| Checkpoint | Concrete outcome required | Failure action |
|---|---|---|
| Covariant variation | Symbolic field equations reproduce the unit constraint and have identically conserved total stress energy | reject the chosen coupling; do not fit it |
| Environment limit | Spherical numerical solutions recover the independently integrated $\chi_b(r)$ to 5% over the observed SPARC--BCG--CLASH support using no host normalization | reject EA-Q0 sourcing or change action family |
| Weak-field limits | deep response within 5% of $\sqrt{a_Qg_b}$ and high-field fractional correction below $10^{-5}$ | reject the kinetic function |
| Mathematical health | positive kinetic eigenvalues, positive squared scalar/vector mode speeds, and no singularity on the measured support | reject EA-Q0 |
| Stage 3 rerun | five folds; SPARC $\chi^2/N\le9.418$, raw CLASH $\le5.0$, macro $\le7.2$, no hard-bound parameter | do not tune Stage 4; move to the environmental MOG control |
| Frozen Stage 4 replay | all 34 hosts; $\chi^2/N\le3.0$, RMS $\le0.17$ dex, $|\bar\Delta|\le0.10$ dex | reject unification even if SPARC/CLASH pass |
| Relativistic metric | same constants predict dynamics and lensing, $|\gamma-1|\le2.3\times10^{-5}$ locally, and $|c_T/c-1|\le10^{-15}$ | reject relativistic completion |

The derivation checkpoint comes first. It gets one complete weak-field cycle and
seven active research days. Lack of progress toward a conserved, stable $Q$
source is a reason to abandon this coupling, not to fit the old H7s formula
again.

## What Stage 4 did not establish

- The result is profile-constrained, not 34 directly observed X-ray profiles.
  Only 10 frozen hosts have direct eRASS or pointed Chandra/XMM coverage.
- The point construction was inspected during development, so it is
  confirmatory against pre-existing thresholds but is not a new blind test.
- The 23 DynPop/NSA points are calibrated proxies; the 11 direct Tian points
  separately pass the continue gate but retain a small negative bias.
- No cosmological perturbation, CMB, structure-growth, Bullet-Cluster, binary-
  pulsar, or compact-object calculation has yet been passed.

Those limitations are carried forward as requirements rather than hidden in a
new parameter.
