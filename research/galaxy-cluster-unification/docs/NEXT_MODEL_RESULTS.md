# Results of the potential, boundary, tide, and void-wall tests

Status: completed staged development analysis, 2026-07-26.

## Verdict

**None of the new tests supports a void-origin explanation of galaxy rotation
curves.** The self-potential screen P0 is a somewhat better phenomenological
predictor than our first acceleration-screened free-$p$ law when whole galaxies
are held out, but it does not beat fixed RAR and its advantage does not reproduce
on the radial holdout. Every proposed causal extension either worsens prediction
or fails its physical scale check.

![Ordinary CF4 tide versus required outer acceleration](../results/cf4_tide_test/tide_vs_required.png)

## Scientific originality conclusion

The literature audit found direct prior art for all broad ingredients:
low-acceleration rotation laws, environmental external-field effects,
potential/density screening, oblate disk screening surfaces, rotation-curve
upturns at screening radii, and voids as unscreened modified-gravity
laboratories. The safe contribution of this project is therefore the particular
matched and preregistered test sequence, not a claim to have invented those
mechanisms.

The most direct overlap is Naik et al.'s chameleon $f(R)$ work, which already
predicts an oblate screening surface and a rotation-curve upturn near the disk
screening radius. Chae et al. already connect SPARC external-field fits to
independently reconstructed large-scale structure. Malandrino et al. supply the
external irregular void boundaries used here. Full citations and conservative
claim language are in `docs/PRIOR_ART_AND_NEXT_TESTS.md`.

## Results table

All strict tests hold out entire galaxies. Values are pooled held-out
$\chi^2$ per radial point over the same 131 SPARC galaxies and 3,034 points.

| Test | Baseline | Candidate | Candidate - baseline | Result |
|---|---:|---:|---:|---|
| P0 potential screen vs first acceleration screen, CF4-balanced folds | 27.610 | 22.140 | -5.470, 95% bootstrap CI [-13.248, 1.115] | Better point estimate; uncertainty crosses zero |
| P0 potential screen vs fixed RAR, CF4-balanced folds | 21.340 | 22.140 | +0.800, CI [-0.866, 2.531] | Competitive, not superior |
| P1 CF4 shifts potential threshold vs P0 | 22.140 | 23.347 | +1.207, CI [-0.108, 2.702] | Fail; prediction worsens |
| B1 potential boundary layer vs P0 | 22.140 | 23.011 | +0.871, CI [-0.761, 2.821] | Fail; prediction worsens |
| W1 published void-wall depth shifts threshold vs wall-balanced P0 | 22.278 | 26.420 | +4.142, CI [-2.025, 11.936] | Fail; prediction worsens |
| Wall-balanced P0 vs fixed RAR | 21.340 | 22.278 | +0.938, CI [-1.309, 3.208] | RAR remains the better point estimate |

P1 and B1 form the declared two-hypothesis family. Their Holm-adjusted
one-sided bootstrap p-values for improvement are both 1.0.

## P0: potential depth is useful only as a comparator

P0 replaces activation by local $g_{\rm bar}$ with activation by the integrated
baryonic potential-depth proxy

$$
|\Phi_{\rm bar}(R)|=\int_R^{R_{\max}}g_{\rm bar}(r)\,dr
+g_{\rm bar}(R_{\max})R_{\max}.
$$

Across CF4-balanced whole-galaxy folds, $p$ is stable from 0.412 to 0.426, still
below the flat-curve value $p=1/2$. The transition depth is usually
$1.31\times10^{-6}$ to $1.56\times10^{-6}$ in $|\Phi|/c^2$, with one fold at
$6.06\times10^{-6}$.

The independent radial split is less favorable: P0 gives outer
$\chi^2/N=5.871$, compared with 5.849 for the old free-$p$ law and 4.748 for
fixed RAR. Its fitted width is 0.0302 dex, effectively the registry's 0.03-dex
lower bound, and $p=0.392$. The boundary contact means the smooth potential
screen is trying to become an abrupt transition. P0 should remain a diagnostic
benchmark, not be described as evidence for a physical screening surface.

## P1: environment moving the transition fails

P1 tests

$$
\log_{10}\chi_{t,i}=\log_{10}\chi_{t,0}+\zeta\mathcal V_i,
$$

using the primary grouped CF4 underdensity. It worsens held-out prediction by
1.207 $\chi^2$ per point. Fold $\zeta$ values are
[-0.051, 0.672, 0.465, 0.117, -0.066], so the predicted positive sign is not
stable. This rejects the proposed threshold-shift coupling just as the earlier
amplitude-shift coupling failed.

## B1: a boundary pressure bump fails

B1 adds

$$
g_B=\kappa a_\star\frac{dS_\Phi}{d\ln R}.
$$

It worsens held-out prediction by 0.871 $\chi^2$ per point. Fold $\kappa$ values
are [0.157, 0.546, -1.644, 0.182, 0.175]. The large negative value in one fold
violates the hypothesized inward boundary sign. A free galaxy-specific boundary
coefficient was not tried because it would abandon the universal prediction.

## W1: actual void walls do not rescue the theory

The project downloaded and hashed the Malandrino et al. Bayesian catalog of 100
high-significance Local-Universe voids at repository commit
`bbbc34594d92eeef32897d67d291d54eb384be6e`. The score uses the authors'
volume-preserving Voronoi-overlap boundary $>0.37$, the full irregular cloud
shape, and half-voxel-corrected distance to the actual wall. No galaxy midpoint
is used.

Seventy-two of all 175 SPARC galaxies, and 51 of the 131 analysis galaxies, lie
inside these boundaries. The five held-out folds contain 10 or 11 inside-void
galaxies each. W1's $\zeta_w$ is positive in all folds
[0.394, 0.494, 0.075, 0.897, 0.429], but the model worsens prediction by 4.142
$\chi^2$ per point and its one-sided bootstrap p-value for improvement is 0.873.
Consistent fitted sign without predictive gain is not evidence; it can arise
from training-set association or confounding.

## T0: ordinary void gravity is far too small and often the wrong sign

T0 solves

$$
\nabla\cdot\mathbf g_\delta=-\frac{3}{2}\Omega_mH_0^2\delta
$$

on the frozen grouped CF4 grid with a zero-padded FFT and no fitted force
normalization. A uniform peculiar acceleration is excluded because it moves a
galaxy and its contents together. The internal upper bound uses the most
compressive tidal eigenvalue at each galaxy, an orientation deliberately chosen
to favor the hypothesis.

- Median maximum inward tide at the outer measured radius:
  $2.79\times10^{-16}\ \mathrm{m\,s^{-2}}$.
- Median observed outer acceleration excess:
  $2.11\times10^{-11}\ \mathrm{m\,s^{-2}}$.
- Median tide/required ratio: $1.33\times10^{-5}$, a 4.88-order shortfall.
- Best ratio in the sample: $2.87\times10^{-4}$, still a 3.54-order shortfall.
- Underdense galaxy locations have positive median tidal trace, the expansive
  or anti-binding sign expected for a density deficit.

This directly rules out ordinary smooth void tides as the required inward
acceleration. An arbitrary amplification factor is not allowed to relabel that
failed mechanism.

## Decision

Stop adding SPARC formula variants. The data now say:

1. A universal low-acceleration law remains useful phenomenology.
2. Potential depth is worth retaining as a matched comparator but not as a
   supported physical screen.
3. Local CF4 density, a CF4-shifted transition, a boundary gradient, and actual
   published void-wall depth all fail out-of-sample prediction.
4. Ordinary void tides miss the required acceleration by thousands to millions.

A future physical theory would need field equations that generate a
galaxy-centered nonlinear response while respecting equivalence-principle,
Solar-System, lensing, and cosmological constraints. Its next empirical test
must be a non-overlapping rotation/velocity-field sample such as LITTLE THINGS,
with SPARC no longer treated as confirmatory data. Without that derivation and
external prediction, further curve-fitting would be data dredging rather than
theory development.
