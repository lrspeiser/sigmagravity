# Sigma v15 spent covariant-invariant inference

## Decision

The tested static local baryonic invariants are **not sufficient** to generate
the missing cluster Weyl field with one transferable operator. The primary
three-scale scalar family is the best cross-cluster result, with normalized
full-field error `0.771407`, well above the frozen `0.500` source-sufficiency
gate. Adding nonlinear total-baryon tidal terms worsens the score to `0.797320`;
adding gas--star overlap and orientation reaches `0.788787`. Neither clears the
frozen ten-percent improvement requirement.

The disclosed post-failure resolution sensitivity adds 5 and 10 kpc structure.
It improves the best score only to `0.749848`, a `2.795%` change rather than the
required ten percent. Total-tidal and component-resolved variants again lose to
the scale-only family. The primary failure is therefore robust over Gaussian
scales from 5 to 150 kpc; it was not caused simply by erasing member-galaxy
peaks.

This is a spent inverse problem, not an observational validation or a gravity
theory. It constrains the source information that a successor action must
generate. No holdout was opened and no physical constant was fitted.

## Why this experiment follows the v14 reset

Sigma v3C had already shown that an arbitrary real isotropic Fourier transfer
could not convert the measured AQUAL Hessian into the compact-halo comparator:
its joint normalized error was `0.800`, with low radial phase coherence. The v14
local gauge-tidal carrier then failed its mathematical action gate in three
distinct completions.

Before selecting another field carrier, v15 asks a measurement-first question:

> Which rotationally covariant information in the measured baryons actually
> predicts the missing convergence and shear across more than one spent
> cluster?

The answer separates a missing physical variable from a bad choice of field
notation.

## One-potential feature construction

The baryonic maps are converted to dimensionless convergence using

\[
\kappa_b={\Sigma_b\over\Sigma_{\rm crit}(z_l,z_s)},
\qquad z_s=2.
\]

Every scalar source feature `s(x)` is made into a metric-compatible Hessian
triplet. In Fourier space,

\[
\widehat\kappa_s=\widehat s,
\qquad
\widehat\gamma_1=D_1(\mathbf k)\widehat s,
\qquad
\widehat\gamma_2=D_2(\mathbf k)\widehat s,
\]

where

\[
D_1={k_E^2-k_N^2\over k_E^2+k_N^2},
\qquad
D_2={2k_Ek_N\over k_E^2+k_N^2}.
\]

A symmetric trace-free baryonic tensor `(Q1,Q2)` first produces the rotational
scalar

\[
\widehat\kappa_Q=D_1\widehat Q_1+D_2\widehat Q_2,
\]

and the same E-mode construction generates its shear. Consequently, one fitted
coefficient acts on convergence and both shear components. The regression
cannot rotate, amplify, or suppress the two shear components independently and
cannot introduce a lens-only field.

The nested primary families are:

| Family | Features | Baryonic information |
|---|---:|---|
| `scalar_scale` | 3 | Total convergence smoothed at 25, 75, and 150 kpc |
| `total_tidal` | 12 | Scalar family plus total gradient magnitude, traceless Hessian, and gradient-axis tensor at every scale |
| `component_overlap` | 24 | Total-tidal family plus gas--star mixing, signed gas-minus-star contrast, cross-gradient tensor, and difference-gradient tensor |

The target is `compact halo - AQUAL`. There is no intercept. Ridge strength is
selected from a frozen grid by symmetric leave-one-cluster-out transfer:
AS295 predicts PLCK G287 and PLCK G287 predicts AS295. The final joint fit uses
one coefficient vector and remains descriptive only.

## Primary frozen result

Unmodified AQUAL has full-field NRMSE `0.902148` in AS295 and `0.895304` in
PLCK G287 on the common central mask.

| Family | Selected ridge alpha | Symmetric cross-cluster NRMSE | Change versus scalar |
|---|---:|---:|---:|
| `scalar_scale` | 1 | **0.771407** | reference |
| `total_tidal` | 1 | 0.797320 | 3.36% worse |
| `component_overlap` | 1 | 0.788787 | 2.25% worse |

For the selected scalar family:

| Train -> test | Full-field NRMSE | Missing-field power closed | Missing-shear alignment |
|---|---:|---:|---:|
| AS295 -> PLCK G287 | 0.728249 | 35.83% | 0.5304 |
| PLCK G287 -> AS295 | 0.812274 | 25.28% | 0.3547 |

The selected operator closes the preregistered minimum residual power in both
directions by a narrow margin, but it fails the absolute `0.500` field gate and
the shear-alignment gate in AS295. Its three leading terms are simply the three
smoothed total-baryon maps with comparable positive contributions.

The additional families can improve their same-cluster descriptive fits. For
example, the component family reaches self-fit NRMSE `0.751916` in AS295 and
`0.597734` in PLCK G287. That information does not transfer: the corresponding
cross scores are `0.862678` into AS295 and `0.707217` into PLCK G287. This is
exactly why a joint fit alone would overstate the evidence for component
physics.

The independently trained operators nevertheless produce similar broad fields:
their pooled prediction cosines are `0.999997`, `0.974167`, and `0.973347` for
the scalar, total-tidal, and component families. The main obstacle is therefore
not wildly different per-cluster coefficient signs. It is that the tested
local bases do not contain enough of the required spatial phase and shear
orientation, especially in AS295.

## Compact-scale sensitivity

The primary map figure showed compact comparator peaks absent from the
25--150 kpc prediction. V15B was frozen after that failure and added 5 and 10
kpc bases without changing the clusters, targets, mask, feature definitions, or
transfer rule.

| Family | Symmetric cross-cluster NRMSE | Change versus primary winner |
|---|---:|---:|
| `scalar_scale` | **0.749848** | 2.795% better |
| `total_tidal` | 0.757466 | 1.807% better |
| `component_overlap` | 0.750792 | 2.672% better |

The scalar sensitivity selects zero ridge penalty and alternating coefficients
across 5, 10, 25, 75, and 150 kpc. Individual contribution RMS values are much
larger than the final field and cancel one another. This is a flexible band-pass
reconstruction, not an identified elegant response. It still fails the absolute
gate, closes only `23.96%` of missing power in the PLCK-to-AS295 direction, and
has shear alignment `0.4385` there.

V15B therefore does not overturn the primary result. Compact measured baryons
help modestly but do not supply the missing transferable field.

![Spent invariant inference](../results/sigma_v15_spent_invariant_inference/spent_invariant_inference.png)

## Physical inference

The result rejects the following as a sufficient weak-field closure on these
spent maps:

\[
\Delta W=F\!\left(
\Sigma_b,\nabla\Sigma_b,\nabla_i\nabla_j\Sigma_b,
\Sigma_g\Sigma_\star,\nabla\Sigma_g\otimes\nabla\Sigma_\star
\right)
\]

when `F` is represented by the tested shared local, multiscale, one-potential
basis. It does not prove that every nonlinear local theory fails, but it removes
the empirical justification for adding another algebraic density, gradient,
tidal, or gas--star multiplier to the action.

The next missing variable must be sought in information not contained in this
static 700 kpc local snapshot:

1. **Wider baryonic boundary conditions.** The halo comparator includes fitted
   external shear, while the v15 feature context stops at 700 kpc.
2. **Dynamical state.** Merger history, gas shocks, member velocities, and
   relaxation time cannot be reconstructed uniquely from one projected mass
   map.
3. **A different constraint structure.** Any causal state carrier must avoid
   the already-falsified localized retarded-memory, material-triad,
   preferred-clock, and local gauge-tensor mechanisms.

The next spent calculation should first decompose the required field into an
internal E-mode sector and harmonic boundary modes, then expand the measured
baryon context without touching a holdout. Only if that fails should a new
dynamical-state postulate be written and its required observational inputs
enumerated.

## Claim boundary

- AS295 and PLCK G287 are spent clusters.
- The compact-halo target was fitted to the same image catalogs and is not
  direct pixel truth.
- External shear may encode matter outside the registered context.
- Feature coefficients and ridge penalties are diagnostic and are not Sigma
  constants.
- The one-potential construction is a two-dimensional integrability condition,
  not a four-dimensional covariant action.
- Two-cluster transfer can reject a simple universal operator but cannot
  establish universality.

## Reproduction

```powershell
python scripts/infer_sigma_v15_spent_invariants.py
python scripts/infer_sigma_v15_spent_invariants.py `
  --config configs/sigma_v15b_spent_invariant_resolution_sensitivity.json `
  --output results/sigma_v15b_spent_invariant_resolution_sensitivity
python -m pytest -q tests/test_sigma_v15_spent_invariant_inference.py
python -m ruff check src/voidscreen/sigma_covariant_feature_inference.py `
  scripts/infer_sigma_v15_spent_invariants.py `
  tests/test_sigma_v15_spent_invariant_inference.py
```

Machine-readable primary and sensitivity evidence is in
`results/sigma_v15_spent_invariant_inference/report.json` and
`results/sigma_v15b_spent_invariant_resolution_sensitivity/report.json`.
