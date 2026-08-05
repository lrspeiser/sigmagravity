# Sigma V19CC cross-scale prediction gates

## Decision

The project will spend its next theory budget on breadth across galaxies and
clusters, not on detailed Solar-System tuning.  Local tests remain mandatory,
but they are a later veto after a candidate has demonstrated that it can
predict both matter motion and two-dimensional lensing geometry.

V19CC turns the existing broad-phenomenology roadmap into measurable tests.
It does not select a source invariant, action, formula, wavelength, coupling,
or target.  In particular, I4 and I5 remain registered source candidates until
the protected V19BQ/V19BS chain decides whether either is actually observable
and transferable in both merger systems.

## What “broad first” now means

The primary galaxy gate is at least 48 untouched galaxies, including at least
32 WALLABY systems and at least six galaxies in every frozen mass, gas,
surface-brightness, and bulge class.  Catalog rotation curves are not enough:
at least 12 systems need observation-space cube or velocity-field tests, eight
need high-resolution inner curves, and eight need simultaneous radial and
vertical dynamics.

The primary cluster gate is at least six untouched systems, including at least
two relaxed and two disturbed or merging clusters.  Every cluster needs a
complete baryon model and raw multiple-image constraints.  A halo or
convergence map inferred under conventional gravity is a comparator, not the
truth the theory is asked to reproduce.

The same frozen action must be no worse than 1.05 times fixed MOND/RAR on the
overall galaxy rotation score and no worse than 1.25 times it in any morphology
class.  On raw strong lensing it must recover every held-out image root, be no
worse than 1.25 times the same-catalog halo comparator in positional RMS, and
close at least 75% of the baryons-only-to-halo gap.

## The next four no-retuning tests

Once the core action and constants are frozen, the following tests use exactly
the same metric.  They may introduce ordinary measurement nuisances under the
same external priors as the comparators, but no new gravity amplitude, range,
orientation, exponent, lag, or object label.

| Gate | Raw question | Frozen success criterion |
|---|---|---|
| Resolved cluster weak lensing | Does the metric predict each background source's two shear components, plus magnification where available, outside the strong-lens core? | Six or more clusters; total covariance deviance no worse than 1.25 times the halo comparator; every registered cluster stratum no worse than 1.50; at least 75% baryon-to-halo gap closure; random-point and B-mode nulls pass. |
| Galaxy-galaxy weak lensing | Does the same galaxy law predict the stacked field at larger projected radii? | At least three baryonic-mass bins and two morphology or surface-density bins; the same 1.25 aggregate, 1.50 per-bin, and 75% gap-closure gates. |
| Joint dynamics and lensing | Can one metric predict motion and light in the same objects? | At least eight systems spanning rotation- and pressure-supported classes; combined score no worse than 1.25 times the halo comparator, with neither domain worse than baryons-only. |
| Merger direction and offsets | Does the baryonic source state place and orient curvature correctly before lensing is seen? | Two spent development mergers plus at least two untouched mergers; median axial error no more than 30 degrees; raw lensing deviance no worse than 1.25 times the halo comparator; 75% gap closure and the full image-root/topology gate. |

For covariance-weighted observables, the primary discrepancy is

\[
D_M=(y-y_M)^T C^{-1}(y-y_M),
\]

and baryon-to-halo gap closure is

\[
F_{\rm close}=1-\frac{D_\Sigma-D_{\rm halo}}
{D_{\rm baryon}-D_{\rm halo}}.
\]

The value is not clipped to make a plot look better.  A negative value means
Sigma made the baryonic prediction worse; a value above one means it beat that
specific halo comparator.

## Point of view on other dark-matter-attributed effects

Weak lensing is the most immediate extension because it asks the same metric
question as strong lensing, just at larger radii and lower curvature.  A model
that fits rotation curves but misses weak shear has not found one law for
galaxies.  A model that gets cluster amplitude but misses the shear direction
has the wrong spatial response; another amplitude is not a repair.

Merger offsets are the sharpest near-term test of the source-state idea.  A
pressure, entropy-gradient, stress, or relative-current source can point away
from the total baryonic-density peak.  That is useful only if its direction is
fixed before lensing is opened and transfers to new mergers.

Gas-poor dwarf satellites and ellipticals are a serious risk for an
instantaneous thermodynamic source.  If their apparent extra gravity is real,
the same action would need to derive a persistent or environmental response
from earlier baryonic evolution; we cannot add a “gas-poor” switch.  The test
must predict dispersion, escape speed, tides, and host-distance dependence
without assigning a halo to every satellite.

Stellar-stream gaps and strong-lens flux anomalies are even harder.  A smooth
long-range response may alter global precession, but it does not automatically
create the compact kicks usually attributed to dark subhalos.  Sigma can claim
that territory only if its field equations independently produce stable,
compact response structures with a predicted abundance and mass scale.

Dynamical friction is an energy-accounting test.  A static field prescription
has no place for the lost orbital energy to go.  A viable action must derive
field stress-energy, radiation or excitation, backreaction, and the sign and
rate of the drag.

Growth, cosmic shear, cluster abundance, and the CMB are not calculable from
the current empirical bridge.  They require the final covariant action,
background solution, and perturbation equations.  The CMB is likely the
strongest eventual challenge because a baryon-only theory must sustain the
early gravitational potentials normally supplied by cold dark matter.

## Reproduction

```powershell
python scripts/check_sigma_v19cc_cross_scale_prediction_gates.py
python -m pytest tests/test_sigma_v19cc_cross_scale_prediction_gates.py -q
```

The machine-readable result is
`results/sigma_v19cc_cross_scale_prediction_gates/report.json`.
