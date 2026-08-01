# Complete-action Sigma consistency test

## The question in ordinary language

Until now, Sigma changed gravity but gravity did not push back on Sigma, and we
did not count the energy stored in the Sigma field itself.  A complete physical
theory cannot ignore those two effects.

Think of Sigma as a flexible landscape:

- **Backreaction** asks whether the gravity flowing across the landscape changes
  its shape.
- **Stored field energy** asks whether building that landscape adds weight that
  must itself gravitate and bend light.

Both effects were derived from one shared action and one shared normalization.
No separate “extra cluster mass” dial was fitted.

## Short answer

Neither effect closes the galaxy/cluster gap.

Letting gravity reshape Sigma changes the combined result by less than one part
in a thousand.  Making the Sigma field heavy enough to matter in the cluster
makes it much too heavy in the galaxy first.  The elegant conclusion is useful:
keep Sigma as an environmental switch, but do not try to repair cluster lensing
by treating its stored energy as a hidden halo.

## How to read the numbers

The earlier reports used “dex,” which is useful to specialists but difficult to
picture.  This report also converts each score into an ordinary mismatch factor:

- 1.0 means an exact match.
- 1.2 means the prediction and target typically differ by a factor of 1.2.
- 1.7 means a typical difference by a factor of about 1.7.
- 2.0 means a factor-of-two difference.

A factor is symmetric in logarithmic space: a factor-1.7 miss could mean 1.7
times too high or about $1/1.7=0.59$ times too low.

## One action, not two adjustable fixes

The static energy contains the AQUAL gravity term and the Sigma-field term:

$$
E=\int d^3x\left[
\frac{a_0^2}{8\pi G}\mathcal F(X,\Sigma)
+K_\Sigma\left(\frac{L_\Sigma^2}{2}|\nabla\Sigma|^2
+V_{\rm eff}(\Sigma,\rho)\right)
\right].
$$

Their relative strength is

$$
\chi=\frac{a_0^2}{8\pi G K_\Sigma}.
$$

Larger $\chi$ means ordinary gravity pushes the Sigma landscape around more
easily.  Smaller $\chi$ makes Sigma stiffer, but it also means a given Sigma
pattern stores more energy.  Those effects cannot be tuned independently
without abandoning the single action.

## Test 1: gravity pushing back on Sigma

The previous one-way calculation had a combined score of 0.230381.  After
varying the same action with respect to Sigma, the best result was 0.230362.
That numerical change is about 0.0085 percent and is not practically meaningful.

The understandable mismatch factors were:

| Feedback $\chi$ | Galaxy factor | Cluster factor | What happened |
|---:|---:|---:|---|
| 0.001 | 1.73 | 1.67 | Essentially the original result |
| 0.003 | 1.74 | 1.66 | Tiny cluster improvement traded for a tiny galaxy worsening |
| 0.010 | 1.76 | 1.64 | The same trade becomes more visible |
| 0.030 | 1.89 | 1.55 | Cluster improves, but the galaxy moves substantially farther away |

All galaxy and cluster field equations converged.  At the best value,
$\chi=0.003$, Sigma's own mass was only 0.000063 percent of the galaxy's
baryonic mass inside 20 kpc and 0.0000033 percent of the cluster baryonic mass
inside 100 kpc.  It was far too light to affect lensing.

## Test 2: allowing Sigma's stored energy to gravitate

The positive field energy relative to the empty-space minimum is

$$
u_\Sigma=K_\Sigma\left[
\frac{L_\Sigma^2}{2}|\nabla\Sigma|^2
+\frac14(1-\Sigma^2)^2
\right].
$$

This energy was converted to mass using $E=mc^2$, accumulated inside each
radius, and added to the gravitational source before solving the AQUAL force
law.

For the universal candidate, decreasing $\chi$ produced this progression:

| $\chi$ | Sigma mass / galaxy baryons at 20 kpc | Sigma mass / cluster baryons at 100 kpc | Galaxy mismatch | Cluster mismatch |
|---:|---:|---:|---:|---:|
| $10^{-8}$ | 0.20 | 0.015 | 1.99 | 1.62 |
| $10^{-9}$ | 2.03 | 0.155 | 4.09 | 1.46 |
| $10^{-10}$ | 20.3 | 1.55 | 21.7 | 2.34 |

The key relationship is visible without specialized statistics.  By the time
the field contributes cluster-sized mass, the galaxy already contains about
twenty times more Sigma-field mass than baryonic mass.  The galaxy prediction
then misses by more than a factor of twenty.

This happened throughout the frozen 800-row stress-energy grid.  Its best row
was simply the row where the field energy was negligible, reproducing the
previous action result.

## What this proves and does not prove

It establishes that, for this action and field potential:

- ignoring feedback in the previous exploratory run was an excellent numerical
  approximation;
- positive Sigma stress-energy is not the missing universal bridge;
- allowing an independent field-mass normalization would hide the problem
  rather than solve it, so none was introduced.

It does not test every possible scalar potential, a nonspherical cluster, or a
complete cosmological solution.  It is still a spherical weak-field calculation.

## Next elegant direction

The scalar-energy route should be set aside.  The cleaner next relativistic
direction is to let the same modified potential bend matter and light through a
physical metric, as tensor-vector-scalar theories do, while Sigma only controls
when that low-acceleration sector activates.

Before adding the relativistic fields, the current universal equation should be
tested on a small ladder of galaxies with different masses and sizes.  In plain
terms: we need to learn whether the same switch turns on at the right place for
a small galaxy, a Milky-Way-sized galaxy, and a giant galaxy.  If it does not,
there is no reason to make the equation relativistically more elaborate.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_sigma_complete_action.py
python -m pytest tests/test_sigma_actions.py tests/test_sigma_complete_action_results.py -q
```

The machine-readable artifacts are under `results/sigma_complete_action/`.
