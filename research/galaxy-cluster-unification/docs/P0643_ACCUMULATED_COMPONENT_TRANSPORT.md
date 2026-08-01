# P0643 finite path-accumulation result

## Why this experiment exists

P0642 showed that a dimensionless measure of field-direction disagreement does
not separate clusters from dwarf galaxies.  A small, irregular galaxy can be
as fractionally asymmetric as a merging cluster.

P0643 tests the more specific idea that unresolved gravitational vectors retain
memory while propagating, so disagreement matters only after it persists over
a finite path.  The candidate source tensor is

\[
\mathsf N_b=
\nu\!\left({|\mathbf g_N|\over a_0}\right)I+
\lambda {a_0\over a_0+|\mathbf g_N|}
C_{\rm cancel}
\left[1-e^{-\ell/L_c}\right]\hat{\mathbf h}\hat{\mathbf h},
\]

with

\[
C_{\rm cancel}=1-
{|\mathbf g_\star+\mathbf g_{\rm gas}|\over
 |\mathbf g_\star|+|\mathbf g_{\rm gas}|},
\qquad
\ell={|\mathbf g_N|\over\|\nabla\mathbf g_N\|_F}.
\]

The field is then defined by

\[
\nabla^2\Phi=\nabla\cdot(\mathsf N_b\nabla\Phi_N).
\]

The primary universal coherence length is `Lc = 10 kpc` and the exponential
power is one.  Ten kiloparsecs was reused from the already-spent P0629 length
screen before any P0643 score.  The other lengths and powers are sensitivity
rows and cannot replace a failed primary.

## What passed

All ten preregistered gates pass:

- a co-centered radial two-component system has activation
  `3.04e-16`;
- a cluster-scale displaced pair accumulates 41.85 times more response than a
  self-similar galaxy-scale pair;
- the 13 registered galaxy maps have median activation `0.001127`;
- the four registered cluster maps have median activation `0.021022`;
- the cluster/galaxy median ratio is 18.655;
- that ratio remains 14.95--19.03 under the low, nominal, and high baryonic
  mass maps;
- exact 90-degree rotation and a large translation leave the summary unchanged
  to numerical precision;
- the worst-case one-AU screening proxy is `4.05e-7` even if the later universal
  amplitude reaches 20, below the preregistered `2e-6` bound; and
- no velocity or lensing target was opened.

The mechanism is transparent in the measured length: the median weighted
tidal path is about 1.08 kpc in the galaxies and 177 kpc in the clusters.  The
exponential is still rising for a typical dwarf but nearly saturated for a
cluster.

## What this does and does not mean

This is the first candidate in the project that produces the desired
galaxy/cluster *domain lever* from registered baryonic maps without reading the
answers.  It does not show that the new response has the right magnitude,
direction, or lens topology.

There is also a price: `Lc` is a new universal dimensional constant.  The model
is more predictive than a separate setting for every object, but it is not yet
a first-principles theory until the coherence length is derived from a field's
mass, propagation speed, relaxation time, or another deeper constant.

An exploratory resolution replay after the gate result found that increasing
the cluster grid from 257 to 513 cells lowered individual activations by roughly
12--23 percent, while the tested galaxy activations changed by roughly 6--17
percent from 65 to 129 cells.  The order-of-magnitude domain separation remains,
but a formal convergence gate is required before raw lensing.

## Required next experiment

Before unsealing P0640:

1. run a frozen resolution and thickness convergence audit;
2. implement the tensor source in the real Poisson/QUMOND solver;
3. use already-spent raw lenses to choose one universal `lambda` and compare
   against an equally flexible external-shear/multipole control;
4. hash the equation, constants, mass assumptions, and predicted deflection
   maps for all four new clusters; and
5. unseal once, scoring image positions, multiplicity, complete roots, critical
   curves, and topology.

The same frozen equation must then be evaluated on all 13 sealed velocity
fields and Solar-System controls.

## Reproduction

```powershell
python scripts/run_p0643_accumulated_component_transport.py
python -m pytest tests/test_p0643_accumulated_component_transport.py -q
```
