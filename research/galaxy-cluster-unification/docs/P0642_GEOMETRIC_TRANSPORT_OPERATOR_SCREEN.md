# P0642 geometric transport operator screen

## Question

Can a field-direction property calculated from baryons alone provide the new
geometric variable that was absent from P0623--P0630, before any new velocity
or lensing answer is opened?

P0642 tests three dimensionless fields:

\[
C_{\rm path}=1-|\langle\hat{\mathbf g}_N\rangle_\gamma|^2,
\qquad
C_{\rm cancel}=1-
{ |\mathbf g_{\star}+\mathbf g_{\rm gas}| 
 \over |\mathbf g_{\star}|+|\mathbf g_{\rm gas}| },
\]

and

\[
C_{\rm hybrid}=1-(1-C_{\rm path})(1-C_{\rm cancel}).
\]

The path average follows the Newtonian field for the local, gauge-independent
length

\[
\ell={|\mathbf g_N|\over\|\nabla\mathbf g_N\|_F}.
\]

The intended field equation is

\[
\nabla^2\Phi=\nabla\!\cdot\!\left[
\left(\nu I+\lambda {a_0\over a_0+|\mathbf g_N|}
C_b\,\hat{\mathbf h}\hat{\mathbf h}\right)\nabla\Phi_N
\right].
\]

This is more restrictive than the earlier map-routing recipes.  It supplies a
local tensor source and the final field is obtained from one scalar potential.
No lens residual selects a direction.

## Blindness boundary

The computation uses the registered H I, stellar-light, X-ray-gas, and cluster
member-light maps from P0639 and P0641.  These are independent baryonic inputs.
It does **not** open:

- any P0633 LITTLE THINGS velocity field;
- either sealed P0640 raw-lensing container;
- any multiple-image coordinate, family assignment, critical curve, or lens
  residual; or
- any inferred dark-matter map for the four validation clusters.

The protocol and numerical thresholds were written before the first P0642
score.

## Result

The first screen is a rejection, not a discovery.

- A co-centered two-component radial source gives essentially exact
  component-cancellation zero, but the numerical path null is larger than its
  preregistered tolerance.
- A binary source activates the path term, although its gain over the radial
  control misses the preregistered factor-five threshold.
- A displaced gas--star pair strongly activates component cancellation.
- Rotation and translation covariance pass to numerical precision.
- The worst-case one-AU coefficient at `lambda=5` misses the deliberately
  strict Solar gate by about one percent.
- Most importantly, the median screened activation of all three operators is
  **not larger in clusters than galaxies**.  The observed dwarf maps contain
  enough real gas--star offsets, clumps, and asymmetry to activate the same
  geometry.

The component-cancellation variant is the only provisional synthetic survivor,
but no operator is advanced because the complete gate set fails.  It would be
misleading to tune `lambda` on lensing after this result.

## What was learned

The 50--125 kpc gas--star centroid offsets in P0641 are real inputs, but a
dimensionless disagreement variable erases the fact that they persist over
hundreds of kiloparsecs.  Dwarf galaxies can have a comparable *fractional*
directional disagreement on much smaller scales.  Therefore a viable version
of the proposed mechanism needs one of the following, stated before lensing is
scored:

1. an accumulated path term that retains physical or dynamically derived path
   length;
2. a time-dependent memory/relaxation field;
3. a field invariant that distinguishes multi-center cluster topology from
   local disk clumpiness; or
4. a proof that the geometric term should affect photon and matter metric
   potentials differently.

The next experiment should not weaken P0642's gates.  It should preregister a
new accumulation law, test it on already-spent systems, compare it with a
matched external-shear control, and freeze its universal constants before the
new RELICS constraints are unsealed.

## Reproduction

```powershell
python scripts/run_p0642_geometric_transport_operator_screen.py
python -m pytest tests/test_p0642_geometric_transport_operator_screen.py -q
```
