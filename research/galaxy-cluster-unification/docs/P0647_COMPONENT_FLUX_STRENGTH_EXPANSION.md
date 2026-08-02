# P0647 component-flux strength expansion

## Question and frozen decision rule

P0646 found that a conservative gas-minus-star flux improved spent RX J2129
image positions, but its best strength was the top of the tested grid. P0647
held every other part of the formula fixed and tested

\[
\lambda\in\{3.5,5,6.5,8,10,12.5\}.
\]

The same five source-family folds and six ordinary lens-geometry parameters
were refit at every strength. Before these scores existed, the protocol stated
that the family would be rejected unless it produced a strict, root-complete
minimum inside the grid. The descriptive spent-heldout split was not used to
select a value.

## Result: reject the strength family

The lowest eligible cross-validation error occurs at `lambda=12.5`, the largest
value tested. It improves the audited zero-field CV RMS from `2.760255` to
`2.308115 arcsec`, or `16.38%`, and beats the P0646 matched isotropic control by
`15.92%`. Those are large numerical changes, but they do not identify a
universal constant because the optimum is still moving at the boundary.

The result fails two of twelve frozen gates:

- the selected strength is not inside the grid; and
- it has no root-complete neighbor on each side and therefore is not a strict
  local minimum.

The exact CV replay also exposes topology/optimizer fragility. `lambda=3.5`
recovers `14/15` held-out image roots and `lambda=5` recovers `13/15`, whereas
the higher four rows recover all `15/15`. P0646 had recovered all roots at five
under its earlier deterministic replay. A physical formula whose conclusion
changes with optimizer basin is not ready for a blind test, even when its RMS
is attractive.

## Descriptive checks

At the boundary value, the full training refit recovers `15/15` roots with
`0.444347 arcsec` RMS. The already-spent heldout split recovers `7/7` roots with
`1.924577 arcsec` RMS, `6.31%` worse than the P0599 comparison. The one-AU
screening coefficient is `2.53e-7`, below the `2e-6` proxy limit, but this is
not a metric/PPN derivation or a Cassini analysis.

No P0633 galaxy velocity outcome or P0640 cluster-lensing outcome was opened.

## Interpretation and next move

P0647 does **not** say that the gas/star component direction is meaningless.
It says that a single unbounded multiplier on this flux is not an identifiable
universal law on the available spent lens. Extending the upper bound again
would violate the frozen rejection rule and reward flexibility rather than
explanation.

The defensible next experiment changes the mathematical question. Useful
options are a bounded response whose ceiling is fixed from field physics, or a
genuinely geometric transport law in which the total response follows from a
path/curvature invariant rather than an arbitrary multiplier. Any such law
must first face the same exact-root stability test and a matched conventional
multipole control on spent data.

## Reproduction

```powershell
python scripts/run_p0647_component_flux_strength_expansion.py
python -m pytest tests/test_p0647_component_flux_strength_expansion.py -q
```
