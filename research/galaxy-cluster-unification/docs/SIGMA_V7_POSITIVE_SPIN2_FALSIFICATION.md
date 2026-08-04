# Sigma v7 positive-spin-2 carrier falsification

## Decision

The positive massive-spin-2 carrier is retired as the current Sigma route under
the project's three-formulation reset rule.  No additional coefficient, screen,
or lensing term will be appended to v7C.

This is a mechanism-level project decision, not a claim that dRGT massive
gravity or Hassan--Rosen bimetric gravity is mathematically false.  A future
complete coupled-metric proposal would be evaluated as a new theory from its
action, not as another v7 fit.

## Common gate

Each formulation was required to provide one positive, Solar-safe, universal
carrier that changes massive dynamics and photon lensing through the same
physical metric, produces useful cluster-scale amplitude and geometry, and uses
no object label or lens-only multiplier.

| Formulation | Material change | Decisive result |
|---|---|---|
| v7A | Unscreened linear positive Fierz--Pauli carrier | High-field force bounds require residue `<=7.5e-6`; maximum lensing is `1.0000075` and decreases rather than activates with distance. |
| v7B | Ghost-free spherical Vainshtein screen | Equal-mean-density galaxy and cluster archetypes have identical screening for every universal range; the healthy exterior lensing ceiling is `1.5`, below the factor-`3` carrier target. |
| v7C | Three-dimensional cubic Hessian scalar response | The scalar is healthy and `7.223%` nonadditive, but its leading conformal metric contribution has `delta W=0`; the non-conformal metric and coupled tensor equations were not frozen. |

These are materially different implementations and independent failure modes,
but they miss the same combined carrier outcome.  The executable synthesis
confirms three distinct candidates and three failed useful-lensing gates without
opening a raw holdout.

## What was learned

1. A positive linear pole is healthy but cannot be both unscreened locally and
   large enough at clusters.
2. Spherical Vainshtein screening keys off mean enclosed density and cannot by
   itself encode the multi-component geometry that the raw lensing tests need.
3. A nonlinear Hessian can encode component overlap and orientation, but
   nonadditivity in an auxiliary scalar does not imply nonadditivity in the
   physical Weyl potential.
4. The physical metric must be derived before a field is compared with a lens
   map.  This is now a mandatory pre-data gate for every successor.

## Constraints on the successor

The next mechanism must:

- be Weyl-active at the same derived order at which it changes massive motion;
- freeze the physical metric before any source-map solve;
- respond to three-dimensional baryonic geometry rather than only `M/r^3`;
- recover the Solar limit without erasing the required cluster response;
- keep at most five universal constants and use no object labels, private force
  parameters, or lens-only multiplier.

A pure conformal scalar, another positive Yukawa pole, or another spherical
density screen does not qualify as a materially new mechanism.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v7a_positive_carrier.py
python scripts/audit_sigma_v7b_spherical_vainshtein.py
python scripts/audit_sigma_v7c_cubic_hessian.py
python scripts/audit_sigma_v7c_metric_projection.py
python scripts/audit_sigma_v7_positive_spin2_falsification.py
```

Machine-readable evidence is stored in
`results/sigma_v7_positive_spin2_falsification/report.json`.
