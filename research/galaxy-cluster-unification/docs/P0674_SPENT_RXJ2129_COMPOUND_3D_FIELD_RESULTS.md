# P0674 spent RX J2129 compound 3D field results

## Frozen result: pass

All 25 structural gates pass on the unchanged physical source and boundary:

- scalar/compound normalized residuals: `8.773e-6 / 9.514e-6`;
- nonlinear iterations: `10 / 10`;
- maximum boundary mismatch: `0`;
- minimum compound constitutive eigenvalue: `0.01562`;
- scalar/compound median strong-lens physical deflection:
  `3.066 / 2.906 arcsec` before `Dds/Ds`;
- compound/scalar deflection RMS ratio: `0.951217`;
- compound-minus-scalar relative RMS: `0.0577633`; and
- scalar/compound normalized deflection curl: `4.07e-16 / 4.44e-16`.

The compound law clears the preregistered `5%` nonperturbative-response gate,
improving on P0672's `0.168%` response by a factor of about 34. The mean effect
is not extra strength, however: the compound field's RMS is `4.88%` smaller
than scalar AQUAL. Whether its angular redistribution can nevertheless create
the missing lens topology must be decided by the next frozen raw-image audit.

No raw lens score, photon amplitude, gravitational slip, or per-object gravity
parameter was used. P0633 and P0640 remain sealed.

## Reproduction

```powershell
python scripts/run_p0674_spent_rxj2129_compound_3d_field_solve.py
python -m pytest tests/test_p0674_spent_rxj2129_compound_3d_field_solve.py -q
```
