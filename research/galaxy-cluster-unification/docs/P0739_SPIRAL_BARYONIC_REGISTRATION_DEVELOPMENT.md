# P0739: velocity-blind spiral baryonic registration

P0739 asked whether four morphology-diverse development galaxies could be turned into common face-on gas and stellar mass maps without seeing a velocity target or invoking a gravity formula.

The result is intentionally **FAIL**, and that failure is useful. Every leakage, coordinate, map-finiteness, mass-closure, edge-mass, and resolution gate passed. The one failed frozen gate was joint stellar/H I footprint coverage inside the published H I radius:

| Galaxy | Joint footprint inside H I R995 |
|---|---:|
| NGC2403 | 87.15% |
| NGC3198 | 93.10% |
| NGC5055 | 80.24% |
| NGC7793 | 99.34% |

The required minimum was 90%. The SINGS IRAC mosaics do not extend across the full H I disks of NGC2403 and NGC5055. Filling those missing pixels with zero would falsely assert that no stars are present and could bias every later gravity prediction.

No validation or holdout image array was opened. No THINGS moment-1 velocity field, moment-2 dispersion field, or SPARC observed speed entered the extraction. The next step is therefore an observational coverage repair, not a change to the gravity equation or a relaxation of the gate.

Reproduce with:

```powershell
python scripts/run_p0739_spiral_baryonic_registration.py
```

The command exits nonzero because the frozen result is a scientifically meaningful failure.

