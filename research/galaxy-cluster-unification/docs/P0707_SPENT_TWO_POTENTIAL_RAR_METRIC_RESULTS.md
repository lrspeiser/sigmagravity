# P0707: spent two-potential RAR metric joint screen

P0707 is the first candidate in the current branch to pass every frozen spent
galaxy, raw-lensing, topology, numerical, parameter-accounting, and Solar gate.

| Domain | Result | Comparator |
|---|---:|---:|
| DDO154 time-potential RMSE | 1.884 km/s | algebraic MOND 2.916 km/s |
| DDO154 weighted RMSE | 1.381 km/s | algebraic MOND 1.226 km/s |
| RX J2129 training / held-out roots | 15/15 / 7/7 | complete |
| RX J2129 training / held-out RMS | 0.601 / 2.670 arcsec | compact halo 2.536 arcsec |
| RX J2129 held-out / compact-halo ratio | 1.053 | gate <=1.25 |
| missing / observable-surplus families | 0 / 2 | gates 0 / <=2 |
| parity-diverse / critical families | 7/7 / 7/7 | complete |
| nuisance parameters at bounds | 0 | gate 0 |

The ordinary DDO154 RMSE is 35.4% below the algebraic-MOND comparator. Its
weighted RMSE is 12.7% higher, inside the frozen 20% spent-development gate.
The cluster held-out positional error is 5.3% above the object-specific compact
halo while using zero per-cluster gravity settings.

The important conceptual change is that no galaxy/cluster label selects a
branch. Matter and photons follow the same metric but probe different canonical
combinations of its two potentials. This is standard weak-field metric
bookkeeping; the proposed baryonic equations for those potentials are new and
post hoc.

P0707 cannot yet support a scientific claim. DDO154 and RX J2129 were fully
spent while the formula was developed. P0633 galaxy kinematics and P0640 raw
lensing constraints remain sealed. Robustness and a hash-locked prediction
manifest must be completed before the one permitted external evaluation.

