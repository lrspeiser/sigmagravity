# P0669 amplitude-multipole 3D results

## Frozen result: pass

All 17 progression gates pass on 13 registered galaxy baryon maps and four
registered cluster baryon maps:

- nominal galaxy median `sigma`: `5.22480e-5`;
- nominal cluster median `sigma`: `0.00305689`;
- nominal cluster/galaxy ratio: `58.507x`;
- weakest cluster/galaxy ratio over the three mass scenarios: `53.290x`;
- nominal galaxy/cluster amplitude gates: `0.117863 / 0.186721`;
- maximum surface-to-volume mass error: `1.72e-15`; and
- conservative minimum constitutive eigenvalue bound: `8.78e-7`.

No per-object gravity parameter or new universal constant was introduced. No
spent-cluster lensing outcome, sealed galaxy kinematic outcome, or sealed
cluster lensing constraint was opened.

## Interpretation

The result supports a narrow statement: interpreting the measured multipole
quantity as a routed-power fraction, and applying its square root to a field
amplitude, creates a stable coefficient-level cluster channel without making
the typical registered galaxy coefficient large. This is substantially more
specific than switching between MOND/RAR and another law by object class: the
same equation reads only baryonic component centroids, quadrupoles, local
accelerations, and tidal lengths.

It does **not** show that the field has the correct magnitude, direction, image
topology, or lensing normalization. Those require the nonlinear tensor field,
zero-slip line-of-sight deflection, and a raw spent-cluster image test.

## Reproduction

```powershell
python scripts/run_p0669_amplitude_multipole_3d_activation.py
python -m pytest tests/test_amplitude_activation_3d.py \
  tests/test_p0669_amplitude_multipole_3d_activation.py -q
```
