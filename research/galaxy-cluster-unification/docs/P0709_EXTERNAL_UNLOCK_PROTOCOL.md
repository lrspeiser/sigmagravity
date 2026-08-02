# P0709: one-time external unlock protocol

P0708 committed predictions for 13 LITTLE THINGS galaxies and four RELICS
clusters without reading their kinematic or strong-lensing outcomes. P0709
specifies the irreversible next operation: one external evaluation of those
predictions, with no formula or gravity-parameter changes.

The protocol fixes the exact NRAO moment-1 and moment-2 products, the immutable
Iorio et al. source package containing the circular-speed tables, both raw
strong-lensing constraint containers, and the archived compact-halo
comparators. The same glafic v2 release is the primary compact-halo comparator
for all four clusters; a Zitrin release is only a sensitivity check and may not
be selected per cluster from its score.

The galaxy score uses published circular speeds and a separate resolved
velocity-field score. Predictions are projected with the already frozen
photometric geometry and convolved with the measured radio beam. No distance,
inclination, PA, stellar mass-to-light ratio, or gravity parameter may be
kinematically refit. Morphology bins are split at medians derived only from the
open baryonic maps; those medians are materialized in the unlock manifest.

The cluster score fits only one ordinary source-plane position per family.
Lens centers, shear, ellipticity, mass sheets, radial amplitudes, and all
gravity parameters stay fixed. If public catalogs do not contain the arc
orientation or parity required for an independent critical-curve observation,
the result must say that the gate is not independently observable. A published
lens-model curve may not be silently relabeled as raw data.

The workflow deliberately uses two commits. First commit this protocol and its
freezing script. Then run the script, commit and push the generated
`results/p0633_external_validation/unlock_manifest.json`, and only after that
download or parse any target outcome. Any later formula change is exploratory
work on a spent sample, not P0633 validation.
