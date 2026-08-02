# P0702: mass-planarity controller math audit

P0702 tests one geometric fact that was missing from the earlier field laws:
a galaxy disk is a thin sheet in three dimensions, while cluster baryons occupy
all three directions. The controller uses the ordered eigenvalues of the
mass-weighted spatial covariance tensor,

`P = (1 - lambda_1/lambda_2) (1 - lambda_1/lambda_3)`.

It approaches one only if one direction is thin relative to both other
directions. A filament has two thin directions and therefore receives a value
near zero; a sphere also receives zero. This avoids classifying every elongated
object as a disk.

## Result

Every frozen mathematical gate passed. Synthetic sheet, filament, and spherical
maps produced planarity values of `0.991831`, `0`, and `4.93e-32`. Axis
permutation changed the value by zero at numerical precision, and translating
the finite synthetic map changed it by `6.82e-6`.

The already-spent input geometry gives `P=0.998220` for DDO154 and `P=0.092736`
for RX J2129. These are mechanism-development diagnostics, not validation. The
formula was proposed after the P0697 and P0699 outcomes were known, and both
planarity values were inspected before this audit was frozen.

The next source equation is therefore fixed as

`S = P S_coh + (1-P)[C S_coh + (1-C) S_local]`,

or equivalently `C_eff=P+(1-P)C`. It has no new constant, fitted threshold,
exponent, length scale, or per-object gravity parameter. P0633 galaxy kinematics
and P0640 raw lensing constraints remain sealed.
