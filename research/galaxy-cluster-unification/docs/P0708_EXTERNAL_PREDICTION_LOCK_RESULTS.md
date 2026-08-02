# P0708: outcome-blind external prediction lock

P0708 generated the complete frozen prediction set for the untouched P0633
sample without opening a velocity, dispersion, image-coordinate, family, or
lens-model target.

The lock contains all 13 LITTLE THINGS galaxies and all four RELICS clusters.
Each galaxy has P0707 time-potential, Newtonian, AQUAL, and QUMOND line-of-sight
velocity fields derived from the same registered gas-plus-stellar map. Each
cluster has physical `Dds/Ds=1` deflection maps for the P0707 Weyl potential,
baryon-only GR, AQUAL, and QUMOND, all derived from the same registered stellar
and gas map.

Every frozen gate passed:

- 17 of 17 systems have complete finite prediction hashes;
- maximum AQUAL residual: `9.90e-6`;
- maximum QUMOND residual: `1.50e-13`;
- maximum candidate 65-to-33-cell relative RMS: `0.0591`;
- maximum mass-conservation error: `4.44e-16`;
- per-object gravity, fitted slip, and fitted photon parameters: zero.

The canonical JSON encoding of the universal parameter vector has SHA-256
`bf3f12d6b32ee3f1b0e3bf48a9603c4aafbcd34b2cbdd3de021d689514099a15`.
The candidate is authorized for one external unlock only after this lock and a
separate unlock manifest have both been committed and pushed.

The generated products are also the reference payloads for the future public
simulator API: immutable inputs, typed model identifiers, field arrays,
resolution diagnostics, and content hashes are kept separate from later score
and leaderboard records.

