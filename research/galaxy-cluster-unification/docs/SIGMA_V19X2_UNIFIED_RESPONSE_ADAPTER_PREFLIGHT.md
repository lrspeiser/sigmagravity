# Sigma V19X2 unified-response adapter preflight

## Purpose

The original V19X commissioning runner assumes that every one of the 5,082
response cells lives under the original V19W `completed` directory.  The live
archive audit proved that this will not be true: terminal recovery must fill
manifest omissions in the separate V19W4 scratch archive, while valid original
cells retain precedence.

The new `sigma_v19x2_unified_response_adapter.py` removes that implementation
blocker without freezing or running V19X2.  It consumes the V19W4 unified index
and resolves each cell through its explicit `cell_directory`, whether the row
is labeled `base_v19w` or `v19w4_recovery`.

An unfrozen orchestration scaffold now carries those validated records into
the unchanged V19X aperture membership, `combine_spectra`, grouping,
cluster-wide abundance fit and selected-region temperature fit sequence.  Its
validated-cell index preserves the source archive and absolute cell directory,
and the final commissioning authorization remains conjunctive: one failed
integrated or regional fit blocks all 494 production regions.

## Independent checks

The adapter does not merely trust the V19W4 report.  Before any spectrum can be
combined it requires:

1. the terminal V19W4 status and every V19W4 gate to pass;
2. exactly 5,082 cells and 20,328 response products;
3. byte-identical preservation of the base archive;
4. a hash- and size-exact unified index;
5. a cell path inside one of the two frozen archive roots;
6. exact task identity, event counts and cell-report hash;
7. exact name, size and SHA-256 for every PHA, ARF and RMF; and
8. an exact positive source-PHA channel-count audit.

Tests use synthetic base and recovery checkpoints and demonstrate that the
adapter accepts both, rejects path traversal/out-of-root paths, and detects a
changed report or product.

## Freeze boundary

This is an implementation preflight, not the V19X2 protocol.  No configuration
or runner is frozen because the terminal V19W4 report and unified index do not
yet exist.  No spectrum is combined, no temperature is fit, no source-state
quantity is derived, and no lensing, halo or gravity payload is opened.

The scaffold itself refuses any configuration whose freeze state is not
`frozen_after_terminal_v19w4_pass`, and it requires exact hashes for its future
runner and adapter.  The executed report records the complete configuration
hash; the configuration does not attempt the impossible operation of embedding
and validating its own changing file hash. It therefore cannot become
executable by mistaking the current draft for an authorized protocol.

After V19W exits and V19W4 passes, the successor configuration will hash:

- the terminal V19W4 config, runner, report and unified index;
- this adapter;
- the unchanged V19X spectral membership, combination, grouping, plasma-model,
  fitting and gate rules; and
- the two explicit allowed archive roots.

Only that post-recovery configuration may authorize the integrated and
selected-region commissioning fits.

The freeze step is now executable rather than manual. The
`freeze_sigma_v19x2_unified_spectral_combination.py` command refuses an absent
or failed V19W4 report, verifies its index and immutable-base gates, copies the
four scientific sections from V19X without structural change, hashes every
parent, and emits the only configuration state the successor runner accepts.
It deliberately cannot emit a configuration while V19W4 is pending.

## Reproduction

```powershell
python -m pytest tests/test_sigma_v19x2_unified_response_adapter.py tests/test_sigma_v19x2_unified_spectral_combination_scaffold.py tests/test_freeze_sigma_v19x2_unified_spectral_combination.py -q
```
