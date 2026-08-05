# Sigma V19X3B V19W5 regional-successor preflight

## Decision

A separately named successor is now prepared for all 494 regional spectra. It
will accept only a terminal V19X2 commissioning result that is itself bound to
the CCD7-hardened V19W5 archive. The preflight passes while the original V19W
response process is still active.

V19X3B does not replace or edit the historical V19X3 files. V19X4 hashes those
files, so changing them would retroactively invalidate the preregistered gas
and source-scoring chain. V19X3B instead wraps the byte-exact V19X3 regional
engine and changes only the response-authority handoff.

## Frozen execution chain

V19X3B cannot be frozen or run until all of these events occur in order:

1. the unchanged V19W base process exits and produces its terminal report;
2. V19W5 recovers every missing/invalid response cell and passes its complete
   5,082-cell, 20,328-product double audit;
3. V19X2 is mechanically frozen from that V19W5 report and passes both
   integrated and both target-blind selected-region fits; and
4. the V19X3B freezer verifies the V19X2 config, runner, report, V19W5 parents,
   archive labels, integrated abundances and every inherited scientific rule.

The runner then uses the existing checkpointed V19X3 implementation for 366
Bullet and 128 Abell 2146 regions. Every region is attempted. Every region must
have a finite temperature, normalization and fixed-abundance best fit, and at
least 12 regions per cluster must pass the complete individual quality gate.
Low-quality finite fits remain in the uncertainty posterior; they are never
dropped or selectively refit.

## What changed and what did not

| Item | V19X3B rule |
|---|---|
| Response authority | V19W5 only |
| Recovery label | `v19w5_recovery` only |
| Response cells/products | 5,082 / 20,328 |
| Regional scientific engine | Byte-exact inherited V19X3 runner |
| Plasma model, grouping and fit gates | Unchanged from V19X2/V19X3 |
| Galaxy, lensing, halo or gravity data | Sealed |
| Existing V19X3/V19X4/V19BP files | Preserved byte-for-byte |

After V19X3B passes, the correct next step is to freeze separately named V19X4
and V19BP successors that hash its terminal report. The existing preregistered
files must not be edited in place merely to redirect a parent.

This is an implementation result, not a gas measurement and not evidence for
Sigma Gravity.

## Verification

```powershell
python scripts/check_sigma_v19x3b_v19w5_regional_successor_preflight.py
python -m pytest tests/test_sigma_v19x3b_v19w5_full_regional_spectral_production.py -q
```

The frozen report is
`results/sigma_v19x3b_v19w5_regional_successor_preflight/report.json`.
