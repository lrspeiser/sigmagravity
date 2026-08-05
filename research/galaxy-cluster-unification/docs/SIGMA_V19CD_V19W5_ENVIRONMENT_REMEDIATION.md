# Sigma V19CD V19W5 environment remediation

## Decision

The protected V19W production process ended normally enough to preserve 4,698
independently valid response cells, leaving 384 of the frozen 5,082-cell
manifest for the already preregistered hardened recovery.  The automatic
V19BR handoff then failed closed before one recovery cell was executed:
`dmkeypar` was absent from the watcher's inherited `PATH` even though the
executable exists in the frozen `sigma-ciao-4.18` environment.

This is an execution-environment failure, not a scientific failure and not a
failed recovery cell.  V19CD permits one manual successor launch under narrow
conditions.  It does not relax V19W5's rule against retrying a failed recovery:
the first launch never reached a recovery attempt, and its partial geometry
workspace and failure report are retained unchanged.

## What is allowed

V19CD verifies the exact hashes and failure signatures, confirms that the old
workspace contains zero completed, failed, partial or quarantined recovery
cells, and requires all CIAO commands plus `numpy` and `pycrates` to resolve
from one declared conda environment.  It then:

1. launches the byte-identical V19W5 runner and config through `conda run`;
2. uses a new scratch directory rather than reusing the partial workspace;
3. requires the full 5,082-cell, 20,328-product double audit and byte-identical
   protected-base checks; and
4. only after that pass, resumes the byte-identical V19BR source-only chain.

No manifest row, mask, response location, background rule, concurrency,
scientific threshold, source invariant or gravity parameter changes.  The
original failed workspace and failure reports are not deleted or overwritten.

## Why this does not open a loophole

A data-quality failure, a failed recovered cell, a failed V19BQ source gate or
a failed V19BS disposition cannot use V19CD.  Its parent hashes and exception
text identify only the missing-`PATH` launch error, and the gate requires zero
cell attempts.  Any other state fails closed.

The downstream chain still opens no lensing, halo map, galaxy rotation target,
action or holdout.  It measures whether one preregistered baryonic source
invariant transfers across Bullet and Abell 2146.  Action derivation remains
forbidden until the separate frozen V19BS rule authorizes it.

## Reproduction

```powershell
python scripts/check_sigma_v19cd_v19w5_environment_remediation.py
python -m pytest tests/test_sigma_v19cd_v19w5_environment_remediation.py -q
```

The execution entry point is:

```powershell
python scripts/run_sigma_v19cd_v19w5_environment_remediation.py --execute
```
