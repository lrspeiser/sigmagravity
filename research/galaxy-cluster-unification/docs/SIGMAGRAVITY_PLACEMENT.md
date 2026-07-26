# SigmaGravity placement and origin

Status: placed in the SigmaGravity repository on 2026-07-26.

## Location

This research angle is contained in:

```text
research/galaxy-cluster-unification/
```

It is a self-contained Python subproject. Its source package, tests,
configurations, imported data, documentation, and results remain below that
directory. It does not import or modify SigmaGravity's main theory code by
default.

## Origin

The directory was copied from the standalone Void Screening Lab at Git commit:

```text
32aae83 Document joint galaxy cluster bridge results
```

The receiving SigmaGravity checkout was at:

```text
5d1597f Sync final PDFs with author-section removal
```

All 379 files tracked by the standalone project were copied, together with its
ignored generated result directories. Machine-specific `.venv`, Git metadata,
and Python/test/lint caches were not copied. They are disposable runtime state,
not research inputs, and the virtual environment contains absolute paths that
would not survive relocation reliably.

## Separation rule

The folder is organizationally inside SigmaGravity but scientifically isolated:

- imported data keep their original provenance and hashes;
- formulas and decision rules remain in the subproject's `configs/` and `docs/`;
- results are generated and interpreted inside this folder;
- no result should be treated as support for SigmaGravity's main theory unless
  a separate, explicit comparison is preregistered.

The original standalone checkout is retained as a recovery copy. New work for
this research angle should use this SigmaGravity subfolder as the canonical
location.
