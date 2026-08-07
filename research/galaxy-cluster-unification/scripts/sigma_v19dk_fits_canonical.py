#!/usr/bin/env python3
"""Canonicalize volatile FITS provenance while retaining exact science arrays."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

_DATASUM_COMMENT = "SIGMA V19DK deterministic data checksum"
_CHECKSUM_COMMENT = "SIGMA V19DK deterministic HDU checksum"


def canonicalize_fits(path: Path, stable_history: str) -> None:
    """Remove run-specific cards and rebuild deterministic FITS checksums."""
    temporary = path.with_name(path.name + ".canonical.tmp")
    if temporary.exists():
        raise RuntimeError(f"canonical temporary file already exists: {temporary}")
    with fits.open(path, memmap=False, lazy_load_hdus=False) as hdus:
        for hdu in hdus:
            header = hdu.header
            for key in ("CHECKSUM", "DATASUM", "DATE", "HISTORY"):
                header.remove(key, remove_all=True, ignore_missing=True)
            header.add_history(stable_history)
            hdu.add_datasum(when=_DATASUM_COMMENT)
            hdu.add_checksum(when=_CHECKSUM_COMMENT, override_datasum=True)
        hdus.writeto(temporary, overwrite=False, checksum=False, output_verify="exception")
    temporary.replace(path)
