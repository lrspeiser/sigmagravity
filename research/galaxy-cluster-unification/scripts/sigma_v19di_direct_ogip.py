#!/usr/bin/env python3
"""Write direct-array Sigma V19DI responses as OGIP ARF/RMF products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


def _copy_nonstructural_header(source: fits.Header, target: fits.Header) -> None:
    structural_prefixes = (
        "NAXIS",
        "TTYPE",
        "TFORM",
        "TUNIT",
        "TDIM",
        "TNULL",
        "TSCAL",
        "TZERO",
        "TLMIN",
        "TLMAX",
    )
    structural_names = {
        "XTENSION",
        "BITPIX",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "THEAP",
        "CHECKSUM",
        "DATASUM",
        "NUMGRP",
        "NUMELT",
    }
    for card in source.cards:
        key = card.keyword
        if not key or key in structural_names or key.startswith(structural_prefixes):
            continue
        if key in {"COMMENT", "HISTORY", "CONTINUE"}:
            continue
        try:
            target[key] = (card.value, card.comment)
        except (TypeError, ValueError):
            continue


def _sparse_row(
    row: np.ndarray, first_channel: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained = np.flatnonzero(np.asarray(row) >= threshold)
    if retained.size == 0:
        return (
            np.asarray([], dtype=np.int16),
            np.asarray([], dtype=np.int16),
            np.asarray([], dtype=np.float32),
        )
    breaks = np.flatnonzero(np.diff(retained) > 1)
    group_starts = np.concatenate(([0], breaks + 1))
    group_stops = np.concatenate((breaks + 1, [retained.size]))
    starts = retained[group_starts] + first_channel
    lengths = retained[group_stops - 1] - retained[group_starts] + 1
    values = np.concatenate(
        [row[start : start + length] for start, length in zip(starts - first_channel, lengths, strict=True)]
    )
    return (
        np.asarray(starts, dtype=np.int16),
        np.asarray(lengths, dtype=np.int16),
        np.asarray(values, dtype=np.float32),
    )


def write_arf(
    template: Path,
    output: Path,
    energy_lo: np.ndarray,
    energy_hi: np.ndarray,
    specresp: np.ndarray,
    exposure: float,
) -> None:
    with fits.open(template, memmap=False) as hdus:
        primary = hdus[0].copy()
        source_header = hdus["SPECRESP"].header.copy()
    columns = [
        fits.Column(name="ENERG_LO", format="E", unit="keV", array=np.asarray(energy_lo, dtype=np.float32)),
        fits.Column(name="ENERG_HI", format="E", unit="keV", array=np.asarray(energy_hi, dtype=np.float32)),
        fits.Column(name="SPECRESP", format="E", unit="cm**2", array=np.asarray(specresp, dtype=np.float32)),
    ]
    response = fits.BinTableHDU.from_columns(columns, name="SPECRESP")
    _copy_nonstructural_header(source_header, response.header)
    response.header["EXPOSURE"] = (float(exposure), "Summed source PHA exposure")
    response.header["ONTIME"] = (float(exposure), "Summed source PHA exposure")
    response.header["LIVETIME"] = (float(exposure), "Summed source PHA exposure")
    response.header["HDUCLASS"] = "OGIP"
    response.header["HDUCLAS1"] = "RESPONSE"
    response.header["HDUCLAS2"] = "SPECRESP"
    response.header["HDUVERS"] = "1.1.0"
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, response]).writeto(output, overwrite=False, checksum=True)


def write_rmf(
    template: Path,
    output: Path,
    energy_lo: np.ndarray,
    energy_hi: np.ndarray,
    matrix: np.ndarray,
    exposure: float,
    threshold: float,
) -> dict[str, Any]:
    with fits.open(template, memmap=False) as hdus:
        primary = hdus[0].copy()
        matrix_header = hdus["MATRIX"].header.copy()
        ebounds = hdus["EBOUNDS"].copy()
    first_channel = int(matrix_header.get("TLMIN4", 1))
    starts: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for row in np.asarray(matrix, dtype=np.float64):
        row_starts, row_lengths, row_values = _sparse_row(
            row, first_channel, threshold
        )
        starts.append(row_starts)
        lengths.append(row_lengths)
        values.append(row_values)
    n_groups = np.asarray([len(item) for item in starts], dtype=np.int16)
    columns = [
        fits.Column(name="ENERG_LO", format="E", unit="keV", array=np.asarray(energy_lo, dtype=np.float32)),
        fits.Column(name="ENERG_HI", format="E", unit="keV", array=np.asarray(energy_hi, dtype=np.float32)),
        fits.Column(name="N_GRP", format="I", array=n_groups),
        fits.Column(name="F_CHAN", format="PI()", array=np.asarray(starts, dtype=object)),
        fits.Column(name="N_CHAN", format="PI()", array=np.asarray(lengths, dtype=object)),
        fits.Column(name="MATRIX", format="PE()", array=np.asarray(values, dtype=object)),
    ]
    response = fits.BinTableHDU.from_columns(columns, name="MATRIX")
    _copy_nonstructural_header(matrix_header, response.header)
    response.header["DETCHANS"] = int(matrix.shape[1])
    response.header["TLMIN4"] = first_channel
    response.header["TLMAX4"] = first_channel + int(matrix.shape[1]) - 1
    response.header["EXPOSURE"] = (float(exposure), "Summed source PHA exposure")
    response.header["HDUCLASS"] = "OGIP"
    response.header["HDUCLAS1"] = "RESPONSE"
    response.header["HDUCLAS2"] = "RSP_MATRIX"
    response.header["HDUVERS"] = "1.3.0"
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, response, ebounds]).writeto(
        output, overwrite=False, checksum=True
    )
    return {
        "energy_rows": int(matrix.shape[0]),
        "channels": int(matrix.shape[1]),
        "groups": int(np.sum(n_groups)),
        "retained_matrix_elements": int(sum(len(item) for item in values)),
        "threshold": float(threshold),
        "first_channel": first_channel,
    }


def link_pha(source: Path, background: Path, arf: Path, rmf: Path) -> None:
    with fits.open(source, mode="update", memmap=False) as hdus:
        header = hdus["SPECTRUM"].header
        header["BACKFILE"] = background.name
        header["ANCRFILE"] = arf.name
        header["RESPFILE"] = rmf.name
        for hdu in hdus:
            hdu.add_checksum(override_datasum=True)
        hdus.flush(output_verify="exception")


def validate_written_response(
    arf: Path,
    rmf: Path,
    expected_energy_rows: int,
    expected_channels: int,
) -> dict[str, Any]:
    with fits.open(arf, memmap=False, checksum=True) as hdus:
        arf_rows = len(hdus["SPECRESP"].data)
        arf_finite = bool(
            np.all(np.isfinite(np.asarray(hdus["SPECRESP"].data["SPECRESP"])))
        )
    with fits.open(rmf, memmap=False, checksum=True) as hdus:
        matrix_rows = len(hdus["MATRIX"].data)
        channels = int(hdus["MATRIX"].header["DETCHANS"])
        ebounds_rows = len(hdus["EBOUNDS"].data)
    gates = {
        "arf_rows_exact": arf_rows == expected_energy_rows,
        "arf_finite": arf_finite,
        "rmf_rows_exact": matrix_rows == expected_energy_rows,
        "rmf_channels_exact": channels == expected_channels,
        "ebounds_rows_exact": ebounds_rows == expected_channels,
    }
    return {
        "arf_rows": arf_rows,
        "rmf_rows": matrix_rows,
        "channels": channels,
        "ebounds_rows": ebounds_rows,
        "gates": gates,
        "passed": all(gates.values()),
    }
