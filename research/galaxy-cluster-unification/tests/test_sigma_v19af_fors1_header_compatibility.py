from __future__ import annotations

import sys
from pathlib import Path

import pytest
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_sigma_v19af_fors1_header_compatibility import (
    expected_primary_size,
    header_cards,
    parse_primary_header,
    read_primary_header_bytes,
    windows_path_to_wsl,
)


def synthetic_header() -> bytes:
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = 16
    header["NAXIS"] = 2
    header["NAXIS1"] = 10
    header["NAXIS2"] = 20
    return header.tostring(endcard=True, padding=True).encode("ascii")


def test_primary_header_reader_stops_at_end_without_reading_image(tmp_path: Path) -> None:
    raw = synthetic_header()
    path = tmp_path / "synthetic.fits"
    path.write_bytes(raw + b"\x01" * 2880)
    recovered = read_primary_header_bytes(path)
    assert recovered == raw
    header = parse_primary_header(recovered)
    assert (header["NAXIS1"], header["NAXIS2"]) == (10, 20)
    assert header_cards(recovered)[-1][:8].strip() == "END"
    assert expected_primary_size(header, len(recovered)) == len(recovered) + 2880


def test_truncated_header_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "truncated.fits"
    path.write_bytes(b"SIMPLE  =                    T")
    with pytest.raises(RuntimeError):
        read_primary_header_bytes(path)


def test_windows_drive_path_is_encoded_for_wsl() -> None:
    path = Path(r"C:\research data\frame.fits.Z")
    assert windows_path_to_wsl(path).startswith("/mnt/c/")
    assert windows_path_to_wsl(path).endswith("research data/frame.fits.Z")
