import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import acquire_sigma_v19cy_a2319_response_support as acquire


def test_frozen_acquisition_scope_and_totals():
    config = acquire.load_and_validate_config(acquire.DEFAULT_CONFIG)
    assert config["expected_files"] == 4
    assert config["expected_bytes"] == 1_214_248_655
    assert not config["authorization"]["access_validation_or_holdout_assets"]
    assert not config["authorization"]["fit_A2319_spectrum_or_velocity"]
    assert not config["authorization"]["generate_A2319_response_or_background"]


def test_existing_exact_file_is_reused_and_hashed(tmp_path: Path):
    path = tmp_path / "support.bin"
    path.write_bytes(b"sigma")
    record = acquire.acquire_one({"bytes": 5, "filename": path.name, "url": "https://invalid.test"}, path)
    assert record["reused"]
    assert record["sha256"] == "38de90475bb334fb3dea5d54f250500aba60fe2c6158115d342b06bcb46e39bf"


def test_fits_structure_accepts_declared_extension_and_columns(tmp_path: Path):
    path = tmp_path / "ehk.fits"
    columns = [
        fits.Column(name="TIME", format="D", array=np.array([1.0])),
        fits.Column(name="CORTIME", format="E", array=np.array([8.0], dtype=np.float32)),
        fits.Column(name="T_SAA_SXS", format="E", array=np.array([1.0], dtype=np.float32)),
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="EHK")]).writeto(path)
    result = acquire.fits_structure(
        path,
        {"extension": "EHK", "required_columns": ["TIME", "CORTIME", "T_SAA_SXS"]},
    )
    assert result["rows"] == 1
    assert result["audited_extension"] == "EHK"


def test_fits_structure_rejects_missing_column(tmp_path: Path):
    path = tmp_path / "bad.fits"
    columns = [fits.Column(name="TIME", format="D", array=np.array([1.0]))]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="EHK")]).writeto(path)
    try:
        acquire.fits_structure(path, {"extension": "EHK", "required_columns": ["TIME", "CORTIME"]})
    except RuntimeError as error:
        assert "CORTIME" in str(error)
    else:
        raise AssertionError("missing required FITS column was accepted")
