import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import acquire_sigma_v19cy_a2319_response_support as acquire


REPORT = (
    acquire.ROOT
    / "results/sigma_v19cy_direct_icm_velocity_evidence/"
    "development_response_support_acquisition.json"
)


def test_frozen_acquisition_scope_and_totals():
    config = acquire.load_and_validate_config(acquire.DEFAULT_CONFIG)
    assert config["protocol_version"].endswith("1.0.1")
    assert config["expected_files"] == 4
    assert config["expected_bytes"] == 1_214_248_655
    assert not config["authorization"]["access_validation_or_holdout_assets"]
    assert not config["authorization"]["fit_A2319_spectrum_or_velocity"]
    assert not config["authorization"]["generate_A2319_response_or_background"]
    assert config["closed_failure_history"]["science_event_or_energy_read"] is False


def test_existing_exact_file_is_reused_and_hashed(tmp_path: Path):
    path = tmp_path / "support.bin"
    path.write_bytes(b"sigma")
    record = acquire.acquire_one({"bytes": 5, "filename": path.name, "url": "https://invalid.test"}, path)
    assert record["reused"]
    assert record["sha256"] == "38de90475bb334fb3dea5d54f250500aba60fe2c6158115d342b06bcb46e39bf"


def test_terminal_acquisition_report_is_exact_and_science_blind():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["protocol_version"].endswith("1.0.1")
    assert report["status"] == (
        "official_provisional_resolve_nxb_support_acquired_hashed_and_structurally_verified"
    )
    assert report["files"] == 4
    assert report["bytes"] == 1_214_248_655
    assert report["science_energy_distribution_read_or_fit"] is False
    assert report["response_or_background_generated"] is False
    assert report["scientific_velocity_fit_performed"] is False
    assert report["validation_or_holdout_accessed"] is False
    records = {record["role"]: record for record in report["records"]}
    assert records["resolve_nxb_ehk"]["sha256"] == (
        "fa91c6e543c2979b2afd9a4cfdca768524767dd3cb9b71ac3c66647b8db81a23"
    )
    assert records["resolve_nxb_events"]["sha256"] == (
        "4c1611170b7b666ef183a224dc727c594a7988b820357d55978d77f6d02ac8e9"
    )
    assert records["resolve_nxb_diagonal_response"]["sha256"] == (
        "a21a12cf1a1c87f148178a0a6a01e1a7e16341ea2f887e9b8344984560dfd9ac"
    )
    assert records["resolve_nxb_empirical_model"]["sha256"] == (
        "4521d7249196f8da18191f5fb98837063f01d95db305de90cff0d72826a8f6de"
    )


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
