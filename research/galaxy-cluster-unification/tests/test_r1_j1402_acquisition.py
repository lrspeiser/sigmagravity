import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "r1_j1402_acquisition" / "report.json"


def load_report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_locked_acquisition_is_complete_and_checksum_verified() -> None:
    report = load_report()
    assert report["summary"]["receipt_count"] == 46
    assert report["summary"]["verified_bytes"] == 1_672_342_355
    assert report["summary"]["checksum_failures"] == []
    assert all(report["checks"].values())
    assert report["gate_pass"]


def test_dinos_coordinate_declaration_is_preserved_not_simplified() -> None:
    dinos = load_report()["Dinos"]
    assert dinos["bands"] == ["F435W", "F555W", "F814W"]
    assert dinos["image_sizes_pixels"] == [120, 140, 140]
    assert len(dinos["transform_matrices"]) == 3
    assert all(abs(value) > 1e-8 for value in dinos["transform_determinants"])
    assert all(0.049 < value < 0.051 for value in dinos["transform_pixel_scales_arcsec"])
    assert dinos["scalar_pixel_size_field"] == 0.04
    assert "not silently substituted" in dinos["coordinate_note"]
    assert dinos["lens_models"] == ["PEMD", "SHEAR_GAMMA_PSI"]
    assert dinos["shapelet_orders"] == [6, 6, 6]


def test_KCWI_raw_science_headers_match_the_frozen_setup() -> None:
    report = load_report()
    science = report["KCWI"]["science_headers"]
    assert len(science) == 4
    assert sum(float(item["exposure_seconds"]) for item in science) == 7200.0
    assert {item["camera"] for item in science} == {"BLUE"}
    assert {item["grating"] for item in science} == {"BL"}
    assert {item["filter"] for item in science} == {"KBlue"}
    assert {item["slicer"] for item in science} == {"Small"}
    assert {item["ampmode"] for item in science} == {"TUP"}
    assert not report["authorization"]["fit_lens_response"]
    assert not report["authorization"]["reduce_KCWI"]
    assert not report["authorization"]["infer_gravity_response"]
