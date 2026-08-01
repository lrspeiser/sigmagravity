import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INITIAL_REPORT = ROOT / "results" / "r1_j1402_dinos_environment" / "report.json"
REPORT = ROOT / "results" / "r1_j1402_dinos_environment_corrected" / "report.json"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_historical_environment_and_interface_gate_passes() -> None:
    item = report()
    assert all(item["checks"].values())
    assert item["gate_pass"]
    assert item["decision"] == "environment_interface_gate_pass_authorize_stored_chain_replay"
    assert item["versions"]["python"].startswith("3.10.")
    assert item["versions"]["lenstronomy"] == "1.11.5"
    assert item["versions"]["numpy"] == "1.26.4"
    assert item["versions"]["astropy"] == "5.3.4"


def test_locked_interfaces_use_HDF5_coordinates_masks_and_normalized_PSFs() -> None:
    rows = report()["bands"]
    assert [item["band"] for item in rows] == ["F435W", "F555W", "F814W"]
    assert all(item["Dolphin_passes_HDF5_dictionary_exactly"] for item in rows)
    assert all(item["lenstronomy_image_instantiated"] for item in rows)
    assert all(
        item["operative_PSF_normalized_to_float32_tolerance"] for item in rows
    )
    assert any(not item["operative_PSF_normalized_to_1e_12"] for item in rows)
    assert all(item["operative_mask_retained_pixels"] >= 100 for item in rows)


def test_initial_float64_tolerance_failure_is_preserved() -> None:
    initial = json.loads(INITIAL_REPORT.read_text(encoding="utf-8"))
    assert not initial["gate_pass"]
    assert not initial["checks"]["locked_lenstronomy_normalizes_all_three_PSFs"]
    assert report()["initial_failed_environment_report"] == (
        "results/r1_j1402_dinos_environment/report.json"
    )


def test_environment_authorizes_only_stored_chain_replay() -> None:
    auth = report()["authorization"]
    assert auth["evaluate_only_stored_chain_coordinates"]
    assert not auth["optimize_nonlinear_model"]
    assert not auth["compute_lens_response"]
    assert not auth["reduce_KCWI"]
    assert not auth["infer_gravity_response"]
    assert not auth["authorize_R2"]
