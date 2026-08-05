import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19cq_v19x2_recovery_root_remediation.json"
REPORT = ROOT / "results" / "sigma_v19cq_v19x2_recovery_root_remediation" / "report.json"


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_v19cq_freezes_one_index_derived_path_change() -> None:
    payload = config()
    correction = payload["config_correction"]
    assert correction["json_path_changed"] == "execution.response_archives.v19w5_recovery"
    assert correction["value_before"] == "/home/henry/sigma-v19w5-response-recovery/v100"
    assert correction["value_after"] == "/home/henry/sigma-v19cd-v19w5-response-recovery/v100"
    assert payload["failure_parents"]["v19w5_unified_index"]["required_archive_counts"] == {
        "base_v19w": 4698,
        "v19w5_recovery": 384,
    }
    assert not payload["authorization"]["change_scientific_section_or_value"]
    assert not payload["authorization"]["run_v19bs_or_derive_action"]


def test_v19cq_completed_result_is_strictly_bounded() -> None:
    payload = report()
    assert payload["status"] == "v19x2_recovery_root_remediation_completed"
    assert payload["changed_paths"] == [{
        "path": "execution.response_archives.v19w5_recovery",
        "before": "/home/henry/sigma-v19w5-response-recovery/v100",
        "after": "/home/henry/sigma-v19cd-v19w5-response-recovery/v100",
    }]
    assert payload["scientific_sections_unchanged"]
    assert payload["unified_index_audit"]["recovery_rows"] == 384
    assert payload["unified_index_audit"]["recovery_roots"] == [
        "/home/henry/sigma-v19cd-v19w5-response-recovery/v100"
    ]
    assert not payload["unified_index_audit"]["invalid_recovery_directories"]
    assert all(payload["gate_results"].values())
    assert not any(payload["authorization_boundary"].values())


def test_v19cq_reaches_a_registered_terminal_disposition() -> None:
    payload = report()
    assert payload["decision"] in {
        "v19x2_valid_scientific_gate_failure_no_full_source_chain",
        "run_frozen_v19bs_disposition_next",
    }
    assert payload["v19x2_report"]["status"] in {
        "unified_spectral_combination_commissioning_gate_failed",
        "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized",
    }
