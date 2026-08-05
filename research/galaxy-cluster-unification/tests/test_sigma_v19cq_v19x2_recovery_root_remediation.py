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


def test_v19cq_records_strictly_bounded_root_repair() -> None:
    payload = report()
    assert payload["status"] == "v19x2_recovery_root_remediation_failed_closed"
    assert "v19x2_reaches_registered_scientific_disposition': False" in payload["exception"]
    assert not any(payload["authorization_boundary"].values())


def test_v19cq_reaches_next_precombination_environment_failure() -> None:
    x2 = json.loads(
        (ROOT / "results" / "sigma_v19x2_unified_spectral_combination_commissioning" / "report.json").read_text(encoding="utf-8")
    )
    assert x2["status"] == "unified_spectral_combination_commissioning_execution_failed"
    assert x2["execution_exception"] == "FileNotFoundError: [Errno 2] No such file or directory: 'combine_spectra'"
    assert not x2["gates"]["execution_completed"]
