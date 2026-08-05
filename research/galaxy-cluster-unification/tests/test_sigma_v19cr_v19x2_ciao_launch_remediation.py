import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19cr_v19x2_ciao_launch_remediation.json"
REPORT = ROOT / "results" / "sigma_v19cr_v19x2_ciao_launch_remediation" / "report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v19cr_freezes_environment_only_remediation() -> None:
    payload = load(CONFIG)
    assert payload["environment"]["launch_prefix"][:3] == [
        "/home/henry/miniforge3/bin/conda", "run", "--no-capture-output"
    ]
    assert payload["environment"]["required_executables"] == ["combine_spectra", "dmgroup"]
    assert payload["authorization"]["change_launch_environment_only"]
    assert not payload["authorization"]["change_x2_config_runner_or_scientific_rules"]
    assert not payload["authorization"]["run_v19bs_or_derive_action"]


def test_v19cr_completed_result_passes_every_boundary() -> None:
    payload = load(REPORT)
    assert payload["status"] == "v19x2_ciao_launch_remediation_completed"
    assert all(payload["preflight"].values())
    assert all(payload["environment_probe"]["checks"].values())
    assert payload["preexecution_scratch_audit"]["only_permitted_files"]
    assert not payload["preexecution_scratch_audit"]["combined_or_fit_products"]
    assert all(payload["gate_results"].values())
    assert not any(payload["authorization_boundary"].values())


def test_v19cr_reaches_registered_scientific_or_source_disposition() -> None:
    payload = load(REPORT)
    assert payload["decision"] in {
        "v19x2_valid_scientific_gate_failure_no_full_source_chain",
        "run_frozen_v19bs_disposition_next",
    }
    assert payload["v19x2_report"]["status"] in {
        "unified_spectral_combination_commissioning_gate_failed",
        "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized",
    }
