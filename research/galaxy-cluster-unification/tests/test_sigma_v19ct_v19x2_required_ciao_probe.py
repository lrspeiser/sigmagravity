import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19ct_v19x2_required_ciao_probe.json"
REPORT = ROOT / "results" / "sigma_v19ct_v19x2_required_ciao_probe" / "report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v19ct_freezes_only_unused_probe_removal() -> None:
    payload = load(CONFIG)
    correction = payload["effective_config_correction"]
    assert correction["before"] == ["sherpa", "astropy", "numpy"]
    assert correction["after"] == ["sherpa", "numpy"]
    assert not correction["scientific_config_or_code_changed"]
    assert not payload["authorization"]["change_installed_environment"]
    assert not payload["authorization"]["change_x2_config_runner_scientific_rules_or_data"]


def test_v19ct_completed_result_passes_every_boundary() -> None:
    payload = load(REPORT)
    assert payload["status"] == "v19x2_required_ciao_probe_completed"
    assert all(payload["wrapper_preflight"].values())
    assert all(payload["preflight"].values())
    assert all(payload["environment_probe"]["checks"].values())
    assert all(payload["gate_results"].values())
    assert payload["effective_config_changes"] == [{
        "path": "environment.required_python_modules",
        "before": ["sherpa", "astropy", "numpy"],
        "after": ["sherpa", "numpy"],
    }]
    assert not any(payload["authorization_boundary"].values())


def test_v19ct_reaches_registered_scientific_or_source_disposition() -> None:
    payload = load(REPORT)
    assert payload["decision"] in {
        "v19x2_valid_scientific_gate_failure_no_full_source_chain",
        "run_frozen_v19bs_disposition_next",
    }
    assert payload["v19x2_report"]["status"] in {
        "unified_spectral_combination_commissioning_gate_failed",
        "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized",
    }
