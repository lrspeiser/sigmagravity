import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19cu_astropy_base_environment_remediation.json"
REPORT = ROOT / "results" / "sigma_v19cu_astropy_base_environment_remediation" / "report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v19cu_freezes_minimal_dependency_only() -> None:
    payload = load(CONFIG)
    remediation = payload["environment_remediation"]
    assert remediation["install_spec"] == "astropy-base=8.0.1"
    assert remediation["full_astropy_metapackage_rejected"]
    assert payload["authorization"]["install_pinned_minimal_runtime_dependency"]
    assert not payload["authorization"]["install_full_astropy_metapackage"]
    assert not payload["authorization"]["change_x2_config_runner_scientific_rules_or_data"]


def test_v19cu_completed_result_passes_environment_and_source_gates() -> None:
    payload = load(REPORT)
    assert payload["status"] == "astropy_base_environment_remediated_and_source_chain_disposed"
    assert all(payload["environment_remediation_preflight"].values())
    assert payload["installed_package_gate"]
    assert payload["environment_package_changes"]["astropy-base"]["before"] is None
    assert payload["environment_package_changes"]["astropy-base"]["after"]["version"] == "8.0.1"
    assert all(payload["environment_probe"]["checks"].values())
    assert all(payload["gate_results"].values())
    assert not any(payload["authorization_boundary"].values())


def test_v19cu_reaches_registered_scientific_or_source_disposition() -> None:
    payload = load(REPORT)
    assert payload["decision"] in {
        "v19x2_valid_scientific_gate_failure_no_full_source_chain",
        "run_frozen_v19bs_disposition_next",
    }
    assert payload["v19x2_report"]["status"] in {
        "unified_spectral_combination_commissioning_gate_failed",
        "unified_spectral_combination_commissioning_passed_and_full_regional_fits_authorized",
    }
