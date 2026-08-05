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


def test_v19ct_records_the_real_astropy_import_dependency() -> None:
    payload = load(REPORT)
    assert payload["status"] == "v19x2_required_ciao_probe_failed_closed"
    assert "import_closure_exact_and_astropy_absent': False" in payload["exception"]
    assert not any(payload["authorization_boundary"].values())


def test_v19ct_failure_precedes_environment_or_scientific_change() -> None:
    payload = load(REPORT)
    assert "v19cs_failure_exact': True" in payload["exception"]
    assert "authorization_probe_only': True" in payload["exception"]
