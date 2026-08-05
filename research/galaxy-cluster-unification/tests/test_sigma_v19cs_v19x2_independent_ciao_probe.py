import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19cs_v19x2_independent_ciao_probe.json"
REPORT = ROOT / "results" / "sigma_v19cs_v19x2_independent_ciao_probe" / "report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v19cs_freezes_independent_probe_without_scientific_change() -> None:
    payload = load(CONFIG)
    assert payload["authorization"]["remove_invalid_historical_terminal_pass_requirement"]
    assert payload["authorization"]["require_independent_live_environment_probe"]
    assert not payload["authorization"]["change_x2_config_runner_or_scientific_rules"]
    assert not payload["authorization"]["run_v19bs_or_derive_action"]


def test_v19cs_records_the_overbroad_probe_failure() -> None:
    payload = load(REPORT)
    assert payload["status"] == "v19x2_independent_ciao_probe_failed_closed"
    assert "V19CR environment probe failed" in payload["exception"]
    assert not any(payload["authorization_boundary"].values())


def test_v19cs_probe_log_identifies_only_the_unused_module() -> None:
    log = (ROOT / "results" / "sigma_v19cs_v19x2_independent_ciao_probe" / "environment_probe.log").read_text(encoding="utf-8")
    assert "ModuleNotFoundError: No module named 'astropy'" in log
