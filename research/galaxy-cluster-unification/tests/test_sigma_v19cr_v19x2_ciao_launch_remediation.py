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


def test_v19cr_records_the_historical_parent_mismatch() -> None:
    payload = load(REPORT)
    assert payload["status"] == "v19x2_ciao_launch_remediation_failed_closed"
    assert "prior_environment_contract_exact': False" in payload["exception"]
    assert not any(payload["authorization_boundary"].values())


def test_v19cr_failure_precedes_environment_probe_and_x2_rerun() -> None:
    payload = load(REPORT)
    assert "scratch_precedes_combination_and_fit': True" in payload["exception"]
    assert "v19x2_environment_failure_exact': True" in payload["exception"]
