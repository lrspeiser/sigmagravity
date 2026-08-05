from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19bo_gas_source_stream_preflight.json"
REPORT = ROOT / "results" / "sigma_v19bo_gas_source_stream_preflight" / "report.json"
RUNNER = ROOT / "scripts" / "check_sigma_v19bo_gas_source_stream_preflight.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19bo", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_passes_with_terminal_and_target_data_sealed() -> None:
    report = load_runner().build_report(CONFIG)
    assert all(report["gates"].values())
    assert not report["terminal_v19x4_opened"]
    assert not report["observed_source_score_computed"]
    assert not report["lensing_halo_action_or_gravity_payload_opened"]


def test_frozen_report_matches_current_stream() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"] == "passed_gas_source_stream_preflight"
    assert all(report["gates"].values())
