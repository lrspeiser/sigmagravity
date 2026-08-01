from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_clash_coverage_target_and_queue_are_frozen() -> None:
    config = json.loads((ROOT / "configs/r1_clash_observable_acquisition_targets.json").read_text())
    assert len(config["target_systems"]) == 20
    assert config["frozen_next_queue"] == ["RXJ1532"]
    assert config["numeric_outcomes"]["current_raw_or_likelihood_catalogs"] == 19
    assert config["numeric_outcomes"]["resolved_catalog_or_shortfall_dispositions"] == 20
    assert config["authorization"]["infer_weyl_response_before_20_of_20"] is False


def test_clash_coverage_ledger_measures_nineteen_with_one_hard_shortfall(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ledger.csv"
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_r1_clash_observable_coverage.py"),
            "--output",
            str(output),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text())
    ledger = pd.read_csv(output)
    assert len(ledger) == 20
    assert report["raw_or_likelihood_catalogs_acquired"] == 19
    assert report["normalized_position_likelihoods_ready"] == 11
    assert report["rerunnable_model_chains_local"] == 6
    assert report["remaining_systems"] == 1
    assert report["frozen_next_queue"] == ["RXJ1532"]
    assert report["primary_source_hard_shortfall_systems"] == ["RXJ1532"]
    assert report["resolved_catalog_or_shortfall_dispositions"] == 20
    assert report["coverage_or_hard_shortfall_gate_passed"] is True
    assert report["coverage_gate_passed"] is False
    assert ledger["gravity_target_used"].eq(False).all()
    assert report["authorization"]["infer_weyl_response"] is False
