from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_cycle1_extensions_exhaust_sand_without_false_promotion(tmp_path: Path) -> None:
    images = tmp_path / "images.csv"
    ledger = tmp_path / "ledger.csv"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_r1_replacement_search_cycle1_extensions.py"),
            "--image-output",
            str(images),
            "--ledger-output",
            str(ledger),
            "--report-output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    image_rows = pd.read_csv(images)
    rows = pd.read_csv(ledger).set_index("system")
    result = json.loads(report.read_text(encoding="utf-8"))

    assert len(image_rows) == 237
    assert rows.loc["MACS J0416", "resolved_bcg_dynamics_bins"] == 0
    assert rows.loc["RXJ 1133", "resolved_bcg_dynamics_bins"] == 3
    assert rows.loc["RXJ 1133", "critical_radius_constraints"] == 2
    assert rows.loc["Abell 1201", "resolved_bcg_dynamics_bins"] == 8
    assert rows.loc["Abell 1201", "critical_radius_constraints"] == 1
    assert not rows["structural_promotion_pass"].any()
    assert result["summary"]["cumulative_unique_hosts_source_screened"] == 16
    assert result["summary"]["remaining_hosts_to_30_target"] == 14
    assert result["decision"]["cycle_1_complete"]
