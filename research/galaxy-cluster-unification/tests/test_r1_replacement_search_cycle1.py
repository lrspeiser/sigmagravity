from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_cycle1_promotes_macs1206_but_not_abell_s1063(tmp_path: Path) -> None:
    images = tmp_path / "images.csv"
    ledger = tmp_path / "ledger.csv"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_r1_replacement_search_cycle1.py"),
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
    result = json.loads(report.read_text(encoding="utf-8"))
    rows = pd.read_csv(ledger).set_index("system")

    assert result["summary"]["non_disturbed_promotion_systems"] == ["MACS J1206"]
    assert result["summary"]["promotion_gap"] == 1
    assert rows.loc["MACS J1206", "dynamics_bins"] == 6
    assert rows.loc["MACS J1206", "strict_inner_image_positions"] == 11
    assert rows.loc["MACS J1206", "structural_radial_rank_upper_bound"] == 11
    assert rows.loc["MACS J1206", "non_disturbed_structural_promotion"]
    assert rows.loc["Abell S1063", "dynamics_bins"] == 9
    assert rows.loc["Abell S1063", "strict_inner_image_positions"] == 1
    assert rows.loc["Abell S1063", "structural_radial_rank_upper_bound"] == 1
    assert not rows.loc["Abell S1063", "non_disturbed_structural_promotion"]
    assert not rows["full_r1_ready"].any()
