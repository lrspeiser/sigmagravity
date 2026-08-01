from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_geometric_screen_enforces_three_mode_structural_rank(tmp_path: Path) -> None:
    output = tmp_path / "rank.csv"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_r1_lensing_geometric_rank.py"),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    rows = pd.read_csv(output).set_index("system")
    assert result["summary"]["geometric_prescreen_systems"] == []
    assert result["summary"]["non_disturbed_prescreen_systems"] == []
    assert result["summary"]["full_marginalized_jacobians_completed"] == 0
    assert rows.loc["A383", "strict_inner_image_positions"] == 2
    assert rows.loc["A383", "structural_radial_rank_upper_bound"] == 2
    assert rows.loc["A383", "family_wide_position_dof_after_source_coordinates"] == 4
    assert rows.loc["A2537", "family_wide_position_dof_after_source_coordinates"] == 8
    assert rows.loc["A2537", "structural_radial_rank_upper_bound"] == 1
    assert rows.loc["MS2137", "family_wide_position_dof_after_source_coordinates"] == 4
    assert rows.loc["MS2137", "structural_radial_rank_upper_bound"] == 1
    assert not rows.loc["MACS J0417", "geometric_prescreen_pass"]
