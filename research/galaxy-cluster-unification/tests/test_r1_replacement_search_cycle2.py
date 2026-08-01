from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_cycle2_group_bridge_fails_overlap_without_false_promotion(tmp_path: Path) -> None:
    images = tmp_path / "images.csv"
    ledger = tmp_path / "ledger.csv"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_r1_replacement_search_cycle2.py"),
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
    row = rows.loc["SDSS J0100+1818"]
    rxj = rows.loc["RX J2129"]
    result = json.loads(report.read_text(encoding="utf-8"))

    assert len(image_rows.loc[image_rows["system"] == "SDSS J0100+1818"]) == 18
    assert len(image_rows.loc[image_rows["system"] == "RX J2129"]) == 25
    assert len(
        image_rows.loc[
            (image_rows["system"] == "SDSS J0100+1818")
            & image_rows["inside_dynamics_support"]
        ]
    ) == 1
    assert row["resolved_bgg_dynamics_bins"] == 6
    assert row["strict_inner_image_positions"] == 1
    assert row["structural_radial_rank_upper_bound"] == 1
    assert not row["structural_promotion_pass"]
    assert not row["non_disturbed_structural_promotion"]
    assert rxj["resolved_bgg_dynamics_bins"] == 4
    assert rxj["strict_inner_image_positions"] == 3
    assert rxj["inner_source_families"] == 3
    assert rxj["family_wide_position_dof_after_source_coordinates"] == 12
    assert rxj["structural_promotion_pass"]
    assert rxj["non_disturbed_structural_promotion"]
    assert not rxj["full_r1_ready"]
    assert result["summary"]["cumulative_unique_hosts_source_screened"] == 18
    assert result["summary"]["remaining_hosts_to_30_target"] == 12
    assert result["summary"]["cumulative_non_disturbed_structural_promotions"] == 2
