from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_dynamics_audit_separates_published_likelihood_from_raw_cubes(
    tmp_path: Path,
) -> None:
    availability_path = tmp_path / "availability.csv"
    products_path = tmp_path / "products.csv"
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_r1_dynamics_public_data.py"),
            "--availability-output",
            str(availability_path),
            "--product-output",
            str(products_path),
            "--report-output",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    availability = pd.read_csv(availability_path).set_index("system")
    products = pd.read_csv(products_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert availability.loc["MACS J1206", "profile_points"] == 6
    assert availability.loc["Abell S1063", "profile_points"] == 9
    assert availability["source_package_present"].all()
    assert not availability["published_numerical_profile_table"].any()
    assert not availability["published_measurement_covariance"].any()
    assert availability["all_required_level3_cubes_public"].all()
    assert not availability["published_likelihood_ready"].any()
    assert not availability["full_r1_ready"].any()
    assert set(products["calibration_level"]) == {3}
    assert set(products["dp_id"]) == {
        "ADP.2017-06-19T11:32:26.411",
        "ADP.2017-03-23T15:58:03.937",
        "ADP.2017-03-28T12:46:01.331",
    }
    assert report["summary"]["systems_with_public_level3_raw_cubes"] == 2
    assert report["summary"]["systems_published_likelihood_ready"] == 0
