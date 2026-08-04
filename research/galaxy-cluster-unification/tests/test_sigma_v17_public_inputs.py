from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plckg287_public_spectroscopy_is_complete_and_provenanced() -> None:
    raw = ROOT / "data" / "raw" / "sigma_v17_dynamical_stress" / "PLCKG287"
    assert (raw / "ReadMe").stat().st_size == 13185
    assert (raw / "tabled1.dat").stat().st_size == 7038
    assert (raw / "tablee1.dat").stat().st_size == 44660
    assert sha256(raw / "ReadMe") == (
        "dd997117959be829c7bf9188326e25d6c8f0240122aef1184018bdc45e5395d0"
    )
    assert sha256(raw / "tabled1.dat") == (
        "6fcfaa93d0418414f1b36c79c03f0bc5c9e9e897ebebe283642b4754fe362947"
    )
    assert sha256(raw / "tablee1.dat") == (
        "b1751cbc6d6d23575497a0dcbf692c59025ff05e780f803d5b85afafe63f0b38"
    )
    assert len((raw / "tabled1.dat").read_text(encoding="ascii").splitlines()) == 153
    assert len((raw / "tablee1.dat").read_text(encoding="ascii").splitlines()) == 639

    report_path = ROOT / "results" / "sigma_v17_public_data_acquisition" / "provenance.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["lensing_target_opened"] is False
    assert report["formula_selection_authorized"] is False
    assert report["selected_member_summary"] == {
        "photometric_members": 24,
        "selected_members": 153,
        "spectroscopic_members": 129,
    }
    assert report["full_spectroscopy_summary"]["secure_quality_3"] == 402
    assert report["full_spectroscopy_summary"]["spectroscopic_rows"] == 639


def test_plckg287_velocity_table_uses_only_measured_member_redshifts() -> None:
    path = (
        ROOT
        / "results"
        / "sigma_v17_public_data_acquisition"
        / "plckg287_selected_spectroscopic_members.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 129
    assert all(float(row["spectroscopic_redshift"]) > 0.0 for row in rows)
    velocities = np.array([float(row["rest_frame_velocity_km_s"]) for row in rows])
    assert abs(float(np.median(velocities))) < 1.0e-10
    assert float(np.std(velocities, ddof=1)) > 1000.0
