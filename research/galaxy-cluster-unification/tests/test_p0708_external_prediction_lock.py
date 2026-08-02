from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0708_prediction_lock_passes_before_external_unlock() -> None:
    report = json.loads(
        (ROOT / "results/p0708_external_prediction_lock/report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "pass"
    assert report["all_prediction_lock_gates_pass"] is True
    assert report["candidate_authorized_for_one_external_unlock"] is True
    assert all(report["gate_results"].values())
    assert report["systems"] == 17
    assert report["universal_parameter_sha256"] == (
        "bf3f12d6b32ee3f1b0e3bf48a9603c4aafbcd34b2cbdd3de021d689514099a15"
    )
    assert report["sealed_P0633_kinematics_opened"] is False
    assert report["sealed_P0640_lensing_constraints_opened"] is False


def test_p0708_manifest_has_complete_unique_prediction_hashes() -> None:
    frame = pd.read_csv(
        ROOT / "results/p0708_external_prediction_lock/system_prediction_manifest.csv"
    )
    assert len(frame) == 17
    assert set(frame["domain"]) == {"galaxy", "cluster"}
    assert frame["system"].is_unique
    assert frame["prediction_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert frame["finite"].all()
