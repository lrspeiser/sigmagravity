import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_j1402_metadata_freezes_exact_science_and_calibration_gate() -> None:
    report = json.loads(
        (ROOT / "results/r1_j1402_kcwi_metadata/report.json").read_text(encoding="utf-8")
    )
    inventory = pd.read_csv(ROOT / "data/derived/r1_j1402_kcwi_night_inventory.csv")

    assert report["metadata_only"] is True
    assert report["science_arrays_downloaded"] is False
    assert report["target"]["science_frame_count"] == 4
    assert report["target"]["science_exposure_seconds"] == 7200.0
    assert report["target"]["setup_consistent"] is True
    assert set(inventory.loc[inventory["is_target_science"], "koaid"]) == set(
        report["target"]["exact_science_ids"]
    )
    assert report["same_configuration_calibrations"]["minimum_gate_pass"] is True
    assert report["decision"] == "calibration_identity_gate_pass_freeze_exact_acquisition_protocol"
