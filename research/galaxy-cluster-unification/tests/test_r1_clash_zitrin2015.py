from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_zitrin_protocol_preserves_model_dependence_and_candidate_policy() -> None:
    protocol = json.loads((ROOT / "configs/r1_clash_zitrin2015_ingest_protocol.json").read_text())
    assert protocol["frozen_before_local_product_ingest"] is True
    assert len(protocol["systems"]) == 7
    assert protocol["pre_registered_checks"]["published_table2_record_count"] == 579
    assert protocol["pre_registered_checks"]["metric_neutral_position_covariance_expected"] is False
    assert protocol["authorization"]["declare_metric_neutral_coordinate_likelihood_ready"] is False
    assert "c, p, or ?" in protocol["pre_registered_row_policy"]["measured_position_row"]


def test_zitrin_audit_adds_six_and_preserves_rxj1532_shortfall() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_r1_clash_zitrin2015.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((ROOT / "results/r1_clash_zitrin2015/report.json").read_text())
    images = pd.read_csv(ROOT / "data/derived/r1_clash_zitrin2015_image_observables.csv")
    systems = pd.read_csv(ROOT / "data/derived/r1_clash_zitrin2015_system_summary.csv")
    controls = np.load(ROOT / "data/derived/r1_clash_zitrin2015_model_control_covariances.npz")

    assert len(images) == 175
    assert len(systems) == 7
    assert report["published_table_rows"] == 579
    assert report["systems_with_raw_observable_catalog"] == 6
    assert report["measured_position_rows"] == 131
    assert report["metric_neutral_observable_rows"] == 129
    assert report["independently_redshift_anchored_families"] == 45
    assert report["hard_shortfall_systems"] == ["RXJ1532"]
    assert report["next_cycle_threshold_met"] is True
    assert report["all_seven_success_outcome_met"] is False
    assert images["metric_neutral_coordinate_likelihood_ready"].eq(False).all()
    assert images["gravity_target_used"].eq(False).all()
    rxj1532 = images.loc[images["system"] == "RXJ1532"]
    assert len(rxj1532) == 3
    assert rxj1532["measured_position_row"].eq(False).all()
    assert set(rxj1532["candidate_flag"]) == {True}
    assert systems.loc[systems["system"] != "RXJ1532", "raw_observable_catalog_acquired"].all()
    assert not bool(systems.loc[systems["system"] == "RXJ1532", "raw_observable_catalog_acquired"].iloc[0])
    assert all(np.allclose(controls[key], np.eye(len(controls[key])) * 1.4**2) for key in controls.files if key.endswith("arcsec2"))
