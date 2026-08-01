from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "p0637_little_things_photometric_metadata.json"
RESULTS = ROOT / "results" / "p0637_little_things_photometric_metadata"


def test_all_frozen_targets_have_one_complete_photometric_input_row():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_csv(RESULTS / "photometric_inputs.csv")
    assert frame["galaxy"].is_unique
    assert set(frame["galaxy"]) == set(config["targets"])
    assert frame.notna().all().all()


def test_one_universal_geometry_and_stellar_normalization_rule_is_used():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_csv(RESULTS / "photometric_inputs.csv")
    assert set(frame["intrinsic_axis_ratio_q0"]) == {
        config["universal_geometry"]["intrinsic_axis_ratio_q0"]
    }
    assert set(frame["nominal_universal_v_band_mass_to_light"]) == {
        config["stellar_normalization"]["nominal_v_band_mass_to_light_solar"]
    }
    assert frame["inclination_rounding_delta_deg"].abs().max() <= 1.0


def test_no_kinematic_outcome_or_per_galaxy_gravity_setting_was_used():
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ready"
    assert report["sealed_target_observables_opened"] is False
    assert report["per_galaxy_gravity_parameters_fit"] is False
    assert report["errors"] == []
