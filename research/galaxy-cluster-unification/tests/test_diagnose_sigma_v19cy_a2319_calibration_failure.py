import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_sigma_v19cy_a2319_calibration_failure as diagnosis


def test_frozen_diagnosis_scope_and_parent_failure_are_exact() -> None:
    config, failed, applied, topology, _, _ = diagnosis.validate_inputs()
    assert failed["decision"] == "stop_before_cluster_event_application"
    assert not failed["line_shape_gate_passed"]
    assert failed["selected_candidate"] is None
    assert not failed["cluster_sky_event_accessed"]
    assert applied["terminal_gate_passed"]
    assert topology["topology_gate_passed"]
    assert config["diagnosed_candidate"] == "branch_linear_common_mode"
    assert config["time_resolved_test"]["bins_per_branch"] == 4
    assert config["diagnostic_thresholds"]["material_centroid_change_ev"] == 0.1
    for key in (
        "read_cluster_sky_event_rows",
        "apply_gain_to_cluster_sky_events",
        "fit_cluster_velocity",
        "access_validation_or_holdout_assets",
        "open_lensing_halo_or_gravity_targets",
        "change_gravity_formula_or_parameters",
        "derive_or_select_action",
    ):
        assert not config["authorization"][key]


def _resolved(values: list[float], total: float = 0.0) -> dict:
    return {
        "total_fit": {"centroid_shift_ev": total},
        "quartiles": [
            {"midpoint": float(index), "fit": {"centroid_shift_ev": value}}
            for index, value in enumerate(values)
        ],
    }


def test_diagnostic_flags_separate_slope_curvature_and_offset() -> None:
    config = diagnosis.load_json(diagnosis.DEFAULT_CONFIG)
    topology = {
        "branches": [
            {"name": "slope", "obsid": "000101000"},
            {"name": "curve", "obsid": "000102000"},
            {"name": "offset", "obsid": "000103000"},
        ]
    }
    targets = diagnosis.load_json(diagnosis.LINE_CONFIG)["published_targets"]
    failed = {
        "fit_results": {
            "branch_linear_common_mode": {
                obsid: {
                    "centroid_shift_ev": targets[obsid]["centroid_shift_ev"] + 0.2
                }
                for obsid in ("000101000", "000102000", "000103000")
            }
        }
    }
    resolved = {
        "slope": _resolved([0.0, 0.05, 0.10, 0.20]),
        "curve": _resolved([0.0, 0.20, 0.20, 0.0]),
        "offset": _resolved([0.0, 0.0, 0.0, 0.0]),
    }
    flags = diagnosis.diagnostic_flags(config, failed, topology, resolved)
    assert flags["000101000"]["slope_flag"]
    assert not flags["000101000"]["curvature_flag"]
    assert flags["000102000"]["curvature_flag"]
    assert not flags["000102000"]["reference_offset_flag"]
    assert flags["000103000"]["reference_offset_flag"]
