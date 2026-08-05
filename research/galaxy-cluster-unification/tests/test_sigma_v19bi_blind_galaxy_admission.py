from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sigma_v19bi_blind_galaxy_admission.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bi", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19bi_passes_every_admission_protocol_gate() -> None:
    report = MODULE.build_report()
    assert report["decision"] == "passed_survey_level_blind_galaxy_admission_protocol"
    assert all(report["gate_results"].values())


def test_v19bi_preserves_all_galaxy_strata_and_multiple_observables() -> None:
    report = MODULE.build_report()
    assert len(report["registered_strata"]) == 8
    final = report["final_holdout_requirements"]
    assert final["minimum_unique_galaxies"] >= 48
    assert final["minimum_per_required_stratum"] >= 6
    assert final["minimum_raw_cube_or_velocity_field_systems"] >= 12
    assert final["minimum_high_resolution_inner_curve_systems"] >= 8
    assert final["minimum_radial_plus_vertical_systems"] >= 8
    assert {row["id"] for row in report["systematic_controls"]} == {
        "WALLABY_FORWARD_CUBE_CONTROL",
        "PHANGS_INNER_GEOMETRY_CONTROL",
        "DISKMASS_VERTICAL_CONTROL",
    }


def test_v19bi_does_not_relabel_nuisances_as_gravity_parameters() -> None:
    report = MODULE.build_report()
    boundary = report["nuisance_boundary"]
    assert not set(boundary["measurement_nuisances"]).intersection(
        boundary["gravity_parameters"]
    )
    comparator = report["fair_comparator_contract"]
    assert comparator["no_per_galaxy_gravity_parameter_for_sigma"]
    assert comparator["fixed_MOND_RAR_parameters_frozen_before_targets"]
    assert comparator["same_baryon_draws_and_measurement_nuisances_for_all_gravity_models"]


def test_v19bi_keeps_new_targets_sealed_and_solar_later() -> None:
    report = MODULE.build_report()
    state = report["admission_state"]
    assert state["selected_galaxies"] == 0
    assert state["new_kinematic_targets_opened"] == 0
    assert not state["action_selected"]
    assert not state["universal_constants_selected"]
    assert not report["priority"]["detailed_solar_optimization_now"]


def test_v19bi_committed_report_matches_rebuild() -> None:
    expected = MODULE.build_report()
    path = ROOT / "results" / "sigma_v19bi_blind_galaxy_admission" / "report.json"
    actual = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected
