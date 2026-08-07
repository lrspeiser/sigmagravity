from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_v19dp_and_v19dq_runner_hashes_are_frozen() -> None:
    for stem in (
        "sigma_v19dp_unmerged_regional_joint_likelihood_preflight",
        "sigma_v19dq_ccd7_background_recovery_preflight",
    ):
        config = load(f"configs/{stem}.json")
        runner = ROOT / config["implementation"]["runner"]
        assert sha256(runner) == config["implementation"]["runner_sha256"]


def test_v19dp_localizes_the_only_failure_to_abell_ccd7() -> None:
    report = load(
        "results/sigma_v19dp_unmerged_regional_joint_likelihood_preflight/"
        "report.json"
    )
    assert report["status"] == "unmerged_regional_joint_likelihood_preflight_failed"
    assert report["aggregate_pass"] is False
    rows = {row["cluster"]: row for row in report["regions"]}
    assert rows["BULLET"]["passed"] is True
    abell = rows["ABELL2146"]
    assert [name for name, passed in abell["gates"].items() if not passed] == [
        "reduced_statistic_at_most_1_5"
    ]
    assert np.isclose(abell["primary"]["fit"]["reduced_statistic"], 1.949652289568236)
    leave_out = {
        row["omitted_cell"]: row["reduced_statistic"]
        for row in abell["leave_one_observation_out"]
    }
    assert np.isclose(
        leave_out["ABELL2146_bin62_obs10464_ccd7"], 1.3271405193503274
    )
    assert leave_out["ABELL2146_bin62_obs10888_ccd7"] < abell["primary"]["fit"][
        "reduced_statistic"
    ]
    assert report["full_regional_joint_likelihood_successor_authorized"] is False


def test_v19dq_recovers_real_background_events_without_a_zero_pha() -> None:
    report = load(
        "results/sigma_v19dq_ccd7_background_recovery_preflight/report.json"
    )
    assert report["status"] == "ccd7_real_background_recovery_preflight_passed"
    assert report["aggregate_pass"] is True
    assert all(report["gates"].values())
    boundary = {row["obsid"]: row for row in report["background_boundary"]}
    assert boundary[10464]["pre_reprojection_ccd7_rows"] == 1_354_493
    assert boundary[10888]["pre_reprojection_ccd7_rows"] == 1_390_824
    assert boundary[10464]["post_reprojection_ccd7_rows"] == 0
    assert boundary[10888]["post_reprojection_ccd7_rows"] == 0
    assert all(row["passed"] for row in boundary.values())

    cells = {row["obsid"]: row for row in report["remediated_cells"]}
    expected = {
        10464: (9498, 1219, 345, 0.086949646),
        10888: (9509, 1204, 84, 0.015882602),
    }
    for obsid, (all_events, band_events, source_events, scale) in expected.items():
        row = cells[obsid]
        background = row["materialized_event_subsets"]["background"]
        source = row["materialized_event_subsets"]["source"]
        assert background["all_energy_rows"] == all_events
        assert background["band_500_7000_rows"] == band_events
        assert source["band_500_7000_rows"] == source_events
        assert row["zero_background_steps"] is None
        assert np.isclose(row["blanksky_scaling"]["effective_background_scale"], scale)
        assert row["background_pha_channel_audit"]["exact"] is True
        assert all(row["gates"].values())


def test_v19dq_repairs_the_joint_fit_without_new_free_parameters() -> None:
    report = load(
        "results/sigma_v19dq_ccd7_background_recovery_preflight/report.json"
    )
    rows = {row["cluster"]: row for row in report["joint_preflight"]["regions"]}
    bullet = rows["BULLET"]
    abell = rows["ABELL2146"]
    assert bullet["passed"] is True
    assert abell["passed"] is True
    assert np.isclose(bullet["primary"]["fit"]["reduced_statistic"], 0.763083471042196)
    assert np.isclose(abell["primary"]["fit"]["reduced_statistic"], 1.0312109869145865)
    assert np.isclose(abell["primary"]["parameters"]["temperature_keV"], 10.172862448781858)
    assert abell["primary"]["free_parameter_count"] == 3
    assert abell["primary"]["shared_parameters"] == [
        "temperature_keV",
        "abundance_solar",
        "normalization",
    ]
    assert abell["maximum_leave_one_out_temperature_relative_shift"] < 0.10
    assert all(row["reduced_statistic"] < 1.12 for row in abell["leave_one_observation_out"])


def test_v19dq_authorizes_only_the_full_ccd7_recovery_stage() -> None:
    report = load(
        "results/sigma_v19dq_ccd7_background_recovery_preflight/report.json"
    )
    assert report["full_ccd7_background_archive_recovery_successor_authorized"] is True
    assert report["full_494_region_joint_likelihood_successor_authorized"] is False
    assert report["next_required_stage"] == (
        "rebuild_and_audit_all_254_affected_ccd7_background_products"
    )
    assert report["all_494_regions_run"] is False
    assert report["thermal_stress_or_baroclinicity_constructed"] is False
    assert report["lensing_halo_action_gravity_or_holdout_payload_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
