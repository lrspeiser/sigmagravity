from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19f_chandra_source_reduction.json"
COMMON = ROOT / "scripts" / "sigma_v19f_chandra_common.py"
REPRO = ROOT / "results" / "sigma_v19f_chandra_repro" / "report.json"
CLEANING = ROOT / "results" / "sigma_v19f_chandra_cleaning" / "report.json"


def load_common():
    spec = importlib.util.spec_from_file_location("sigma_v19f_common_test", COMMON)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_is_target_blind_and_has_complete_frozen_ancestry() -> None:
    common = load_common()
    config, acquisition, runtime = common.validate_protocol(CONFIG)
    assert set(config["clusters"]) == {"BULLET", "ABELL2146"}
    assert sum(len(row["obsids"]) for row in config["clusters"].values()) == 20
    assert acquisition["lensing_target_opened"] is False
    assert runtime["gates"]["runtime_gate_passed"] is True
    assert runtime["smoke"] == {
        "failed": 0,
        "log": "results/sigma_v17a_ciao_environment/ciao_smoke.log",
        "log_sha256": "b9224e6a1132ff52c4ea2279c647a9ac210f19b4f79706edb47d2e08c17b678b",
        "passed": 37,
        "run": 37,
        "runner": "/home/henry/miniforge3/envs/sigma-ciao-4.18/test/smoke/bin/run_smoke_tests.sh",
        "skipped": 0,
    }


def test_protocol_declares_the_single_faint_mode_exception() -> None:
    common = load_common()
    config, _, _ = common.validate_protocol(CONFIG)
    modes = {
        (cluster, int(obsid)): mode
        for cluster, values in config["clusters"].items()
        for obsid, mode in values["expected_archive_datamode"].items()
    }
    assert [key for key, mode in modes.items() if mode == "FAINT"] == [
        ("BULLET", 554)
    ]
    assert sum(mode == "VFAINT" for mode in modes.values()) == 19


def test_resolved_detector_choices_are_inherited_without_target_fields() -> None:
    common = load_common()
    config, _, _ = common.validate_protocol(CONFIG)
    resolved = common.resolved_shared_config(config)
    assert set(resolved["clusters"]) == {"BULLET", "ABELL2146"}
    assert resolved["event_reprocessing"]["pix_adj"] == "edser"
    assert resolved["flare_filtering"]["time_bin_seconds"] == 250
    assert resolved["flare_filtering"]["minimum_retained_fraction"] == 0.5
    assert resolved["point_sources"]["wavdetect_sigthresh"] == 1e-6
    assert resolved["background"]["normalization_energy_keV"] == [9.0, 12.0]
    assert resolved["background"]["random_seed"] == "the integer ObsID"


def test_reprojection_report_passes_declared_detector_gates() -> None:
    common = load_common()
    config, _, _ = common.validate_protocol(CONFIG)
    report = load(REPRO)
    assert report["config_sha256"] == common.sha256(CONFIG)
    assert report["observation_count"] == 20
    assert report["runtime_gate_passed"] is True
    assert report["lensing_target_opened"] is False
    assert report["event_images_inspected"] is False
    assert report["shock_front_fitted"] is False
    assert report["source_constructed"] is False
    for row in report["observations"]:
        expected = common.declared_mode(config, row["cluster"], int(row["obsid"]))
        assert row["archive_datamode"] == expected
        assert row["event"]["header"]["DATAMODE"] == expected
        assert row["check_vf_pha_requested"] is (expected == "VFAINT")
        assert row["check_vf_pha_history_present"] is True
        assert row["check_vf_pha_history_value"] == (
            "yes" if expected == "VFAINT" else "no"
        )
        assert row["event"]["current_caldb_comment_present"] is True


def test_cleaning_report_passes_source_only_background_gates() -> None:
    common = load_common()
    common.validate_protocol(CONFIG)
    report = load(CLEANING)
    assert report["config_sha256"] == common.sha256(CONFIG)
    assert report["repro_report_sha256"] == common.sha256(REPRO)
    assert report["observation_count"] == 20
    assert report["minimum_retained_exposure_fraction"] >= 0.5
    assert report["event_images_visually_inspected"] is False
    assert report["astrometry_completed"] is False
    assert report["shock_front_fitted"] is False
    assert report["source_constructed"] is False
    assert report["lensing_target_opened"] is False
    assert all(row["blanksky_scaling"] for row in report["observations"])


def test_v19f_runner_modules_are_importable() -> None:
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        for filename in (
            "run_sigma_v19f_chandra_repro.py",
            "run_sigma_v19f_chandra_cleaning.py",
        ):
            spec = importlib.util.spec_from_file_location(filename, scripts / filename)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(scripts))
