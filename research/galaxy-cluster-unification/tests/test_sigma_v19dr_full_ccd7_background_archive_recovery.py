from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


REPORT_PATH = (
    "results/sigma_v19dr_full_ccd7_background_archive_recovery/report.json"
)


def test_v19dr_protocol_and_implementation_are_frozen() -> None:
    config = load_json(
        "configs/sigma_v19dr_full_ccd7_background_archive_recovery.json"
    )
    report = load_json(REPORT_PATH)
    runner = ROOT / config["implementation"]["runner"]

    assert sha256(runner) == config["implementation"]["runner_sha256"]
    assert sha256(runner) == report["runner_sha256"]
    assert sha256(
        ROOT / "configs/sigma_v19dr_full_ccd7_background_archive_recovery.json"
    ) == report["config_sha256"]


def test_v19dr_terminal_report_passes_every_registered_gate() -> None:
    report = load_json(REPORT_PATH)

    assert report["status"] == (
        "full_256_cell_ccd7_background_archive_recovery_passed"
    )
    assert report["aggregate_pass"] is True
    assert all(report["gates"].values())
    assert report["selected_cells"] == 256
    assert report["completed_cells"] == 256
    assert report["failures"] == {}
    assert report["source_band_event_range"] == [22, 532]
    assert report["background_band_event_range"] == [20, 1219]
    assert report["total_background_band_events"] == 45_252


def test_v19dr_recovery_index_contains_all_and_only_the_frozen_cells() -> None:
    report = load_json(REPORT_PATH)
    item = report["recovery_index"]
    path = ROOT / item["path"]
    rows = load_csv(item["path"])

    assert len(rows) == item["rows"] == 256
    assert sha256(path) == item["sha256"]
    assert Counter(int(row["obsid"]) for row in rows) == Counter(
        {10464: 128, 10888: 128}
    )
    assert {int(row["ccd_id"]) for row in rows} == {7}
    assert {row["cluster"] for row in rows} == {"ABELL2146"}
    assert len({row["cell_name"] for row in rows}) == 256
    assert all(int(row["background_band_events"]) > 0 for row in rows)
    assert all(row["all_cell_gates_passed"] == "True" for row in rows)
    assert all(
        np.isclose(
            float(row["effective_background_scale"]),
            float(row["blanksky_scale"]),
            rtol=1e-6,
            atol=0.0,
        )
        for row in rows
    )


def test_v19dr_unified_archive_is_complete_and_has_exact_replacements() -> None:
    report = load_json(REPORT_PATH)
    item = report["unified_product_index"]
    path = ROOT / item["path"]
    rows = load_csv(item["path"])
    recovered = [row for row in rows if row["archive"] == "v19dr_real_ccd7_background"]

    assert len(rows) == item["rows"] == 5_082
    assert sha256(path) == item["sha256"]
    assert len({row["cell_name"] for row in rows}) == 5_082
    assert len(recovered) == 256
    assert {row["cluster"] for row in recovered} == {"ABELL2146"}
    assert {int(row["obsid"]) for row in recovered} == {10464, 10888}
    assert {int(row["ccd_id"]) for row in recovered} == {7}


def test_v19dr_authorizes_only_the_next_source_likelihood() -> None:
    report = load_json(REPORT_PATH)

    assert report["full_494_region_joint_likelihood_successor_authorized"] is True
    assert report["all_494_regions_run"] is False
    assert report["thermal_stress_or_baroclinicity_constructed"] is False
    assert report["lensing_halo_action_gravity_or_holdout_payload_opened"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
