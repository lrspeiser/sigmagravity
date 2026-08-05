from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19x_spectral_combination_commissioning.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19x_freezes_existing_parents_and_no_spectral_outcome() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert config["integrity"]["v19w_final_report_existed_at_freeze"] is False
    assert config["integrity"]["combined_spectrum_or_fit_statistic_known_at_freeze"] is False
    assert config["integrity"]["gravity_formula_or_parameter_changed"] is False
    assert config["fit_sequence"]["published_temperature_used_as_gate"] is False


def test_v19x_commissioning_regions_are_unique_source_count_maxima() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["parents"]["v19u_manifest"]
    sums: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"cells": 0, "source": 0, "background": 0, "scaled": 0.0}
    )
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["cluster"], int(row["bin_id"]))
            sums[key]["cells"] += 1
            sums[key]["source"] += int(row["source_band_events"])
            sums[key]["background"] += int(row["background_band_events"])
            sums[key]["scaled"] += float(row["scaled_background_events"])
    for cluster, registered in config["registered_workload"]["clusters"].items():
        candidates = [(key, value) for key, value in sums.items() if key[0] == cluster]
        maximum = max(value["source"] for _, value in candidates)
        winners = [(key, value) for key, value in candidates if value["source"] == maximum]
        assert len(winners) == 1
        (winner_cluster, winner_bin), values = winners[0]
        selected = registered["commissioning_region"]
        assert winner_cluster == cluster
        assert winner_bin == selected["bin_id"]
        assert values["cells"] == selected["cells"]
        assert values["source"] == selected["source_events_0p5_7_keV"]
        assert values["background"] == selected["background_events_0p5_7_keV"]
        assert values["scaled"] == pytest.approx(
            selected["scaled_background_events_0p5_7_keV"], rel=0, abs=1e-12
        )


def test_v19x_cannot_run_before_v19w_authorizes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    runtime = config["runtime_authorization"]
    assert runtime["may_start_before_authorization"] is False
    assert runtime["required_completed_cells"] == 5082
    assert runtime["required_product_index_rows"] == 5082
    report_path = ROOT / runtime["required_v19w_report"]
    if not report_path.exists():
        pytest.skip("V19W production is still running")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == runtime["required_status"]
    assert report["completed_cells"] == runtime["required_completed_cells"]
    assert report["product_index"]["rows"] == runtime["required_product_index_rows"]
    assert report["regional_spectral_fitting_authorized"] is True
