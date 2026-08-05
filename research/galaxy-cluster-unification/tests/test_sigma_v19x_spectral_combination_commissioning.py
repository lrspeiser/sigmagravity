from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19x_spectral_combination_commissioning.json"
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x_spectral_combination_commissioning as runner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19x_freezes_existing_parents_and_no_spectral_outcome() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol_version"].endswith("1.0.1")
    for key, value in config["parents"].items():
        if key.endswith("_sha256"):
            continue
        expected = config["parents"].get(f"{key}_sha256")
        if expected is not None:
            assert sha256(ROOT / value) == expected
    assert sha256(ROOT / config["execution"]["runner"]) == config["execution"][
        "runner_sha256"
    ]
    runner.validate_parent_hashes(config)
    assert config["integrity"]["v19w_final_report_existed_at_freeze"] is False
    assert config["integrity"]["combined_spectrum_or_fit_statistic_known_at_freeze"] is False
    assert config["integrity"]["gravity_formula_or_parameter_changed"] is False
    assert config["fit_sequence"]["published_temperature_used_as_gate"] is False
    assert "625 exact event-energy rows correspond to 651" in config[
        "pre_execution_count_gate_correction"
    ]
    assert config["integrity"][
        "v19x_v100_combined_product_existed_before_count_gate_correction"
    ] is False


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


def test_v19x_runner_builds_the_exact_frozen_apertures() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = runner.load_manifest(config)
    plan = runner.build_aperture_plan(config, manifest)
    assert len(plan["BULLET"]["integrated"]) == 3812
    assert len(plan["ABELL2146"]["integrated"]) == 1270
    assert {int(row["bin_id"]) for row in plan["BULLET"]["regional"]} == {169}
    assert {int(row["bin_id"]) for row in plan["ABELL2146"]["regional"]} == {62}


def test_v19x_runner_refuses_a_missing_authorization_report(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="final authorization report"):
        runner.validate_runtime_authorization(config, tmp_path / "absent.json")


def test_v19x_cell_validation_checks_manifest_counts_and_all_hashes(
    tmp_path: Path,
) -> None:
    manifest = {
        "cluster": "TEST",
        "bin_id": "7",
        "obsid": "42",
        "ccd_id": "3",
        "source_band_events": "11",
        "background_band_events": "5",
    }
    name = runner.cell_name(manifest)
    completed = tmp_path / "completed" / name
    products_dir = completed / "products"
    products_dir.mkdir(parents=True)
    products = {}
    index = {
        "cluster": "TEST",
        "bin_id": "7",
        "obsid": "42",
        "ccd_id": "3",
    }
    roles = {
        "source_pha": ("source.pi", "source_pha_sha256"),
        "background_pha": ("background.pi", "background_pha_sha256"),
        "arf": ("source.arf", "arf_sha256"),
        "rmf": ("source.rmf", "rmf_sha256"),
    }
    for role, (filename, index_key) in roles.items():
        path = products_dir / filename
        path.write_bytes(f"{role}-fixture".encode())
        digest = sha256(path)
        products[role] = {
            "name": filename,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        index[index_key] = digest
    report = {
        "cell_name": name,
        "cluster": "TEST",
        "bin_id": 7,
        "obsid": 42,
        "ccd_id": 3,
        "preflight": {"source_band_events": 11, "background_band_events": 5},
        "source_pha_channel_audit": {"pha_total_counts": 13, "exact": True},
        "products": products,
        "gates": {"all_fixture_checks": True},
    }
    (completed / "cell_report.json").write_text(json.dumps(report), encoding="utf-8")
    validated = runner.validate_cell(manifest, index, tmp_path)
    assert validated["source_pha_total_counts"] == 13
    assert validated["source_band_events"] == 11
    report["preflight"]["source_band_events"] = 12
    (completed / "cell_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source event-energy count mismatch"):
        runner.validate_cell(manifest, index, tmp_path)
