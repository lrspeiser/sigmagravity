from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_sigma_v19x2_unified_spectral_combination_commissioning as runner


def fixture_config(tmp_path: Path) -> dict:
    return {
        "runtime_authorization": {
            "required_v19w4_report": "unused.json",
            "required_unified_cells": 2,
            "required_unified_products": 8,
        },
        "parents": {
            "v19w4_config_sha256": "a" * 64,
            "v19w4_runner_sha256": "b" * 64,
        },
        "execution": {
            "response_archives": {
                "base_v19w": str(tmp_path / "base"),
                "v19w4_recovery": str(tmp_path / "recovery"),
            }
        },
        "registered_workload": {
            "clusters": {
                "BULLET": {
                    "total_cells": 1,
                    "commissioning_region": {"cells": 1},
                },
                "ABELL2146": {
                    "total_cells": 1,
                    "commissioning_region": {"cells": 1},
                },
            }
        },
    }


def fixture_manifest() -> list[dict[str, str]]:
    return [
        {
            "cluster": "BULLET",
            "bin_id": "1",
            "obsid": "11",
            "ccd_id": "0",
            "source_band_events": "10",
            "background_band_events": "2",
        },
        {
            "cluster": "ABELL2146",
            "bin_id": "2",
            "obsid": "22",
            "ccd_id": "1",
            "source_band_events": "12",
            "background_band_events": "3",
        },
    ]


def fixture_validated(tmp_path: Path) -> dict:
    records = {}
    for index, manifest in enumerate(fixture_manifest()):
        key = runner.adapter.task_key(manifest)
        archive = "base_v19w" if index == 0 else "v19w4_recovery"
        records[key] = {
            "cluster": key[0],
            "bin_id": key[1],
            "obsid": key[2],
            "ccd_id": key[3],
            "cell_name": runner.adapter.cell_name(manifest),
            "archive": archive,
            "cell_directory": tmp_path / archive / runner.adapter.cell_name(manifest),
            "source_band_events": int(manifest["source_band_events"]),
            "background_band_events": int(manifest["background_band_events"]),
            "source_pha_total_counts": 20 + index,
            "source_pha": tmp_path / f"source{index}.pi",
            "source_pha_sha256": str(index) * 64,
        }
    return records


def fixture_plan(manifest: list[dict[str, str]]) -> dict:
    return {
        "BULLET": {"integrated": [manifest[0]], "regional": [manifest[0]]},
        "ABELL2146": {"integrated": [manifest[1]], "regional": [manifest[1]]},
    }


def passing_combination(label: str, cells: list[dict], *_args) -> dict:
    return {
        "label": label,
        "cells": len(cells),
        "grouped_pha_links": {"A": "B"},
        "expected_grouped_pha_links": {"A": "B"},
        "frozen_snapshot": {"files": 4},
        "full_pha_count_conservation_exact": True,
    }


def passing_fit(_config, cluster: str, combination: dict, abundance) -> dict:
    return {
        "cluster": cluster,
        "label": combination["label"],
        "fit_completed": True,
        "parameters": {"abundance_solar": 0.3 if abundance is None else abundance},
        "gates": {"all_passed": True},
    }


def test_scaffold_orchestrates_mixed_archive_with_unchanged_fit_order(
    tmp_path: Path, monkeypatch
) -> None:
    config = fixture_config(tmp_path)
    manifest = fixture_manifest()
    plan = fixture_plan(manifest)
    validated = fixture_validated(tmp_path)
    index = tmp_path / "unified.csv"
    index.write_text("fixture\n", encoding="utf-8")
    report_path = tmp_path / "v19w4.json"
    report_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner.adapter,
        "authorize_unified_index",
        lambda *_args, **_kwargs: (
            {"unified_product_index": {"sha256": "c" * 64}},
            index,
        ),
    )
    monkeypatch.setattr(runner.inherited_v19x, "load_manifest", lambda _config: manifest)
    monkeypatch.setattr(
        runner.inherited_v19x, "build_aperture_plan", lambda *_args: plan
    )
    monkeypatch.setattr(
        runner.adapter, "validate_unified_archive", lambda *_args: validated
    )
    monkeypatch.setattr(runner.inherited_v19x, "combine_aperture", passing_combination)
    calls = []

    def recording_fit(config_arg, cluster, combination, abundance):
        calls.append((cluster, combination["label"], abundance))
        return passing_fit(config_arg, cluster, combination, abundance)

    monkeypatch.setattr(runner.inherited_v19x, "fit_spectrum", recording_fit)
    result = runner.execute(config, tmp_path / "output", tmp_path / "scratch", report_path)
    assert result["status"].startswith("unified_spectral_combination_commissioning_passed")
    assert result["full_494_region_combination_and_fit_authorized"]
    assert result["validated_response_archive_counts"] == {
        "base_v19w": 1,
        "v19w4_recovery": 1,
    }
    assert calls == [
        ("BULLET", "BULLET_integrated", None),
        ("ABELL2146", "ABELL2146_integrated", None),
        ("BULLET", "BULLET_bin1", 0.3),
        ("ABELL2146", "ABELL2146_bin2", 0.3),
    ]


def test_scaffold_does_not_authorize_after_a_fit_gate_failure(
    tmp_path: Path, monkeypatch
) -> None:
    config = fixture_config(tmp_path)
    manifest = fixture_manifest()
    plan = fixture_plan(manifest)
    validated = fixture_validated(tmp_path)
    monkeypatch.setattr(runner.inherited_v19x, "combine_aperture", passing_combination)

    def one_failed_fit(config_arg, cluster, combination, abundance):
        row = passing_fit(config_arg, cluster, combination, abundance)
        if cluster == "ABELL2146" and abundance is not None:
            row["gates"]["all_passed"] = False
        return row

    monkeypatch.setattr(runner.inherited_v19x, "fit_spectrum", one_failed_fit)
    result = runner.combine_and_fit(
        config,
        tmp_path / "output",
        tmp_path / "scratch",
        manifest,
        plan,
        validated,
    )
    assert result["status"] == "unified_spectral_combination_commissioning_gate_failed"
    assert not result["gates"]["both_regional_fits_pass"]
    assert not result["full_494_region_combination_and_fit_authorized"]


def test_validated_index_preserves_archive_and_cell_directory(tmp_path: Path) -> None:
    records = list(fixture_validated(tmp_path).values())
    result = runner.write_validated_index(records, tmp_path / "validated.csv")
    with (tmp_path / "validated.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert result["rows"] == 2
    assert [row["archive"] for row in rows] == ["base_v19w", "v19w4_recovery"]
    assert all(Path(row["cell_directory"]).is_absolute() for row in rows)


def test_runner_refuses_an_unfrozen_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"freeze_state": "draft", "implementation": {}}),
        encoding="utf-8",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        runner.validate_frozen_runner(config)
    except RuntimeError as exc:
        assert "not frozen after a terminal V19W4 pass" in str(exc)
    else:
        raise AssertionError("unfrozen V19X2 configuration was accepted")
