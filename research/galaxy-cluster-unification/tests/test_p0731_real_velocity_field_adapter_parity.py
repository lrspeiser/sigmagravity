from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0731_real_velocity_field_adapter_parity"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_real_velocity_adapter_parity_result() -> None:
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    config_path = ROOT / "configs" / "p0731_real_velocity_field_adapter_parity.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert report["status"] == "pass"
    assert report["systems"] == len(config["systems"]) == 13
    assert report["models"] == len(config["models"]) == 4
    assert report["evaluations"] == 52
    assert report["failedGates"] == []
    assert all(report["gateResults"].values())
    assert report["configSha256"] == sha256(config_path)

    with (RESULTS / "per_galaxy_model_scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 52
    assert {row["galaxy"] for row in rows} == set(config["systems"])
    assert {row["model"] for row in rows} == {
        model["id"] for model in config["models"]
    }
    assert all(row["field_artifact_hashes_valid"] == "True" for row in rows)
    assert all(row["valid_pixel_support_exact"] == "True" for row in rows)
    assert min(int(row["adapter_valid_pixels"]) for row in rows) >= 100
    assert all(int(row["per_object_gravity_parameters"]) == 0 for row in rows)


def test_p0731_published_artifacts_exist() -> None:
    for name in (
        "SUMMARY.md",
        "model_summary.csv",
        "observation_bundle_manifest.csv",
        "per_galaxy_model_scores.csv",
        "report.json",
        "score_and_parity_summary.png",
    ):
        assert (RESULTS / name).is_file()
