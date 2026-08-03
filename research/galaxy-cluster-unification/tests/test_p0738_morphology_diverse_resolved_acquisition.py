from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0738_morphology_diverse_resolved_acquisition.json"
RESULT = ROOT / "results/p0738_morphology_diverse_resolved_acquisition"
RAW = ROOT / "data/raw/p0738_things_sings_resolved"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_sample_has_the_declared_diversity_and_split() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    systems = config["systems"]
    assert config["status"] == "frozen_before_download_or_fits_content_inspection"
    assert len(systems) == 8
    assert sum(system["split"] == "development" for system in systems) == 4
    assert sum(system["split"] == "validation" for system in systems) == 2
    assert sum(system["split"] == "holdout" for system in systems) == 2
    assert {system["id"] for system in systems if system["split"] == "holdout"} == {
        "NGC2841",
        "NGC7331",
    }
    assert min(system["sparc"]["hubbleType"] for system in systems) == 3
    assert max(system["sparc"]["hubbleType"] for system in systems) == 7


def test_acquisition_manifest_passes_without_opening_holdout_arrays() -> None:
    report = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["fileCount"] == 40
    assert report["totalBytes"] == 380151360
    assert report["splitCounts"] == {"development": 4, "validation": 2, "holdout": 2}
    assert all(report["gateResults"].values())
    assert report["holdoutArraysOpened"] is False
    assert report["gravityParameters"] == 0
    assert report["velocityTargetsUsedForBaryonicExtraction"] is False
    holdout = [row for row in report["files"] if row["split"] == "holdout"]
    assert len(holdout) == 10
    assert not any(row["arrayOpened"] for row in holdout)


def test_all_raw_files_reproduce_the_manifest_bytes_and_hashes() -> None:
    report = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    for record in report["files"]:
        path = RAW / record["relativePath"]
        assert path.is_file()
        assert path.stat().st_size == record["expectedBytes"] == record["actualBytes"]
        assert _sha256(path) == record["sha256"]


def test_withheld_velocity_products_are_explicitly_separate_from_baryons() -> None:
    report = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    roles = {(row["kind"], row["scientificRole"]) for row in report["files"]}
    assert ("moment0", "baryonic_input") in roles
    assert ("moment1", "withheld_target") in roles
    assert ("moment2", "withheld_target") in roles
    assert ("irac1", "baryonic_input") in roles
    assert ("irac1_weight", "baryonic_input") in roles

