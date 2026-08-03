from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p0740_allwise_w1_coverage_supplement.json"
RESULT = ROOT / "results/p0740_allwise_w1_coverage_supplement"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_uniform_sample_and_frozen_tile_selection() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    systems = config["systems"]
    assert config["status"] == "frozen_before_wise_image_download_or_array_inspection"
    assert len(systems) == 8
    assert sum(item["split"] == "development" for item in systems) == 4
    assert sum(item["split"] == "validation" for item in systems) == 2
    assert sum(item["split"] == "holdout" for item in systems) == 2
    assert sum(len(item["coaddIds"]) for item in systems) == 14
    assert config["source"]["products"] == ["intensity", "uncertainty"]


def test_acquisition_passes_without_opening_any_pixel_array() -> None:
    report = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["systems"] == 8
    assert report["coadds"] == 14
    assert report["files"] == 28
    assert report["bytes"] == 108662400
    assert all(report["checks"].values())
    assert report["arraysOpened"] == {"development": 0, "validation": 0, "holdout": 0}
    assert report["velocityOrDispersionArraysOpened"] == 0
    assert report["gravityParameters"] == 0
    assert report["aggregate"]["minimumUnionWcsFootprintFraction"] >= 0.99


def test_downloaded_cutouts_match_manifest_hashes() -> None:
    files = pd.read_csv(RESULT / "file_manifest.csv")
    assert len(files) == 28
    assert files["array_opened"].sum() == 0
    for row in files.itertuples(index=False):
        path = ROOT / row.relative_path
        assert path.is_file()
        assert path.stat().st_size == row.bytes
        assert _sha256(path) == row.sha256

