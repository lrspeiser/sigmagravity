from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v18d_collisionless_stress_transfer.json"
REPORT = ROOT / "results" / "sigma_v18d_collisionless_stress_transfer" / "report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_has_one_transferable_one_metric_feature() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["fit"]["feature_count"] == 1
    assert config["fit"]["intercept"] is False
    assert config["fit"]["coefficient_constraint"] == "nonnegative"
    assert config["fit"]["per_cluster_amplitude_scale_shear_or_orientation"] is False
    assert config["map_protocol"]["one_metric_rule"].startswith("target, baryon baseline")
    assert config["integrity"]["source_maps_frozen_before_target_opened"] is True
    assert config["integrity"]["holdout_opened"] is False


def test_report_preserves_transfer_and_integrity_boundaries() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "completed Sigma v18D collisionless-stress transfer diagnostic"
    assert len(report["primary_result"]["directions"]) == 2
    directions = {(row["train_cluster"], row["test_cluster"]) for row in report["primary_result"]["directions"]}
    assert directions == {("MACS0416", "PLCKG287"), ("PLCKG287", "MACS0416")}
    assert report["inverse_coefficient_is_physical_constant"] is False
    assert report["per_cluster_amplitude_scale_shear_or_orientation"] is False
    assert report["lensing_target_opened"] is True
    assert report["holdout_opened"] is False


def test_report_hashes_frozen_sources_and_is_resolution_scored() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == digest(CONFIG)
    assert report["input_hashes"]["source_config"] == digest(ROOT / config["parents"]["source_config"])
    assert report["input_hashes"]["source_report"] == digest(ROOT / config["parents"]["source_report"])
    assert report["resolution_stability"]["maximum_change"] >= 0.0
    assert isinstance(report["gate_results"]["advance"], bool)
