from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19b_replacement_cluster_screen.json"
REPORT = ROOT / "results" / "sigma_v19b_replacement_cluster_screen" / "report.json"
SCRIPT = ROOT / "scripts" / "audit_sigma_v19b_replacement_cluster_screen.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sigma_v19b_screen", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_history_significance_uses_no_shock_boundary() -> None:
    module = load_module()
    bullet = {
        "basis": "mach_excess_above_unity",
        "measured_value": 2.57,
        "uncertainty_toward_no_shock": 0.23,
    }
    el_gordo = {
        "basis": "published_detection_confidence",
        "measured_value": 0.98,
        "uncertainty_toward_no_shock": None,
    }
    assert math.isclose(
        module.history_significance_sigma(bullet),
        (2.57 - 1.0) / 0.23,
        rel_tol=1e-12,
    )
    assert 2.0 < module.history_significance_sigma(el_gordo) < 2.1


def test_protocol_seals_all_lensing_answers() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["blindness"]["all_replacement_lensing_targets_remain_sealed"]
    assert config["authorization"]["formula_selection_authorized"] is False
    assert config["authorization"]["lensing_or_halo_payload_access_authorized"] is False
    assert config["authorization"]["holdout_access_authorized"] is False
    for candidate in config["candidates"].values():
        assert (
            candidate["later_lensing_suitability_metadata"][
                "coordinates_or_model_read_during_screen"
            ]
            is False
        )


def test_report_selects_only_source_gate_survivors() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    candidates = report["candidate_results"]
    assert report["selected_development_pair"] == ["ABELL2146", "BULLET"]
    assert candidates["BULLET"]["source_acquisition_eligible"] is True
    assert candidates["ABELL2146"]["source_acquisition_eligible"] is True
    assert candidates["ELGORDO"]["source_acquisition_eligible"] is False
    assert candidates["MACS0025"]["source_acquisition_eligible"] is False
    assert candidates["BULLET"]["history_statistic_sigma"] > 6.8
    assert math.isclose(
        candidates["ABELL2146"]["history_statistic_sigma"], 6.5, rel_tol=1e-12
    )
    assert candidates["ELGORDO"]["history_statistic_sigma"] < 5.0


def test_development_selection_is_not_a_final_sample_claim() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    gates = report["gate_results"]
    candidates = report["candidate_results"]
    assert gates["replacement_development_pair_identified"] is True
    assert gates["selected_pair_final_lensing_sample_metadata_ready"] is False
    assert (
        candidates["ABELL2146"]["final_sample_lensing_metadata_gates"][
            "minimum_spectroscopic_families"
        ]
        is False
    )
    assert report["gravity_parameters_fit"] == 0
    assert report["lensing_or_halo_payload_used"] is False


def test_source_construction_stays_closed_until_local_products_exist() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    gates = report["gate_results"]
    assert gates["source_archive_acquisition_authorized"] is True
    assert gates["selected_pair_local_source_products_ready"] is False
    assert gates["source_construction_authorized"] is False
    assert gates["selected_pair_has_assumption_independent_clocks"] is False
    assert report["new_lensing_target_opened"] is False
    assert report["holdout_opened"] is False


def test_report_hashes_frozen_protocol_and_parents() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["input_hashes"]["config"] == digest(CONFIG)
    for key in ("assembly_readiness_config", "assembly_readiness_report"):
        assert report["input_hashes"][key] == digest(ROOT / config["parents"][key])
