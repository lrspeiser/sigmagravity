from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19ce_wallaby_counterpart_mixture_propagation.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19ce", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CONFIG = ROOT / "configs" / "sigma_v19ce_wallaby_counterpart_mixture_propagation.json"
REPORT = ROOT / "results" / "sigma_v19ce_wallaby_counterpart_mixture_propagation" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v19ce_normalization_math_and_undefined_rule() -> None:
    candidates = [
        {"object_id": "a", "spatial_lr_k": "3"},
        {"object_id": "b", "spatial_lr_k": "1"},
    ]
    diagnostics = {
        "a": {
            "quality_controlled_foreground_contamination": "false",
            "foreground_astrometric_evidence": "false",
        },
        "b": {
            "quality_controlled_foreground_contamination": "true",
            "foreground_astrometric_evidence": "true",
        },
    }
    soft = {
        "quality_controlled_weight": 0.1,
        "other_foreground_evidence_weight": 1.0,
    }
    weights, raw, total = MODULE.normalized_weights(
        candidates, diagnostics, soft, "k"
    )
    assert raw == [3.0, 0.1]
    assert total == 3.1
    assert abs(sum(value for value in weights if value is not None) - 1) < 1e-15
    zero = {
        "quality_controlled_weight": 0.0,
        "other_foreground_evidence_weight": 0.0,
    }
    all_foreground = {
        key: {**value, "quality_controlled_foreground_contamination": "true"}
        for key, value in diagnostics.items()
    }
    missing, _, total = MODULE.normalized_weights(
        candidates, all_foreground, zero, "k"
    )
    assert missing == [None, None]
    assert total == 0


def test_v19ce_report_carries_full_source_inventory_and_no_selection() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["decision"] == (
        "counterpart_uncertainty_ready_for_target_blind_marginalization"
    )
    assert all(report["gate_results"].values())
    assert report["release_rows"] == 711
    assert report["candidate_rows"] == 18_550
    assert report["release_scenarios"] == 11_376
    assert report["defined_scenarios"] + report["undefined_scenarios"] == 11_376
    assert report["maximum_defined_normalization_error"] <= 1e-12
    assert not report["access_boundary_audit"]["counterpart_treatment_or_kernel_selected"]


def test_v19ce_outputs_are_hash_exact_and_keep_every_candidate() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for item in report["outputs"].values():
        path = ROOT / item["path"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]
    candidate_path = ROOT / report["outputs"]["candidate_mixture_weights"]["path"]
    with candidate_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18_550
    assert len({(row["source_row_id"], row["object_id"]) for row in rows}) == 18_550
    assert all(row["counterpart_selected"] == "false" for row in rows)
    weight_columns = [key for key in rows[0] if key.startswith("p_")]
    assert len(weight_columns) == 16


def test_v19ce_config_forbids_target_feedback_and_new_parameters() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["mixture_definition"]["new_tunable_hyperparameters"] == 0
    assert not config["mixture_definition"]["hard_counterpart_selected"]
    assert not config["mixture_definition"]["treatment_selected"]
    assert not config["mixture_definition"]["kernel_selected"]
    future = config["future_use_contract"]
    assert future["source_mixture_frozen_before_target_access"]
    assert future["primary_result_must_marginalize_counterpart_identity"]
    assert future["candidate_identity_may_not_be_selected_by_gravity_fit"]
