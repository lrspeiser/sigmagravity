from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19aa_member_counterpart_association.json"
REPORT = ROOT / "results" / "sigma_v19aa_member_counterpart_association" / "report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def test_v19aa_report_and_output_provenance() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["config_sha256"] == sha256(CONFIG)
    assert report["implementation"] == config["implementation"]
    assert report["gates"]["all_integrity_gates_pass"] is True
    assert report["catalog_counts"] == {
        "unique_hsc_detections": 779,
        "unique_nsc_detections": 226,
        "reciprocal_hsc_nsc_pairs": 174,
        "unified_candidates": 831,
    }
    assert report["clusters"]["ABELL2146"]["association_states"] == {
        "ambiguous": 46,
        "no_candidate": 4,
        "secure": 13,
    }
    assert report["clusters"]["BULLET"]["association_states"] == {
        "ambiguous": 77,
        "no_candidate": 1,
    }

    for name in ("unified_candidates", "candidate_posteriors", "member_associations"):
        path = ROOT / report["outputs"][name]
        assert sha256(path) == report["outputs"][f"{name}_sha256"]


def test_v19aa_every_prior_normalizes_with_explicit_null() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    members = csv_rows(ROOT / report["outputs"]["member_associations"])
    posteriors = csv_rows(ROOT / report["outputs"]["candidate_posteriors"])
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in posteriors:
        grouped[(row["cluster"], row["object_id"])].append(row)

    assert len(members) == 141
    tolerance = float(config["gates"]["posterior_normalization_tolerance"])
    for member in members:
        key = (member["cluster"], member["object_id"])
        assert int(member["candidate_count"]) == len(grouped[key])
        for prior in config["association"]["counterpart_prior_sensitivity"]:
            suffix = f"{float(prior):.2f}"
            total = float(member[f"null_posterior_q_{suffix}"]) + sum(
                float(row[f"posterior_q_{suffix}"]) for row in grouped[key]
            )
            assert total == pytest.approx(1.0, abs=tolerance)


def test_v19aa_secure_rows_obey_every_frozen_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    members = csv_rows(ROOT / report["outputs"]["member_associations"])
    secure = config["association"]["secure_match_gates"]

    global_ids = [row["global_map_candidate_id"] for row in members if row["global_map_candidate_id"]]
    assert len(global_ids) == len(set(global_ids))
    for row in members:
        if row["association_state"] != "secure":
            continue
        assert row["secure_counterpart_id"] == row["top_candidate_id"]
        assert row["global_map_candidate_id"] == row["top_candidate_id"]
        assert float(row["top_posterior_min"]) >= float(
            secure["minimum_posterior_across_prior_grid"]
        )
        assert float(row["top_to_second_likelihood_ratio"]) >= float(
            secure["minimum_top_to_second_likelihood_ratio"]
        )
        assert as_bool(row["top_dual_survey"]) or as_bool(
            row["top_repeated_detection_support"]
        )
        assert not as_bool(row["top_probable_foreground_star_diagnostic"])


def test_v19aa_claim_boundary_remains_measurement_only() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["lensing_or_halo_payload_opened"] is False
    assert report["photometric_transformation_performed"] is False
    assert report["stellar_mass_inference_performed"] is False
    assert report["mass_current_constructed"] is False
    assert report["gravity_formula_or_parameter_changed"] is False
