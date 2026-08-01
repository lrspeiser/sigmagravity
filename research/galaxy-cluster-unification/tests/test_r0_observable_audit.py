from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0_audit_covers_columns_and_stops_before_r1(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.csv"
    instances = tmp_path / "instances.csv"
    coverage = tmp_path / "coverage.csv"
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_r0_observables.py"),
            "--matrix-output",
            str(matrix),
            "--coverage-output",
            str(coverage),
            "--instance-output",
            str(instances),
            "--report-output",
            str(report),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    provenance = pd.read_csv(matrix)
    scalar_provenance = pd.read_csv(instances)
    systems = pd.read_csv(coverage)

    assert result["provenance_matrix"]["all_required_scored_columns_covered"]
    assert result["instance_provenance"]["rows"] == 19030
    assert result["instance_provenance"]["rows_by_dataset"] == {
        "BCG": 238,
        "CLASH": 588,
        "SPARC": 18204,
    }
    assert result["instance_provenance"]["systems_by_dataset"] == {
        "BCG": 34,
        "CLASH": 20,
        "SPARC": 131,
    }
    assert result["instance_provenance"]["unique_score_input_files"] == 133
    assert result["instance_provenance"]["all_exact_input_hashes_present"]
    assert result["instance_provenance"]["every_scalar_has_lineage_and_covariance_disposition"]
    assert result["instance_provenance"]["bytes"] == instances.stat().st_size
    assert result["instance_provenance"]["sha256"] == _sha256(instances)
    assert result["clash"]["systems"] == 20
    assert result["clash"]["scored_summary_points"] == 84
    assert result["bcg"]["frozen_systems"] == 34
    assert result["bcg"]["direct_single_radius_Jeans_summaries"] == 11
    assert result["bcg"]["calibrated_single_radius_proxies"] == 23
    assert result["same_object_pilot_gate"]["eligible_systems"] == 0
    assert not result["same_object_pilot_gate"]["passes"]
    assert result["stage_decision"]["R1_sample_freeze"] == "not_authorized"
    assert set(provenance["dataset"]) == {"SPARC", "CLASH", "BCG"}
    assert set(scalar_provenance["dataset"]) == {"SPARC", "CLASH", "BCG"}
    assert not scalar_provenance.duplicated(
        ["dataset", "source_variant", "system", "system_point_index_zero_based", "scored_column"]
    ).any()
    assert scalar_provenance["scored_value"].notna().all()
    assert np.isfinite(scalar_provenance["scored_value"].to_numpy(dtype=float)).all()
    assert scalar_provenance["score_input_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert len(systems) == 54
    assert not systems["same_object_pilot_eligible"].any()


def test_clash_target_is_classified_as_model_dependent() -> None:
    config = json.loads(
        (ROOT / "configs" / "r0_observable_audit.json").read_text(encoding="utf-8")
    )
    target = next(
        row
        for row in config["records"]
        if row["dataset"] == "CLASH" and row["scored_column"] == "log_gtot"
    )
    lineage = config["lineages"][target["lineage_id"]]
    assert "NFW" in lineage["transformation"]
    assert "GR" in lineage["metric_or_dynamics_assumptions"]
    assert lineage["alternative_theory_forward_modeling"].startswith("not_from_scored_summary")
