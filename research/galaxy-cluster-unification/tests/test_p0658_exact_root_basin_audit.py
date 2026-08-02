from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p0658_exact_root_basin_audit"


def report() -> dict:
    return json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_reproduces_frozen_fit_without_parameter_changes():
    result = report()
    assert result["reproduced_optimizer_cost"] == 5.692377432355883
    coverage = result["coverage"]
    assert coverage["validation_images"] == 4
    assert coverage["starts_per_image"] == 25
    assert coverage["algorithms"] == 4
    assert coverage["attempts_per_image"] == 100
    assert coverage["total_attempts"] == 400
    assert coverage["gravity_parameters_changed"] == 0
    assert coverage["geometry_parameters_changed"] == 0
    assert coverage["source_positions_changed"] == 0


def test_both_original_failures_remain_without_local_roots():
    result = report()
    summary = pd.read_csv(RESULTS / "recovery_summary.csv").set_index("image_id")
    assert result["status"] == "local_topology_failure_supported"
    assert result["originally_failed_images_recovered"] == []
    assert result["originally_failed_images_unrecovered"] == ["1b", "6b"]
    assert summary.loc["1b", "accepted_attempts"] == 0
    assert summary.loc["6b", "accepted_attempts"] == 0
    assert bool(summary.loc["1b", "recovered_by_audit"]) is False
    assert bool(summary.loc["6b", "recovered_by_audit"]) is False


def test_one_b_has_only_a_distant_exact_branch():
    attempts = pd.read_csv(RESULTS / "root_attempts.csv")
    one_b = attempts[attempts.image_id == "1b"]
    best = one_b.sort_values("closure_arcsec").iloc[0]
    assert best.closure_arcsec < 1e-12
    assert best.displacement_from_observed_arcsec > 5.28
    assert bool(best.accepted) is False


def test_six_b_has_a_large_irreducible_local_closure():
    attempts = pd.read_csv(RESULTS / "root_attempts.csv")
    six_b = attempts[attempts.image_id == "6b"]
    assert six_b.closure_arcsec.min() > 0.97
    assert six_b.accepted.sum() == 0


def test_control_images_are_recovered_by_every_attempt():
    summary = pd.read_csv(RESULTS / "recovery_summary.csv").set_index("image_id")
    for image_id in ("2a", "7a"):
        assert summary.loc[image_id, "accepted_attempts"] == 100
        assert summary.loc[image_id, "distinct_local_roots"] == 1


def test_no_advancement_blindness_and_hashes_are_preserved():
    result = report()
    assert result["candidate_selected_or_advanced"] is False
    assert result["universal_rescore_required_before_any_candidate_change"] is False
    assert result["sealed_P0633_kinematics_opened"] is False
    assert result["sealed_P0640_lensing_constraints_opened"] is False
    assert result["protocol_sha256"] == digest(
        ROOT / "configs/p0658_exact_root_basin_audit.json"
    )
    assert result["source_sha256"] == digest(
        ROOT / "scripts/run_p0658_exact_root_basin_audit.py"
    )
    assert (RESULTS / "exact_root_basin_audit.png").stat().st_size > 20000
