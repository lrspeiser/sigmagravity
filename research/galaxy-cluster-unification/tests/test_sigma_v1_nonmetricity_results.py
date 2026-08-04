from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_sigma_v1_report_matches_inputs_and_decision() -> None:
    report = json.loads(
        (ROOT / "results" / "sigma_v1_nonmetricity_cycle" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    config = ROOT / "configs" / "sigma_v1_nonmetricity_cycle.json"
    galaxy = ROOT / "results" / "p0711_external_galaxy_rotation_validation" / "report.json"
    cluster = (
        ROOT
        / "results"
        / "p0714_ready_subset_raw_lensing"
        / "cluster_model_scores.csv"
    )
    assert report["input_hashes"] == {
        "config": _sha256(config),
        "galaxy_report": _sha256(galaxy),
        "raw_cluster_scores": _sha256(cluster),
    }
    assert report["gate_results"] == {
        "mathematical_and_limit_checks": True,
        "galaxy": True,
        "raw_cluster_lensing": False,
        "independent_lensing_response": False,
    }
    assert report["weak_field_derivation"]["reduced_theory"] == "standard-mu AQUAL"
    assert report["advances"] is False


def test_inherited_raw_cluster_rows_are_complete_and_failed() -> None:
    rows = pd.read_csv(
        ROOT
        / "results"
        / "sigma_v1_nonmetricity_cycle"
        / "inherited_raw_cluster_scores.csv"
    )
    assert set(rows["cluster"]) == {"AS295", "PLCKG287"}
    assert set(rows["model"]) == {"AQUAL_simple_mu_diagnostic"}
    assert (rows["root_convergence_fraction"] == 1.0 / 3.0).all()
    assert not rows["all_heldout_topologies_correct"].any()
