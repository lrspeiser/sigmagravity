from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19cl_registered_cluster_morphology_acquisition.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19cl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v19cl_reproduces_registered_catalogs_and_passes_gates() -> None:
    report, _, _ = MODULE.build_report()
    assert report["decision"] == "balanced_source_morphology_pool_established_source_completeness_required"
    assert all(report["gate_results"].values())
    assert report["source_integrity"]["primary_rows"] == 964
    assert report["source_integrity"]["secondary_rows"] == 150


def test_v19cl_has_preregistered_relaxed_disturbed_and_complexity_diversity() -> None:
    report, primary, _ = MODULE.build_report()
    assert report["crossmatch"]["primary_exact_one_to_one"] == 24
    assert report["crossmatch"]["primary_all_required_metrics_finite"] == 18
    assert report["morphology_class_counts"]["confirmed_relaxed"] >= 3
    assert report["morphology_class_counts"]["confirmed_disturbed"] >= 3
    assert report["morphology_class_counts"]["discordant_extreme"] >= 2
    assert all(row["candidate_id"] for row in primary)


def test_v19cl_is_coordinate_free_and_opens_no_gravity_or_lensing_outcome() -> None:
    report, _, _ = MODULE.build_report()
    assert not report["crossmatch"]["coordinate_matching_used"]
    audit = report["access_boundary_audit"]
    assert audit["clusters_selected_or_admitted"] == 0
    assert not audit["raw_lensing_coordinates_or_halo_maps_opened"]
    assert not audit["sigma_gravity_scored_or_fit"]
    assert not audit["action_or_constants_changed"]
    assert not audit["detailed_solar_optimization_performed"]


def test_v19cl_committed_report_matches_rebuild() -> None:
    expected, _, _ = MODULE.build_report()
    actual = json.loads((ROOT / "results" / "sigma_v19cl_registered_cluster_morphology_acquisition" / "report.json").read_text(encoding="utf-8"))
    assert actual == expected
