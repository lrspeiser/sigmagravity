from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19df_macsj0018_component_current.json"
RUNNER = ROOT / "scripts" / "run_sigma_v19df_macsj0018_component_current.py"
REPORT = ROOT / "results" / "sigma_v19df_macsj0018_component_current" / "report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v19df", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_source_only_and_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    assert authorization["open_or_digitize_figure_pixels"] is False
    assert authorization["open_unreleased_ksz_or_noise_pixels"] is False
    assert authorization["use_lensing_halo_or_dark_matter_map"] is False
    assert authorization["fit_or_change_gravity_formula_or_constants"] is False
    assert authorization["derive_or_select_covariant_action"] is False
    assert authorization["open_validation_or_holdout_system"] is False
    assert config["ksz_gas_branch"]["raw_branch_authorized"] is False
    assert config["ksz_gas_branch"]["figure_digitization_forbidden"] is True


def test_runner_hash_is_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["implementation"]["runner_sha256"] == digest(RUNNER)


def test_coordinate_and_direction_helpers() -> None:
    runner = load_runner()
    assert np.isclose(runner.ra_hms_to_deg("00:18:33.4"), 4.639166666666667)
    assert np.isclose(runner.dec_dms_to_deg("+16:26:13"), 16.436944444444443)
    assert np.isclose(runner.axial_difference(289.0, 243.0), 46.0)
    assert np.isclose(runner.axial_difference(5.0, 175.0), 10.0)


def test_terminal_report_preserves_the_negative_source_result() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "macsj0018_public_component_current_not_admitted"
    assert report["catalog"]["counts"] == {
        "literature_all": 168,
        "keck_all": 117,
        "inside_box_after_internal_duplicate_removal": 161,
        "cross_table_matches": 5,
        "final_rows": 156,
    }
    assert report["catalog"]["source_group_counts"] == {"literature": 98, "keck": 58}
    assert report["gates"]["catalog_reproduces_156_rows"] is True
    assert report["gates"]["permutation_p_at_most_0p05"] is False
    assert report["gates"]["literature_keck_axial_difference_at_most_30_deg"] is False
    assert report["gates"]["bootstrap_95_direction_half_width_at_most_45_deg"] is False
    assert report["member_velocity_gradient_admitted"] is False
    assert report["analysis_grade_ksz_gas_products_publicly_available"] is False
    assert report["component_resolved_current_source_admitted"] is False
    assert report["figure_pixels_digitized"] is False
    assert report["ksz_or_noise_pixels_opened"] is False
    assert report["lensing_halo_or_dark_matter_map_opened"] is False
    assert report["gravity_formula_or_constant_fit"] is False
    assert report["covariant_action_selected_or_derived"] is False
    assert report["validation_or_holdout_opened"] is False


def test_catalog_and_cross_match_outputs_are_exact() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    catalog_path = ROOT / report["catalog"]["output"]
    match_path = ROOT / report["catalog"]["cross_matches"]
    with catalog_path.open(encoding="utf-8", newline="") as handle:
        catalog = list(csv.DictReader(handle))
    with match_path.open(encoding="utf-8", newline="") as handle:
        matches = list(csv.DictReader(handle))
    assert len(catalog) == 156
    assert len(matches) == 5
    assert len({row["object_id"] for row in catalog}) == 156
    assert all(row["source_group"] in {"literature", "keck"} for row in catalog)
    assert digest(catalog_path) == report["catalog"]["output_sha256"]
    assert digest(match_path) == report["catalog"]["cross_matches_sha256"]
