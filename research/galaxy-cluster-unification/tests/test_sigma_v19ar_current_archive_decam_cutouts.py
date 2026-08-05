from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19ar_current_archive_decam_cutouts.py"
CONFIG = ROOT / "configs" / "sigma_v19ar_current_archive_decam_cutouts.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ar", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_config_and_parent_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(config, require_frozen=True)
    assert config["status"] == "frozen_before_v19ar_pixel_retrieval"
    assert len(hashes) == len(config["parent_artifacts"]) + 2
    assert not any(
        config["authorization"][name]
        for name in (
            "rank_or_select_exposures_by_science_values",
            "fit_or_compare_photometry",
            "choose_psf_or_deblend_model",
            "query_ambiguous_candidates",
            "infer_mass_or_current",
            "read_lensing_or_halo_payload",
            "change_gravity_physics_or_parameters",
            "open_holdout",
        )
    )


def test_resolved_plan_counts_hashes_and_routes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan_path = ROOT / config["frozen_resolved_plan"]["path"]
    assert digest(plan_path) == config["frozen_resolved_plan"]["sha256"]
    rows = read_rows(plan_path)
    assert len(rows) == 139
    assert sum(int(row["measurement_rows"]) for row in rows) == 1032
    assert len({row["group_id"] for row in rows}) == 139
    assert len({row["output_path"] for row in rows}) == 139
    assert len({row["retrieval_url"] for row in rows}) == 139
    assert len({row["exposure"] for row in rows}) == 82
    assert len({row["source_md5"] for row in rows}) == 82
    rules = [row["identity_selection_rule"] for row in rows]
    assert rules.count("unique_latest_same_observation_c4d_ooi_instcal") == 62
    assert rules.count("unique_latest_same_observation_assoc_ooi_instcal") == 77


def test_every_selected_identity_is_the_unique_latest_same_observation_product() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = read_rows(ROOT / config["frozen_resolved_plan"]["path"])
    by_exposure = {row["exposure"]: row for row in rows}
    for exposure, row in by_exposure.items():
        payload_path = ROOT / row["identity_payload_path"]
        assert digest(payload_path) == row["identity_payload_sha256"]
        query_name, _rule = MODULE.identity_query_name(exposure, row["sia_assoc_id"])
        candidates = [
            item
            for item in MODULE.AQ.vohdu_rows(payload_path.read_bytes())
            if MODULE.identity_matches(exposure, query_name, row["filter"], item)
        ]
        for item in candidates:
            MODULE.AQ.validate_candidate(item, row["filter"])
        assert len({str(item["original_filename"]) for item in candidates}) == 1
        newest = max(str(item["file_updated"]) for item in candidates)
        winners = [item for item in candidates if str(item["file_updated"]) == newest]
        assert len(winners) == 1
        assert winners[0]["md5sum"] == row["source_md5"]
        assert str(winners[0]["original_filename"]) == row["source_original_filename"]


def test_every_group_reproduces_one_unique_all_anchor_header_wcs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = read_rows(ROOT / config["frozen_resolved_plan"]["path"])
    groups = MODULE.AQ.AP.AO.parent_groups(config)
    payload_cache: dict[str, bytes] = {}
    for row in rows:
        path = ROOT / row["archive_header_payload_path"]
        if str(path) not in payload_cache:
            payload_cache[str(path)] = path.read_bytes()
        assert digest(path) == row["archive_header_payload_sha256"]
        resolved = MODULE.AQ.AP.resolve_unique_header(
            payload_cache[str(path)],
            groups[(row["exposure"], row["sia_extension"])],
            float(config["archive_header_resolution"]["wcs_containment_tolerance_pixel"]),
        )
        assert resolved["fits_hdu_index"] == int(row["fits_hdu_index"])
        assert resolved["header_extname"] == row["header_extname"]
        assert resolved["header_ccdnum"] == row["header_ccdnum"]


def test_all_legacy_v1_groups_change_to_current_processing_without_pixel_input() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    current = read_rows(ROOT / config["frozen_resolved_plan"]["path"])
    previous = read_rows(
        ROOT / "data" / "derived" / "sigma_v19aq_all_archive_decam_cutouts" / "resolved_plan.csv"
    )
    previous_by_group = {row["group_id"]: row for row in previous}
    changed = [
        row for row in current if row["source_md5"] != previous_by_group[row["group_id"]]["source_md5"]
    ]
    assert len(changed) == 25
    assert len({row["exposure"] for row in changed}) == 14
    assert all(row["exposure"].endswith("_v1") for row in changed)
    assert not (ROOT / config["outputs"]["cutout_directory"]).exists()
    report = json.loads(
        (ROOT / "results" / "sigma_v19aq_all_archive_decam_cutouts" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed_closed_on_corrupt_legacy_archive_product"
    assert report["gates"]["v19aq_passed"] is False
