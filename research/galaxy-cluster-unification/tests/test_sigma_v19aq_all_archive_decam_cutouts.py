from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19aq_all_archive_decam_cutouts.py"
CONFIG = ROOT / "configs" / "sigma_v19aq_all_archive_decam_cutouts.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19aq", SCRIPT)
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
    assert config["status"] == "frozen_before_v19aq_pixel_retrieval"
    assert len(hashes) == len(config["parent_artifacts"]) + 2
    assert not any(
        config["authorization"][name]
        for name in (
            "rank_or_select_exposures",
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
    assert rules.count("exact_frozen_archive_basename") == 25
    assert rules.count("stale_c4d_prefix_unique_latest_instcal") == 37
    assert rules.count("frozen_assoc_id_unique_latest_ooi_instcal") == 77


def test_every_identity_and_header_payload_is_hash_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = read_rows(ROOT / config["frozen_resolved_plan"]["path"])
    for row in rows:
        exact = ROOT / row["exact_identity_payload_path"]
        header = ROOT / row["archive_header_payload_path"]
        assert digest(exact) == row["exact_identity_payload_sha256"]
        assert digest(header) == row["archive_header_payload_sha256"]
        if row["fallback_identity_payload_path"]:
            fallback = ROOT / row["fallback_identity_payload_path"]
            assert digest(fallback) == row["fallback_identity_payload_sha256"]
        else:
            assert row["identity_selection_rule"] == "exact_frozen_archive_basename"
        expected = config["archive_header_resolution"]["retrieval_endpoint"].format(
            md5=row["source_md5"], fits_hdu_index=row["fits_hdu_index"]
        )
        assert row["retrieval_url"] == expected


def test_every_group_reproduces_one_unique_all_anchor_header_wcs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = read_rows(ROOT / config["frozen_resolved_plan"]["path"])
    groups = MODULE.AP.AO.parent_groups(config)
    payload_cache: dict[str, bytes] = {}
    for row in rows:
        path = ROOT / row["archive_header_payload_path"]
        payload = payload_cache.setdefault(str(path), path.read_bytes())
        resolved = MODULE.AP.resolve_unique_header(
            payload,
            groups[(row["exposure"], row["sia_extension"])],
            float(config["archive_header_resolution"]["wcs_containment_tolerance_pixel"]),
        )
        assert resolved["fits_hdu_index"] == int(row["fits_hdu_index"])
        assert resolved["header_extname"] == row["header_extname"]
        assert resolved["header_ccdnum"] == row["header_ccdnum"]


def test_no_v19aq_pixels_existed_at_freeze_and_v19ap_failure_is_preserved() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert not (ROOT / config["outputs"]["cutout_directory"]).exists()
    report = json.loads(
        (
            ROOT
            / "results"
            / "sigma_v19ap_header_wcs_resolved_decam_cutouts"
            / "report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "failed_closed_on_deterministic_header_only_sia_response"
    assert report["deterministic_failure"]["response_bytes"] == 28800
    assert report["gates"]["v19ap_passed"] is False
