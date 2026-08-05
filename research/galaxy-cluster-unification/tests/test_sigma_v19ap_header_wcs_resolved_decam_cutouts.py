from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19ap_header_wcs_resolved_decam_cutouts.py"
CONFIG = ROOT / "configs" / "sigma_v19ap_header_wcs_resolved_decam_cutouts.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_config_and_parent_hashes() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(config, require_frozen=True)
    assert config["status"] == "frozen_before_v19ap_pixel_retrieval"
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


def test_resolved_plan_counts_routes_and_hash() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan_path = ROOT / config["frozen_resolved_plan"]["path"]
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == config[
        "frozen_resolved_plan"
    ]["sha256"]
    rows = read_rows(plan_path)
    assert len(rows) == 139
    assert sum(int(row["measurement_rows"]) for row in rows) == 1032
    assert len({row["group_id"] for row in rows}) == 139
    assert len({row["output_path"] for row in rows}) == 139
    methods = [row["retrieval_method"] for row in rows]
    assert methods.count("nsc_sia_group_cutout") == 102
    assert methods.count("archive_header_wcs_selected_hdu") == 37


def test_every_archive_group_has_a_reproducible_unique_header_wcs() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = read_rows(ROOT / config["frozen_resolved_plan"]["path"])
    groups = MODULE.AO.parent_groups(config)
    archive_rows = [
        row for row in rows if row["retrieval_method"] == "archive_header_wcs_selected_hdu"
    ]
    assert len({row["archive_header_payload_path"] for row in archive_rows}) == 22
    assert len({row["retrieval_url"] for row in archive_rows}) == 37
    for row in archive_rows:
        payload_path = ROOT / row["archive_header_payload_path"]
        assert hashlib.sha256(payload_path.read_bytes()).hexdigest() == row[
            "archive_header_payload_sha256"
        ]
        resolved = MODULE.resolve_unique_header(
            payload_path.read_bytes(),
            groups[(row["exposure"], row["sia_extension"])],
            float(config["archive_header_resolution"]["wcs_containment_tolerance_pixel"]),
        )
        assert int(row["fits_hdu_index"]) == resolved["fits_hdu_index"]
        assert row["header_extname"] == resolved["header_extname"]
        assert row["header_ccdnum"] == resolved["header_ccdnum"]
        expected = config["archive_header_resolution"]["retrieval_endpoint"].format(
            md5=row["source_md5"], fits_hdu_index=row["fits_hdu_index"]
        )
        assert row["retrieval_url"] == expected


def test_v19ao_failure_was_preserved_without_silent_selection() -> None:
    report = json.loads(
        (ROOT / "results" / "sigma_v19ao_resilient_decam_cutouts" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed_closed_before_any_group_was_accepted"
    assert report["counts"]["groups_accepted"] == 0
    assert report["gates"]["no_group_was_silently_dropped"] is True
    assert report["metadata_only_diagnosis"][
        "unique_header_containing_all_14_frozen_anchors"
    ] == 35
