import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19ao_resilient_decam_cutouts.py"
CONFIG = ROOT / "configs" / "sigma_v19ao_resilient_decam_cutouts.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ao", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def load_plan():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with (ROOT / config["frozen_hybrid_plan"]["path"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        return config, list(csv.DictReader(handle))


def test_frozen_runner_base_parents_and_hybrid_plan_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(config, require_frozen=True)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]
    assert hashes["frozen_base_runner"] == config["implementation"]["frozen_base_runner_sha256"]
    for artifact in config["parent_artifacts"]:
        assert hashes[artifact["path"]] == artifact["sha256"]
    plan = ROOT / config["frozen_hybrid_plan"]["path"]
    assert MODULE.sha256(plan) == config["frozen_hybrid_plan"]["sha256"]


def test_hybrid_plan_preserves_every_group_and_membership():
    _config, rows = load_plan()
    assert len(rows) == 139
    assert len({row["group_id"] for row in rows}) == 139
    assert len({row["retrieval_url"] for row in rows}) == 139
    assert len({row["output_path"] for row in rows}) == 139
    assert sum(int(row["measurement_rows"]) for row in rows) == 1032
    methods = {name: sum(row["retrieval_method"] == name for row in rows) for name in {
        "nsc_sia_group_cutout", "archive_selected_hdu"
    }}
    assert methods == {"nsc_sia_group_cutout": 102, "archive_selected_hdu": 37}


def test_every_archive_identity_is_unique_exact_and_raw_payload_hashed():
    _config, rows = load_plan()
    archive = [row for row in rows if row["retrieval_method"] == "archive_selected_hdu"]
    assert len({row["source_md5"] for row in archive}) == 22
    assert len({(row["source_md5"], row["source_hdu_index"]) for row in archive}) == 37
    for row in archive:
        assert row["exposure"].endswith("_d2")
        assert row["retrieval_url"].endswith(f"?hdus=0,{row['source_hdu_index']}")
        payload = ROOT / row["identity_payload_path"]
        assert payload.is_file()
        assert MODULE.sha256(payload) == row["identity_payload_sha256"]


def test_source_route_is_identifier_based_and_science_selection_is_forbidden():
    config, rows = load_plan()
    for row in rows:
        expected = "archive_selected_hdu" if row["exposure"].endswith("_d2") else "nsc_sia_group_cutout"
        assert row["retrieval_method"] == expected
    authorization = config["authorization"]
    assert authorization["download_every_frozen_group"]
    assert authorization["open_pixels_for_fits_wcs_and_finite_value_integrity_only"]
    assert not authorization["rank_or_select_exposures"]
    assert not authorization["fit_or_compare_photometry"]
    assert not authorization["choose_psf_or_deblend_model"]
    assert not authorization["query_ambiguous_candidates"]
    assert not authorization["infer_mass_or_current"]
    assert not authorization["read_lensing_or_halo_payload"]
    assert not authorization["change_gravity_physics_or_parameters"]
