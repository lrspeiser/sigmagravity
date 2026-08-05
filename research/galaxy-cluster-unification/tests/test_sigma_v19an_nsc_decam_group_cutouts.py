import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acquire_sigma_v19an_nsc_decam_group_cutouts.py"
CONFIG = ROOT / "configs" / "sigma_v19an_nsc_decam_group_cutouts.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19an", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_runner_parents_and_group_plan_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(CONFIG, config, require_frozen=True)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]
    for artifact in config["parent_artifacts"]:
        assert hashes[artifact["path"]] == artifact["sha256"]
    plan_path = ROOT / config["frozen_group_plan"]["path"]
    assert MODULE.sha256(plan_path) == config["frozen_group_plan"]["sha256"]


def test_group_plan_is_deterministic_complete_and_within_bounds():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan, _groups = MODULE.build_plan(config)
    plan_path = ROOT / config["frozen_group_plan"]["path"]
    MODULE.assert_plan_matches(plan_path, plan)
    assert len(plan) == 139
    assert len({row["exposure"] for row in plan}) == 82
    assert len({row["access_url"] for row in plan}) == 139
    assert sum(int(row["measurement_rows"]) for row in plan) == 1032
    assert max(float(row["size_ra_deg"]) for row in plan) <= 0.25
    assert max(float(row["size_dec_deg"]) for row in plan) <= 0.15


def test_frozen_plan_csv_has_unique_paths_and_preserves_every_group():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with (ROOT / config["frozen_group_plan"]["path"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 139
    assert len({row["group_id"] for row in rows}) == 139
    assert len({row["output_path"] for row in rows}) == 139
    assert all(row["access_url"].startswith("https://datalab.noirlab.edu/svc/cutout?") for row in rows)


def test_config_authorizes_only_complete_structural_acquisition():
    authorization = json.loads(CONFIG.read_text(encoding="utf-8"))["authorization"]
    assert authorization["download_every_group_cutout"]
    assert authorization["open_pixels_for_fits_wcs_and_finite_value_integrity_only"]
    assert not authorization["rank_or_select_exposures"]
    assert not authorization["fit_or_compare_photometry"]
    assert not authorization["choose_psf_or_deblend_model"]
    assert not authorization["query_ambiguous_candidates"]
    assert not authorization["infer_mass_or_current"]
    assert not authorization["read_lensing_or_halo_payload"]
    assert not authorization["change_gravity_physics_or_parameters"]
