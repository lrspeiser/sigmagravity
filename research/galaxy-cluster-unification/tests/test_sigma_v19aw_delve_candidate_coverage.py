import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19aw_delve_candidate_coverage.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19aw_delve_candidate_coverage.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19aw", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_protocol_is_frozen_before_candidate_matching():
    config = json.loads(CONFIG.read_text())
    assert config["status"] == "frozen_before_full_field_acquisition_or_candidate_matching"
    assert config["source"]["table"] == "delve_dr3.coadd_objects"
    assert config["source"]["field_radius_deg"] == 0.07
    assert config["matching"]["radius_arcsec"] == 0.5
    assert config["gates"]["exact_field_rows"] == 2351
    assert config["gates"]["minimum_complete_candidate_fraction"] == 0.9
    assert not config["authorization"]["select_or_rank_counterparts"]
    assert not config["authorization"]["infer_mass_or_current"]


def test_signed_flux_coverage_keeps_negative_flux():
    config = json.loads(CONFIG.read_text())
    row = {}
    for band in "griz":
        row[f"gap_flux_{band}"] = "-2.5"
        row[f"gap_flux_err_{band}"] = "1.0"
        row[f"nepochs_{band}"] = "2"
    assert MODULE.finite_photometry(
        row,
        config["matching"]["required_photometry_bands"],
        config["coverage_observable"],
    )
    row["gap_flux_err_z"] = ""
    assert not MODULE.finite_photometry(
        row,
        config["matching"]["required_photometry_bands"],
        config["coverage_observable"],
    )


def test_matching_preserves_multiple_and_null_states():
    config = json.loads(CONFIG.read_text())
    field_rows = []
    for object_id, ra in (("1", 10.0), ("2", 10.00002)):
        row = {
            "coadd_object_id": object_id,
            "ra": str(ra),
            "dec": "20.0",
            "alphawin_j2000": str(ra),
            "deltawin_j2000": "20.0",
        }
        for band in "griz":
            row[f"gap_flux_{band}"] = "1.0"
            row[f"gap_flux_err_{band}"] = "0.2"
            row[f"nepochs_{band}"] = "1"
        field_rows.append(row)
    candidates = [
        {"candidate_id": "c1", "candidate_ra_deg": "10", "candidate_dec_deg": "20"},
        {"candidate_id": "c2", "candidate_ra_deg": "11", "candidate_dec_deg": "20"},
    ]
    matches, coverage = MODULE.match_candidates(config, field_rows, candidates)
    assert len(matches) == 2
    assert coverage[0]["matching_state"] == "multiple"
    assert coverage[0]["complete_signed_flux_griz_matches"] == 2
    assert coverage[1]["matching_state"] == "no_match"
    assert not coverage[1]["has_complete_signed_flux_griz_match"]


def test_adql_is_one_field_query_with_no_candidate_coordinates():
    config = json.loads(CONFIG.read_text())
    adql = MODULE.build_adql(config)
    assert "FROM delve_dr3.coadd_objects" in adql
    assert "q3c_radial_query" in adql
    assert "ORDER BY coadd_object_id" in adql
    assert "candidate" not in adql.lower()
