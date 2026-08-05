from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sigma_v19bz_hi_optical_spatial_information_audit.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19bz_hi_optical_spatial_information_audit.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19bz", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPORT_PATH = ROOT / "results" / "sigma_v19bz_hi_optical_spatial_information_audit" / "report.json"
CANDIDATE_PATH = ROOT / "data" / "derived" / "sigma_v19bz_hi_optical_spatial_information_audit" / "candidate_spatial_scores.csv"
RELEASE_PATH = ROOT / "data" / "derived" / "sigma_v19bz_hi_optical_spatial_information_audit" / "release_information.csv"
CANDIDATE_SHA256 = "bc6dfb4cd0a30d72269a44aae20c356c982bf8bd14f0c5e1b6f686fd61adfe20"
RELEASE_SHA256 = "deb0185973d8d99948062bc8f4efecc5e0b2c5f33289c824a5bb70a4ce000cad"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_header(size: int = 21) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = size
    header["NAXIS2"] = size
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = (size + 1) / 2
    header["CRPIX2"] = (size + 1) / 2
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = -20.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["BMAJ"] = 3.0 / 3600.0
    header["BMIN"] = 3.0 / 3600.0
    return header


def test_v19bz_honestly_declares_post_source_exploration() -> None:
    cfg = config()
    honesty = cfg["honesty_boundary"]
    assert honesty["source_data_were_inspected_before_this_audit_was_frozen"]
    assert not honesty["gravity_or_kinematic_target_was_inspected"]
    assert not honesty["this_is_a_preregistered_theory_or_holdout_gate"]
    assert cfg["spatial_information_model"]["hard_counterpart_assignment"] == "forbidden"


def test_v19bz_localized_signal_ranks_near_candidate_first() -> None:
    header = synthetic_header()
    data = np.zeros((21, 21), dtype=np.float64)
    data[10, 10] = 10.0
    wcs = WCS(header).celestial
    near_ra, near_dec = wcs.pixel_to_world_values(10.0, 10.0)
    far_ra, far_dec = wcs.pixel_to_world_values(2.0, 2.0)
    result = MODULE.score_candidate_positions(
        data,
        header,
        np.asarray([near_ra, far_ra]),
        np.asarray([near_dec, far_dec]),
        [0.0, 0.5, 1.0, 2.0],
    )
    for values in result["likelihood_ratios"].values():
        assert values[0] > values[1]


def test_v19bz_uniform_signal_gives_equal_spatial_information() -> None:
    header = synthetic_header()
    data = np.ones((21, 21), dtype=np.float64)
    wcs = WCS(header).celestial
    ra, dec = wcs.pixel_to_world_values(np.asarray([8.0, 12.0]), np.asarray([10.0, 10.0]))
    result = MODULE.score_candidate_positions(
        data,
        header,
        np.asarray(ra),
        np.asarray(dec),
        [0.0],
    )
    assert np.allclose(result["likelihood_ratios"][0.0], [1.0, 1.0])


def test_v19bz_keeps_every_target_and_assignment_channel_sealed() -> None:
    boundary = config()["access_boundary"]
    assert not boundary["skymapper_extendedness_used_as_weight"]
    assert not boundary["hard_counterpart_selected"]
    assert not boundary["candidate_removed"]
    assert not boundary["wallaby_kinematic_table_row_read"]
    assert not boundary["rotation_speed_or_velocity_field_read"]
    assert not boundary["gravity_formula_residual_or_halo_result_read"]
    assert not boundary["development_validation_holdout_split_selected"]
    assert not boundary["gravity_action_or_constant_changed"]
    assert not boundary["lensing_payload_opened"]
    assert not boundary["solar_system_optimization_performed"]


def test_v19bz_real_audit_finds_spatial_information_insufficient() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert all(report["gate_results"].values())
    assert report["decision"] == "source_spatial_information_insufficient_for_hard_counterpart"
    assert not report["information_sufficient_for_hard_counterpart"]
    audit = report["information_audit"]
    assert audit["robust_margin_ge_3"] == 3
    assert np.isclose(audit["robust_margin_fraction"], 3 / 711)
    assert audit["same_top_all_kernel_branches"] == 492
    assert audit["duplicate_release_names"] == 119
    assert audit["duplicate_top_stable_all_releases_and_kernels"] == 82
    assert audit["primary_margin_grid_counts"] == {
        "1.5": 119,
        "2": 61,
        "3": 24,
        "5": 10,
        "10": 3,
    }
    assert audit["field_summary"]["Norma"]["robust_margin_ge_3"] == 0


def test_v19bz_outputs_are_exact_and_retain_all_candidate_release_pairs() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert sha256(CANDIDATE_PATH) == CANDIDATE_SHA256
    assert sha256(RELEASE_PATH) == RELEASE_SHA256
    assert report["outputs"]["candidate_scores"]["sha256"] == CANDIDATE_SHA256
    assert report["outputs"]["release_information"]["sha256"] == RELEASE_SHA256
    with CANDIDATE_PATH.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        fields = list(reader.fieldnames)
        rows = list(reader)
    assert len(rows) == 18_550
    assert len({(row["source_row_id"], row["object_id"]) for row in rows}) == 18_550
    assert not any(
        token in field.lower()
        for field in fields
        for token in ("selected", "posterior", "counterpart_probability", "gravity", "velocity")
    )
    with RELEASE_PATH.open(encoding="utf-8", newline="") as stream:
        release_rows = list(csv.DictReader(stream))
    assert len(release_rows) == 711
