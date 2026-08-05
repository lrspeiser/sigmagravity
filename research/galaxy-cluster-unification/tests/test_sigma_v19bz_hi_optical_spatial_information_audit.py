from __future__ import annotations

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


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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
