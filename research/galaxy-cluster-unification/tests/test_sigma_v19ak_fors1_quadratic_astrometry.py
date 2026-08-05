import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19ak_fors1_quadratic_astrometry.py"
CONFIG = ROOT / "configs" / "sigma_v19ak_fors1_quadratic_astrometry.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ak", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_runner_base_parent_and_science_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(CONFIG, config)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]
    assert hashes["frozen_base_runner"] == config["implementation"]["frozen_base_runner_sha256"]


def test_quadratic_centroid_recovers_synthetic_gaussian():
    yy, xx = np.mgrid[0:41, 0:41]
    true_x, true_y = 20.37, 19.62
    image = 100.0 + 5000.0 * np.exp(-((xx - true_x) ** 2 + (yy - true_y) ** 2) / (2 * 1.8**2))
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))["centroid"]
    result = MODULE.refine_centroid(image, 20.0, 20.0, settings)
    assert result["accepted"]
    assert abs(result["refined_x_pixel"] - true_x) < 0.08
    assert abs(result["refined_y_pixel"] - true_y) < 0.08


def test_nonconcave_peak_is_rejected():
    yy, xx = np.mgrid[0:41, 0:41]
    image = 100.0 + (xx - 20.0) ** 2 + (yy - 20.0) ** 2
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))["centroid"]
    result = MODULE.refine_centroid(image, 20.0, 20.0, settings)
    assert not result["accepted"]
    assert result["rejection_reason"] == "nonconcave_peak"


def test_evidence_gates_match_v19ai():
    current = json.loads(CONFIG.read_text(encoding="utf-8"))
    previous = json.loads(
        (ROOT / "configs" / "sigma_v19ai_fors1_subpixel_astrometry.json").read_text(encoding="utf-8")
    )
    assert current["science_products"] == previous["science_products"]
    assert current["gates"] == previous["gates"]
    for name in (
        "maximum_shift_from_v19ah_peak_pixel",
        "minimum_fwhm_pixel",
        "maximum_fwhm_pixel",
        "maximum_ellipticity",
    ):
        assert current["centroid"][name] == previous["centroid"][name]


def test_config_forbids_detection_rematching_and_science_inference():
    authorization = json.loads(CONFIG.read_text(encoding="utf-8"))["authorization"]
    assert not authorization["detect_or_rematch_sources"]
    assert not authorization["inspect_member_or_candidate_coordinates_or_cutouts"]
    assert not authorization["fit_photometry_or_deblending"]
    assert not authorization["infer_stellar_mass_or_current"]
    assert not authorization["change_gravity_physics_or_parameters"]
