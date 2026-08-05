import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19ai_fors1_subpixel_astrometry.py"
CONFIG = ROOT / "configs" / "sigma_v19ai_fors1_subpixel_astrometry.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19ai", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_runner_parent_and_science_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(CONFIG, config)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]


def test_subpixel_centroid_recovers_synthetic_gaussian():
    yy, xx = np.mgrid[0:41, 0:41]
    true_x, true_y = 20.37, 19.62
    image = 100.0 + 5000.0 * np.exp(-((xx - true_x) ** 2 + (yy - true_y) ** 2) / (2 * 1.8**2))
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))["centroid"]
    result = MODULE.refine_centroid(image, 20.0, 20.0, settings)
    assert result["accepted"]
    assert abs(result["refined_x_pixel"] - true_x) < 0.03
    assert abs(result["refined_y_pixel"] - true_y) < 0.03


def test_nonfinite_aperture_is_rejected():
    image = np.ones((41, 41), dtype=float)
    image[20, 20] = np.nan
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))["centroid"]
    result = MODULE.refine_centroid(image, 20.0, 20.0, settings)
    assert not result["accepted"]
    assert result["rejection_reason"] == "nonfinite_aperture"


def test_config_forbids_detection_rematching_and_science_inference():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    authorization = config["authorization"]
    assert not authorization["detect_or_rematch_sources"]
    assert not authorization["inspect_member_or_candidate_coordinates_or_cutouts"]
    assert not authorization["fit_photometry_or_deblending"]
    assert not authorization["infer_stellar_mass_or_current"]
    assert not authorization["change_gravity_physics_or_parameters"]
