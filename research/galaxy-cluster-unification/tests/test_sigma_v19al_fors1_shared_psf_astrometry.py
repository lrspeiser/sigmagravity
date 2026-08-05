import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19al_fors1_shared_psf_astrometry.py"
CONFIG = ROOT / "configs" / "sigma_v19al_fors1_shared_psf_astrometry.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19al", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_runner_base_parent_and_science_hashes_match():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = MODULE.validate_config(CONFIG, config)
    assert hashes["runner"] == config["implementation"]["runner_sha256"]
    assert hashes["frozen_base_runner"] == config["implementation"]["frozen_base_runner_sha256"]


def test_free_and_shared_psf_fits_recover_synthetic_star():
    yy, xx = np.mgrid[0:41, 0:41]
    true_x, true_y, true_sigma = 20.37, 19.62, 1.8
    image = 100.0 + 0.2 * (xx - 20) - 0.1 * (yy - 20)
    image += 5000.0 * np.exp(-((xx - true_x) ** 2 + (yy - true_y) ** 2) / (2 * true_sigma**2))
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))["centroid"]
    preliminary = MODULE.fit_star(image, 20.0, 20.0, settings, shared_sigma=None)
    fixed = MODULE.fit_star(image, 20.0, 20.0, settings, shared_sigma=true_sigma)
    assert preliminary["accepted"] and fixed["accepted"]
    assert abs(preliminary["shared_psf_sigma_pixel"] - true_sigma) < 0.03
    assert abs(fixed["refined_x_pixel"] - true_x) < 0.03
    assert abs(fixed["refined_y_pixel"] - true_y) < 0.03


def test_image_fingerprint_is_value_stable_and_discriminating():
    first = np.arange(400, dtype=float).reshape(20, 20)
    second = first.copy()
    assert MODULE.image_fingerprint(first) == MODULE.image_fingerprint(second)
    second[0, 0] += 1
    assert MODULE.image_fingerprint(first) != MODULE.image_fingerprint(second)


def test_evidence_gates_match_v19ai():
    current = json.loads(CONFIG.read_text(encoding="utf-8"))
    previous = json.loads(
        (ROOT / "configs" / "sigma_v19ai_fors1_subpixel_astrometry.json").read_text(encoding="utf-8")
    )
    assert current["science_products"] == previous["science_products"]
    assert current["gates"] == previous["gates"]


def test_config_forbids_detection_science_inference_and_gravity_change():
    authorization = json.loads(CONFIG.read_text(encoding="utf-8"))["authorization"]
    assert not authorization["detect_or_rematch_sources"]
    assert not authorization["inspect_member_or_candidate_coordinates_or_cutouts"]
    assert not authorization["fit_science_photometry_or_member_deblending"]
    assert not authorization["infer_stellar_mass_or_current"]
    assert not authorization["change_gravity_physics_or_parameters"]
